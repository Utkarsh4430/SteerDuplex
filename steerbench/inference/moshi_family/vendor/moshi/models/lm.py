# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Modified for SteerBench inference (see ../../NOTICE.md): PersonaPlex system/voice-prompt
# LMGen and dep_q=16 loader merged into the Kyutai moshi 0.2.13 sources; unused experimental decoding hooks removed.

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import typing as tp
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from os.path import splitext
from typing import Optional, Union, List, Tuple, Iterator

import numpy as np
import sphn
import torch
from torch import nn

from ..conditioners import ConditionFuser, ConditionProvider, ConditionTensors
from ..modules.streaming import State, StreamingContainer, StreamingModule
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..utils.compile import CUDAGraphed
from ..utils.quantize import replace_linear_with_qlinear
from ..utils.sampling import sample_token
from .lm_utils import ScaledEmbedding, _delay_sequence, _init_layer, _undelay_sequence

logger = logging.getLogger(__name__)


def scatter_with_mask_(tensor: torch.Tensor, dim: int,
                       index: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> None:
    """Scatter but skipping the updates that are masked."""
    old_value = tensor.gather(dim, index)
    value = torch.where(mask, value, old_value)
    tensor.scatter_(dim, index, value)


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]
    text_logits: torch.Tensor  # [B, 1, T, text_card]
    text_mask: torch.Tensor  # [B, 1, T]


class LMModel(StreamingContainer):
    """Transformer-based language model on multiple streams of codes.

    Args:
        n_q (int): Number of parallel streams to model as input.
        dep_q (int): Number of parallel streams to model in the depformer.
        card (int): Cardinality, vocabulary size.
        text_card (int): Cardinality of the text vocabulary.
        text_card_out (int or None): Cardinality of output text, if different from the input.
        demux_second_text_stream: (bool): Whether two text streams are muxed together with a cartesian product.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_emb (bool): Whether to normalize embeddings.
        bias_proj (bool): Use bias for output projections.
        depformer_*: params used for the Depformer Transformer, all the other will be shared.
        depformer_multi_linear (bool): if True, uses one linear layer per codebook to project the
            output of the main transformer to the Depformer latent space.
        depformer_dim_feedforward (int| list[int]| None): If None, defaults to hidden_scale * depformer_dim.
        depformer_weights_per_step_schedule (list[int] | None): mapping `CODEBOOK_INDEX -> WEIGHT_INDEX`, allowing
        depformer_low_rank_embeddings (int | None): if provided, uses low rank embeddings, with a linear
        existing_text_padding_id (int): token to use for the padding.
        same_initial (bool): if True, uses the same initial tokens for both text and audio mode.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(
        self,
        delays: tp.List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        text_card_out: int | None = None,
        demux_second_text_stream: bool = False,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_weights_per_step_schedule: list[int] | None = None,
        depformer_low_rank_embeddings: int | None = None,
        depformer_pos_emb: str = "sin",
        depformer_norm: str | None = None,
        existing_text_padding_id: int = 3,
        existing_text_end_padding_id: int = 0,
        extra_heads_num_heads: int = 0,
        extra_heads_dim: int = 6,
        context: tp.Optional[int] = None,
        causal: bool = True,
        condition_provider: tp.Optional[ConditionProvider] = None,
        fuser: tp.Optional[ConditionFuser] = None,
        quantize: bool = False,
        device=None,
        dtype=None,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        text_card_out = text_card if text_card_out is None else text_card_out
        assert len(delays) == self.num_codebooks, f"expected {self.num_codebooks} delays, got {len(delays)}."
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.existing_text_end_padding_id = existing_text_end_padding_id
        self.context = context
        self.depformer_weights_per_step_schedule = depformer_weights_per_step_schedule
        if depformer_weights_per_step_schedule is not None:
            assert len(depformer_weights_per_step_schedule) == dep_q
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )
        self.emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]
        )
        # Unlike for audio, here we authorize the model to output the special token.
        self.text_emb = EmbeddingFactory(text_card + 1, dim, demux_second_stream=demux_second_text_stream)

        self.text_linear = nn.Linear(dim, text_card_out, bias=bias_proj)
        depformer_prefix = "depformer_"
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            quantize=quantize,
            context=context,
            causal=causal,
            checkpointing=gradient_checkpointing,
            **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        self.depformer_multi_linear = depformer_multi_linear
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        kwargs_dep["cross_attention"] = False
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            # One linear layer per codebook to project different informations from the main model.
            num_in = dep_q
            if depformer_weights_per_step_schedule:
                num_in = max(depformer_weights_per_step_schedule) + 1
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False) for _ in range(num_in)]
            )
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False)]
            )
        EmbeddingFactory = partial(EmbeddingFactory, low_rank=depformer_low_rank_embeddings)
        if dep_q > 0:
            # Only using up to dep_q - 1 because the last codebook is never an input to Depformer.
            self.depformer_emb = nn.ModuleList(
                [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
            )
            self.depformer_text_emb = EmbeddingFactory(
                text_card + 1,
                depformer_dim,
                demux_second_stream=demux_second_text_stream,
            )
            if depformer_norm is None:
                self.depformer_norms = nn.ModuleList([nn.Identity() for _ in range(dep_q)])
            else:
                self.depformer_norms = nn.ModuleList(
                    [create_norm_fn(depformer_norm, depformer_dim) for _ in range(dep_q)])
            if depformer_dim_feedforward is None:
                depformer_dim_feedforward = int(hidden_scale * depformer_dim)
            self.depformer = StreamingTransformer(
                d_model=depformer_dim,
                dim_feedforward=depformer_dim_feedforward,
                norm=norm,
                weights_per_step_schedule=depformer_weights_per_step_schedule,
                causal=causal,
                quantize=quantize,
                checkpointing=gradient_checkpointing,
                device=device,
                dtype=dtype,
                **kwargs_dep,
            )
            # Depformer follow its own cycle of streaming entirely contained in one time step
            # and should not follow the streaming of the steps dimensions.
            self.depformer.set_streaming_detached(True)
        else:  # No-Depformer --- e.g., an ASR model
            self.depformer_emb = None
            self.depformer_text_emb = None
            self.depformer = None

        self.extra_heads = nn.ModuleList(
            [nn.Linear(dim, extra_heads_dim, bias=False) for _ in range(extra_heads_num_heads)]
        )

        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )
        self.to(device=device, dtype=dtype)
        # We always keep the condition provider as float32.
        self.condition_provider = condition_provider
        self.fuser = fuser
        if self.condition_provider is not None:
            self.condition_provider.to(device=device)
        if self.fuser is not None:
            self.fuser.to(device=device)
        self._init_weights()
        if quantize:
            replace_linear_with_qlinear(self)

    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        """Token id for the start of sequence (text)."""
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        """Token id for text padding."""
        return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """Token id for optionally marking the last padding step for a word."""
        return self.existing_text_end_padding_id

    @property
    def zero_token_id(self) -> int:
        """Special value in the input tokens, indicating that no sampling should
        happen for that value, and no input should be given to the model."""
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2

    @property
    def device(self) -> torch.device:
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def dtype(self) -> torch.dtype:
        first_param = next(iter(self.text_emb.parameters()))
        return first_param.dtype

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        """From which channel does audio start? Always 1 in practice.

        The state has shape [B, text_channels + audio_channels, T] where text_channels=1
        in practice, but in the future we might want to support >1.
        """
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        # Returns the initial token that will be fed to the model to predict the very first timestep.
        # The output shape will be [B, K, 1].
        device = next(iter(self.parameters())).device
        zero = torch.full(
            [1, 1, 1], self.zero_token_id, device=device, dtype=torch.long
        )
        special = torch.full_like(zero, self.initial_token_id)

        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special
        text_token = text_special
        audio_token = audio_token.expand(-1, self.num_audio_codebooks, -1)
        token = torch.cat([text_token, audio_token], dim=1)
        return token

    def forward(
            self, codes: torch.Tensor,
            condition_tensors: tp.Optional[ConditionTensors] = None) -> LMOutput:
        """Given an input tensor of codes [B, K, T] and list of conditions, returns the logits
        along with masks indicating the valid positions at which to compute the loss.
        The logits time steps are aligned with those in the input `code`.
        Should only be used for training, not inference (use `LMGen` for that).

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps. When text is supported,
                the first 'codebook' corresponds to the text, and the remaining codebooks are for the  audio.
            condition_tensors (dict[str, ConditionType], optional): pre-computed conditioning tensors.
        Returns:
            LMOutput: Language model outputs, containing either text or audio logits, or both.
                logits (torch.Tensor, or None) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor, or None) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
                text_logits (torch.Tensor, or None) of shape [B, 1, T, text_card].
                text_mask (torch.Tensor, or None) of shape [B, 1, T], mask over the valid positions for the text.
        """
        B, K, T = codes.shape
        assert K == self.num_codebooks, (K, self.num_codebooks)
        # Delaying codes and removing the last time step that will never be an input.
        initial = self._get_initial_token().expand(B, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial)
        # Inserting the empty tokens for the first time step.
        delayed_codes = torch.cat([initial, delayed_codes], dim=2)

        sum_condition: torch.Tensor | None = None
        cross_attention_src: torch.Tensor | None = None
        if condition_tensors is None:
            assert self.fuser is None
        else:
            assert self.fuser is not None
            sum_condition = self.fuser.get_sum(condition_tensors)
            cross_attention_src = self.fuser.get_cross(condition_tensors)

        transformer_out, text_logits = self.forward_text(delayed_codes[:, :, :-1], sum_condition, cross_attention_src)
        assert transformer_out.shape[0] == delayed_codes.shape[0]
        assert transformer_out.shape[1] == delayed_codes.shape[2] - 1
        logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)

        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens. We will with NaN values invalid positions
        # to ensure they properly handled.
        logits, logits_mask = _undelay_sequence(
            self.delays[self.audio_offset:self.audio_offset + self.dep_q],
            logits, fill_value=float('NaN'))
        logits_mask &= (codes[:, self.audio_offset: self.audio_offset + self.dep_q] != self.zero_token_id)
        text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float('NaN'))
        text_logits_mask &= (codes[:, :1] != self.zero_token_id)
        return LMOutput(logits, logits_mask, text_logits, text_logits_mask)

    def embed_codes(self, sequence: torch.Tensor) -> torch.Tensor:
        """Embed input codes into a single vector per timestep (no transformer)."""
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks."
        input_sequence = sequence
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](
                input_sequence[:, cb_index + self.audio_offset]
            )
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self.text_emb(input_sequence[:, 0])
        input_ = text_emb if input_ is None else input_ + text_emb
        return input_

    def forward_codes(
        self,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed codes and run through transformer (no conditions). Used by LMGen."""
        return self.forward_embeddings(self.embed_codes(sequence))

    def forward_embeddings(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run pre-computed embeddings through transformer. Used by LMGen for cached voice prompt replay."""
        transformer_out = self.transformer(input_)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = self.text_linear(transformer_out)
        text_logits = text_logits[:, None]
        return transformer_out, text_logits

    def forward_text(
        self,
        sequence: torch.Tensor, sum_condition: torch.Tensor | None = None,
        cross_attention_src: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks."
        input_sequence = sequence
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](
                input_sequence[:, cb_index + self.audio_offset]
            )
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self.text_emb(input_sequence[:, 0])

        input_ = text_emb if input_ is None else input_ + text_emb
        if sum_condition is not None:
            input_ = input_ + sum_condition.to(input_)
        if cross_attention_src is not None:
            cross_attention_src = cross_attention_src.to(input_)
        transformer_out = self.transformer(input_, cross_attention_src=cross_attention_src)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = self.text_linear(transformer_out)
        text_logits = text_logits[:, None]
        return transformer_out, text_logits

    def forward_depformer_training(
        self,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        assert self.depformer_text_emb
        assert self.depformer_emb
        assert self.depformer

        B, K, T = sequence.shape
        Ka = self.dep_q
        assert (
            K == self.num_codebooks
        ), f"Codebooks for Depformer training should be passed all at once, got {K}."
        depformer_inputs = []
        for cb_index in range(Ka):
            if self.depformer_multi_linear:
                linear_index = cb_index
                if self.depformer_weights_per_step_schedule is not None:
                    linear_index = self.depformer_weights_per_step_schedule[cb_index]
                transformer_in = self.depformer_in[linear_index](transformer_out)
            else:
                transformer_in = self.depformer_in[0](transformer_out)
            if cb_index == 0:
                token_in = self.depformer_text_emb(sequence[:, 0])
            else:
                token_in = self.depformer_emb[cb_index - 1](sequence[:, cb_index + self.audio_offset - 1])
            depformer_inputs.append(token_in + transformer_in)
        depformer_input = torch.stack(depformer_inputs, 2)
        # depformer_input is [B, T, K, depformer_dim], reshaping to [B * T, K, D]
        depformer_input = depformer_input.view(B * T, Ka, -1)
        depformer_output = self.depformer(depformer_input)
        all_logits = []
        for cb_index in range(Ka):
            logits = self.linears[cb_index](self.depformer_norms[cb_index](depformer_output[:, cb_index]))
            all_logits.append(logits.view(B, T, -1))
        logits = torch.stack(all_logits, 1)
        assert logits.dim() == 4, logits.shape  # [B, Ka, T, card]
        return logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        assert self.depformer_text_emb is not None
        assert self.depformer_emb is not None
        assert self.depformer is not None
        B, K, S = sequence.shape
        assert (
            K == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        assert (
            S == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        assert (
            transformer_out.shape[1] == 1
        ), "Transformer out should be a for a single step."
        last_token_input: tp.Optional[torch.Tensor] = None
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            in_index = depformer_cb_index
            if self.depformer_weights_per_step_schedule is not None:
                in_index = self.depformer_weights_per_step_schedule[in_index]
            depformer_input = self.depformer_in[in_index](depformer_input)
        else:
            depformer_input = self.depformer_in[0](depformer_input)
        if depformer_cb_index == 0:
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        assert last_token_input is not None
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1
        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](self.depformer_norms[depformer_cb_index](dep_output))
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits

    def _init_weights(self):
        """Initialization of the transformer module weights.
        Mostly truncated gaussian, with `std = 1 / sqrt(dim_in)`.
        Embeddings are also initialized with `1 / sqrt(dim)` rather than `1`.
        Some layers are not going to be properly initialized:
            - in_proj in MHA.
            - depth transformer layers.
        This is to match how our models were trained so far.
        """

        for emb_layer in self.emb:
            _init_layer(emb_layer)
        if self.depformer_emb is not None:
            for emb_layer in self.depformer_emb:
                _init_layer(emb_layer)
        _init_layer(self.text_emb)
        if self.depformer_text_emb is not None:
            _init_layer(self.depformer_text_emb)
        _init_layer(self.text_linear)

        for tr_layer in self.transformer.layers:
            tr_layer.apply(_init_layer)

        for linear in self.linears:
            _init_layer(linear)


# ---------------------------------------------------------------------------
# PersonaPlex constants and helper functions (from NVIDIA PersonaPlex, MIT License)
# ---------------------------------------------------------------------------

AUDIO_TOKENS_PER_STREAM = 8
FRAME_RATE_HZ = 12.5
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS    = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)


def create_sinewave(duration: float, sample_rate: int) -> np.ndarray:
    """Return a 440 Hz 'silent' sinewave of the given duration."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = 0.5
    return amplitude * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


def normalize_audio(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalize **mono** audio to a target LUFS level."""
    import pyloudnorm as pyln
    if wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav[0]
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    return pyln.normalize.loudness(wav, loudness, target_lufs)


def load_audio(filepath: str, sample_rate: int):
    """Load and resample audio from a file."""
    sample_pcm, sample_sr = sphn.read(filepath)
    sample_pcm = sphn.resample(
        sample_pcm, src_sample_rate=sample_sr, dst_sample_rate=sample_rate
    )
    return sample_pcm


def _iterate_audio(sample_pcm, sample_interval_size, max_len=sys.maxsize, pad=True):
    cnt = 0
    while sample_pcm.shape[-1] > 0 and cnt < max_len:
        sample = sample_pcm[:, :sample_interval_size]
        sample_pcm = sample_pcm[:, sample_interval_size:]
        if sample_pcm.shape[-1] == 0 and pad:
            sample = np.concatenate(
                [
                    sample,
                    np.zeros(
                        (
                            sample.shape[0],
                            sample_interval_size - sample.shape[-1],
                        )
                    ),
                ],
                axis=1,
            )
        cnt += 1
        yield sample[0:1]


def encode_from_sphn(mimi, samples, max_batch=sys.maxsize):
    """Encode audio samples using Mimi codec."""
    device = next(mimi.parameters()).device
    current_batch = []
    done_flag = False
    max_batch = 1

    while True:
        try:
            sample = next(samples)
            tensor = torch.tensor(sample, dtype=torch.float32, device=device)
            tensor = tensor.unsqueeze(0)
            current_batch.append(tensor)
        except StopIteration:
            done_flag = True

        if (not done_flag) and len(current_batch) < max_batch:
            continue
        if not current_batch:
            break

        batch = torch.cat(current_batch, dim=0)
        encoded = mimi.encode(batch)
        separated = torch.unbind(encoded, dim=0)
        reshaped = [x.unsqueeze(0) for x in separated]
        detached = [x.detach().clone() for x in reshaped]

        current_batch = []
        yield from detached

        if done_flag:
            break


@torch.no_grad()
def create_loss_report(
    state_cache: torch.Tensor,
    lm_model: LMModel,
    text_logits: torch.Tensor,
    audio_logits: torch.Tensor,
    target: torch.Tensor,
    sampled_text_token: torch.Tensor,
    sampled_audio_tokens: torch.Tensor,
    target_position: int,
):
    """Create a loss report for inference-time loss monitoring."""
    report = {}
    B = state_cache.shape[0]
    report["losses"] = torch.zeros(B, lm_model.dep_q + 1, device=state_cache.device)
    text_target = target[:, 0, 0]
    text_loss = torch.nn.functional.cross_entropy(
        text_logits[:, 0, 0],
        text_target,
        reduction="none",
        ignore_index=-100,
    )
    report["losses"][:, 0] = text_loss
    for k in range(lm_model.dep_q):
        audio_target = target[:, lm_model.audio_offset + k, 0]
        audio_loss = torch.nn.functional.cross_entropy(
            audio_logits[:, k],
            audio_target,
            reduction="none",
            ignore_index=-100,
        )
        report["losses"][:, k + 1] = audio_loss
    return report


# ---------------------------------------------------------------------------
# PersonaPlex LMGen — inference with system prompt injection
# (from NVIDIA PersonaPlex, MIT License)
# ---------------------------------------------------------------------------

@dataclass
class _LMGenState:
    cache: torch.Tensor
    provided: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_embeddings: CUDAGraphed
    graphed_depth: CUDAGraphed
    text_action_logprobs: torch.Tensor
    offset: int = 0
    _set_exec_mask_graphed: CUDAGraphed | None = None
    last_emitted_codes: torch.Tensor | None = None
    last_emitted_provided: torch.Tensor | None = None
    last_emitted_text_logprob: torch.Tensor | None = None
    last_emitted_timestep: int | None = None

    @property
    def batch_size(self) -> int:
        return self.cache.shape[0]

    @property
    def device(self) -> torch.device:
        return self.cache.device

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def reset(self, reset_mask: torch.Tensor | None = None) -> None:
        self.offset = 0
        self.provided.zero_()
        self.text_action_logprobs.fill_(float("nan"))
        self.last_emitted_codes = None
        self.last_emitted_provided = None
        self.last_emitted_text_logprob = None
        self.last_emitted_timestep = None

    def set_exec_mask(self, exec_mask: torch.Tensor) -> None:
        # LMGen advances a shared circular cache. Current callers use a single
        # synchronized stream, so there is no per-sample mask to maintain here.
        pass

class LMGen(StreamingModule[_LMGenState]):
    def __init__(
        self,
        lm_model: LMModel,
        device: str | torch.device,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        check: bool = False,
        report_loss: bool = False,
        return_logits: bool = False,
        audio_silence_frame_cnt: int = 1,
        text_prompt_tokens: Optional[list[int]] = None,
        save_voice_prompt_embeddings: bool = False,
        sample_rate: int = 32000,
        frame_rate: float = FRAME_RATE_HZ,
    ):
        assert not lm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = lm_model
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.text_prompt_tokens = text_prompt_tokens
        self.audio_silence_frame_cnt = audio_silence_frame_cnt
        self.voice_prompt = None
        self.zero_text_code = 3
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._frame_size = int(self._sample_rate / self._frame_rate)
        self._zero_frame = torch.zeros(1, 1, self._frame_size, device=device)
        duration = self._frame_size / self._sample_rate
        sine = create_sinewave(duration, self._sample_rate)
        self._sine_frame = torch.tensor(sine, device=device).unsqueeze(0).unsqueeze(0)
        self.check = check
        self.report_loss = report_loss
        if report_loss:
            return_logits = True
        self.return_logits = return_logits
        self.max_delay = max(lm_model.delays)
        self.delays_cuda = torch.tensor(
            lm_model.delays, device=lm_model.device, dtype=torch.long
        )
        self.save_voice_prompt_embeddings = save_voice_prompt_embeddings
        # Callers that reuse one fixed voice prompt many times can keep the
        # exact embeddings and KV cache in memory instead of re-running Mimi.
        self.cache_voice_prompt_embeddings_in_memory = False
        self.voice_prompt_audio: Optional[torch.Tensor] = None
        self.voice_prompt_cache: Optional[torch.Tensor] = None
        self.voice_prompt_embeddings: Optional[torch.Tensor] = None

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 3),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        )
        provided = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 3),
            False,
            device=lm_model.device,
            dtype=torch.bool
        )
        text_action_logprobs = torch.full(
            (batch_size, self.max_delay + 3),
            float("nan"),
            device=lm_model.device,
            dtype=torch.float32,
        )

        disable = lm_model.device.type != 'cuda'
        graphed_main = CUDAGraphed(lm_model.forward_codes, disable=disable)
        graphed_embeddings = CUDAGraphed(lm_model.forward_embeddings, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        return _LMGenState(
            cache,
            provided,
            initial,
            graphed_main,
            graphed_embeddings,
            graphed_depth,
            text_action_logprobs,
        )

    @torch.no_grad()
    def prepare_step_input(self,
                           input_tokens: torch.Tensor = None,
                           moshi_tokens: torch.Tensor = None,
                           text_token: torch.Tensor = None,
                           provided_mask: torch.Tensor | None = None,
                           ):
        # provided_mask: optional [B] bool tensor for RAGGED per-row batching.
        # When None (default), the moshi/text target writes mark EVERY row
        # "provided" (forced) exactly as before -> byte-identical to the legacy
        # path. When given, only rows where provided_mask[i] is True are forced;
        # rows where it is False have provided=False for the moshi/text streams
        # so process_transformer_output samples them (per-row force-vs-sample).
        # The USER input stream is ALWAYS forced (it is the conditioning signal),
        # so provided_mask never gates the input_tokens block below.
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        needed_tokens = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        CT = state.cache.shape[2]

        if input_tokens is not None:
            assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = input_tokens.shape
            assert S == 1, "Only support being given steps one by one."
            assert (
                Ki == needed_tokens
            ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

            for q_other in range(input_tokens.shape[1]):
                k = AUDIO_TOKENS_PER_STREAM + 1 + q_other
                delay = lm_model.delays[k]
                write_position = (state.offset + delay) % CT
                state.cache[:, k, write_position : write_position + 1] = input_tokens[:, q_other]
                state.provided[:, k, write_position : write_position + 1] = True

        if moshi_tokens is not None:
            assert moshi_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = moshi_tokens.shape
            assert S == 1, "Only support being given steps one by one."
            assert (
                Ki == needed_tokens
            ), f"We expect {needed_tokens} tokens from the moshi stream, got {Ki}."

            for q_moshi in range(moshi_tokens.shape[1]):
                k = 1 + q_moshi
                delay = lm_model.delays[k]
                write_position = (state.offset + delay) % CT
                state.cache[:, k, write_position : write_position + 1] = moshi_tokens[:, q_moshi]
                if provided_mask is None:
                    state.provided[:, k, write_position : write_position + 1] = True
                else:
                    state.provided[:, k, write_position : write_position + 1] = provided_mask.view(-1, 1)

        if text_token is not None:
            write_position = (state.offset + lm_model.delays[0]) % CT
            state.cache[:, 0, write_position] = text_token
            if provided_mask is None:
                state.provided[:, 0, write_position] = True
            else:
                state.provided[:, 0, write_position] = provided_mask

        for k, delay in enumerate(lm_model.delays):
            if state.offset <= delay:
                state.cache[:, k, state.offset % CT] = state.initial[:, k, 0]
                state.provided[:, k, state.offset % CT] = True

        if state.offset == 0:
            state.cache[:, :, 0] = state.initial[:, :, 0]
            state.offset += 1
            return None

        model_input_position = (state.offset - 1) % CT
        target_position = state.offset % CT
        input_ = state.cache[:, :, model_input_position : model_input_position + 1]
        target_ = state.cache[:, :, target_position : target_position + 1]
        provided_ = state.provided[:, :, target_position : target_position + 1]

        if self.check:
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()
        return input_, provided_, target_, model_input_position, target_position

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor = None, moshi_tokens: torch.Tensor = None, text_token: torch.Tensor = None,
             return_embeddings: bool = False, provided_mask: torch.Tensor | None = None) \
        -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self._streaming_state
        lm_model = self.lm_model
        prepared_inputs = self.prepare_step_input(
            input_tokens, moshi_tokens, text_token, provided_mask=provided_mask,
        )
        if prepared_inputs is None:
            return (None, None) if self.report_loss or self.return_logits else None
        input_, provided_, target_, model_input_position, target_position = prepared_inputs
        if self.check:
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()
        embeddings = None
        if return_embeddings:
            embeddings = self.lm_model.embed_codes(input_)
        transformer_out, text_logits = state.graphed_main(input_)
        output = self.process_transformer_output(
            transformer_out,
            text_logits,
            provided_,
            target_,
            model_input_position,
            target_position,
        )
        if return_embeddings:
            return output, embeddings
        return output

    @torch.no_grad()
    def step_embeddings(self, embeddings: torch.Tensor):
        state = self._streaming_state
        lm_model = self.lm_model
        needed_input_tokens = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        _dummy_audio_token = lm_model._get_initial_token()
        while True:
            prepared_inputs = self.prepare_step_input(
                input_tokens=_dummy_audio_token[:, 1:1+needed_input_tokens], moshi_tokens=_dummy_audio_token[:, 1+needed_input_tokens:], text_token=self.zero_text_code,
            )
            if prepared_inputs is not None:
                break
        _, provided_, target_, model_input_position, target_position = prepared_inputs
        transformer_out, text_logits = state.graphed_embeddings(embeddings)
        return self.process_transformer_output(
            transformer_out,
            text_logits,
            provided_,
            target_,
            model_input_position,
            target_position,
        )

    @torch.no_grad()
    def process_transformer_output(self, transformer_out, text_logits, provided_, target_, model_input_position, target_position):
        state = self._streaming_state
        lm_model = self.lm_model

        sampled_text_token = sample_token(
            text_logits.float(),
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )
        assert sampled_text_token.dim() == 3, sampled_text_token.shape
        assert sampled_text_token.shape[2] == 1
        assert sampled_text_token.shape[1] == 1, "Only one text stream supported."
        sampled_text_token = sampled_text_token[:, 0, 0]

        next_text_token = torch.where(provided_[:, 0, 0], target_[:, 0, 0], sampled_text_token)
        _dep_tok = target_[:, lm_model.audio_offset:, 0]
        _dep_prov = provided_[:, lm_model.audio_offset:, 0]

        # Record the likelihood under the exact sampling distribution. LMGen
        # samples target_position now but emits it only after max_delay steps;
        # keeping this in the same circular clock lets callers verify
        # online/replay parity for the action that is eventually returned.
        scaled_text_logits = text_logits[:, 0, 0].float() / float(self.temp_text)
        if self.top_k_text > 0 and self.top_k_text < scaled_text_logits.shape[-1]:
            top_values, top_indices = torch.topk(
                scaled_text_logits, k=self.top_k_text, dim=-1
            )
            top_logprobs = torch.log_softmax(top_values, dim=-1)
            matches = top_indices == next_text_token.unsqueeze(-1)
            selected_logprob = torch.where(
                matches,
                top_logprobs,
                torch.full_like(top_logprobs, float("-inf")),
            ).amax(dim=-1)
        else:
            selected_logprob = torch.log_softmax(
                scaled_text_logits, dim=-1
            ).gather(-1, next_text_token.unsqueeze(-1)).squeeze(-1)
        state.text_action_logprobs[:, target_position] = selected_logprob

        if self.return_logits:
            sampled_audio_tokens, audio_logits = state.graphed_depth(next_text_token, transformer_out, _dep_tok, _dep_prov)
        else:
            sampled_audio_tokens = state.graphed_depth(next_text_token, transformer_out, _dep_tok, _dep_prov)

        state.cache[:, 0, target_position] = torch.where(
            ~state.provided[:, 0, target_position],
            sampled_text_token,
            state.cache[:, 0, target_position],
        )
        state.cache[:, 1 : lm_model.dep_q + 1, target_position] = torch.where(
            ~state.provided[:, 1 : lm_model.dep_q + 1, target_position],
            sampled_audio_tokens,
            state.cache[:, 1 : lm_model.dep_q + 1, target_position],
        )

        report = {}
        if self.report_loss:
            report = create_loss_report(
                state_cache=state.cache,
                lm_model=lm_model,
                text_logits=text_logits,
                audio_logits=audio_logits,
                target=target_,
                sampled_text_token=sampled_text_token,
                sampled_audio_tokens=sampled_audio_tokens,
                target_position=target_position,
            )

        if state.offset <= self.max_delay:
            state.last_emitted_codes = None
            state.last_emitted_provided = None
            state.last_emitted_text_logprob = None
            state.last_emitted_timestep = None
            state.provided[:, :, model_input_position] = False
            state.offset += 1
            if self.report_loss:
                return None, report
            if self.return_logits:
                return None, None
            else:
                return None

        B = state.cache.shape[0]
        CT = state.cache.shape[2]
        emitted_timestep = state.offset - self.max_delay
        full_index = (
            ((emitted_timestep + self.delays_cuda) % CT)
            .view(1, -1, 1)
            .expand(B, -1, 1)
        )
        emitted_codes = state.cache.gather(dim=2, index=full_index)
        emitted_provided = state.provided.gather(dim=2, index=full_index)
        text_position = int((emitted_timestep + lm_model.delays[0]) % CT)
        state.last_emitted_codes = emitted_codes
        state.last_emitted_provided = emitted_provided
        state.last_emitted_text_logprob = state.text_action_logprobs[:, text_position]
        state.last_emitted_timestep = int(emitted_timestep)
        out = emitted_codes[:, : lm_model.dep_q + 1]

        state.provided[:, :, model_input_position] = False
        state.offset += 1
        if self.report_loss:
            return out, report
        elif self.return_logits and not self.report_loss:
            return out, (text_logits.clone(), audio_logits.clone())
        else:
            return out

    def last_emitted_step(self) -> dict[str, torch.Tensor | int] | None:
        """Return metadata for the logical frame returned by the last step().

        Moshi's delayed codebooks mean that the frame returned by ``step`` is
        not the frame passed to that same call. Trajectory replay must use
        these undelayed full-stream codes and exclude targets that were forced.
        """
        state = self._streaming_state
        if state is None or state.last_emitted_codes is None:
            return None
        return {
            "codes": state.last_emitted_codes,
            "provided": state.last_emitted_provided,
            "text_logprob": state.last_emitted_text_logprob,
            "timestep": int(state.last_emitted_timestep),
        }

    def load_voice_prompt(self, voice_prompt: str):
        self.voice_prompt = voice_prompt
        raw_audio = load_audio(voice_prompt, self._sample_rate)
        raw_audio = normalize_audio(raw_audio, self._sample_rate, -24.0)
        if raw_audio.ndim == 1:
            raw_audio = raw_audio[None, :]
        self.voice_prompt_audio = raw_audio
        self.voice_prompt_cache: Optional[torch.Tensor] = None
        self.voice_prompt_embeddings: Optional[torch.Tensor] = None

    def load_voice_prompt_embeddings(self, path: str):
        self.voice_prompt = path
        state = torch.load(path)
        self.voice_prompt_audio = None
        self.voice_prompt_embeddings = state["embeddings"].to(self.lm_model.device)
        self.voice_prompt_cache = state["cache"].to(self.lm_model.device)

    def _encode_zero_frame(self) -> torch.Tensor:
        return torch.as_tensor(
            SILENCE_TOKENS, dtype=torch.long, device=self.lm_model.device,
        ).view(1, 8, 1)

    def _encode_sine_frame(self) -> torch.Tensor:
        return torch.as_tensor(
            SINE_TOKENS, dtype=torch.long, device=self.lm_model.device,
        ).view(1, 8, 1)

    def _encode_voice_prompt_frames(self, mimi):
        return encode_from_sphn(
            mimi,
            _iterate_audio(
                self.voice_prompt_audio,
                sample_interval_size=self._frame_size,
                pad=True,
            ),
            max_batch=1,
        )

    def _step_voice_prompt_frame(self, voice_prompt_frame_tokens, saved_embeddings=None):
        out = self.step(
            moshi_tokens=voice_prompt_frame_tokens,
            text_token=self.zero_text_code,
            input_tokens=self._encode_sine_frame(),
            return_embeddings=self.save_voice_prompt_embeddings,
        )
        if out is not None and self.save_voice_prompt_embeddings:
            _, embeddings = out
            saved_embeddings.append(embeddings)

    def _step_voice_prompt_core(self, mimi) -> Iterator[None]:
        if self.voice_prompt_embeddings is not None:
            for next_embed in self.voice_prompt_embeddings:
                yield
                self.step_embeddings(next_embed)
            state = self._streaming_state
            state.cache.copy_(self.voice_prompt_cache)
            return

        elif self.voice_prompt_audio is not None:
            saved_embeddings = []
            for voice_prompt_frame_tokens in self._encode_voice_prompt_frames(mimi):
                yield
                self._step_voice_prompt_frame(voice_prompt_frame_tokens, saved_embeddings)
            yield

            if self.save_voice_prompt_embeddings:
                embeddings = torch.stack(saved_embeddings, dim=0).detach()
                cache = self._streaming_state.cache.detach().clone()
                if self.cache_voice_prompt_embeddings_in_memory:
                    self.voice_prompt_embeddings = embeddings
                    self.voice_prompt_cache = cache
                else:
                    torch.save(
                        {"embeddings": embeddings.cpu(), "cache": cache.cpu()},
                        splitext(self.voice_prompt)[0] + ".pt",
                    )
        print('Done loading voice prompt.')

    def _step_voice_prompt(self, mimi):
        for _ in self._step_voice_prompt_core(mimi):
            pass

    async def _step_voice_prompt_async(self, mimi, is_alive: Optional[Callable] = None):
        for _ in self._step_voice_prompt_core(mimi):
            if is_alive is not None and not await is_alive():
                break

    def _step_audio_silence_core(self) -> Iterator[None]:
        for _ in range(self.audio_silence_frame_cnt):
            yield
            self.step(
                moshi_tokens=self._encode_zero_frame(),
                text_token=self.zero_text_code,
                input_tokens=self._encode_sine_frame(),
            )
        print('Done loading audio silence.')

    def _step_audio_silence(self):
        for _ in self._step_audio_silence_core():
            pass

    async def _step_audio_silence_async(self, is_alive: Optional[Callable] = None):
        for _ in self._step_audio_silence_core():
            if is_alive is not None and not await is_alive():
                break

    def _step_text_prompt_core(self) -> Iterator[None]:
        for text_prompt_token in self.text_prompt_tokens:
            yield
            self.step(
                moshi_tokens=self._encode_zero_frame(),
                text_token=text_prompt_token,
                input_tokens=self._encode_sine_frame(),
            )
        print('Done loading text prompt.')

    def _step_text_prompt(self):
        for _ in self._step_text_prompt_core():
            pass

    async def _step_text_prompt_async(self, is_alive: Optional[Callable] = None):
        for _ in self._step_text_prompt_core():
            if is_alive is not None and not await is_alive():
                break

    async def step_system_prompts_async(self, mimi, is_alive: Optional[Callable] = None):
        await self._step_voice_prompt_async(mimi, is_alive)
        await self._step_audio_silence_async(is_alive)
        await self._step_text_prompt_async(is_alive)
        await self._step_audio_silence_async(is_alive)

    def step_system_prompts(self, mimi):
        self._step_voice_prompt(mimi)
        self._step_audio_silence()
        self._step_text_prompt()
        self._step_audio_silence()

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_provided: torch.Tensor
    ) -> torch.Tensor:
        (B,) = text_token.shape
        prev_token = text_token
        lm_model = self.lm_model
        depformer_tokens: list[torch.Tensor] = []
        depformer_logits: list[torch.Tensor] = []
        assert not lm_model.depformer.is_streaming
        with lm_model.depformer.streaming(B):
            for cb_index in range(lm_model.dep_q):
                input_ = prev_token[:, None, None]
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                if self.return_logits:
                    assert logits.shape == (B, 1, 1, lm_model.card), logits.shape
                    ret_logits = logits.squeeze(dim=1).squeeze(dim=1)
                    assert ret_logits.shape == (B, lm_model.card), ret_logits.shape
                    depformer_logits.append(ret_logits.float())
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                )
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]
                prev_token = torch.where(
                    audio_provided[:, cb_index],
                    audio_tokens[:, cb_index],
                    next_token,
                )
                depformer_tokens.append(next_token)

        assert len(depformer_tokens) == lm_model.dep_q, (
            len(depformer_tokens),
            lm_model.dep_q,
        )
        tokens = torch.stack(depformer_tokens, dim=1)
        assert tokens.shape == (B, lm_model.dep_q), tokens.shape
        if self.return_logits:
            all_logits = torch.stack(depformer_logits, dim=1)
            assert all_logits.shape == (B, lm_model.dep_q, lm_model.card), all_logits.shape
            return tokens, all_logits
        else:
            return tokens
