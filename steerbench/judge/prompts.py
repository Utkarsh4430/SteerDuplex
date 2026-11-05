"""SteerBench judge prompts, verbatim from the judge that produced the reference numbers.

Do not edit: any change to these strings is a new judge and breaks comparability with published numbers.
`{...}` fields are filled with str.format; literal braces are doubled.
"""

TEXT_JUDGE_PROMPT = """\
You are grading a voice assistant's text response against a checklist of binary rubrics.

You are given:
  - the user's utterance (text)
  - the model's RESPONSE TRANSCRIPT (text — this is what the voice assistant actually said)
  - a list of TEXT RUBRICS to grade

First, write a brief 1-2 sentence summary of what the model response actually
does (what it says, what style markers are present in the text). Then grade
each rubric: for each one, give a one-sentence explanation pointing at concrete
words or phrasing in the transcript, then a pass=true/false verdict.

USER UTTERANCE:
{user_utterance}

MODEL RESPONSE TRANSCRIPT:
{moshi_text}

TEXT RUBRICS:
{text_rubrics_block}

Return strict JSON:
{{
  "response_summary": "<1-2 sentences about what the transcript does>",
  "text_results": [
    {{"item": "<copy of rubric text>", "explanation": "<one sentence citing concrete words/phrases>", "pass": true|false}},
    ...
  ]
}}
"""

AUDIO_JUDGE_PROMPT = """\
You are grading the STYLE of a voice assistant's audio response against a checklist of binary rubrics.

You will hear TWO audio clips:
  1. REFERENCE — an example of the target style for this evaluation.
     Use it ONLY as a style anchor (tone, prosody, pace, accent, emotional
     delivery). IGNORE its content/words entirely — do NOT compare what is said.
  2. MODEL RESPONSE — the audio you are grading.

For each rubric, judge whether MODEL RESPONSE exhibits the same style
qualities the reference exhibits. First, write a 1-2 sentence summary of the
model response's vocal style (pace, pitch, energy, accent, tone) and how it
relates to the reference's style. Then grade each rubric: for each one, give
a one-sentence explanation citing what you heard in the model response, then
a pass=true/false verdict.

AUDIO RUBRICS:
{audio_rubrics_block}

Return strict JSON:
{{
  "style_summary": "<1-2 sentences about model response style vs reference>",
  "audio_results": [
    {{"item": "<copy of rubric text>", "explanation": "<one sentence citing what you heard>", "pass": true|false}},
    ...
  ]
}}
"""

# Text parts that label the two clips in the audio-judge request (same order as sent).
REFERENCE_CLIP_LABEL = "REFERENCE audio (style anchor; ignore content):"
RESPONSE_CLIP_LABEL = "MODEL RESPONSE audio (the one you are grading):"

EMPTY_TRANSCRIPT_PLACEHOLDER = "(empty)"


def format_rubrics(rubrics: list[dict]) -> str:
    if not rubrics:
        return "  (none)"
    return "\n".join(f"  - {r['item']}" for r in rubrics)
