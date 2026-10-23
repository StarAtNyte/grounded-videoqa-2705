# Public Grounded VideoQA winner research

## Findings (2025 challenge)

- **Winner: SGVR@KAIST** — [challenge results](https://perception-test-challenge.github.io/challenge2025.html), [technical report](https://drive.google.com/file/d/1YahPyJe2PEaj2VAWl0g9yusI-nkqvbtA/view). The report describes Gemini 2.5 Pro + CORTEX trigger-moment reasoning, Molmo-7B point-to-box grounding, and bidirectional SAM2 tracking. Fallbacks are forward sliding-window search and reverse search; OCR questions use a visual descriptive alias rather than individual letters.
- **Runner-up: TutuAI** — [technical report](https://drive.google.com/file/d/1hJO45Wl56s3uggbjM-DiIxWKCVyoqAj4/view). No public implementation was located. Their report describes a simpler two-stage, training-free pipeline: Seed1.6-VL-thinking directly emits the *first frame of appearance* and normalized box, then SAM2.1 Hiera-Large tracks bidirectionally. They report HOTA 0.4304.

## Code availability

No official SGVR or TutuAI challenge repository/checkpoint was found. The most relevant public implementation is [Grounded-VideoLLM](https://github.com/WHB139426/Grounded-Video-LLM), which provides inference/training code and checkpoints for temporal grounding, but is not the challenge submission and does not directly serialize Perception Test box tracks.

The official dataset/tool repository is [google-deepmind/perception_test](https://github.com/google-deepmind/perception_test).

## Actionable controlled experiment

The highest-value next experiment is **TutuAI-style direct grounding**, kept opt-in and evaluated on the same Q8/Q9/Q10 shards:

1. Sample the video at 1 FPS (or a documented equivalent).
2. Ask one VLM to return `<frame>idx</frame><bbox>x1 y1 x2 y2</bbox>` for the first object appearance.
3. Convert the returned normalized box to the official format.
4. Initialize two SAM2.1-Hiera-Large propagations and merge forward/backward tracks.
5. Preserve a per-question raw response, parse diagnostics, artifact hash, local `src/grounded_videoqa/tools/validate.py` output, and HOTA.

Do not replace the current baseline or scale this route until its controlled mean exceeds `0.2062`. The 2025 winner reports are architectural guidance, not validation/test labels or executable code.
