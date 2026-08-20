# Grounded VideoQA 2705 on Modal

This repo uses the official Google DeepMind archives, stored in a Modal Volume:

- test videos: `grounded_question_test_videos.zip` (~11.3 GB)
- test annotations: `grounded_question_test_annotations.zip` (~17.5 kB)

Raw data, predictions, and model caches stay out of git.

## Leaderboard status

_Last checked: 2026-08-19._ This table tracks the public leaderboard and all
known project submissions, including private submissions. Private entries are
counted in **Project submissions** but do not receive a public leaderboard rank
until made public.

| Entry | Visibility | HOTA | Public rank | Verification |
|---|---|---:|---:|---|
| `StarAtNyte` reported EvalAI result | Public (reported) | **0.2120** | **1** | Submission ID/artifact identity still unconfirmed |
| `StarAtNyte` submission `579655` | **Private** | **0.212013** | Not ranked | EvalAI result confirmed; artifact SHA-256 recorded below |

**Project submissions counted: 1 reported public + 1 confirmed private = 2.**

Private submission `579655` used phase 5350, `zero-shot`, contact
`khanalnitij20@gmail.com`, and artifact `test_submission.json` with SHA-256
`def19142c2ac8132d04e175cd2bf1085aecb86b194b7f16f98fbc5388be7b160`. It is
not public and does not appear on the leaderboard. EvalAI returned HOTA
`0.21201318502426147`.
When a private EvalAI submission is created, add its submission ID, score, and
artifact SHA-256 here; keep it in the count even if it is later made public.
See `SUBMISSION_RESULT.md` for the evidence caveat and `AGENTS.md` for upload
rules.

## Setup

```bash
python -m pip install modal
modal token new
```

## Download and inspect

```bash
modal run modal_app.py::download --split test
modal run modal_app.py::inspect --split test
```

Use `valid` first for pipeline checks. The volume is named
`grounded-videoqa-2705-data`; it survives function/container restarts.

## Canonical advanced architecture

Run the implemented local stack with:

```bash
modal run modal_app.py::predict_advanced --split test \\
  --video-id video_6358 --question-id 4 --backend sam2 \\
  --sam2-checkpoint /data/checkpoints/sam2.1_hiera_large.pt \\
  --sam2-config PRINTED_CONFIG_PATH \\
  --spatial-refine
```

This executes TimeLens2 (when), Molmo2 VideoTrack (candidate identities), and
SAM2.1 (trajectory). Molmo2-ER spatial correction and Qwen/Keye arbitration are
separate optional escalation stages; run them after this path is validated on
train/validation, because loading all models simultaneously wastes GPU memory.


Configure the VLM secret once:

```bash
modal secret create gemini-api GOOGLE_API_KEY="$GOOGLE_API_KEY"
modal run modal_app.py::reason --split test --video-id video_6358 --question-id 4
```

The reasoning result is cached in `/data/cache/test/...`. Prediction will use
its descriptive object phrase and trigger frame automatically:

```bash
modal run modal_app.py::predict --split test \\
  --video-id video_6358 --question-id 4 --backend csrt
```

Prepare the production tracker and use the printed config path:

```bash
modal run modal_app.py::prepare_sam2
```

Then pass the printed checkpoint and config to `predict --backend sam2`.

## Smoke-test prediction

```bash
modal run modal_app.py::predict \
  --split test --video-id VIDEO_ID --question-id QUESTION_ID
```

For restartable full-split inference, use the batch endpoint. Partition large
runs with `--start` and `--limit`:

```bash
modal run modal_app.py::batch_predict --split valid \\
  --output valid_submission.json --start 0 --limit 10
modal run modal_app.py::validate_submission --split valid \\
  --output valid_submission.json --allow-partial
```

After freezing the configuration, run test partitions similarly:

```bash
modal run modal_app.py::batch_predict --split test \\
  --output test_submission.json --start 0 --limit 50
```

The batch manifest is restartable and commits after every question.

The current default runs Grounding DINO-Tiny at an anchor and propagates
bidirectionally with a bounded-memory CSRT fallback. It emits the official
sparse frame cadence (normally every 30 frames). For the real tracker, use
SAM2.1:

```bash
modal run modal_app.py::predict --split valid \\
  --video-id VIDEO_ID --question-id QUESTION_ID --backend sam2 \\
  --sam2-checkpoint /data/checkpoints/sam2.1_hiera_large.pt \\
  --sam2-config /data/checkpoints/sam2.1_hiera_l.yaml
```

The grounding stage is still intentionally replaceable: the serious system
should feed it VLM trigger moments and descriptive object phrases rather than
only the raw question.

`predict_molmo2_sam2` anchors SAM2 on Molmo2's single first-evidence point by
default; this remains the documented baseline to revisit. Pass `--multi-point`
to additionally seed SAM2 with every other Molmo2-reported timestamp for the
same object ID (VideoMolmo-inspired). `predict_cortex_sam2 --proposal-grounding`
now also rejects a Grounding-DINO box that disagrees with Molmo's selected
point. Neither flag is on by default; see `MULTIPOINT_GATING_EXPERIMENT.md`
for the controlled-slice results that kept both off.

The generated JSON follows the format shown in the official
`grounded_vqa.ipynb`: normalized `[xmin, ymin, xmax, ymax]` boxes, frame IDs,
track IDs, and scores. Validate it locally with:

```bash
python validate.py predictions.json
```

## Modal cost/safety notes

- Download only the split needed; do not download Unified VideoQA archives.
- Run heavy model functions sequentially and cache outputs in the Volume.
- Never use validation/test annotations as training or prompting data.
- Controlled shard artifacts are stored in the persistent Volume; use `CONTROLLED_EXPERIMENT.md` for commands and evidence.
- Do not commit the Volume contents, extracted videos, checkpoints, or outputs.
