# Grounded VideoQA 2705 on Modal

This repo uses the official Google DeepMind archives, stored in a Modal Volume:

- test videos: `grounded_question_test_videos.zip` (~11.3 GB)
- test annotations: `grounded_question_test_annotations.zip` (~17.5 kB)

Raw data, predictions, and model caches stay out of git.

> **⚠️ DATASET UPDATE**: Organizers updated test/valid datasets on **Aug 11, 2026**.
> Re-download both splits before any new inference: `modal run -m grounded_videoqa.jobs.modal_app::download --split test`
> and `modal run -m grounded_videoqa.jobs.modal_app::download --split valid`.

## Leaderboard status

_Last checked: 2026-08-25._ This table tracks the public leaderboard and all
known project submissions, including private submissions. Private entries are
counted in **Project submissions** but do not receive a public leaderboard rank
until made public.

| Entry | Visibility | HOTA | Public rank | Verification |
|---|---|---:|---:|---|
| `Banana Bats` / `fusion` submission `580596` | **Public** | **0.662001** | **1** | Live EvalAI test leaderboard, split `7004` |
| `tibo` submission `580552` | **Public** | **0.494200** | **2** | Live EvalAI test leaderboard, split `7004` |
| `SRUKR` submission `578838` | **Public** | **0.429367** | **3** | Live EvalAI test leaderboard, split `7004` |
| `StarAtNyte` / `Submission` submission `580601` (submitted Aug 25) | **Public** | **0.367646** | **4** | Finished EvalAI result; complete artifact and SHA-256 verified locally |
| `Zero-shot video grounding pipeline` submission `579806` (submitted Aug 22) | Private; can be made public | **0.251900** | **5 if public at current scores** | Submission identity and full-precision result verified through the authenticated EvalAI API; artifact hash is not available locally |
| `StarAtNyte` submission `579655` | **Private** | **0.212013** | Not ranked | EvalAI result confirmed; artifact SHA-256 recorded below |

**Project submissions counted: 3 confirmed test results.** Submission `580601`
was created and initially verified private, but the authoritative EvalAI state
on 25 Aug later showed it public at rank 4. Its HOTA is
`0.3676459789276123`, improving the previous project best by `0.115746`, while
trailing Banana Bats by `0.294355` HOTA. Our score is 55.53% of the leader's.

Submission `580601` is the completed v2 production pipeline: Gemini 3.7 Flash
with high thinking performs independent zero-shot video reasoning for every
question; Molmo2-ER grounds answer objects; and SAM2.1 Hiera-Large propagates
stable tracks at the official sparse cadence. The one question for which
Molmo2-ER returned no track (`video_8728` question `6`) was independently
re-grounded with Molmo2-VideoPoint-4B before the final merge. The submitted
artifact contains 932 videos, 1,859 questions, 2,934 tracks, and zero empty
answers. Its SHA-256 is
`35f9526e6e577ed56d3669114edcefda1bccc55b386c8e09156b9b306cb08777`.

The `0.251900` submission is the established zero-shot v1 production pipeline:
CORTEX-style Gemini video reasoning identifies the answer object and a strong
trigger frame; Molmo spatial grounding localizes it; SAM2.1 Hiera-Large tracks
it bidirectionally; and the serializer emits normalized boxes on the official
sparse frame IDs with stable track identities. Its EvalAI metadata is method
`Zero-shot video grounding pipeline` and description
`Grounded VideoQA baseline pipeline (zero-shot)`. It is not evidence for the
uncompleted v2 Molmo2-ER experiment below.

Private submission `579655` used phase 5350, `zero-shot`, contact
`khanalnitij20@gmail.com`, and artifact `test_submission.json` with SHA-256
`def19142c2ac8132d04e175cd2bf1085aecb86b194b7f16f98fbc5388be7b160`. It is
not public and does not appear on the leaderboard. EvalAI returned HOTA
`0.21201318502426147`.
When a private EvalAI submission is created, add its submission ID, score, and
artifact SHA-256 here; keep it in the count even if it is later made public.
See `SUBMISSION_RESULT.md` for the evidence caveat and `AGENTS.md` for upload
rules.

See `RESULTS_LEDGER_2026-08-25.md` for the consolidated submission,
leaderboard, controlled-experiment, and competitor-method record.

## V2 Pipeline (HOTA improvement effort)

The v2 pipeline upgrades the stack whose best reported HOTA is 0.25 (v1) toward the
2025 winner's HOTA 0.497. See `V2_PIPELINE_EXPERIMENT.md` for full commands.

**Key upgrades:**
1. **Reasoning**: `gemini-3.6-flash` → `gemini-3.7-flash` with high thinking through Vertex AI
2. **CORTEX prompt v2**: question-family routing, OCR alias, visibility gap semantics
3. **Grounding**: `Molmo-7B-O-0924` → `Molmo2-ER` (pixel-accurate spatial specialist)
   or `Molmo2-VideoPoint-4B` (video-specific pointing, fine-tuned on VideoPoint data)
4. **Tracking**: SAM2.1-Hiera-Large (unchanged, but with better anchor frames)

Quick start for v2 inference:
```bash
# 1. Re-download updated datasets
modal run -m grounded_videoqa.jobs.modal_app::download --split test
modal run -m grounded_videoqa.jobs.modal_app::download --split valid

# 2. Re-reason test split with Gemini 3.7 Flash through Vertex AI
modal run -m grounded_videoqa.jobs.modal_app::batch_reason_pro --split test --limit 1900 \
  --use-vertex --model gemini-3.7-flash --thinking-level high

# 3. Full test inference
modal run -m grounded_videoqa.jobs.modal_app::batch_predict_v2 --split test \
  --output v2_test_submission.json --spatial-model molmo2er

# 4. Validate and submit
python -m grounded_videoqa.tools.validate  # local schema check
# Upload to EvalAI phase 5350
```

```bash
python -m pip install modal
modal token new
```

## Download and inspect

```bash
modal run -m grounded_videoqa.jobs.modal_app::download --split test
modal run -m grounded_videoqa.jobs.modal_app::inspect --split test
```

Use `valid` first for pipeline checks. The volume is named
`grounded-videoqa-2705-data`; it survives function/container restarts.

## Canonical advanced architecture

Run the implemented local stack with:

```bash
modal run -m grounded_videoqa.jobs.modal_app::predict_advanced --split test \\
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
modal run -m grounded_videoqa.jobs.modal_app::reason --split test --video-id video_6358 --question-id 4
```

The reasoning result is cached in `/data/cache/test/...`. Prediction will use
its descriptive object phrase and trigger frame automatically:

```bash
modal run -m grounded_videoqa.jobs.modal_app::predict --split test \\
  --video-id video_6358 --question-id 4 --backend csrt
```

Prepare the production tracker and use the printed config path:

```bash
modal run -m grounded_videoqa.jobs.modal_app::prepare_sam2
```

Then pass the printed checkpoint and config to `predict --backend sam2`.

## Smoke-test prediction

```bash
modal run -m grounded_videoqa.jobs.modal_app::predict \
  --split test --video-id VIDEO_ID --question-id QUESTION_ID
```

For restartable full-split inference, use the batch endpoint. Partition large
runs with `--start` and `--limit`:

```bash
modal run -m grounded_videoqa.jobs.modal_app::batch_predict --split valid \\
  --output valid_submission.json --start 0 --limit 10
modal run -m grounded_videoqa.jobs.modal_app::validate_submission --split valid \\
  --output valid_submission.json --allow-partial
```

After freezing the configuration, run test partitions similarly:

```bash
modal run -m grounded_videoqa.jobs.modal_app::batch_predict --split test \\
  --output test_submission.json --start 0 --limit 50
```

The batch manifest is restartable and commits after every question.

The current default runs Grounding DINO-Tiny at an anchor and propagates
bidirectionally with a bounded-memory CSRT fallback. It emits the official
sparse frame cadence (normally every 30 frames). For the real tracker, use
SAM2.1:

```bash
modal run -m grounded_videoqa.jobs.modal_app::predict --split valid \\
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
python -m grounded_videoqa.tools.validate predictions.json
```

## Modal cost/safety notes

- Download only the split needed; do not download Unified VideoQA archives.
- Run heavy model functions sequentially and cache outputs in the Volume.
- Never use validation/test annotations as training or prompting data.
- Controlled shard artifacts are stored in the persistent Volume; use `CONTROLLED_EXPERIMENT.md` for commands and evidence.
- Do not commit the Volume contents, extracted videos, checkpoints, or outputs.
