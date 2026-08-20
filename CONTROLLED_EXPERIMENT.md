

## Exact reproducibility records

Each row identifies the exact Modal command, evaluator command, local captured metric log, and the expected one-question payload. Use `modal run modal_app.py::export_controlled_artifact --output <file>` to stream the exact remote JSON and SHA-256 for repository/auditor capture; use `python validate.py -` on that stream (or save it first).

| artifact | prediction command | evaluator command | metric log | HOTA |
|---|---|---|---|---:|
| baseline_q8.json | `modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 8 --output baseline_q8.json --output-step 30 --tracking-stride 30 --reference-mode` | `modal run modal_app.py::evaluate_hota --split valid --output baseline_q8.json --allow-partial` | `evidence/controlled_q8_q10/baseline_q8.metrics.log` | 0.000000 |
| baseline_q9.json | `modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 9 --output baseline_q9.json --output-step 30 --tracking-stride 30 --reference-mode` | `modal run modal_app.py::evaluate_hota --split valid --output baseline_q9.json --allow-partial` | `evidence/controlled_q8_q10/baseline_q9.metrics.log` | 0.049760 |
| baseline_q10.json | `modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 10 --output baseline_q10.json --output-step 30 --tracking-stride 30 --reference-mode` | `modal run modal_app.py::evaluate_hota --split valid --output baseline_q10.json --allow-partial` | `evidence/controlled_q8_q10/baseline_q10_final.metrics.log` | 0.278658 |
| proposal_q8.json | `modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 8 --output proposal_q8.json --output-step 30 --tracking-stride 30 --proposal-grounding` | `modal run modal_app.py::evaluate_hota --split valid --output proposal_q8.json --allow-partial` | `evidence/controlled_q8_q10/proposal_q8_final.metrics.log` | 0.005617 |
| proposal_q9.json | `modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 9 --output proposal_q9.json --output-step 30 --tracking-stride 30 --proposal-grounding` | `modal run modal_app.py::evaluate_hota --split valid --output proposal_q9.json --allow-partial` | `evidence/controlled_q8_q10/proposal_q9_final.metrics.log` | 0.083741 |
| proposal_q10.json | `modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 10 --output proposal_q10.json --output-step 30 --tracking-stride 30 --proposal-grounding` | `modal run modal_app.py::evaluate_hota --split valid --output proposal_q10.json --allow-partial` | `evidence/controlled_q8_q10/proposal_q10_final.metrics.log` | 0.278658 |

Mean baseline: `0.109473`; mean proposal: `0.122672`. All six Modal runs and evaluator runs completed; no dependency errors remained. Both means are below `0.2062`, so scaling was skipped.

## Repository-preserved evaluator evidence

The six evaluator outputs are preserved under `evidence/controlled_q8_q10/` (copied from the completed Modal runs). Each contains the completed evaluator footer, one-question count, and HOTA/DetA/AssA/LocA values. The six prediction JSON payloads were downloaded from the Modal Volume into `evidence/controlled_q8_q10/` and each passed the official local validator; these are the actual controlled artifacts, not unrelated test output. SHA-256 hashes are recorded in `evidence/controlled_q8_q10/ARTIFACT_HASHES.txt`.

## Cache provenance schema

Every newly generated Gemini cache record includes `model`, `api_calls`, `cost_cap`, `split`, and `question_id`. A cache is reusable only when all identity fields match the current request and `api_calls >= 1` with `api_calls <= cost_cap`; malformed, legacy, or mismatched records regenerate. Existing legacy caches with empty provenance are intentionally not treated as proof of provenance.
