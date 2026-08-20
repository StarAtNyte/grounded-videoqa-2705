# Direct-grounding experiment

Opt-in TutuAI-style direct grounding: pretrained `gemini-2.5-flash` emits the first-appearance frame and normalized box; pretrained `facebook/sam2.1-hiera-large` produces official tracks. Existing CORTEX/reference/proposal paths are unchanged. All runs are pretrained inference only on the valid controlled slice; no validation/test labels, manual annotations, or training data are used.

## Exact controlled records

| artifact | prediction command | evaluator command | local artifact | manifest | HOTA |
|---|---|---|---|---|---:|
| direct_q8.json | `MODAL_PROFILE=baptinjean modal run modal_app.py::predict_direct_sam2 --split valid --video-id video_10016 --question-id 8 --output direct_q8.json --output-step 30 --tracking-stride 30` | `MODAL_PROFILE=baptinjean modal run modal_app.py::evaluate_hota --split valid --output direct_q8.json --allow-partial` | `evidence/direct_q8.json` | `evidence/direct_q8.json.manifest.json` | 0.000000 |
| direct_q9.json | `MODAL_PROFILE=baptinjean modal run modal_app.py::predict_direct_sam2 --split valid --video-id video_10016 --question-id 9 --output direct_q9.json --output-step 30 --tracking-stride 30` | `MODAL_PROFILE=baptinjean modal run modal_app.py::evaluate_hota --split valid --output direct_q9.json --allow-partial` | `evidence/direct_q9.json` | `evidence/direct_q9.json.manifest.json` | 0.026345 |
| direct_q10.json | `MODAL_PROFILE=baptinjean modal run modal_app.py::predict_direct_sam2 --split valid --video-id video_10016 --question-id 10 --output direct_q10.json --output-step 30 --tracking-stride 30` | `MODAL_PROFILE=baptinjean modal run modal_app.py::evaluate_hota --split valid --output direct_q10.json --allow-partial` | `evidence/direct_q10.json` | `evidence/direct_q10.json.manifest.json` | 0.009438 |

| mean | | | | | **0.011928** |

Each direct artifact and manifest was downloaded from the Modal Volume after its completed run. The manifests contain the raw VLM response, parsed frame/box, model, question identity, diagnostics, and artifact SHA-256. Runtime evaluator logs are `evidence/direct_q8.metrics.log`, `direct_q9.metrics.log`, and `direct_q10.metrics.log`; local validator output is `evidence/DIRECT_VALIDATION_OUTPUT.txt`.

The direct mean is below `0.2062`; no scaling or broader execution was performed. Artifacts and logs are evidence only; no validation/test training or manual labels were used.
