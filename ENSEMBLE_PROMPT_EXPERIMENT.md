# Proposal/point prompt ensemble experiment

This opt-in candidate compared both a clamped Grounding-DINO box prompt and a Molmo point prompt for each grounded candidate, then selected the short-track trial using the existing leakage-free heuristic. Existing baseline/reference modes remain unchanged. No labels or manual annotations were used.

Commands (gpudashboard0):

```bash
python -m py_compile advanced_pipeline.py modal_app.py
modal run modal_app.py::batch_reason --split valid --start 0 --limit 3 --cost-cap 3 --model gemini-3.6-flash
modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id {8,9,10} --output ensemble_qN.json --output-step 30 --tracking-stride 30 --proposal-grounding
modal run modal_app.py::evaluate_hota --split valid --output ensemble_qN.json --allow-partial
python validate.py evidence/ensemble_qN.json
```

| shard | HOTA |
|---|---:|
| Q8 | 0.000000 |
| Q9 | 0.004479 |
| Q10 | 0.279286 |
| mean | **0.094588** |

All three artifacts pass validation. SHA-256 values are in `evidence/ENSEMBLE_ARTIFACT_HASHES.txt`; evaluator logs and validator output are beside each artifact. The candidate fails the `0.2062` gate and consumed no EvalAI submission. It is not promoted as a default. The experiment used only pretrained inference and cached reasoning; no validation/test labels, manual annotations, or training were used.
