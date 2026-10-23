# Proposal/point prompt ensemble experiment

This candidate used pretrained `allenai/Molmo-7B-O-0924` spatial grounding, `IDEA-Research/grounding-dino-tiny` proposal detection, and `facebook/sam2.1-hiera-large` tracking. It compared a clamped DINO box prompt and a Molmo point prompt for each grounded candidate, then selected the short-track trial using the existing heuristic. The Gemini reasoning cache used pretrained `gemini-3.6-flash`. All runs were pretrained inference only on `video_10016` valid Q8/Q9/Q10; no validation/test labels, manual annotations, or training data were used.

Exact commands (Modal workspace `gpudashboard0`):

```bash
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::batch_reason --split valid --start 0 --limit 3 --cost-cap 3 --model gemini-3.6-flash
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 8 --output ensemble_q8.json --output-step 30 --tracking-stride 30 --proposal-grounding
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output ensemble_q8.json --allow-partial
python -m grounded_videoqa.tools.validate evidence/ensemble_q8.json
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 9 --output ensemble_q9.json --output-step 30 --tracking-stride 30 --proposal-grounding
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output ensemble_q9.json --allow-partial
python -m grounded_videoqa.tools.validate evidence/ensemble_q9.json
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 10 --output ensemble_q10.json --output-step 30 --tracking-stride 30 --proposal-grounding
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output ensemble_q10.json --allow-partial
python -m grounded_videoqa.tools.validate evidence/ensemble_q10.json
```

| shard | HOTA |
|---|---:|
| Q8 | 0.000000 |
| Q9 | 0.004479 |
| Q10 | 0.279286 |
| mean | **0.094588** |

All three artifacts pass validation. SHA-256 values are in `evidence/ENSEMBLE_ARTIFACT_HASHES.txt`; evaluator logs and validator output are beside each artifact. The candidate fails the `0.2062` gate and consumed no EvalAI submission. It is not promoted as a default.
