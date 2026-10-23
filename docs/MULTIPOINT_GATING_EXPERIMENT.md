# Multi-point and proposal-gating experiments

All runs used pretrained inference only on `video_10016`, valid Q8/Q9/Q10. The multi-point candidate uses `allenai/Molmo2-8B` plus `facebook/sam2.1-hiera-large`; the proposal-gating candidate uses `gemini-3.6-flash` cached reasoning, `allenai/Molmo-7B-O-0924`, `IDEA-Research/grounding-dino-tiny`, and `facebook/sam2.1-hiera-large`. No validation/test labels, manual annotations, or training data were used. Existing baseline/reference modes remain unchanged and both options are opt-in.

## Molmo2 multi-point candidate

Exact commands (Modal workspace `gpudashboard0`):

```bash
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::predict_molmo2_sam2 --split valid --video-id video_10016 --question-id 8 --output molmo2_multipoint_q8.json --output-step 30 --tracking-stride 30 --multi-point
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output molmo2_multipoint_q8.json --allow-partial
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::predict_molmo2_sam2 --split valid --video-id video_10016 --question-id 9 --output molmo2_multipoint_q9.json --output-step 30 --tracking-stride 30 --multi-point
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output molmo2_multipoint_q9.json --allow-partial
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::predict_molmo2_sam2 --split valid --video-id video_10016 --question-id 10 --output molmo2_multipoint_q10.json --output-step 30 --tracking-stride 30 --multi-point
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output molmo2_multipoint_q10.json --allow-partial
```

| shard | HOTA |
|---|---:|
| Q8 | 0.001350 |
| Q9 | 0.179848 |
| Q10 | 0.233892 |
| mean | **0.138363** |

Artifacts are `evidence/molmo2_multipoint_q8.json`, `_q9.json`, `_q10.json`; each has validator output, and `evidence/MOLMO2_MULTIPOINT_HASHES.txt` records hashes. Candidate rejected: mean is below `0.2062`; no EvalAI submission.

## Proposal/point gating candidate

Exact commands:

```bash
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 8 --output proposal_gated_q8.json --output-step 30 --tracking-stride 30 --proposal-grounding
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output proposal_gated_q8.json --allow-partial
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 9 --output proposal_gated_q9.json --output-step 30 --tracking-stride 30 --proposal-grounding
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output proposal_gated_q9.json --allow-partial
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 10 --output proposal_gated_q10.json --output-step 30 --tracking-stride 30 --proposal-grounding
MODAL_PROFILE=gpudashboard0 modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output proposal_gated_q10.json --allow-partial
```

| shard | HOTA |
|---|---:|
| Q8 | 0.000000 |
| Q9 | 0.004479 |
| Q10 | 0.279286 |
| mean | **0.094588** |

Artifacts are `evidence/proposal_gated_q8.json`, `_q9.json`, `_q10.json`; each has validator output, evaluator logs, and `evidence/PROPOSAL_GATED_HASHES.txt`. Candidate rejected: mean is below `0.2062`; no EvalAI submission.
