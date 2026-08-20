# Grounding/tracking fixes

All runs used pretrained inference only on `video_10016`, valid Q8/Q9/Q10: `allenai/Molmo-7B-O-0924` for spatial grounding, `IDEA-Research/grounding-dino-tiny` for proposals, and `facebook/sam2.1-hiera-large` for tracking. No validation/test labels, manual annotations, or training data were used.

Exact fixed-proposal commands (Modal workspace `baptinjean`):

```bash
MODAL_PROFILE=baptinjean modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 8 --output proposal_fixed_q8.json --output-step 30 --tracking-stride 30 --proposal-grounding
MODAL_PROFILE=baptinjean modal run modal_app.py::evaluate_hota --split valid --output proposal_fixed_q8.json --allow-partial
MODAL_PROFILE=baptinjean modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 9 --output proposal_fixed_q9.json --output-step 30 --tracking-stride 30 --proposal-grounding
MODAL_PROFILE=baptinjean modal run modal_app.py::evaluate_hota --split valid --output proposal_fixed_q9.json --allow-partial
MODAL_PROFILE=baptinjean modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 10 --output proposal_fixed_q10.json --output-step 30 --tracking-stride 30 --proposal-grounding
MODAL_PROFILE=baptinjean modal run modal_app.py::evaluate_hota --split valid --output proposal_fixed_q10.json --allow-partial
```

Implemented three concrete fixes:

- Molmo2 track parsing now preserves repeated `(time, object_id, x, y)` records instead of discarding IDs and keeping only the final coordinate pair.
- Grounding-DINO boxes are clamped to image bounds and near-full-frame/degenerate proposals are rejected.
- SAM2 trial/full propagation enables bounded occlusion recovery with the configured tracking stride.
- Restored `run_advanced`'s missing `tracking_stride=30` default and required `cv2` import.

Controlled proposal rerun on `video_10016`, output/tracking stride 30:

| shard | previous proposal | fixed proposal |
|---|---:|---:|
| Q8 | 0.005617 | 0.005617 |
| Q9 | 0.083741 | 0.581634 |
| Q10 | 0.278658 | 0.001795 |
| mean | 0.122672 | 0.196349 |

The mean improved substantially but remains below `0.2062`; no scale-up was performed. Q10 regression shows proposal selection is still unstable: DINO can select a box inconsistent with Molmo's selected point. The next highest-value fix is proposal/point agreement gating plus retaining a point-only fallback when the proposal does not contain the selected Molmo point. Artifacts, metric logs, hashes, and validator outputs are under `evidence/`.
