# Grounding/tracking fixes

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
