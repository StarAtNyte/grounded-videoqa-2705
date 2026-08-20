# Multi-point SAM2 conditioning + proposal/point agreement gating

Two opt-in, backward-compatible changes, tested on the same controlled slice
(`video_10016`, valid split, Q8/Q9/Q10) used by every prior experiment in this
repo, against the same `0.2062` scale gate.

## 1. Multi-point Molmo2 -> SAM2 (VideoMolmo-inspired)

`sam2_backend.py Sam2Bidirectional.track()` gained an optional trailing
`extra_points: list[(frame_idx, x, y)]` argument: each is seeded as its own
positive prompt before propagation, conditioning both directions on more than
one observation instead of a single anchor point. This mirrors VideoMolmo's
multi-point fusion (arXiv:2506.05336) without adopting its heavier
Molmo-7B-lineage stack (see `OPEN_MODEL_RESEARCH.md`).

`advanced_pipeline.py run_molmo2_sam2()` gained `multi_point: bool = False`.
Default False preserves the original single first-evidence-point anchor
(the documented baseline, kept for later revisit per project decision).
When True, every other Molmo2-reported timestamp for the same object ID is
converted to sparse-clip frame indices and passed as `extra_points`.

Commands:

```bash
MODAL_PROFILE=baptinjean modal run modal_app.py::predict_molmo2_sam2 --split valid --video-id video_10016 --question-id 8 --output molmo2_multipoint_q8.json --output-step 30 --tracking-stride 30 --multi-point
MODAL_PROFILE=baptinjean modal run modal_app.py::evaluate_hota --split valid --output molmo2_multipoint_q8.json --allow-partial
# ...same for question-id 9 and 10
```

| shard | prior best (single point, fixed parser) | multi-point |
|---|---:|---:|
| Q8 | 0.004267 | 0.001350 |
| Q9 | 0.038440 | 0.179848 |
| Q10 | 0.249480 | 0.233892 |
| mean | 0.097396 | **0.138363** |

Multi-point conditioning improves the controlled mean by ~42% over the best
prior standalone-Molmo2 variant (large Q9 gain, small Q10 regression, Q8
regression). Still below `0.2062` and below the currently deployed CORTEX
pipeline's live test result (`0.2120`, `SUBMISSION_RESULT.md`). Not scaled to
full test; `multi_point` stays opt-in and off by default.

## 2. Proposal/point agreement gating (CORTEX + Grounding-DINO)

`run_cortex_sam2()`'s DINO proposal-box selection now additionally requires
the candidate box to contain Molmo's own selected point before accepting it;
otherwise it falls through to the next-ranked box or the point-only fallback
(`proposal_box = None`). This was the next-highest-value fix flagged in
`GROUNDING_FIX_EXPERIMENT.md`.

Commands:

```bash
MODAL_PROFILE=baptinjean modal run modal_app.py::predict_cortex_sam2 --split valid --video-id video_10016 --question-id 8 --output proposal_gated_q8.json --output-step 30 --tracking-stride 30 --proposal-grounding
MODAL_PROFILE=baptinjean modal run modal_app.py::evaluate_hota --split valid --output proposal_gated_q8.json --allow-partial
# ...same for question-id 9 and 10
```

| shard | prior best (fixed proposal, no gate) | gated |
|---|---:|---:|
| Q8 | 0.005617 | 0.005617 |
| Q9 | 0.581634 | 0.083741 |
| Q10 | 0.001795 | 0.278658 |
| mean | 0.196349 | **0.122672** |

Gating regresses the controlled mean: on Q9 it rejected a box that actually
tracked well (containment with Molmo's point was the wrong filter for that
shard), while Q10 improved. Net effect negative. Not scaled; the gate is
implemented but leaves worse results than the ungated fixed-proposal path on
this slice, so it should not be turned on by default.

## Conclusion

Neither variant exceeds the `0.2062` controlled gate or the currently
submitted result (`0.2120` HOTA, leaderboard rank 1). Per the standing project
rule, no full-test batch was generated and nothing new was submitted. Both
flags remain opt-in, default-off, for future revisit; the single-point Molmo2
anchor and ungated DINO proposal remain the code defaults.
