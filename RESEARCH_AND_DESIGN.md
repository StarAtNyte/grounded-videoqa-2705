# Grounded VideoQA 2705: research and proposed system

## What prior solutions actually show

The previous challenge reports are more useful than generic VideoQA papers because
HOTA rewards *visible detections plus stable identity*, not a text answer.

### 2024 challenge

The official challenge summary (arXiv:2411.19941, section 4.6) reports:

- baseline: MDETR on the middle frame, static boxes: **0.057 HOTA**;
- 2024 winner: **0.270 HOTA**;
- winning recipe: Gemini answer extraction -> Grounding DINO -> SAM2;
- runner-up: LLaVA-OneVision -> OWLv2 -> SAM2.

The important lesson is that the detector/grounder and tracker matter much more
than a static middle-frame box. The decomposition is sound, but it compounds
errors when the answer is temporal, ambiguous, or absent in the chosen frame.

### 2025 challenge

The official workshop page links the SGVR@KAIST winner report and TutuAI
runner-up report.

**SGVR@KAIST (winner, 0.4968 test HOTA)** uses:

1. Gemini 2.5 Pro on video sampled at 3 FPS with a CORTEX prompt;
2. an explicit answer description with visual attributes, spatial relations, and
   an object-level trigger frame;
3. Molmo-7B point/box grounding at that frame;
4. fallback search: forward sliding windows, then reverse search;
5. SAM2 Hiera-Large at 10 FPS, bidirectionally;
6. merging of tracklets / multi-object handling.

Their validation ablation was 0.3687 (sliding window), 0.4011 (reverse search),
and 0.4380 (trigger moment). They specifically report OCR-like objects and
objects distinguished by left/right or color as failure cases. They also call
out multi-trigger questions and comparative physical reasoning as unresolved.

**TutuAI (runner-up, 0.4304 test HOTA)** unifies reasoning and grounding in a
single Seed1.6-VL-thinking call. The prompt asks for the *first frame* where
the object appears and a normalized box, then SAM2.1-Hiera-Large propagates
backward and forward. They used 1 FPS for cost, and note that audio/subtle
motion is lost at that rate.

### Non-obvious dataset/evaluation technicalities

These details materially change the implementation:

- The official annotations store normalized boxes as `[xmin, ymin, xmax,
  ymax]`, but the older notebook's docstring incorrectly mentions another
  order. Preserve the actual array order and assert it with visualization.
- Ground-truth object tracks are not annotated at every decoded frame. In the
  sample data, `frame_ids` are normally `0, 30, 60, ...` (roughly 1 FPS), and
  each track can start at a later sampled frame. The hidden evaluator compares
  predictions on those IDs. Our internal tracker can run at full FPS, but the
  submission assembler should emit the official sampled frame IDs, not assume
  every decoded frame is scored.
- A question's `answers` are indices into `object_tracking`, not text answers.
  Test annotations expose questions but not these answer indices, so the model
  must infer the object set independently at test time.
- The evaluator starts each GT track at its `initial_tracking_box` one-hot
  position. Emitting a box before the inferred first-visible frame creates a
  false positive; emitting a stale box during disappearance also hurts HOTA.
- HOTA is averaged over question sequences and has separate DetA, AssA, and
  LocA components. A model-selection score based only on aggregate HOTA can
  hide identity fragmentation. Save all four metrics and inspect per-family
  failures.
- The official helper keeps at most ten predicted tracks by score. We should
  never emit a large open-vocabulary candidate set: answer-set precision is
  more valuable than detector recall after the semantic stage.
- Video metadata contains a measured, non-integer `frame_rate`; use frame index
  for boxes and timestamps only for temporal reasoning. Do not convert sampled
  frame IDs using an assumed exact 30 FPS.

The winning design is not “run Grounding DINO on the question.” It is:

- answer-conditioned semantic disambiguation;
- temporal anchor selection;
- high-quality initial localization;
- bidirectional segmentation tracking;
- careful handling of visibility, multiple objects, and IDs.

Our current Grounding-DINO-Tiny + forward-CSRT code should remain only as a
smoke test. It is not a serious submission model.

## Proposed system: AnchorGraph-GVQA

A stronger, still practical design is a confidence-aware ensemble around the
same winning decomposition.

### Stage A: temporal answer reasoning

Run a strong video-capable VLM on a 3 FPS contact sheet plus audio when useful.
Require strict JSON, not free-form CoT:

```json
{
  "objects": [{"description": "...", "count": 1,
    "relations": ["..."], "events": ["..."]}],
  "segments": [{"start": 0.0, "end": 12.0}],
  "anchors": [{"frame": 180, "visibility": 0.92}],
  "evidence": [{"frame": 180, "reason": "..."}]
}
```

Do not depend on hidden reasoning. Ask for concise evidence and calibrated
confidence. Make two independent calls (different frame stride/prompt) only
when ambiguity is high; reconcile descriptions and anchors with an LLM or
rules. This improves decoys, “first/last/again”, comparisons, and plural
answers.

Use train annotations to build a **question-family router** (action relation,
spatial relation, appearance, OCR/text, count/comparison, repeated-event,
plural). This is allowed training data and lets us use specialized prompts.
Validation/test questions must never be used to fit the router or thresholds.

### Stage B: anchor proposal and grounding ensemble

Do not trust one trigger frame. Generate 3--5 candidate anchors around each
VLM anchor and around event boundaries. At each candidate:

- Molmo or Florence-2 gives points/boxes from the rich description;
- Grounding DINO / OWLv2 gives open-vocabulary boxes;
- SAM2 image predictor turns points/boxes into masks;
- optional OCR detector is used only for text-object families.

Score candidates using four signals:

1. grounding model confidence;
2. agreement between independent grounders;
3. description-to-crop embedding similarity;
4. temporal/evidence consistency with the VLM event.

Select the best anchor per object, not merely the earliest or middle frame.
Retain alternatives when scores are close; they are useful for validation-time
model selection and for recovery after occlusion.

### Stage C: bidirectional tracklet graph

Use SAM2.1 Hiera-Large (or a newer SAM2-compatible checkpoint) from each
selected mask/box, forward and backward. Process at full required output frame
rate or a safe stride followed by interpolation/refinement; never assume that
10 FPS is enough for fast contacts or tiny objects.

For every tracklet/frame, retain mask, box, SAM confidence, area-change score,
image embedding, and visibility probability. Build an **AnchorGraph**:

- vertices are tracklet fragments from anchors and periodic re-detections;
- edges connect temporally adjacent fragments;
- edge cost combines mask IoU, box IoU, motion extrapolation, DINO/CLIP
  appearance similarity, and object-description similarity;
- Hungarian/min-cost-flow matching preserves IDs and merges forward/backward
  fragments;
- start a new fragment after an occlusion, then reconnect it using appearance
  and motion rather than letting SAM2 drift indefinitely.

This directly targets HOTA's association component and is safer than blindly
concatenating SAM2 outputs.

### Stage D: visibility and answer-set control

HOTA penalizes hallucinated detections. Calibrate a visible/not-visible gate on
train videos using the official labels:

- suppress masks with low tracker confidence, implausible area jumps, or weak
  crop-description similarity;
- allow gaps for occlusion instead of emitting a stale box;
- re-detect around predicted reappearance windows;
- for plural answers, emit exactly the semantically supported objects and keep
  one stable ID per physical object.

For “objects shown more than once” or repeated actions, use multiple temporal
segments and merge only when appearance/geometry supports the same identity.
For comparisons (“tallest”, “leftmost”), reason over candidate tracks first,
then emit only the selected object(s).

## Highest-upside additions beyond the winning recipe

### 1. Learn the semantic selector from train tracks

The train split gives us a rare supervised signal: each question points to
object-track IDs. Build a frozen visual/text embedding candidate ranker from
train examples. For each question, create positive crops from answer tracks and
hard negatives from same-category/non-answer tracks in the same video. Train a
small pairwise ranker on question text, VLM object description, crop embedding,
relative position, size, and temporal evidence. At test time, use it to rerank
Molmo/DINO/SAM proposals. This attacks the dominant error—choosing the wrong
similar object—without pretending 1,859 examples are enough to train a new VLM.

### 2. Calibrate boxes for the metric, not for segmentation beauty

SAM masks often produce tighter boxes than human object boxes. On a train
holdout, learn an expansion model conditioned on mask area, aspect ratio,
object family, and occlusion. Compare mask bounding box, detector box, their
union, and an anisotropically expanded box. Optimize sampled-frame IoU. This
can improve LocA without changing the tracker.

### 3. Treat tracking as a measurement problem

Run SAM2 at full decoded FPS internally, but maintain a sampled-frame output
view matching the annotation cadence. At each scored frame, fuse three
measurements: SAM2 mask box, CoTracker3 point-motion box, and periodic
question-conditioned detector box. Use a Kalman/constant-velocity prior only
as a proposal, never as an unconditional hallucinated detection. Reinitialize
after confidence collapse. CoTracker3 is especially valuable for fast motion
where a segmentation mask can drift between anchor frames.

### 4. Use negative evidence

Ask the VLM for frames where the object is *not* visible and model visibility as
a separate binary sequence. A track should contain gaps for genuine occlusion or
absence, and should reconnect only after appearance and geometry agree. This is
better than SAM2's tendency to continue a plausible-looking mask through a
long occlusion.

### 5. Separate answer cardinality from detection cardinality

First decide whether the answer is one object, a set, or a repeated-event set.
Then run a fixed number of grounding slots. For identical objects, initialize
all slots jointly and use appearance/trajectory assignment; do not independently
select the top box for each object, which causes ID swaps.

### 6. Audio is a selective expert

The test video archive has no separate audio archive, but the MP4s include
audio. Extract audio only for questions involving hitting, pouring, speaking,
noise, or temporal causality. Use audio as evidence for *when* an event happens,
not as a replacement for visual grounding. This recovers the subtle-motion cases
that the 1-FPS runner-up explicitly identified.

## Training and validation strategy

Use the 1,859 official train questions for:

- prompt examples grouped by question family;
- threshold/visibility calibration;
- a lightweight confidence/reranking model;
- tracker re-detection and association tuning.

Use the official evaluation code and a fixed train holdout for all decisions.
Keep validation untouched until the complete pipeline is frozen; use it only for
pipeline selection. Never prompt with validation/test labels or manually label
test videos.

Report HOTA decomposition (detection, association, localization) by question
family and object count. A single aggregate score hides whether a change helps
reasoning or merely causes hallucinated boxes.

## Modal execution plan

Keep the pipeline restartable and cache every stage in the existing Volume:

```text
/data/{split}/videos/*.mp4
/data/{split}/annotations.json
/data/cache/{video_id}/{question_id}/reasoning.json
/data/cache/{video_id}/{question_id}/anchors.json
/data/cache/{video_id}/{question_id}/tracklets/*.npz
/data/predictions/{run_name}.json
```

Recommended Modal functions:

1. `prepare_split`: archive download/extraction and metadata index;
2. `reason_batch`: VLM calls, rate-limited and cached;
3. `ground_video`: anchor ensemble on A100, sequential per GPU;
4. `track_video`: SAM2 + graph association on A100;
5. `assemble_and_validate`: CPU-only output assembly;
6. `evaluate_valid`: official HOTA evaluation and diagnostics.

Use Modal Secrets for API keys, not environment literals. Use a run manifest
(model revisions, prompt hash, git commit, thresholds) so an interrupted run
can resume and results remain reproducible.

## Build order

1. Add official annotation loader and train/valid HOTA evaluator.
2. Replace CSRT with SAM2.1 bidirectional tracking.
3. Implement VLM JSON reasoning + trigger candidates.
4. Add Molmo/Florence/Grounding-DINO grounding and candidate reranking.
5. Add visibility gating, re-detection, and ID association graph.
6. Tune only on train holdout; run validation once the pipeline is frozen.
7. Run test inference and validate the exact EvalAI JSON schema.

## Sources

- Official data/tools: https://github.com/google-deepmind/perception_test
- Official challenge page: https://perception-test-challenge.github.io/
- 2024 challenge summary: https://arxiv.org/abs/2411.19941
- SGVR@KAIST 2025 winner report: linked under Grounded videoQA on the official
  workshop page (Google Drive; report title “Pinpointing Trigger Moment…”)
- TutuAI 2025 runner-up report: linked under Grounded videoQA on the official
  workshop page (title “A Two-Stage VLM-based Approach…”)
- HOTA: https://arxiv.org/abs/2009.07736
