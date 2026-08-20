# Current model research for GVQA 2705

Checked current Hugging Face model cards and implementations, not just generic
leaderboards. None publishes a Perception Test GVQA score, so the final choice
must be decided on a train holdout using the official HOTA evaluator.

## Best candidates today

> Additional checks through August 2026 found two especially important new
> options: TimeLens2 for temporal evidence and Molmo2-ER for spatial/embodied
> pointing. They change the recommended decomposition below.

### 1. Molmo2-ER — best spatial specialist to test first

`allenai/Molmo2-ER` is a 4B Molmo2 derivative specialized for pixel-accurate
pointing, egocentric video, spatial reasoning, and temporal reasoning. Its card
reports a +17 point improvement over the Molmo2 starting point on embodied
benchmarks and competition with much larger closed models. It is only about
19 GB of artifacts and is easy to run alongside a tracker on one A100.

Use Molmo2-ER for anchor points, object cardinality, and difficult spatial
relations. It is a specialist, not a complete HOTA tracker.

### 2. Molmo2-8B — direct video-track specialist

`allenai/Molmo2-8B` is specifically trained for video point and video track
QA. Its model card documents `Molmo2-VideoPoint` and `Molmo2-VideoTrack` data,
and a tracking prompt that returns per-frame `<tracks ...>` coordinates. The
card recommends `max_fps=8` for tracking.

Why it is unusually relevant:

- one model can perform question understanding and temporal localization;
- outputs frame-indexed points and persistent point IDs;
- open weights/data and Apache-2.0 model license;
- approximately 35 GB of repository artifacts, feasible on an A100 80 GB with
  careful loading (avoid downloading duplicate variants);
- based on Qwen3-8B + SigLIP2, so it is newer than the 2025 Molmo used by the
  winner.

Limitation: its documented output is points/tracks, while the challenge wants
boxes. Use every predicted point as a SAM2 prompt, add local point clusters or
box prompts where available, and convert SAM2 masks to calibrated boxes. Do
not make a box by a fixed pixel radius around a point.

### 3. TimeLens2-8B — temporal oracle

`MCG-NJU/TimeLens2-8B` (arXiv:2607.17423, July 2026) is built on Qwen3-VL-8B
and is explicitly trained for multi-span temporal grounding. Its model card
reports 48.0 average mIoU on seven temporal-grounding benchmarks and an
18.1-point improvement over its Qwen3-VL-8B backbone. It returns all relevant
intervals, which is exactly what repeated-event and visibility questions need.

Use it to propose `visible_intervals` and trigger windows. Do not use its time
spans as object boxes. Run it at 2 FPS first, then refine boundary frames with
Molmo2-ER and SAM2.

### 4. Qwen3-VL-8B / 30B-A3B — semantic/grounding judge

The current Qwen3-VL cards advertise native long-context video, improved
spatial/occlusion reasoning, 2D grounding, and text-timestamp alignment. The
8B checkpoint is about 17.5 GB and is practical on one A100. The 32B checkpoint
is about 66.7 GB before runtime overhead and should use quantization or a larger
GPU.

Use Qwen3-VL for:

- answer object descriptions;
- cardinality and comparison questions;
- event intervals and trigger-frame proposals;
- hard-negative selection among candidate tracklets.

It is not a replacement for SAM2. Its box outputs must be parsed, validated,
and independently grounded.

### 5. Keye-VL-2.0-30B-A3B — strongest long-video API/local candidate

The current Keye-VL-2.0 model card reports 58.4 mIoU on Charades-TimeLens and
74.1 on LongVideoBench, with 256K sparse-attention context. It claims parity
with Gemini-class temporal grounding. Its architecture is attractive for
hour-long video, but this challenge's short clips do not require its full
context advantage. Test it as a high-quality reasoning critic if the 30B-A3B
runtime is stable on Modal.

### 6. InternVL3.5 — strong alternative / critic

InternVL3.5 has explicit thinking mode, parallel best-of-N reasoning, visual
PRM support, video inference examples, and grounding/video evaluation sections.
The 8B checkpoint is practical; larger MoE variants may require multiple GPUs.
Use it as a disagreement critic rather than the only model. Its best-of-N can
be useful for ambiguous spatial relations, but majority voting on free-form
answers is unsafe; vote after canonicalizing object attributes and anchor
frames.

### 7. GLM-4.6V-Flash — practical generalist

`zai-org/GLM-4.6V-Flash` is a roughly 9B local model with about 20.6 GB of
artifacts, much more practical than the 106B GLM-4.6V. It supports video and
visual grounding and is a useful independent judge. The model card recommends
SGLang/vLLM for video; Transformers inference is useful for a first smoke test.

### 8. GLM-4.5/4.6V large — powerful but not the first Modal choice

The model card claims strong video event recognition and precise grounding,
with direct bounding-box prompting and thinking mode. However, the listed
repository artifacts are roughly 215 GB and its 106B/12B-active architecture
has substantial memory/runtime requirements. It is a good hosted API candidate
or multi-GPU experiment, not a one-A100 baseline.

### 9. Gemini — still a useful oracle, not a dependency

Gemini remains attractive because it accepts video/audio directly and the
published SGVR solution achieved 0.4968 HOTA with Gemini 2.5 Pro + Molmo +
SAM2. But a local Molmo2/Qwen3-VL cascade removes API cost, improves
reproducibility, and avoids rate limits. Use Gemini only for low-confidence or
model-disagreement cases if budget permits.

## Recommended architecture now

```text
TimeLens2-8B (2 FPS)
    └─ all evidence intervals / event windows
             ↓
Molmo2-ER (8 FPS, interval-local crops)
    └─ spatial points, object count, persistent candidate IDs
             ↓
Qwen3-VL-8B or Keye-VL-2.0 critic
    └─ answer-set adjudication and hard-negative rejection
             ↓
SAM2.1 Hiera-Large from points/boxes, forward + backward
             ↓
CoTracker3 motion and reappearance proposals
             ↓
track graph + visibility gate + train-fitted box calibration
             ↓
official sparse frame IDs and JSON
```

This is a more principled division than asking one VLM to output both semantic
answers and precise trajectories. TimeLens2 answers *when*, Molmo2-ER answers
*where*, and SAM2 answers *how the object moves*.

For difficult cases:

```text
Molmo2 vs Qwen3-VL disagreement → InternVL3.5 or Gemini escalation
```

This is better than using Qwen or Gemini to generate a phrase and then asking
Grounding DINO to solve the entire task. Molmo2 supplies temporal evidence;
Qwen supplies semantic adjudication; SAM2 supplies pixel-accurate continuity.

## Important implementation details

1. **Frame mapping.** Molmo2's `max_fps=8` timestamps are not challenge frame
   indices. Convert timestamp to `round(timestamp * measured_frame_rate)`, then
   clamp to the video. The final output must use the dataset's sampled IDs,
   normally every 30 decoded frames.
2. **Point-to-box.** For each Molmo track ID, use its highest-confidence and
   most-visible points as SAM2 prompts. Prompt at several temporally separated
   points, not just the first point.
3. **Track IDs.** Preserve Molmo IDs as preferred identity labels, but use
   appearance/motion matching to reconnect after an occlusion.
4. **Boxes.** Compare SAM2 mask bounds, detector bounds, and expanded/fused
   bounds on train holdout. Human boxes are not guaranteed to equal tight mask
   bounds.
5. **Video resolution.** Run VLM at a resized copy for efficiency, but keep the
   original width/height and exact frame index for SAM2/output conversion.
6. **Audio.** Extract audio only for event families where it adds evidence;
   do not spend VLM context on audio for static appearance questions.
7. **No test leakage.** Train holdout can tune prompts, thresholds, and the
   ranker. Validation/test labels must not be used for selection or prompting.

## Practical ranking

| Rank | Model | Role | Modal fit |
|---|---|---|---|
| 1 | Molmo2-ER | spatial/video pointing | A100 80GB |
| 2 | TimeLens2-8B | multi-span temporal evidence | A100 80GB |
| 3 | Molmo2-8B | direct video point/track proposals | A100 80GB |
| 4 | Qwen3-VL-8B | semantic/temporal judge | A100 80GB |
| 5 | Keye-VL-2.0-30B-A3B | long-video critic | multi-GPU/quantized |
| 6 | GLM-4.6V-Flash | independent local critic | A100 80GB |
| 7 | InternVL3.5-8B | best-of-N critic | A100 80GB |
| 8 | Gemini | selective oracle | API/cost dependent |

## Sources checked

- Molmo2-8B model card: https://huggingface.co/allenai/Molmo2-8B
- Qwen3-VL-32B model card: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct
- InternVL3.5 model card: https://huggingface.co/OpenGVLab/InternVL3_5-8B
- GLM-4.5V model card: https://huggingface.co/zai-org/GLM-4.5V
- SAM2: https://github.com/facebookresearch/sam2
- CoTracker3: https://github.com/facebookresearch/co-tracker
- Molmo2-ER: https://huggingface.co/allenai/Molmo2-ER
- TimeLens2: https://huggingface.co/MCG-NJU/TimeLens2-8B
- TimeLens2 paper: https://arxiv.org/abs/2607.17423
- VideoChat3 paper: https://arxiv.org/abs/2607.14935
- Keye-VL-2.0-30B-A3B: https://huggingface.co/Kwai-Keye/Keye-VL-2.0-30B-A3B
- GLM-4.6V-Flash: https://huggingface.co/zai-org/GLM-4.6V-Flash
- MolmoAct2 / Molmo2-ER paper: https://arxiv.org/abs/2605.02881
