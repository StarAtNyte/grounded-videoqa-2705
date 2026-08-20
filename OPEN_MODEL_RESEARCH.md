# Open-model alternatives to Gemini for full-test inference

## Highest-value candidate: Molmo2-VideoPoint-4B

Hugging Face: https://huggingface.co/allenai/Molmo2-VideoPoint-4B

This Apache-2.0 model is based on Qwen3-4B and is explicitly fine-tuned for video pointing/counting. Its official example accepts a video and query and emits frame-indexed point/track coordinates (`<points>`/`<tracks>`), eliminating the Gemini reasoning API dependency. The model card reports VideoCountEval accuracy 36.8 and close accuracy 56.5, comparable to Gemini 2.5 Pro on that evaluation. It is the best fit for replacing the Gemini→Molmo cascade with one local video-pointing model, followed by box construction and SAM2.

## Higher-scoring but less plug-and-play candidates

- **Seed1.6-VL-thinking** — the 2025 TutuAI runner-up used it for direct first-frame/box grounding and reported HOTA 0.4304. It is not open-weight and requires an API/provider, but it is the strongest evidence-backed replacement if access is available.
- **RGA3 / UniGR-7B** — https://github.com/qirui-chen/RGA3-release and https://huggingface.co/SurplusDeficit/UniGR-7B. Open code/checkpoint for object-centric video QA with visual grounding/referring and video segmentation. More likely to improve spatial identification than a generic VLM, but its output adapter is substantial.
- **ScanFocus** — https://github.com/TenMinutes209/ScanFocus. Reports strong spatio-temporal grounding numbers (HC-STVG v2 m_vIoU 41.7), but is a benchmark-specific tube grounder, not a question-conditioned Perception Test model; checkpoint/training setup is heavier.
- **DeViL-7B** — https://huggingface.co/gaostar/DeViL-7B and https://github.com/gaostar123/DeViL. Detector-empowered spatio-temporal grounding; reports 43.1 m_vIoU on HC-STVG and 14.33 FPS. More specialized for dense temporal grounding, but integration is less direct than Molmo2-VideoPoint.
- **TimeLens2-8B** — https://huggingface.co/MCG-NJU/TimeLens2-8B and https://github.com/TencentARC/TimeLens. Qwen3-VL-based temporal grounding model returning relevant time spans, not spatial boxes; useful for selecting temporal windows but insufficient alone for GVQA tracks.
- **VideoITG-8B** — https://huggingface.co/nvidia/VideoITG-8B and https://github.com/NVlabs/VideoITG. Apache-2.0 code, instructed temporal grounding and frame selection; temporal rather than direct spatial-box output.
- **Grounded-VideoLLM** — https://github.com/WHB139426/Grounded-Video-LLM. Public inference/training/checkpoints for temporal grounding, but not a direct Perception Test box-track solution.
- **Qwen3-VL / InternVL3** — strong general open video VLMs with visual grounding support, but require custom prompt/parser work for frame-indexed points and are larger than Molmo2-VideoPoint.

## Recommended experiment

Implement a local `Molmo2-VideoPoint-4B` adapter that:

1. Sends each test video/question to the local model (no Gemini API quota).
2. Parses official frame-indexed point/track output.
3. Converts points to small boxes or uses points directly as SAM2 prompts.
4. Runs bidirectional SAM2 and official serialization.
5. Preserves raw model text, parsed coordinates, manifests, heartbeats, and hashes.

First compare it on the existing controlled Q8/Q9/Q10 slice. If it beats 0.2062, use it for full test inference; otherwise document the result and stop. Do not use validation/test labels for training or prompt construction.
