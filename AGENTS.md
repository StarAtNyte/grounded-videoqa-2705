# Challenge 2705 — Grounded VideoQA

Read this file before making changes in this directory.

## Official links

- EvalAI: https://eval.ai/web/challenges/challenge-page/2705/overview
- Phases: https://eval.ai/web/challenges/challenge-page/2705/phases
- Workshop: https://perception-test-challenge.github.io/
- Dataset and tools: https://github.com/google-deepmind/perception_test
- Perception Test paper: https://arxiv.org/abs/2305.13786

## Schedule

- Validation phase: June 5–August 27, 2026, 23:59:59 UTC.
- Test phase: July 15–August 27, 2026, 23:59:59 UTC.
- Technical report deadline: August 31, 2026, 23:59:59 UTC.
- Winner announcement: September 3, 2026.

## Task

Given a video and a natural-language question/query, locate and track throughout the video the object or objects that constitute the answer. The output is not a text answer: it is one or more temporally consistent bounding-box tracks.

The system must jointly solve:

- Language and video understanding to determine the answer object.
- Spatial grounding to find that object in frames.
- Tracking through motion, scale changes, occlusion, and reappearance.
- Multi-object handling when the question's answer contains multiple objects.

Videos are high-resolution RGB with audio, up to roughly 35 seconds, 30 FPS, and at most 1080p.

## Data

| Split | Videos | Questions | Approximate video download |
| --- | ---: | ---: | ---: |
| Train | 586 | 1,859 | 7.3 GB |
| Validation | 1,545 | 3,051 | 19.3 GB |
| Test | 932 | 1,859 | 11.3 GB |

Keep raw videos, extracted frames, predictions, and checkpoints out of version control.

## Evaluation and submission

- Primary metric: HOTA, which measures detection and association quality.
- Output every required frame/object track in the organizer's exact schema.
- Maintain stable track identities; good boxes with frequent identity fragmentation can still score poorly.
- Add a local schema validator and visualize sampled predicted tracks before submission.
- Use the validation phase only as a pipeline and evaluation check, not as training or prompting data.

## Recommended implementation

Suggested modular pipeline:

1. Use a VLM or language-grounding model to infer the visual identity of the answer.
2. Detect or segment candidate objects on strategically selected anchor frames.
3. Initialize a video object tracker from the strongest candidate masks or boxes.
4. Propagate tracks forward and backward.
5. Re-detect after occlusions and reconcile identities.
6. Convert masks to boxes if required and serialize them in the official format.

Candidate component categories include open-vocabulary detection, text-conditioned segmentation, video object segmentation, and multi-object tracking. Run heavy components sequentially to control VRAM. Modal A100 80GB access is confirmed, but the pipeline should remain restartable and cache intermediate results to protect credits.

## Rules

- Do not manually label test videos.
- Process each test video/question pair independently; do not exploit correlations among test questions.
- Do not train, self-supervise, or prompt using validation or test data.
- The official train split and eligible external datasets may be used.
- Public model APIs are permitted, but grounding/tracking outputs must still satisfy the required schema.
- A public test submission and technical report are required for prize eligibility.

