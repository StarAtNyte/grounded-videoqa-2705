# Molmo2-8B controlled experiment

Implemented opt-in `predict_molmo2_sam2`: pretrained `allenai/Molmo2-8B` consumes the video/question and emits frame-indexed point tracks; the selected object identity initializes bidirectional `facebook/sam2.1-hiera-large` tracking and official serialization. Existing CORTEX/reference/proposal modes are unchanged. All runs used pretrained inference only; no validation/test labels, manual annotations, or training data were used.

Commands used for the controlled slice:

```bash
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::predict_molmo2_sam2 --split valid --video-id video_10016 --question-id 8 --output molmo2_q8.json --output-step 30 --tracking-stride 30
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output molmo2_q8.json --allow-partial
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::predict_molmo2_sam2 --split valid --video-id video_10016 --question-id 9 --output molmo2_q9.json --output-step 30 --tracking-stride 30
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output molmo2_q9.json --allow-partial
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::predict_molmo2_sam2 --split valid --video-id video_10016 --question-id 10 --output molmo2_q10.json --output-step 30 --tracking-stride 30
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output molmo2_q10.json --allow-partial
```

| mode | Q8 | Q9 | Q10 | mean |
|---|---:|---:|---:|---:|
| baseline | 0.000000 | 0.049760 | 0.278658 | 0.109473 |
| Grounding-DINO proposal | 0.005617 | 0.083741 | 0.278658 | 0.122672 |
| Molmo2-8B direct points | 0.005617 | 0.050312 | 0.284290 | 0.113406 |

Artifacts `evidence/molmo2_q8.json`, `molmo2_q9.json`, and `molmo2_q10.json` pass `python -m grounded_videoqa.tools.validate`; evaluator logs are in the same evidence directory. Molmo2 improves Q10 over both prior modes, matches proposal Q8, but underperforms proposal Q9 and the proposal mean. Since mean `0.113406 < 0.2062`, no full test inference was performed.

## Audit follow-up

The legacy `run_advanced` path had an undefined `tracking_stride`; it now defaults to 30 and imports its required `cv2` dependency without changing caller behavior. Modal regression smoke command `modal run -m grounded_videoqa.jobs.modal_app::predict_advanced --split valid --video-id video_10016 --question-id 8 --output regression_smoke.json --backend sam2` completed and wrote an artifact (diagnostics reported zero candidates, but no runtime exception). Molmo2 manifests were downloaded to `evidence/molmo2_q8.json.manifest.json`, `molmo2_q9.json.manifest.json`, and `molmo2_q10.json.manifest.json`; each contains the raw response and parsed diagnostics. Validator outputs and hashes are preserved alongside the three artifacts.

## Parser/recovery rerun

This corrected-parser candidate also used pretrained `allenai/Molmo2-8B` plus `facebook/sam2.1-hiera-large`, with no validation/test labels, manual annotations, or training data; only model-generated points and the question text were used.

Exact corrected-parser commands (Modal workspace `baptinjean`):

```bash
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::predict_molmo2_sam2 --split valid --video-id video_10016 --question-id 8 --output molmo2_fixed_q8.json --output-step 30 --tracking-stride 30
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output molmo2_fixed_q8.json --allow-partial
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::predict_molmo2_sam2 --split valid --video-id video_10016 --question-id 9 --output molmo2_fixed_q9.json --output-step 30 --tracking-stride 30
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output molmo2_fixed_q9.json --allow-partial
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::predict_molmo2_sam2 --split valid --video-id video_10016 --question-id 10 --output molmo2_fixed_q10.json --output-step 30 --tracking-stride 30
MODAL_PROFILE=baptinjean modal run -m grounded_videoqa.jobs.modal_app::evaluate_hota --split valid --output molmo2_fixed_q10.json --allow-partial
```

After preserving Molmo2 object IDs and enabling recovery, the rerun measured Q8 `0.004267`, Q9 `0.038440`, Q10 `0.249480`, mean `0.097396`. These artifacts, manifests, raw responses, validator outputs, metric logs, and hashes are named `evidence/molmo2_fixed_q{8,9,10}.*`. The ID-preserving parser is correct and auditable, but this particular selection heuristic reduced the controlled mean; it is not promoted or scaled.
