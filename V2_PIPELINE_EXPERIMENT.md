# V2 Pipeline Experiment

## Configuration

- Reasoning: `gemini-2.5-pro` with CORTEX v2 prompt (stronger trigger frame instruction,
  question-family routing, OCR alias rules, calibrated confidence)
- Grounding: `allenai/Molmo2-ER` (default) or `allenai/Molmo2-VideoPoint-4B`
- Tracking: `facebook/sam2.1-hiera-large`, bidirectional

All runs use pretrained inference only. No validation/test labels, manual annotations,
or training data are used.

## Controlled experiment gate

Must exceed 0.2062 mean HOTA on valid Q8/Q9/Q10 before full test inference.

### Step 1: Re-download updated datasets (Aug 11 organizer update)

```bash
# REQUIRED: Organizers updated datasets on Aug 11, 2026
modal run modal_app.py::download --split test
modal run modal_app.py::download --split valid
```

### Step 2: Re-cache CORTEX reasoning with gemini-2.5-pro

```bash
# For controlled slice first
modal run modal_app.py::batch_reason_pro --split valid --start 0 --limit 3 --cost-cap 3

# Check cache
modal run modal_app.py::inspect_cache --split valid --video-id video_10016 --question-id 8
modal run modal_app.py::inspect_cache --split valid --video-id video_10016 --question-id 9
modal run modal_app.py::inspect_cache --split valid --video-id video_10016 --question-id 10
```

### Step 3: Controlled slice with Molmo2-ER (default spatial model)

```bash
modal run modal_app.py::predict_cortex_sam2_v2 --split valid --video-id video_10016 \
    --question-id 8 --output v2_q8.json --output-step 30 --tracking-stride 30 \
    --spatial-model molmo2er
modal run modal_app.py::evaluate_hota --split valid --output v2_q8.json --allow-partial

modal run modal_app.py::predict_cortex_sam2_v2 --split valid --video-id video_10016 \
    --question-id 9 --output v2_q9.json --output-step 30 --tracking-stride 30 \
    --spatial-model molmo2er
modal run modal_app.py::evaluate_hota --split valid --output v2_q9.json --allow-partial

modal run modal_app.py::predict_cortex_sam2_v2 --split valid --video-id video_10016 \
    --question-id 10 --output v2_q10.json --output-step 30 --tracking-stride 30 \
    --spatial-model molmo2er
modal run modal_app.py::evaluate_hota --split valid --output v2_q10.json --allow-partial
```

### Step 4: If gate passes — full test inference

```bash
# Re-reason the full test split with gemini-2.5-pro (1,859 questions)
modal run modal_app.py::batch_reason_pro --split test --limit 1900 --cost-cap 1900

# Full test inference with v2 pipeline
modal run modal_app.py::batch_predict_v2 --split test \
    --output v2_test_submission.json --spatial-model molmo2er

# Validate locally
python validate.py  # or use modal endpoint

# Submit to EvalAI (private first, then public)
# Phase 5350, Method: StarAtNyte
```

### Step 5: If v2 with molmo2er < gate, try molmo2vp4b

```bash
modal run modal_app.py::predict_cortex_sam2_v2 --split valid --video-id video_10016 \
    --question-id 8 --output v2_vp4b_q8.json --output-step 30 --tracking-stride 30 \
    --spatial-model molmo2vp4b
# ... (same for Q9, Q10)
```

## Results

| shard | previous (v1) | v2-molmo2er | v2-molmo2vp4b |
|---|---:|---:|---:|
| Q8 | 0.000000 | | |
| Q9 | 0.049760 | | |
| Q10 | 0.278658 | | |
| mean | 0.109473 | | |

_To be filled in after experiments complete._

## Notes

- Best v1 full-test HOTA: 0.25 (Aug 22 private submission, method
  `Zero-shot video grounding pipeline`)
- Earlier v1 full-test HOTA: 0.21201318502426147 (submission 579655, private)
- Scale gate: 0.2062 on controlled slice
- Key upgrade vs v1: CORTEX v2 prompt + gemini-2.5-pro + Molmo2-ER or VP4B
- CRITICAL: Re-download datasets first (Aug 11 organizer update)
