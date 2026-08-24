# EvalAI submission results

## Best test submission — 0.251900 HOTA

The team confirmed the following finished EvalAI test submission on 24 Aug
2026:

- Method: `Zero-shot video grounding pipeline`
- Method description: `Grounded VideoQA baseline pipeline (zero-shot)`
- Submitted at: Aug 22, 2026 2:36:22 PM (as displayed by EvalAI)
- Status: finished
- Execution time: `11.977851` seconds
- Submission ID: `579806`
- HOTA: **`0.2519000172615051`** (verified with the authenticated EvalAI API)
- Visibility: private at the time of confirmation; eligible to be toggled public
- Provisional public rank on 24 Aug: **1**

The live public test leaderboard at verification time was led by
`Wind_Rain_Tower` at `0.2271479219198227`, followed by `ATD0709` at
`0.22395506501197815`. Therefore `0.2519000172615051` becomes rank 1 when made public.

### What produced the 0.251900 result

This is the established zero-shot v1 production path, rather than the unfinished
v2 experiment:

1. CORTEX-style Gemini video reasoning converts each question into a visual
   answer-object description and selects a high-quality trigger frame.
2. Molmo-7B-O-0924 spatial grounding points to the described answer object on
   the trigger frame.
3. SAM2.1 Hiera-Large propagates the object track forward and backward while
   retaining stable track identity.
4. The submission assembler converts masks to normalized bounding boxes and
   emits the official sparse frame IDs in the organizer's JSON schema.

The method is pretrained/API-based zero-shot inference. It does not train or
prompt on validation/test labels. The submitted-file hash is still unavailable
locally; the submission ID and full-precision score above are authoritative API
results rather than inferred values.

## Earlier private test submission

- EvalAI phase: `5350` (Test Phase)
- Submission ID: `579655`
- Visibility: private
- Method: `StarAtNyte`
- Evaluation mode: `zero-shot`
- Contact: `khanalnitij20@gmail.com`
- Artifact: `test_submission.json` (`932` videos / `1,859` questions)
- SHA-256: `def19142c2ac8132d04e175cd2bf1085aecb86b194b7f16f98fbc5388be7b160`
- Status: evaluated
- HOTA: `0.21201318502426147`

This private submission was explicitly authorized as an experimental attempt. It
is not a public leaderboard result and is not eligible for final competition
ranking while private.
