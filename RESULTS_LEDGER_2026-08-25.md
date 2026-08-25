# Results and experiment ledger — 25 Aug 2026

This file separates full-test EvalAI results from controlled validation-slice
results. Scores from the Q8/Q9/Q10 slice must never be described as full-test
scores.

## Full-test submissions

| Submission | Pipeline | HOTA | Visibility/current state |
|---|---|---:|---|
| `580601` | Gemini 3.7 Flash high-thinking CORTEX v2 + Molmo2-ER + SAM2.1 Hiera-Large; one persistent empty recovered with Molmo2-VideoPoint-4B | **0.3676459789276123** | Submitted private; later public, rank 4 |
| `579806` | Gemini CORTEX reasoning + Molmo-7B grounding + SAM2.1 Hiera-Large | 0.2519000172615051 | Private |
| `579655` | Earlier zero-shot production pipeline | 0.21201318502426147 | Private |

Submission `580601` used artifact SHA-256
`35f9526e6e577ed56d3669114edcefda1bccc55b386c8e09156b9b306cb08777`.
It contains 932 videos, 1,859 questions, 2,934 tracks, and zero empty answers.

## Live test leaderboard

Authenticated and unauthenticated reads of EvalAI challenge-phase split `7004`
returned the same ten public entries on 25 Aug 2026.

| Rank | Team | Method | Submission | HOTA |
|---:|---|---|---:|---:|
| 1 | Banana Bats | `fusion` | 580596 | **0.6620005965232849** |
| 2 | tibo | blank | 580552 | 0.4941996932029724 |
| 3 | SRUKR | blank | 578838 | 0.42936691641807556 |
| 4 | StarAtNyte | `Submission` | 580601 | **0.3676459789276123** |
| 5 | Wind_Rain_Tower | blank | 579935 | 0.2271479219198227 |
| 6 | ATD0709 | `ATD0709` | 578618 | 0.22395506501197815 |
| 7 | BoundedAccord | `Bounded Grounding DINO-Qwen Tracker` | 579691 | 0.2025170475244522 |
| 8 | amanupg | `run-g0` | 580531 | 0.16397108137607574 |
| 9 | HHHHHHHHHH | blank | 580311 | 0.14954611659049988 |
| 10 | Host_96594_Team | `zero-shot random baseline` | 578561 | 0.001085077179595828 |

The gap from StarAtNyte to Banana Bats is `0.2943546175956726` HOTA.
StarAtNyte is at 55.53% of the leader's score.

## Controlled validation experiments

All rows below use valid `video_10016`, questions Q8/Q9/Q10, unless noted.
They are diagnostic experiments, not full validation or test scores.

| Candidate | Q8 | Q9 | Q10 | Mean | Decision |
|---|---:|---:|---:|---:|---|
| v1 reference | 0.000000 | 0.049760 | 0.278658 | 0.109473 | Baseline |
| Grounding-DINO proposal | 0.005617 | 0.083741 | 0.278658 | 0.122672 | Rejected |
| Gemini direct frame/box | 0.000000 | 0.026345 | 0.009438 | 0.011928 | Rejected |
| Proposal/point ensemble | 0.000000 | 0.004479 | 0.279286 | 0.094588 | Rejected |
| Molmo2-8B direct points | 0.005617 | 0.050312 | 0.284290 | 0.113406 | Rejected |
| Molmo2 corrected parser/recovery | 0.004267 | 0.038440 | 0.249480 | 0.097396 | Rejected |
| Molmo2 multi-point | 0.001350 | 0.179848 | 0.233892 | 0.138363 | Rejected |
| Proposal/point agreement gate | 0.000000 | 0.004479 | 0.279286 | 0.094588 | Rejected |
| Fixed proposal/recovery | 0.005617 | 0.581634 | 0.001795 | 0.196349 | Rejected; unstable across questions |
| v2 Molmo2-ER + Gemini 3.7 | 0.000000 | 0.671618 | **0.528002** | **0.399873** | Passed gate; scaled to test |

The v2 controlled mean exceeded the `0.2062` gate but only used three
questions. Its `0.399873` mean overestimated the full-test `0.367646` by
`0.032227`. Q10's `0.528002` is the source of the mistaken “0.5+” impression;
it is one question, not the submission score.

Detailed commands, artifacts, hashes, and evaluator logs remain in:

- `CONTROLLED_EXPERIMENT.md`
- `DIRECT_GROUNDING_EXPERIMENT.md`
- `ENSEMBLE_PROMPT_EXPERIMENT.md`
- `MOLMO2_EXPERIMENT.md`
- `MULTIPOINT_GATING_EXPERIMENT.md`
- `GROUNDING_FIX_EXPERIMENT.md`
- `V2_PIPELINE_EXPERIMENT.md`
- `evidence/`

## Banana Bats method evidence and likely gap

The Banana Bats page supplied by the user describes four routes and labels the
public submission method `fusion` on EvalAI. Its primary route is Gemini + SAM
3 with noun-phrase extraction, object sessions, evidence-window refinement,
10-frame stride, start/end-frame interpolation, question-type-aware stride,
and guard rails. It also lists Gemini-only, Gemini + Molmo 2, and Molmo 2
baselines, implying that `fusion` may combine or arbitrate multiple routes.

Compared with our production run, the highest-confidence architectural gaps
are:

1. **Temporal resolution:** their stated 10-frame stride versus our fixed
   30-frame grounding/tracking stride.
2. **Temporal boundaries:** explicit start/end interpolation and evidence
   windows versus our sparse trigger-anchor propagation.
3. **Tracker/segmenter generation:** SAM 3 object sessions versus SAM2.1.
4. **Routing and fusion:** question-type-aware stride and multiple reasoning
   routes versus one primary Molmo2-ER route with only an empty-result fallback.
5. **Guard rails/refinement:** a final Gemini evidence refinement stage is
   described; our pipeline does not perform an equivalent track-aware final
   arbitration.

These are public method claims, not a recovered implementation. The next
experiments should isolate each factor on a substantially larger train
holdout—not reuse validation/test labels or prompt on test examples—before any
new full-test submission.
