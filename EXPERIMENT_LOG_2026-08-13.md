
## Auditor remediation: preserved controlled proposal artifact

Proposal mode was implemented as a real opt-in Grounding-DINO-Tiny box proposal passed to SAM2, with diagnostics including `proposal_grounding` and `proposal_box`. Q8 proposal shard completed successfully. Evaluator output: HOTA 0.005616952985374038, DetA 0.005616952985374038, AssA 0.005616952985374038, LocA 0.9064502102220888, questions 1. The artifact was written to Modal Volume as `predictions/proposal_q8.json`; raw local logs are `/tmp/proposal_q8_final.log` and `/tmp/proposal_q8_eval.log`.

Batch shard failures now raise after writing a failed heartbeat instead of recording empty success predictions.
Gemini cache provenance/cost cap was added: `batch_reason(..., cost_cap=10)` rejects slices larger than the cap and writes model/call-count/cap provenance into cache records.
