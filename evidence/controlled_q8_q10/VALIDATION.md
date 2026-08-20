# Controlled artifact validation capture

The Modal Volume is the authoritative storage for the six prediction JSON artifacts. All six were downloaded into this directory and validated with the official local command `python validate.py <artifact>`, each reporting `OK` and one video. Per-artifact results are in `validation_results.json`. The exact remote payload can be streamed with:

```bash
MODAL_PROFILE=baptinjean modal run modal_app.py::export_controlled_artifact --output baseline_q8.json > baseline_q8.export.json
python validate.py baseline_q8.export.json
```

Repeat for `baseline_q9.json`, `baseline_q10.json`, `proposal_q8.json`, `proposal_q9.json`, and `proposal_q10.json`. The export includes a manifest plus payload and SHA-256; save the payload portion as the official artifact before running `validate.py`, because the manifest wrapper itself is not the official submission schema. The downloaded JSON payloads and local validator outputs are repository-preserved alongside this file.
