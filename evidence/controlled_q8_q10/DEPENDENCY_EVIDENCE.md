# Evaluator dependency evidence

`modal_app.py` declares `pycocotools` and `tabulate` in the Modal evaluator image. The completed evaluator logs in this directory contain successful HOTA output and `App completed` footers with no dependency error. `evaluate_hota` imports `local_eval`, whose COCO evaluator path uses `pycocotools`. Local environment lacks this optional evaluator dependency; runtime proof is the successful Modal evaluator execution, not a local import.
