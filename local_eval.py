"""Official-style HOTA evaluation for Grounded VideoQA validation outputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def _iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(1e-8,aa+bb-inter)


def evaluate(prediction: dict, labels: dict, only_present: bool = False) -> dict:
    # TrackEval currently uses removed NumPy aliases on modern images.
    if not hasattr(np, "float"): np.float = float
    if not hasattr(np, "int"): np.int = int
    from trackeval.metrics import HOTA
    scores=[]; det=[]; ass=[]; loc=[]
    for vid,item in labels.items():
        pred_q=prediction.get(vid,{}).get("grounded_question",{})
        objects=item.get("object_tracking",[])
        for q in item.get("grounded_question",[]):
            key=str(q["id"])
            if only_present and key not in pred_q and q["id"] not in pred_q:
                continue
            gt=[objects[i] for i in q.get("answers",[])]
            pr=pred_q.get(key,pred_q.get(q["id"],[]))
            gt_by={}; pr_by={}; gt_boxes={}; pr_boxes={}
            for gi,t in enumerate(gt):
                start=next((i for i,x in enumerate(t.get("initial_tracking_box",[])) if x),0)
                for fid,box in zip(t.get("frame_ids",[])[start:],t.get("bounding_boxes",[])[start:]):
                    gt_by.setdefault(int(fid),[]).append(gi); gt_boxes[(int(fid),gi)]=box
            for ti,t in enumerate(pr[:10]):
                for fid,box in zip(t.get("frame_ids",[]),t.get("bounding_boxes",[])):
                    pr_by.setdefault(int(fid),[]).append(ti); pr_boxes[(int(fid),ti)]=box
            frames=sorted(gt_by)
            if not frames: continue
            seq={"num_gt_ids":len(gt), "num_tracker_ids":len(pr[:10]),
                 "num_gt_dets":sum(len(x) for x in gt_by.values()),
                 "num_tracker_dets":sum(len(x) for x in pr_by.values()),
                 "gt_ids":[np.asarray(gt_by[f],dtype=int) for f in frames],
                 "tracker_ids":[np.asarray(pr_by.get(f,[]),dtype=int) for f in frames],
                 "similarity_scores":[]}
            for f in frames:
                g=gt_by[f]; p=pr_by.get(f,[])
                seq["similarity_scores"].append(np.asarray([[_iou(gt_boxes[(f,gi)],pr_boxes[(f,ti)]) for ti in p] for gi in g],dtype=float))
            r=HOTA().eval_sequence(seq)
            scores.append(float(np.mean(r["HOTA"]))); det.append(float(np.mean(r["DetA"]))); ass.append(float(np.mean(r["AssA"]))); loc.append(float(np.mean(r["LocA"])))
    return {"HOTA":float(np.mean(scores)) if scores else 0., "DetA":float(np.mean(det)) if det else 0.,
            "AssA":float(np.mean(ass)) if ass else 0., "LocA":float(np.mean(loc)) if loc else 0., "questions":len(scores)}


def evaluate_files(prediction_path: str, label_path: str, only_present: bool = False) -> dict:
    return evaluate(json.loads(Path(prediction_path).read_text()), json.loads(Path(label_path).read_text()), only_present=only_present)
