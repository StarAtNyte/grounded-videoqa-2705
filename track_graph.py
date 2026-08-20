"""Confidence-aware tracklet fusion for re-detection and occlusion recovery."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gqa_core import DenseTrack, box_iou, fuse_boxes, gate_visibility

@dataclass
class Observation:
    frame_id: int
    box: list[float]
    confidence: float
    source: str


def fuse_observations(observations: list[Observation], output_ids: list[int],
                      min_conf: float = .18) -> DenseTrack:
    """Fuse detector/SAM/flow measurements without inventing missing frames."""
    by_frame: dict[int, list[Observation]] = {}
    for o in observations:
        if o.confidence >= min_conf: by_frame.setdefault(o.frame_id, []).append(o)
    boxes, confs, frames = [], [], []
    for fid in output_ids:
        obs = by_frame.get(fid, [])
        if not obs: continue
        # SAM is spatially precise; detector is generally better at full extent.
        priority = {"sam2": 1.0, "detector": .85, "flow": .65, "csrt": .35}
        weights = [max(1e-3, o.confidence * priority.get(o.source, .5)) for o in obs]
        boxes.append(fuse_boxes([o.box for o in obs], weights)); confs.append(min(1., max(weights)))
        frames.append(fid)
    return DenseTrack(0, frames, boxes, confs, source="fused")


def match_tracklets(left: list[DenseTrack], right: list[DenseTrack],
                    boundary: int, max_cost: float = .75) -> list[tuple[int,int]]:
    """Greedy gated association; replace with min-cost-flow for many tracks."""
    pairs, used = [], set()
    scored = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            ba, bb = a.at(boundary), b.at(boundary)
            if ba is None or bb is None: continue
            cost = 1 - box_iou(ba, bb)
            scored.append((cost, i, j))
    for cost, i, j in sorted(scored):
        if cost <= max_cost and i not in used and j not in used:
            pairs.append((i, j)); used |= {i, j}
    return pairs


def calibrate_box(box: list[float], area_bucket: str = "medium",
                  family: str = "generic") -> list[float]:
    """Conservative defaults; coefficients should be fit on train holdout."""
    scales = {"tiny": 1.18, "small": 1.10, "medium": 1.05, "large": 1.02}
    return __import__("gqa_core").expand_box(box, scales.get(area_bucket, 1.05), scales.get(area_bucket, 1.05))
