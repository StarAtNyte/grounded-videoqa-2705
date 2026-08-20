"""Dataset, geometry, visibility, and submission primitives for GVQA 2705.

Everything here is CPU-testable and independent of a particular VLM/tracker.
The model layer produces dense tracklets; this module makes them metric-safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence
import math

import numpy as np


def sampled_frame_ids(num_frames: int, step: int = 30) -> list[int]:
    """Official GVQA labels are sampled by frame index (normally every 30)."""
    if num_frames <= 0 or step <= 0:
        return []
    return list(range(0, num_frames, step))


def normalize_box(box: Sequence[float], width: int, height: int) -> list[float]:
    """Convert pixel xyxy to clipped normalized xyxy."""
    x1, y1, x2, y2 = map(float, box)
    return [max(0., min(1., x1 / width)), max(0., min(1., y1 / height)),
            max(0., min(1., x2 / width)), max(0., min(1., y2 / height))]


def clip_box(box: Sequence[float]) -> np.ndarray:
    b = np.asarray(box, dtype=np.float32).reshape(4)
    b[[0, 2]] = np.clip(b[[0, 2]], 0, 1)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, 1)
    if b[2] < b[0]: b[0], b[2] = b[2], b[0]
    if b[3] < b[1]: b[1], b[3] = b[3], b[1]
    return b


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = clip_box(a), clip_box(b)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return float(inter / max(area_a + area_b - inter, 1e-8))


def expand_box(box: Sequence[float], x_scale: float = 1.04,
               y_scale: float = 1.04) -> list[float]:
    b = clip_box(box); cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    w, h = (b[2]-b[0])*x_scale, (b[3]-b[1])*y_scale
    return clip_box([cx-w/2, cy-h/2, cx+w/2, cy+h/2]).tolist()


def median_filter_boxes(boxes: Sequence[Sequence[float]], radius: int = 1) -> list[list[float]]:
    """Robust temporal denoising; preserves gaps represented by None."""
    out = []
    for i, box in enumerate(boxes):
        if box is None:
            out.append(None); continue
        vals = [np.asarray(x) for x in boxes[max(0,i-radius):i+radius+1] if x is not None]
        out.append(np.median(vals, axis=0).tolist() if vals else None)
    return out


@dataclass
class DenseTrack:
    object_id: int
    frame_ids: list[int]
    boxes: list[list[float]]
    confidences: list[float] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    source: str = "unknown"

    def __post_init__(self):
        if len(self.frame_ids) != len(self.boxes):
            raise ValueError("frame_ids and boxes length mismatch")
        self.boxes = [clip_box(x).tolist() for x in self.boxes]
        if not self.confidences:
            self.confidences = [1.] * len(self.boxes)
        if len(self.confidences) != len(self.boxes):
            raise ValueError("confidences and boxes length mismatch")

    def at(self, frame_id: int):
        try: return self.boxes[self.frame_ids.index(frame_id)]
        except ValueError: return None

    def to_submission(self, score: float | None = None) -> dict:
        return {"id": int(self.object_id), "score": float(score if score is not None else np.mean(self.confidences)),
                "bounding_boxes": self.boxes, "frame_ids": self.frame_ids}


def fuse_boxes(boxes: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    """Confidence-weighted box fusion, useful for SAM + detector + flow."""
    if not boxes: raise ValueError("cannot fuse empty boxes")
    w = np.maximum(np.asarray(weights, dtype=np.float32), 1e-4); w /= w.sum()
    return clip_box(np.sum(np.asarray([clip_box(x) for x in boxes]) * w[:, None], axis=0)).tolist()


def association_cost(a: DenseTrack, b: DenseTrack, frame_id: int) -> float:
    """Low cost means likely same identity at a re-detection boundary."""
    ba, bb = a.at(frame_id), b.at(frame_id)
    if ba is None or bb is None: return 1e3
    iou_cost = 1 - box_iou(ba, bb)
    return iou_cost


def gate_visibility(boxes: Sequence[Sequence[float] | None], confidences: Sequence[float],
                    min_conf: float = .18, max_area_jump: float = 4.0) -> list[bool]:
    """Conservative gate: suppress low-confidence and implausible jumps."""
    visible, prev_area = [], None
    for box, conf in zip(boxes, confidences):
        if box is None or conf < min_conf:
            visible.append(False); continue
        b = clip_box(box); area = max(1e-7, float((b[2]-b[0])*(b[3]-b[1])))
        ok = prev_area is None or max(area/prev_area, prev_area/area) <= max_area_jump
        visible.append(ok)
        if ok: prev_area = area
    return visible
