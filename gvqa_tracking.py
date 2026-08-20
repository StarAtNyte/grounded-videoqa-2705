"""Tracking backends and metric-safe track assembly.

The OpenCV backend is a deterministic fallback. The SAM2 backend can be
plugged in without changing the output contract.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Iterable
from gqa_core import DenseTrack, normalize_box, clip_box, expand_box, sampled_frame_ids


def _tracker():
    if hasattr(cv2, "TrackerCSRT_create"): return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()


def _frame(cap, index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)); ok, frame = cap.read()
    return frame if ok else None


def csrt_bidirectional(video: str | Path, anchor_frame: int, box_px: Iterable[float],
                       object_id: int = 0, output_step: int = 30,
                       expand: float = 1.04) -> DenseTrack:
    """Bidirectional fallback; emits only official sampled frame IDs.

    Seeking is intentional here: it keeps memory bounded for 1080p videos. The
    production SAM2 backend should replace this function for quality.
    """
    cap = cv2.VideoCapture(str(video)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    anchor = _frame(cap, anchor_frame)
    if anchor is None: raise RuntimeError(f"cannot read anchor frame {anchor_frame}")
    x1,y1,x2,y2 = map(float, box_px)
    # OpenCV's Python tracker binding requires integer tuple values.
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    x1, y1 = min(w - 3, max(0, x1)), min(h - 3, max(0, y1))
    x2, y2 = min(w - 1, max(x1 + 2, x2)), min(h - 1, max(y1 + 2, y2))
    box=(x1, y1, max(2, x2-x1), max(2, y2-y1))
    points: dict[int, tuple[list[float], float]] = {}
    for direction in (1, -1):
        tracker = _tracker(); tracker.init(anchor, box)
        if direction == 1:
            # Sequential decoding is dramatically faster than seeking every
            # frame (especially for long-GOP MP4s on a Modal Volume).
            cap_f = cv2.VideoCapture(str(video)); cap_f.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame)
            indices = range(anchor_frame, n)
            frames = ((fid, cap_f.read()[1]) for fid in indices)
        else:
            cap_f = None
            # Reverse decoding is not supported by OpenCV. We only need
            # scored frames for this fallback; SAM2 handles dense reverse
            # propagation in the production backend.
            indices = [anchor_frame] + [fid for fid in range((anchor_frame // output_step) * output_step, -1, -output_step) if fid != anchor_frame]
            frames = ((fid, _frame(cap, fid)) for fid in indices)
        for fid, frame in frames:
            if frame is None: continue
            if fid != anchor_frame:
                ok, rect = tracker.update(frame)
                if not ok: continue
                x,y,bw,bh = rect; rect=(x,y,x+bw,y+bh)
            else: rect=(x1,y1,x2,y2)
            if fid % output_step == 0 or fid == anchor_frame:
                nb = normalize_box(rect, w, h)
                points[fid] = (expand_box(nb, expand, expand), .75 if fid == anchor_frame else .35)
        if cap_f is not None: cap_f.release()
    cap.release()
    ids = sorted(points)
    return DenseTrack(object_id, ids, [points[i][0] for i in ids], [points[i][1] for i in ids], source="csrt")


def mask_box(mask: np.ndarray, image_width: int, image_height: int,
             expand: float = 1.04) -> list[float] | None:
    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2: return None
    ys, xs = np.where(mask > 0)
    if len(xs) < 4: return None
    return expand_box(normalize_box([xs.min(), ys.min(), xs.max()+1, ys.max()+1], image_width, image_height), expand, expand)
