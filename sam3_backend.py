"""SAM 3.1 concept-prompted video tracking backend.

SAM 3.1 exposes a stateful video predictor: a short text concept is added to a
frame and the predictor returns masklets while propagating through the video.
This adapter deliberately keeps SAM 3 details out of the submission pipeline
and converts all supported output variants into the project's DenseTrack type.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gqa_core import DenseTrack, sampled_frame_ids


_OUTPUT_KEYS = {
    "ids": ("out_obj_ids", "obj_ids", "object_ids", "ids"),
    "masks": ("out_binary_masks", "masks", "masklets", "pred_masks", "mask_logits"),
    "boxes": ("out_boxes_xywh", "boxes_xywh", "out_boxes", "boxes", "pred_boxes"),
    "scores": ("out_probs", "out_obj_scores", "scores", "probs", "pred_scores"),
}


def _value(outputs: Any, names: Iterable[str], default=None):
    if not isinstance(outputs, dict):
        return default
    for name in names:
        if name in outputs and outputs[name] is not None:
            return outputs[name]
    return default


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _items(value: Any, count: int) -> list[Any]:
    """Make scalar/list/tensor outputs indexable without assuming torch."""
    arr = _as_numpy(value)
    if arr.ndim == 0:
        return [arr.item()] * count
    return [arr[i] for i in range(min(count, len(arr)))]


def _mask_box(mask: Any) -> tuple[list[float] | None, tuple[int, int] | None]:
    arr = _as_numpy(mask)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None, None
    if arr.dtype.kind in "fc":
        arr = arr > 0
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        return None, (arr.shape[1], arr.shape[0])
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], (arr.shape[1], arr.shape[0])


def _box_xyxy(box: Any, width: int, height: int) -> list[float] | None:
    arr = _as_numpy(box).reshape(-1)
    if arr.size < 4:
        return None
    vals = [float(x) for x in arr[:4]]
    # Official SAM3 video output is normalized xywh. Accept pixel xyxy and
    # normalized xyxy as fallbacks for older/dev predictor output variants.
    normalized = max(abs(x) for x in vals) <= 1.5
    if normalized:
        x, y, a, b = vals
        if a >= x and b >= y and (a > 1.0 or b > 1.0):
            # normalized xyxy (rare, but used by some wrappers)
            return [x, y, a, b]
        return [x, y, x + a, y + b]
    x, y, a, b = vals
    # SAM-style pixel xywh is the most common non-normalized variant.
    if a <= width and b <= height and (x + a > width or y + b > height or a < x or b < y):
        return [x / width, y / height, (x + a) / width, (y + b) / height]
    return [x / width, y / height, a / width, b / height]


def _frame_outputs(response: Any) -> tuple[int | None, dict]:
    if not isinstance(response, dict):
        return None, {}
    frame = response.get("frame_index", response.get("frame_idx", response.get("frame_id")))
    outputs = response.get("outputs", response.get("output", response))
    return (None if frame is None else int(frame)), outputs if isinstance(outputs, dict) else {}


def tracks_from_outputs(outputs_per_frame: Iterable[tuple[int, dict]], output_step: int = 30,
                        object_ids: set[int] | None = None) -> list[DenseTrack]:
    """Convert SAM3 frame dictionaries into stable, sparse DenseTrack objects."""
    by_object: dict[int, list[tuple[int, list[float], float]]] = defaultdict(list)
    for frame_index, outputs in outputs_per_frame:
        if frame_index is None or (output_step > 1 and frame_index % output_step):
            continue
        ids = _items(_value(outputs, _OUTPUT_KEYS["ids"]), 10**9)
        masks = _items(_value(outputs, _OUTPUT_KEYS["masks"]), len(ids))
        boxes = _items(_value(outputs, _OUTPUT_KEYS["boxes"]), len(ids))
        scores = _items(_value(outputs, _OUTPUT_KEYS["scores"]), len(ids))
        count = max(len(ids), len(masks), len(boxes))
        for i in range(count):
            oid = int(ids[i]) if i < len(ids) else i
            if object_ids is not None and oid not in object_ids:
                continue
            mask_box, shape = _mask_box(masks[i]) if i < len(masks) else (None, None)
            width, height = (shape or (0, 0))
            box = _box_xyxy(boxes[i], width, height) if i < len(boxes) and width and height else None
            if mask_box is not None and width and height:
                box = [mask_box[0] / width, mask_box[1] / height,
                       mask_box[2] / width, mask_box[3] / height]
            if box is None:
                continue
            score = float(scores[i]) if i < len(scores) else 1.0
            if not np.isfinite(score):
                score = 1.0
            by_object[oid].append((int(frame_index), box, score))
    tracks = []
    for oid, records in sorted(by_object.items()):
        records.sort(key=lambda x: x[0])
        frames, boxes, scores = zip(*records)
        tracks.append(DenseTrack(oid, list(frames), list(boxes), list(scores), source="sam3.1"))
    return tracks


class Sam3VideoTracker:
    """Thin stateful wrapper around the official SAM3 video predictor."""

    def __init__(self, predictor=None):
        if predictor is None:
            from sam3.model_builder import build_sam3_video_predictor
            predictor = build_sam3_video_predictor()
        self.predictor = predictor

    def track(self, video: str | Path, prompt: str, anchor_frame: int = 0,
              output_step: int = 30, object_ids: set[int] | None = None) -> list[DenseTrack]:
        if not prompt or not prompt.strip():
            raise ValueError("SAM3 requires a non-empty short concept prompt")
        start = self.predictor.handle_request({
            "type": "start_session", "resource_path": str(video)
        })
        session_id = start["session_id"]
        try:
            self.predictor.handle_request({
                "type": "add_prompt", "session_id": session_id,
                "frame_index": int(anchor_frame), "text": prompt.strip(),
            })
            responses = self.predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": session_id,
            })
            frame_outputs = []
            for response in responses:
                frame, outputs = _frame_outputs(response)
                if frame is not None:
                    frame_outputs.append((frame, outputs))
            return tracks_from_outputs(frame_outputs, output_step, object_ids)
        finally:
            # Closing is supported by current SAM3; tolerate older revisions
            # whose session manager only exposes reset_session.
            try:
                self.predictor.handle_request({"type": "close_session", "session_id": session_id})
            except Exception:
                try:
                    self.predictor.handle_request({"type": "reset_session", "session_id": session_id})
                except Exception:
                    pass


__all__ = ["Sam3VideoTracker", "tracks_from_outputs"]
