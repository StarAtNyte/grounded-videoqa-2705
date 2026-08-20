"""Optional SAM2.1 backend.

Install SAM2 in the Modal image before using this module. The rest of the
pipeline only depends on the returned DenseTrack objects.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
from gqa_core import DenseTrack, sampled_frame_ids
from gvqa_tracking import mask_box


def _timestamp_sample_ids(video: str | Path, num_frames: int, fps: float) -> list[int]:
    """Map one-second annotation cadence onto the video's actual frame PTS."""
    # Official labels use the fixed every-30-frame cadence. Their timestamps
    # preserve source timing but do not change the required frame IDs.
    return sampled_frame_ids(num_frames, 30)


class Sam2Bidirectional:
    @staticmethod
    def _video_fps(video: str | Path) -> float:
        import cv2
        cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.; cap.release()
        return float(fps)

    def __init__(self, checkpoint: str, model_cfg: str):
        from sam2.build_sam import build_sam2_video_predictor
        # SAM2's Hydra loader expects a package-relative config name, even
        # when callers pass the path printed by prepare_sam2.
        cfg = str(model_cfg)
        marker = "configs/"
        if marker in cfg:
            cfg = cfg[cfg.index(marker):]
        self.predictor = build_sam2_video_predictor(cfg, checkpoint)

    def track(self, video: str | Path, anchor_frame: int, box_xyxy: Iterable[float] | None,
              object_id: int, num_frames: int, width: int, height: int,
              output_step: int = 30, point_xy: tuple[float,float] | None = None,
              frame_stride: int = 1, trial_window: int | None = None,
              recover_occlusions: bool = False, recovery_stride: int = 30,
              extra_points: Iterable[tuple[int, float, float]] | None = None) -> DenseTrack:
        """Propagate, then optionally re-anchor degraded spans with same ID.

        extra_points is an optional list of (frame_idx, x, y) evidence points
        from other timestamps of the same tracked identity (e.g. additional
        Molmo2 track evidence beyond the anchor). Each is seeded as its own
        positive prompt before propagation, conditioning both directions on
        more than one observation, similar to VideoMolmo's multi-point fusion
        (arXiv:2506.05336) rather than SAM2's single-frame default.
        """
        state = self.predictor.init_state(video_path=str(video), offload_video_to_cpu=True)
        self.predictor.reset_state(state)
        if point_xy is not None:
            kwargs = dict(inference_state=state, frame_idx=int(anchor_frame),
                          obj_id=int(object_id),
                          points=np.asarray([point_xy], dtype=np.float32),
                          labels=np.asarray([1], dtype=np.int32))
            # A loose box supplies object extent while the positive point
            # disambiguates the target from nearby identical objects.
            if box_xyxy is not None:
                kwargs["box"] = np.asarray(list(box_xyxy), dtype=np.float32)
            self.predictor.add_new_points_or_box(**kwargs)
        else:
            self.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=int(anchor_frame), obj_id=int(object_id),
                box=np.asarray(list(box_xyxy), dtype=np.float32))
        for extra_frame, ex, ey in (extra_points or []):
            extra_frame = max(0, min(int(num_frames) - 1, int(extra_frame)))
            if extra_frame == int(anchor_frame): continue
            try:
                self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=extra_frame, obj_id=int(object_id),
                    points=np.asarray([[ex, ey]], dtype=np.float32),
                    labels=np.asarray([1], dtype=np.int32))
            except (RuntimeError, ValueError):
                continue
        masks: dict[int, np.ndarray] = {}
        # SAM2 versions differ in reverse argument support; use explicit runs.
        # Always propagate both directions. The anchor is clamped by the
        # caller; IndexError from codec/SAM2 tail mismatches is contained
        # below while preserving masks already produced.
        for reverse in (False, True):
            kwargs = {"start_frame_idx": int(anchor_frame),
                      "reverse": reverse}
            if trial_window is not None:
                # SAM2 interprets max_frame_num_to_track as a count from the
                # anchor. Never request beyond the actual container bounds;
                # short sparse clips otherwise probe frame N (out of range).
                remaining = (int(anchor_frame) + 1) if reverse else (int(num_frames) - int(anchor_frame))
                kwargs["max_frame_num_to_track"] = max(1, min(int(trial_window), max(1, remaining - 1)))
            try:
                iterator = self.predictor.propagate_in_video(state, **kwargs)
            except TypeError:
                # Older releases may not expose the trial-window argument.
                if trial_window is not None:
                    kwargs.pop("max_frame_num_to_track", None)
                    iterator = self.predictor.propagate_in_video(state, **kwargs)
                else:
                    iterator = None
            if iterator is None:
                # Older releases expose only forward propagation. A production
                # image should pin SAM2.1; this fallback still gives a track.
                if reverse: continue
                iterator = self.predictor.propagate_in_video(state)
            try:
                for fid, ids, logits in iterator:
                    # Some SAM2 builds lazily discover a shorter decoded clip;
                    # discard any out-of-range tail instead of failing a shard.
                    if int(fid) < 0 or int(fid) >= int(num_frames):
                        continue
                    for index, oid in enumerate(ids):
                        if int(oid) != int(object_id): continue
                        masks[int(fid)] = (logits[index] > 0).detach().cpu().numpy()
            except IndexError:
                # Decoder/container length can differ from OpenCV metadata.
                # Preserve masks already produced and let official filtering
                # serialize a partial, valid track.
                continue
        if recover_occlusions and masks:
            # SAM2Long-style training-free recovery: reliable masks become
            # memory checkpoints. Re-prompt the same object at checkpoints and
            # fill only missing/degenerate propagated masks, preserving ID.
            def mask_coords(mask):
                arr = np.asarray(mask).squeeze()
                if arr.ndim != 2: return None, None
                return np.where(arr)
            def valid_mask(mask):
                ys, xs = mask_coords(mask)
                return ys is not None and len(xs) >= 4 and (xs.max() > xs.min()) and (ys.max() > ys.min())
            reliable = sorted(fid for fid, mask in masks.items() if valid_mask(mask))
            step = max(1, int(recovery_stride))
            for checkpoint in reliable[::step]:
                mask = masks[checkpoint]
                ys, xs = mask_coords(mask)
                if ys is None or len(xs) < 4: continue
                bx = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
                cx, cy = float((bx[0] + bx[2]) / 2), float((bx[1] + bx[3]) / 2)
                try:
                    self.predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=int(checkpoint), obj_id=int(object_id),
                        points=np.asarray([[cx, cy]], dtype=np.float32), labels=np.asarray([1], dtype=np.int32),
                        box=bx)
                    try:
                        iterator = self.predictor.propagate_in_video(
                            state, start_frame_idx=int(checkpoint), reverse=False,
                            max_frame_num_to_track=int(step))
                    except TypeError:
                        iterator = self.predictor.propagate_in_video(
                            state, start_frame_idx=int(checkpoint), reverse=False)
                    for fid, ids, logits in iterator:
                        for index, oid in enumerate(ids):
                            if int(oid) == int(object_id) and int(fid) not in masks:
                                masks[int(fid)] = (logits[index] > 0).detach().cpu().numpy()
                except (RuntimeError, TypeError, ValueError):
                    continue
        if trial_window is not None:
            # Trial tracks are scored on dense propagated frames; official
            # cadence is applied only to the final full-video track.
            frame_ids = sorted(masks)
        else:
            # Propagation frame indices can be off by one from container PTS
            # (e.g. mask at 600 vs official label frame 599). Assign the
            # nearest available mask to each official scoring frame.
            # A sparse temporary clip has one frame per scoring instant, so
            # request every sparse frame before mapping it back to source IDs.
            # Applying the original 30-frame cadence here would retain only
            # every 30th sparse frame and silently discard almost the track.
            target_step = 1 if frame_stride != 1 else 30
            targets = sampled_frame_ids(num_frames, target_step)
            available = sorted(masks)
            frame_ids = [f for f in targets if available and
                         min(abs(a - f) for a in available) <= 2]
        boxes, conf = [], []
        for fid in frame_ids:
            source_fid = fid if fid in masks else min(masks, key=lambda a: abs(a - fid))
            b = mask_box(masks[source_fid], width, height, expand=1.04)
            if b is not None:
                boxes.append(b); conf.append(float(np.mean(masks[source_fid])))
            else:
                boxes.append([0., 0., 0., 0.]); conf.append(0.)
        if frame_stride != 1:
            frame_ids = [min(num_frames * frame_stride - 1, f * frame_stride) for f in frame_ids]
        return DenseTrack(object_id, frame_ids, boxes, conf, source="sam2.1")
