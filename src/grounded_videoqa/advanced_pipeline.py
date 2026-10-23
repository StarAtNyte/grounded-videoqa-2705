"""Canonical GVQA inference: temporal intervals -> video points -> masks."""
from __future__ import annotations
import cv2
import re
import numpy as np
from pathlib import Path
from grounded_videoqa.gqa_core import DenseTrack, normalize_box, sampled_frame_ids


def _video_info(video):
    cap=cv2.VideoCapture(str(video)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    return n,fps,w,h


def _sparse_video(video, stride: int = 30):
    """Create a 1 FPS temporary clip for SAM2; scoring frames are 30 apart."""
    import tempfile
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    path=tempfile.mktemp(suffix=".mp4")
    writer=cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.,fps/stride), (w,h))
    i=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i % stride == 0: writer.write(frame)
        i += 1
    cap.release(); writer.release()
    # Re-open the encoded clip: MP4 codecs can expose fewer frames than were
    # submitted (the source-loop count is not a safe SAM2 bound).
    check = cv2.VideoCapture(path)
    encoded = int(check.get(cv2.CAP_PROP_FRAME_COUNT)); check.release()
    return path, max(0, encoded)


def run_advanced(video: str | Path, question: str, backend: str = "sam2",
                 sam2_checkpoint: str = "", sam2_config: str = "",
                 target: str | None = None, spatial_refine: bool = False,
                 temporal_model=None, track_model=None, tracking_stride: int = 30) -> tuple[list[dict], dict]:
    """Run temporal and point specialists, then a box tracker.

    The model passes are deliberately sequential: TimeLens2 is released before
    Molmo2 is loaded, protecting A100 memory. This function is remote-only.
    """
    import os, cv2
    os.environ.setdefault("HF_HOME", "/data/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/data/hf-cache/hub")
    from grounded_videoqa.advanced_models import TimeLens, MolmoVideoTrack
    n,fps,w,h=_video_info(video)
    temporal_model = temporal_model or TimeLens()
    track_model = track_model or MolmoVideoTrack()
    hint = temporal_model.trigger(str(video), question) if hasattr(temporal_model, "trigger") else {}
    intervals = hint.get("intervals", []) or temporal_model.intervals(str(video), question)
    trigger_seconds = float(hint.get("trigger_seconds", 0.0))
    object_query = target or hint.get("description") or question
    # Ground using a semantically disambiguated description, then select the
    # evidence point nearest the trigger moment rather than midpoint evidence.
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    candidates=track_model.tracks(str(video), object_query, max_fps=8)
    gc.collect(); torch.cuda.empty_cache()
    # A second, spatially-specialized pass corrects the point at the selected
    # anchor. It is optional because it costs another model load.
    if spatial_refine and candidates:
        from grounded_videoqa.advanced_models import MolmoSpatial
        from PIL import Image
        spatial = MolmoSpatial()
        cap=cv2.VideoCapture(str(video))
        for obj in candidates:
            pts=obj.get("points",[])
            if not pts: continue
            t,x,y=pts[len(pts)//2]; af=max(0,min(n-1,round(float(t)*fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES,af); ok,frame=cap.read()
            if not ok: continue
            refined=spatial.point(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),target or question)
            if refined: obj["points"]=[(t,refined[0][0],refined[0][1])]+pts[1:]
        cap.release(); del spatial
        gc.collect(); torch.cuda.empty_cache()
    if not candidates: return [], {"intervals": intervals, "points": 0}
    # Recompute intervals cheaply is avoided; retain all Molmo evidence when
    # the temporal specialist failed, otherwise filter by returned windows.
    valid = (lambda t: (not intervals) or any(a <= float(t) <= b for a,b in intervals))
    tracks=[]
    if backend == "sam2":
        if not sam2_checkpoint or not sam2_config: raise ValueError("SAM2 paths required")
        from grounded_videoqa.sam2_backend import Sam2Bidirectional
        tracker=Sam2Bidirectional(sam2_checkpoint,sam2_config)
    from grounded_videoqa.gvqa_tracking import csrt_bidirectional
    for obj in candidates:
        pts=obj.get("points",[])
        pts=[p for p in pts if valid(p[0])]
        if not pts: continue
        # Anchor at the highest-density/earliest reliable evidence point.
        chosen=min(pts, key=lambda p: abs(float(p[0])-trigger_seconds))
        t,x,y=chosen; anchor=max(0,min(n-1,round(float(t)*fps)))
        px,py=x*w,y*h
        if backend == "sam2":
            # Official annotations are sparse (roughly every 30 frames). Run
            # SAM2 on that sparse clip rather than propagating 30x more frames.
            sparse_path, sparse_n = _sparse_video(video, tracking_stride)
            try:
                sparse_anchor = max(0, min(sparse_n - 1, round(anchor / tracking_stride)))
                tr=tracker.track(sparse_path, sparse_anchor, None, int(obj["id"]),
                                 sparse_n, w, h, 1, point_xy=(px,py), frame_stride=tracking_stride,
                                 recover_occlusions=True, recovery_stride=tracking_stride)
            finally:
                Path(sparse_path).unlink(missing_ok=True)
        else:
            bw,bh=.10*w,.10*h
            tr=csrt_bidirectional(str(video),anchor,(px-bw/2,py-bh/2,px+bw/2,py+bh/2),int(obj["id"]),30,1.04)
        tracks.append(tr.to_submission())
    return tracks,{"intervals":intervals,"trigger_seconds":trigger_seconds,
                    "object_query":object_query,
                    "points":sum(len(x.get("points",[])) for x in candidates),"fps":fps}


def run_molmo2_sam2(video: str | Path, question: str, sam2_checkpoint: str,
                    sam2_config: str, output_step: int = 30,
                    tracking_stride: int = 30, multi_point: bool = False,
                    track_model=None) -> tuple[list[dict], dict]:
    """Molmo2-8B video-point tracking followed by SAM2 refinement.

    multi_point is opt-in and defaults to False: the original single
    first-evidence-point anchor (below) remains the baseline behavior to
    revisit later. When True, every other timestamp Molmo2 reported for the
    same object ID is seeded into SAM2 as an additional positive prompt
    (VideoMolmo-style multi-point conditioning, arXiv:2506.05336), instead of
    anchoring on one point alone.
    """
    import cv2
    from grounded_videoqa.advanced_models import Molmo2VideoPoint
    from grounded_videoqa.sam2_backend import Sam2Bidirectional
    n, fps, w, h = _video_info(video)
    track_model = track_model or Molmo2VideoPoint()
    raw, points = track_model.track_points(str(video), question)
    if not points:
        raise RuntimeError("Molmo2 returned no video points")
    # Molmo2 reports timestamps in seconds for video-point output. If a model
    # build reports frame-like values instead, retain the bounded fallback.
    # Select the most continuously observed object identity, then its first
    # evidence point. This preserves Molmo2 IDs instead of tracking a random
    # final candidate from each timestamp.
    by_id = {}
    for point in points:
        by_id.setdefault(int(point[1]), []).append(point)
    object_id, object_points = max(by_id.items(), key=lambda item: (len(item[1]), -min(p[0] for p in item[1])))
    t, _, x, y = min(object_points, key=lambda p: p[0])
    frame = int(round(t * fps)) if t <= (n / max(fps, 1.0)) * 2 else int(round(t))
    frame = max(0, min(n - 1, frame))
    pixel = (x * w, y * h)

    def _to_frame(pt_t):
        return max(0, min(n - 1, int(round(pt_t * fps)) if pt_t <= (n / max(fps, 1.0)) * 2 else int(round(pt_t))))

    sparse_path, sparse_n = _sparse_video(video, tracking_stride)
    try:
        anchor = max(0, min(sparse_n - 1, round(frame / tracking_stride)))
        extra_points = None
        if multi_point:
            extra_points = []
            for pt_t, _, px, py in object_points:
                if pt_t == t: continue
                pt_frame = _to_frame(pt_t)
                sparse_idx = max(0, min(sparse_n - 1, round(pt_frame / tracking_stride)))
                extra_points.append((sparse_idx, px * w, py * h))
        tracker = Sam2Bidirectional(sam2_checkpoint, sam2_config)
        track = tracker.track(sparse_path, anchor, None, 1, sparse_n, w, h,
                              output_step=output_step, point_xy=pixel,
                              frame_stride=tracking_stride, extra_points=extra_points)
    finally:
        Path(sparse_path).unlink(missing_ok=True)
    return [track.to_submission()], {"mode": "molmo2_8b", "raw_response": raw,
                                     "point_count": len(points), "object_id": object_id,
                                     "object_count": len(by_id), "selected_time": t,
                                     "frame": frame, "point": [x, y],
                                     "tracking_stride": tracking_stride,
                                     "multi_point": multi_point,
                                     "extra_point_count": len(extra_points or [])}


def run_direct_sam2(video: str | Path, grounding: dict, sam2_checkpoint: str,
                    sam2_config: str, output_step: int = 3,
                    tracking_stride: int = 30) -> tuple[list[dict], dict]:
    """Direct VLM frame/box grounding followed by bidirectional SAM2."""
    from grounded_videoqa.sam2_backend import Sam2Bidirectional
    n, fps, w, h = _video_info(video)
    frame = max(0, min(n - 1, int(grounding["frame"])))
    box = tuple(float(x) for x in grounding["box"])
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError("direct grounding box must be normalized xyxy")
    pixel_box = (box[0] * w, box[1] * h, box[2] * w, box[3] * h)
    sparse_path, sparse_n = _sparse_video(video, tracking_stride)
    try:
        anchor = max(0, min(sparse_n - 1, round(frame / tracking_stride)))
        tracker = Sam2Bidirectional(sam2_checkpoint, sam2_config)
        track = tracker.track(sparse_path, anchor, pixel_box, 1, sparse_n, w, h,
                              output_step=output_step,
                              frame_stride=tracking_stride, trial_window=None)
    finally:
        Path(sparse_path).unlink(missing_ok=True)
    return [track.to_submission()], {"mode": "direct_vlm", "frame": frame,
                                     "box": list(box), "fps": fps,
                                     "tracking_stride": tracking_stride}


def run_cortex_sam2(video: str | Path, reasoning: dict, sam2_checkpoint: str,
                    sam2_config: str, output_step: int = 3,
                    grounding_strategy: str = "trigger", tracking_stride: int = 30,
                    reference_mode: bool = False,
                    proposal_grounding: bool = False, spatial_model=None) -> tuple[list[dict], dict]:
    """CORTEX trigger -> Molmo spatial point -> full-rate SAM2 tracking."""
    import gc, cv2, torch
    from PIL import Image
    from grounded_videoqa.advanced_models import MolmoSpatial
    from grounded_videoqa.sam2_backend import Sam2Bidirectional
    n, fps, w, h = _video_info(video)
    objects = reasoning.get("objects", [])
    if not objects: return [], {"reason": "no reasoning objects"}
    spatial = spatial_model or MolmoSpatial()
    candidates = []
    for obj_index, obj in enumerate(objects):
        desc = str(obj.get("description") or "physical object")
        relations = obj.get("relations", []) or []
        events = obj.get("events", []) or []
        # Description alone is ambiguous when several identical objects are
        # present. Preserve CORTEX's relational/event evidence for grounding.
        if relations:
            desc += "; relation: " + ", ".join(map(str, relations))
            relation_text = " ".join(map(str, relations)).lower()
            if "under" in relation_text or "beneath" in relation_text:
                desc += "; do not point to the occluding object; point to the small target underneath it"
        if events:
            desc += "; event: " + ", ".join(map(str, events))
        anchors = obj.get("anchors", []) or []
        visible_intervals = obj.get("visible_intervals", []) or []
        anchor = int(max(anchors, key=lambda a: float(a.get("confidence", 0))).get("frame", n // 2)) if anchors else n // 2
        if grounding_strategy not in {"trigger", "sliding", "reverse"}:
            raise ValueError("grounding_strategy must be trigger, sliding, or reverse")
        if grounding_strategy == "trigger":
            frames = [max(0, min(n - 1, anchor))]
        elif grounding_strategy == "sliding":
            # 2025 winner's naive sliding-window ablation: forward 30-frame
            # probes, selecting the first usable grounding evidence.
            frames = list(range(0, n, 30))
        else:
            # Winner's reverse-play fallback: inspect the early window first,
            # then search backward from the clip tail to avoid early decoys.
            frames = list(range(0, min(n, 101), 30))
            frames += list(range(max(0, n - 1), 100, -30))
        # Trigger mode gets a local fallback search around the proposed moment;
        # other modes already define their complete search order.
        # Visibility intervals are useful for our enhanced trigger search, but
        # the reference implementation must first try exactly one frame.
        if not reference_mode:
            if grounding_strategy == "trigger":
                frames += list(range(min(n - 1, anchor + 30), min(n, anchor + 121), 30))
                frames += list(range(max(0, anchor - 30), max(-1, anchor - 121), -30))
            for a, b in visible_intervals:
                frames += list(range(max(0, int(a)), min(n, int(b) + 1), 5))
            # Bound VLM calls: long visibility intervals otherwise make one
            # question take tens of minutes and defeat restartable batching.
            unique = list(dict.fromkeys(frames))
            if len(unique) > 12:
                keep = [unique[0]] + unique[1:12]
                frames = list(dict.fromkeys(keep))
        observations = []
        def in_visible_interval(fid):
            return any(float(a) <= fid <= float(b) for a, b in visible_intervals)
        cap = cv2.VideoCapture(str(video))
        for fid in dict.fromkeys(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid); ok, frame = cap.read()
            if not ok: continue
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pts = spatial.point_variants(pil, desc)
            for pt in pts:
                x, y = pt
                margin = min(x, y, 1.0 - x, 1.0 - y)
                observations.append((len(pts) == 1, margin, -abs(fid - anchor), fid, pt))
            # Enhanced-only tiled grounding gives tiny/occluded objects more
            # pixels while mapping every point back to source coordinates.
            tiny_hint = re.search(r"marble|small|tiny|hidden|occlud|letter|key|bead", desc, re.I)
            if tiny_hint and not reference_mode:
                W, H = pil.size
                tiles = [(0, 0, int(W*.65), int(H*.65)),
                         (int(W*.35), 0, W, int(H*.65)),
                         (0, int(H*.35), int(W*.65), H),
                         (int(W*.35), int(H*.35), W, H)]
                for left, top, right, bottom in tiles:
                    crop = pil.crop((left, top, right, bottom)).resize((W, H))
                    crop_pts = spatial.point_variants(crop, desc)
                    cw, ch = right-left, bottom-top
                    for cx, cy in crop_pts:
                        x, y = (left + cx*cw)/W, (top + cy*ch)/H
                        if .01 < x < .99 and .01 < y < .99:
                            observations.append((len(crop_pts) == 1, min(x,y,1-x,1-y),
                                                 -abs(fid-anchor), fid, (x,y)))
        cap.release()
        if observations:
            # Prefer evidence inside the reasoner's visibility interval, then
            # nearest the trigger. Margin is only a tie-breaker: maximizing
            # it globally previously selected a later, unrelated object.
            interval_obs = [o for o in observations if in_visible_interval(o[3])]
            pool0 = interval_obs or observations
            interior = [o for o in pool0 if o[1] >= 0.03]
            pool = interior or pool0
            # Retain a small diverse beam rather than trusting one VLM click.
            # Prefer the trigger-nearest evidence first; singleton output is
            # valuable, but not enough to override temporal proximity.
            if grounding_strategy == "sliding":
                # Preserve temporal search order; singleton is still a hard
                # preference, while margin breaks ties within a frame.
                ranked = sorted(pool, key=lambda o: (o[0], -o[3], o[1]), reverse=True)
            elif grounding_strategy == "reverse":
                ranked = sorted(pool, key=lambda o: (o[0], o[3], o[1]), reverse=True)
            else:
                ranked = sorted(pool, key=lambda o: (o[0], o[2], o[1]), reverse=True)
            chosen = []
            for singleton, _, _, used, point in ranked:
                if all((point[0] - q[2][0]) ** 2 + (point[1] - q[2][1]) ** 2 > 0.02 ** 2
                       for q in chosen):
                    chosen.append((obj_index, used, point, singleton))
                if len(chosen) == (1 if reference_mode else 3): break
            candidates.extend(chosen)
    del spatial; gc.collect(); torch.cuda.empty_cache()
    if not candidates: return [], {"reason": "Molmo found no points"}
    # Molmo's point is the intended SAM2 prompt. Keep DINO refinement as a
    # separate benchmark path; replacing a precise point with a broad box can
    # select an adjacent decoy in occlusion scenes.
    tracker = Sam2Bidirectional(sam2_checkpoint, sam2_config)
    proposal_box = None
    if proposal_grounding and candidates:
        # Real open-vocabulary proposal path: Grounding-DINO proposes an
        # answer-conditioned box at the trigger frame; Molmo remains the point
        # adjudicator. This is opt-in and never changes reference mode.
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            query = re.sub(r";.*$", "", desc).strip(" .")
            cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(candidates[0][1]))
            ok, frame = cap.read(); cap.release()
            if ok:
                proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
                det = AutoModelForZeroShotObjectDetection.from_pretrained(
                    "IDEA-Research/grounding-dino-tiny").cuda().eval()
                inputs = proc(images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                              text=query, return_tensors="pt").to("cuda")
                with torch.inference_mode(): out = det(**inputs)
                result = proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=.18, text_threshold=.12,
                    target_sizes=[(h, w)])[0]
                if len(result["boxes"]):
                    # Reject DINO proposals that are outside or nearly the
                    # full frame; these destabilize SAM2 more than a point.
                    # Also reject a proposal that disagrees with Molmo's own
                    # selected point: GROUNDING_FIX_EXPERIMENT.md found DINO
                    # can pick a box inconsistent with Molmo's chosen object,
                    # so a proposal only survives if it actually contains the
                    # point Molmo already committed to.
                    molmo_px, molmo_py = candidates[0][2][0] * w, candidates[0][2][1] * h
                    ranked = sorted(zip(result["boxes"], result.get("scores", [])),
                                    key=lambda pair: float(pair[1]), reverse=True)
                    for raw_box, _score in ranked:
                        x1, y1, x2, y2 = [float(v) for v in raw_box.detach().cpu()]
                        x1, y1 = max(0.0, min(x1, w - 1)), max(0.0, min(y1, h - 1))
                        x2, y2 = max(0.0, min(x2, w)), max(0.0, min(y2, h))
                        area = max(0.0, x2-x1) * max(0.0, y2-y1) / max(1.0, w*h)
                        contains_point = x1 <= molmo_px <= x2 and y1 <= molmo_py <= y2
                        if x2 > x1 and y2 > y1 and area <= 0.65 and contains_point:
                            proposal_box = (x1, y1, x2, y2); break
                    # No ranked box agreed with Molmo's point: keep the
                    # point-only fallback (proposal_box stays None) instead of
                    # forcing a disagreeing box onto SAM2.
                del det
                gc.collect(); torch.cuda.empty_cache()
        except Exception:
            proposal_box = None
    trial_tracks = []
    for obj_index, anchor, point, singleton in candidates:
        px, py = point[0] * w, point[1] * h
        # When DINO supplies a proposal, compare it against a point-only trial
        # rather than assuming either prompt type is universally superior.
        prompt_modes = [(tuple(proposal_box), None)] if proposal_box else []
        prompt_modes.append((None, (px, py)))
        for box_xyxy, prompt_point in prompt_modes:
            # Trial propagation uses a sparse clip to match the official cadence;
            # this avoids expensive full-rate decoding and SAM2 tail-index errors.
            sparse_path, sparse_n = _sparse_video(video, tracking_stride)
            try:
                sparse_anchor = max(0, min(sparse_n - 1, round(anchor / tracking_stride)))
                tr = tracker.track(sparse_path, sparse_anchor, box_xyxy, obj_index + 1,
                                   sparse_n, w, h, output_step=output_step,
                                   point_xy=prompt_point, frame_stride=tracking_stride,
                                   trial_window=60, recover_occlusions=True,
                                   recovery_stride=tracking_stride)
            finally:
                Path(sparse_path).unlink(missing_ok=True)
            trial_tracks.append((tr, obj_index, anchor, point, singleton, box_xyxy))
    # Select the most stable trial: coverage and non-degenerate mask scores
    # are cheap proxies for the short-track quality objective.
    def trial_score(item):
        tr, _, anchor, point, singleton, _box = item
        boxes = tr.boxes
        valid = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
        coverage = len(valid) / max(1, len(boxes))
        confidence = sum(tr.confidences) / max(1, len(tr.confidences))
        # Conservative short-track score. Avoid global center/size priors:
        # tiny or edge-located answer objects are legitimate in GVQA.
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in valid]
        area_penalty = max(areas, default=1.)
        trigger_bonus = 1.0 / (1.0 + abs(anchor - n // 2) / max(1, n))
        return (coverage + 0.15 * confidence + 0.1 * trigger_bonus
                - 0.2 * area_penalty + 0.15 * float(singleton))
    def select_winner(object_trials):
        if reference_mode:
            # Faithful winner-style baseline: use the first successful trigger
            # grounding, not a post-hoc candidate quality oracle.
            return object_trials[0]
        # Molmo singleton output is strong identity evidence. Prefer the
        # strongest singleton in temporal search order; quality scoring only
        # resolves ties among observations from the same search context.
        singleton_trials = [x for x in object_trials if x[4]]
        # CORTEX provides the intended temporal anchor; among singleton
        # observations, choose the one nearest the reasoner's anchor rather
        # than the first probe (which may be an early decoy).
        if singleton_trials:
            object_index = singleton_trials[0][1]
            object_data = objects[object_index]
            target_anchor = int(max((a.get("frame", n // 2)
                                     for a in (object_data.get("anchors", []) or [])),
                                    default=n // 2))
            # Temporal proximity is the primary identity cue, but use the
            # short SAM2 trial quality as a tie-breaker. A tiny target can be
            # absent exactly at CORTEX's frame while visible a few frames away.
            nearest = min(abs(x[2] - target_anchor) for x in singleton_trials)
            local = [x for x in singleton_trials if abs(x[2] - target_anchor) <= nearest + 30]
            short_relation = (object_data.get("relations") and any(
                float(b) - float(a) <= 60
                for a, b in (object_data.get("visible_intervals") or [])))
            # For tiny targets visible only in a short interval, prefer the
            # last successful probe: Molmo may see the occluder at the trigger
            # instant and the target immediately after it is revealed.
            return (max(local, key=lambda x: (x[2], trial_score(x))) if short_relation
                    else max(local, key=trial_score))
        return max(object_trials, key=trial_score)

    # Select one winner per semantic answer object. The previous global winner
    # silently collapsed plural answers to one track even when CORTEX and Molmo
    # had successfully grounded every object.
    winners = []
    for object_index in sorted({x[1] for x in trial_tracks}):
        object_trials = [x for x in trial_tracks if x[1] == object_index]
        if object_trials:
            winners.append(select_winner(object_trials))

    tracks = []
    selected_diagnostics = []
    for winner in winners:
        _, winner_obj, winner_anchor, winner_point, _, winner_box = winner
        px, py = winner_point[0] * w, winner_point[1] * h
        full_sparse = None
        full_video = str(video); full_anchor = winner_anchor; full_n = n; full_stride = 1
        if tracking_stride != 1:
            full_sparse, full_n = _sparse_video(video, tracking_stride)
            full_video = full_sparse
            full_anchor = max(0, min(full_n - 1, round(winner_anchor / tracking_stride)))
            full_stride = tracking_stride
        try:
            full_track = tracker.track(
                full_video, full_anchor, winner_box, winner_obj + 1,
                full_n, w, h, output_step=output_step,
                point_xy=(px, py), frame_stride=full_stride,
                recover_occlusions=not reference_mode, recovery_stride=30)
        finally:
            if full_sparse is not None: Path(full_sparse).unlink(missing_ok=True)
        if tracking_stride != 30:
            keep = [i for i, fid in enumerate(full_track.frame_ids) if fid % 30 == 0]
            full_track.frame_ids = [full_track.frame_ids[i] for i in keep]
            full_track.boxes = [full_track.boxes[i] for i in keep]
            full_track.confidences = [full_track.confidences[i] for i in keep]
        full_valid = sum(b[2] > b[0] and b[3] > b[1] for b in full_track.boxes)
        tracks.append(full_track.to_submission())
        selected_diagnostics.append({"object": winner_obj, "anchor": winner[2],
                                     "point": winner[3], "score": trial_score(winner),
                                     "full_valid": full_valid})
    return tracks, {"objects": len(objects), "grounded": len(candidates),
                    "anchors": [x[2] for x in trial_tracks], "points": [x[3] for x in trial_tracks],
                    "candidate_singletons": [bool(x[4]) for x in trial_tracks],
                    "selected": (selected_diagnostics[0] if len(selected_diagnostics) == 1
                                 else selected_diagnostics),
                    "fps": fps, "output_step": output_step,
                    "tracking_stride": tracking_stride, "reference_mode": reference_mode,
                    "grounding_strategy": grounding_strategy,
                    "proposal_grounding": proposal_grounding,
                    "proposal_box": proposal_box,
                    "selected_prompt": "box" if winner_box else "point"}
