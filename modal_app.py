"""Modal pipeline for Grounded VideoQA 2705.

The pipeline is intentionally split into cacheable stages. The current local
backend is a strong smoke-test baseline (DINO anchor + bidirectional CSRT),
while the output contract is ready for SAM2/CoTracker/VLM backends.

Examples:
  modal run modal_app.py::download --split test
  modal run modal_app.py::inspect --split valid
  modal run modal_app.py::predict --split valid --video-id video_1580 --question-id 6
"""
from __future__ import annotations
import json, pathlib, zipfile
import modal

APP_NAME = "grounded-videoqa-2705"
DATA_ROOT = "/data"
BUCKET = "https://storage.googleapis.com/dm-perception-test/zip_data"
ARCHIVES = {
    "test": ("grounded_question_test_videos.zip", "grounded_question_test_annotations.zip"),
    "valid": ("grounded_question_valid_videos.zip", "grounded_question_valid_annotations.zip"),
    "train": ("grounded_question_train_videos.zip", "grounded_question_train_annotations.zip"),
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("grounded-videoqa-2705-data", create_if_missing=True)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "opencv-contrib-python-headless", "requests", "tqdm",
                 "torch", "torchvision", "accelerate", "bitsandbytes", "Pillow", "scipy", "einops", "pycocotools", "tabulate", "decord2", "molmo_utils", "qwen-vl-utils[decord]>=0.0.14", "transformers==4.57.1", "google-genai",
                 "git+https://github.com/facebookresearch/sam2.git", "git+https://github.com/JonathonLuiten/TrackEval.git")
)
_sources = ("gqa_core", "gvqa_tracking", "sam2_backend", "sam3_backend", "gvqa_reasoning",
            "track_graph", "advanced_models", "advanced_pipeline", "local_eval", "validate")
image = base_image.add_local_python_source(*_sources)
# Molmo2 uses the modern Transformers video processor; keep it on the base
# Transformers 4.57 image rather than the legacy Molmo-7B image.
molmo2_image = image
# Optional winner-aligned Molmo-7B image. Its remote processor requires
# TensorFlow and Transformers 4.50.3 plus the runtime compatibility shim.
molmo7_image = base_image.pip_install(
    "tensorflow-cpu==2.19.1", "transformers==4.50.3").add_local_python_source(*_sources)

# SAM 3.1 currently requires a newer Python/PyTorch/CUDA stack than the
# established SAM2 image. Keep it isolated so existing inference remains
# reproducible. Checkpoint access is gated by Hugging Face and is supplied via
# the optional `huggingface` Modal secret at execution time.
sam3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision", "numpy<2", "Pillow", "opencv-python-headless",
                 "setuptools<81", "einops", "pycocotools", "psutil", "huggingface_hub", "git+https://github.com/facebookresearch/sam3.git")
    .add_local_python_source("gqa_core", "sam3_backend")
)


def _download(url: str, destination: pathlib.Path) -> None:
    import requests
    from tqdm import tqdm
    if destination.exists() and destination.stat().st_size > 0: return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status(); total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                if chunk: f.write(chunk); bar.update(len(chunk))


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60 * 60 * 8)
def download(split: str = "test") -> None:
    """Download and extract one official archive pair into the persistent volume."""
    if split not in ARCHIVES: raise ValueError(f"split must be one of {sorted(ARCHIVES)}")
    root = pathlib.Path(DATA_ROOT) / split; root.mkdir(parents=True, exist_ok=True)
    for archive in ARCHIVES[split]:
        zpath = pathlib.Path(DATA_ROOT) / "archives" / archive
        _download(f"{BUCKET}/{archive}", zpath)
        marker = root / (archive + ".extracted")
        if not marker.exists():
            with zipfile.ZipFile(zpath) as zf: zf.extractall(root)
            marker.touch(); volume.commit()
    print(f"Ready: {root}")


def _find_json(root: pathlib.Path) -> pathlib.Path:
    files = sorted(root.rglob("*.json")); preferred = [p for p in files if "grounded_question" in p.name]
    if not files: raise FileNotFoundError(f"No annotations under {root}")
    return (preferred or files)[0]


def _video_path(root: pathlib.Path, video_id: str) -> pathlib.Path:
    hits = list(root.rglob(video_id + ".mp4"))
    if not hits: raise FileNotFoundError(f"{video_id}.mp4 not found under {root}")
    return hits[0]


def _sam3_prompt_anchor(split: str, video_id: str, question_id: str,
                        question: dict, prompt: str = "",
                        anchor_frame: int = -1) -> tuple[str, int]:
    """Resolve cached evidence and reduce reasoning text to a SAM3 concept."""
    cache = pathlib.Path(DATA_ROOT) / "cache" / split / video_id / f"{question_id}.json"
    if cache.exists():
        reasoning = json.loads(cache.read_text()).get("reasoning", {})
        objects = reasoning.get("objects", [])
        if objects:
            obj = objects[0]
            anchors = obj.get("anchors", [])
            if anchor_frame < 0 and anchors:
                anchor_frame = int(max(anchors, key=lambda x: float(x.get("confidence", 0))).get("frame", 0))
            if not prompt:
                prompt = str(obj.get("description") or "")
    if not prompt:
        prompt = question["question"]
    import re
    prompt = prompt.strip()
    # Reasoners often append the actual imperative after scene context.
    tracked = re.search(r"track\s+(?:(?:the|a|an)\s+)?(.+)$", prompt, flags=re.I)
    if tracked:
        prompt = tracked.group(1)
    prompt = re.sub(r"^(the|a|an)\s+", "", prompt, flags=re.I)
    prompt = re.split(r"\s+(?:that|which|who|on which|where|holding|used to|in order to)\s+",
                      prompt, maxsplit=1, flags=re.I)[0].strip(" .,;:")
    return prompt, max(0, anchor_frame)


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60 * 10)
def extract_frame(split: str, video_id: str, frame_id: int, output: str = "frame.jpg") -> None:
    """Extract a diagnostic frame; used only for model debugging."""
    import cv2
    cap = cv2.VideoCapture(str(_video_path(pathlib.Path(DATA_ROOT) / split, video_id)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id)); ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError("frame unavailable")
    target = pathlib.Path(DATA_ROOT) / "predictions" / output
    target.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(target), frame); volume.commit()
    print(target)


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60 * 10)
def inspect(split: str = "test") -> None:
    root = pathlib.Path(DATA_ROOT) / split; annotation = _find_json(root)
    db = json.loads(annotation.read_text())
    qs = sum(len(v.get("grounded_question", [])) for v in db.values())
    question_examples = [{"video_id": vid, "question_ids": [q["id"] for q in db[vid].get("grounded_question", [])]}
                          for vid in list(db)[:5]]
    print(json.dumps({"annotation": str(annotation), "videos": len(db), "questions": qs,
                      "examples": question_examples}, indent=2))


def _dino_boxes(video: pathlib.Path, frame_id: int, query: str) -> tuple[list[list[float]], int, int]:
    """Question-conditioned anchor proposals. Models are loaded per warm worker."""
    import cv2, torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read(); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    if not ok: raise RuntimeError(f"could not read anchor {frame_id}")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).cuda().eval()
    inputs = processor(images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                       text=query, return_tensors="pt").to("cuda")
    with torch.inference_mode(): out = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=.20, text_threshold=.15,
        target_sizes=[(h, w)])[0]
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)[:5]
    return [boxes[i] for i in order], w, h


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60 * 30)
def prepare_sam2() -> None:
    """Cache the official SAM2.1 Hiera-Large checkpoint in the Volume."""
    import requests, pathlib, sam2
    from tqdm import tqdm
    target = pathlib.Path(DATA_ROOT) / "checkpoints" / "sam2.1_hiera_large.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    if not target.exists():
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status(); total = int(r.headers.get("content-length", 0))
            with target.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(16 * 1024 * 1024):
                    if chunk: f.write(chunk); bar.update(len(chunk))
    package = pathlib.Path(sam2.__file__).parent
    configs = sorted(package.rglob("sam2.1_hiera_l.yaml"))
    volume.commit()
    print(json.dumps({"checkpoint": str(target), "configs": [str(x) for x in configs]}, indent=2))


@app.function(image=image, volumes={DATA_ROOT: volume},
              secrets=[modal.Secret.from_name("gemini-api-key"),
                       modal.Secret.from_name("gemini-api-keys")], timeout=60 * 30)
def reason(split: str, video_id: str, question_id: str, model: str = "gemini-3.6-flash", force: bool = False, cost_cap: int = 1) -> None:
    """Ask a video VLM for structured answer objects and trigger frames.

    Create the secret once with `modal secret create gemini-api
    GOOGLE_API_KEY=...`. Results are cached and can be inspected before
    expensive grounding/tracking.
    """
    import os, time
    from google import genai
    from google.genai import types
    from gvqa_reasoning import SYSTEM_PROMPT, parse_reasoning
    root = pathlib.Path(DATA_ROOT) / split; db = json.loads(_find_json(root).read_text())
    if video_id not in db: raise KeyError(f"unknown video_id: {video_id}")
    matches = [q for q in db[video_id].get("grounded_question", []) if str(q["id"]) == str(question_id)]
    if not matches: raise KeyError(f"unknown question_id: {question_id}")
    video = _video_path(root, video_id); import cv2
    cap = cv2.VideoCapture(str(video)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    cache = pathlib.Path(DATA_ROOT) / "cache" / split / video_id / f"{question_id}.json"
    if cache.exists() and not force:
        try:
            cached = json.loads(cache.read_text())
            prov = cached.get("provenance", {})
            if (prov.get("model") == model and prov.get("split") == split and
                    str(prov.get("question_id")) == str(question_id) and
                    int(prov.get("api_calls", 0)) >= 1 and
                    int(prov.get("api_calls", 0)) <= int(prov.get("cost_cap", 0)) and
                    int(prov.get("cost_cap", 0)) >= 1):
                print(f"cached: {cache}"); return
        except (OSError, ValueError, TypeError):
            pass
        print(f"stale or unproven cache; regenerating: {cache}")
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = next((v for k, v in os.environ.items()
                        if k.startswith(("GOOGLE_API_KEY_", "GEMINI_API_KEY_")) and v), None)
    if not api_key:
        raise RuntimeError("No Gemini API key found in attached Modal secrets")
    if cost_cap < 1: raise ValueError("cost_cap must be >= 1")
    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(video))
    for _ in range(120):
        state = getattr(uploaded, "state", None)
        if not state or getattr(state, "name", "ACTIVE") == "ACTIVE": break
        if getattr(state, "name", "") == "FAILED": raise RuntimeError("Gemini video processing failed")
        time.sleep(2); uploaded = client.files.get(name=uploaded.name)
    q = matches[0]["question"]
    prompt_text = SYSTEM_PROMPT + f"\nVideo has {n} frames at {fps:.4f} FPS.\nQuestion: {q}"
    response = client.models.generate_content(
        model=model, contents=[uploaded, prompt_text],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1))
    parsed = parse_reasoning(response.text, n)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"question": q, "raw": response.text,
                                 "provenance": {"model": model, "api_calls": 1, "cost_cap": cost_cap, "split": split, "question_id": str(question_id)},
                                 "reasoning": {"objects": [{"description": o.description, "count": o.count,
                                 "relations": o.relations, "events": o.events,
                                 "anchors": [{"frame": a.frame, "confidence": a.confidence} for a in o.anchors],
                                 "visible_intervals": o.visible_intervals} for o in parsed.objects],
                                 "answer_cardinality": parsed.answer_cardinality,
                                 "confidence": parsed.confidence, "evidence": parsed.evidence}}, indent=2))
    volume.commit(); print(cache)


@app.function(image=image, volumes={DATA_ROOT: volume},
              secrets=[modal.Secret.from_name("gemini-api-key"),
                       modal.Secret.from_name("gemini-api-keys")], timeout=60 * 60 * 8)
def batch_reason(split: str = "valid", limit: int = 50, start: int = 0,
                 model: str = "gemini-3.6-flash", force: bool = False,
                 cost_cap: int = 10) -> None:
    """Cache Gemini reasoning with a hard per-run call cap."""
    import os, time, cv2
    from google import genai
    from google.genai import types
    from gvqa_reasoning import SYSTEM_PROMPT, parse_reasoning
    root = pathlib.Path(DATA_ROOT) / split
    db = json.loads(_find_json(root).read_text())
    jobs = [(vid, q) for vid in sorted(db) for q in db[vid].get("grounded_question", [])]
    jobs = jobs[start:start + limit]
    if limit > cost_cap:
        raise ValueError(f"limit {limit} exceeds cost_cap {cost_cap}")
    # Modal secrets may expose one key or a numbered key set. Keep keys
    # distinct and rotate clients on quota errors; never print key material.
    keys = []
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(name): keys.append(os.environ[name])
    for name in sorted(os.environ):
        if name.startswith(("GOOGLE_API_KEY_", "GEMINI_API_KEY_")) and os.environ[name]:
            keys.append(os.environ[name])
    keys = list(dict.fromkeys(keys))
    if not keys:
        raise RuntimeError("No Gemini API key found in attached Modal secrets")
    clients = [genai.Client(api_key=key) for key in keys]
    done = 0; api_calls = 0; key_index = 0
    for vid, qitem in jobs:
        qid = str(qitem["id"])
        cache = pathlib.Path(DATA_ROOT) / "cache" / split / vid / f"{qid}.json"
        if cache.exists() and not force:
            try:
                cached = json.loads(cache.read_text()); prov = cached.get("provenance", {})
                if (prov.get("model") == model and prov.get("split") == split and
                        str(prov.get("question_id")) == str(qid) and
                        int(prov.get("api_calls", 0)) >= 1 and
                        int(prov.get("api_calls", 0)) <= int(prov.get("cost_cap", 0)) and
                        int(prov.get("cost_cap", 0)) >= 1):
                    done += 1; continue
            except (OSError, ValueError, TypeError):
                pass
        video = _video_path(root, vid)
        cap = cv2.VideoCapture(str(video)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        if api_calls >= cost_cap:
            raise RuntimeError(f"Gemini cost cap exhausted after {api_calls} calls")
        api_calls += 1
        prompt = SYSTEM_PROMPT + f"\nVideo has {n} frames at {fps:.4f} FPS.\nQuestion: {qitem['question']}"
        last_error = None; response = None
        for attempt in range(max(3, len(clients))):
            client = clients[key_index % len(clients)]
            key_index += 1
            uploaded = client.files.upload(file=str(video))
            for _ in range(120):
                state = getattr(uploaded, "state", None)
                if not state or getattr(state, "name", "ACTIVE") == "ACTIVE": break
                if getattr(state, "name", "") == "FAILED": raise RuntimeError("Gemini video processing failed")
                time.sleep(2); uploaded = client.files.get(name=uploaded.name)
            try:
                response = client.models.generate_content(
                    model=model, contents=[uploaded, prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1))
                break
            except Exception as exc:
                last_error = exc
                text = str(exc).upper()
                quota = "429" in text or "RESOURCE_EXHAUSTED" in text or "QUOTA" in text
                if not quota and attempt >= 2: raise
                if attempt + 1 < max(3, len(clients)):
                    time.sleep(2 if quota else 10 * (attempt + 1))
        if response is None:
            raise RuntimeError(f"Gemini failed after rotating {len(clients)} key(s): {last_error}")
        parsed = parse_reasoning(response.text, n)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"question": qitem["question"], "raw": response.text,
            "provenance": {"model": model, "api_calls": api_calls, "cost_cap": cost_cap, "split": split, "question_id": str(qid)}, 
            "reasoning": {"objects": [{"description": o.description, "count": o.count,
            "relations": o.relations, "events": o.events,
            "anchors": [{"frame": a.frame, "confidence": a.confidence} for a in o.anchors],
            "visible_intervals": o.visible_intervals} for o in parsed.objects],
            "answer_cardinality": parsed.answer_cardinality, "confidence": parsed.confidence,
            "evidence": parsed.evidence}}, indent=2))
        done += 1; volume.commit()
        print(json.dumps({"done": done, "video": vid, "question": qid}), flush=True)


@app.function(image=image, gpu="A100", volumes={DATA_ROOT: volume}, timeout=60 * 60 * 4)
def predict(split: str, video_id: str, question_id: str,
            output: str = "predictions.json", anchor_frame: int = -1,
            backend: str = "csrt", sam2_checkpoint: str = "",
            sam2_config: str = "") -> None:
    """Run an anchor detector and bidirectional track assembly for one pair.

    For serious runs, replace `_dino_boxes` with VLM trigger candidates plus
    Molmo/SAM2. This endpoint already emits sparse official frame IDs and
    supports stable IDs and visibility-aware track backends.
    """
    from gqa_core import sampled_frame_ids
    from gvqa_tracking import csrt_bidirectional
    root = pathlib.Path(DATA_ROOT) / split; db = json.loads(_find_json(root).read_text())
    if video_id not in db:
        examples = list(db)[:10]
        raise KeyError(f"Unknown video_id {video_id!r}. Use an actual ID from inspect; examples: {examples}")
    item = db[video_id]
    questions = item.get("grounded_question", [])
    matches = [q for q in questions if str(q["id"]) == str(question_id)]
    if not matches:
        raise KeyError(f"Unknown question_id {question_id!r} for {video_id}; available: {[q['id'] for q in questions]}")
    question = matches[0]
    video = _video_path(root, video_id)
    import cv2
    cap = cv2.VideoCapture(str(video)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    anchor = n // 2 if anchor_frame < 0 else max(0, min(n - 1, int(anchor_frame)))
    # Prefer cached VLM trigger moments and descriptive object phrases. If
    # reasoning has not been run, preserve the deterministic DINO fallback.
    cache = pathlib.Path(DATA_ROOT) / "cache" / split / video_id / f"{question_id}.json"
    hypotheses = []
    if cache.exists():
        cached = json.loads(cache.read_text()).get("reasoning", {})
        for obj in cached.get("objects", []):
            anchors = obj.get("anchors", [])
            best = max(anchors, key=lambda x: float(x.get("confidence", 0)), default={})
            grounding_text = str(obj.get("description", question["question"]))
            rels = obj.get("relations", []) or []
            evs = obj.get("events", []) or []
            if rels:
                grounding_text += "; " + ", ".join(map(str, rels))
            if evs:
                grounding_text += "; " + ", ".join(map(str, evs))
            count = max(1, int(obj.get("count", 1)))
            # Occlusion-game questions deliberately contain several similar
            # instances; retain multiple detector hypotheses so the tracker
            # can preserve the correct identity instead of committing to the
            # first visually similar object.
            if "similar" in question["question"].lower() or "occlusion" in question["question"].lower():
                count = max(count, 5)
            hypotheses.append((grounding_text,
                               count,
                               max(0, min(n - 1, int(best.get("frame", anchor))))))
    if not hypotheses:
        hypotheses = [(question["question"], 5, anchor)]
    proposals = []
    for description, count, trigger in hypotheses:
        candidate_boxes, _, _ = _dino_boxes(video, trigger, description)
        proposals.extend((trigger, box) for box in candidate_boxes[:count])
    tracks = []
    if backend not in {"csrt", "sam2"}:
        raise ValueError("backend must be csrt or sam2")
    if backend == "sam2":
        if not sam2_checkpoint or not sam2_config:
            raise ValueError("sam2 backend requires --sam2-checkpoint and --sam2-config")
        from sam2_backend import Sam2Bidirectional
        cap = cv2.VideoCapture(str(video)); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        tracker = Sam2Bidirectional(sam2_checkpoint, sam2_config)
        for tid, (trigger, box) in enumerate(proposals):
            track = tracker.track(video, trigger, box, tid, n, width, height, output_step=30)
            tracks.append(track.to_submission())
    else:
        for tid, (trigger, box) in enumerate(proposals):
            track = csrt_bidirectional(video, trigger, box, tid, output_step=30, expand=1.04)
            tracks.append(track.to_submission())
    prediction = {video_id: {"grounded_question": {str(question_id): tracks}}}
    out = pathlib.Path(DATA_ROOT) / "predictions" / output; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prediction)); volume.commit()
    print(json.dumps({"output": str(out), "video": video_id, "question": question_id,
                      "anchor": anchor, "sampled_frames": len(sampled_frame_ids(n)),
                      "tracks": len(tracks)}, indent=2))


@app.function(image=image, gpu="A100-80GB", volumes={DATA_ROOT: volume}, timeout=60 * 60 * 8)
def predict_advanced(split: str, video_id: str, question_id: str,
                     output: str = "advanced_predictions.json", backend: str = "sam2",
                     sam2_checkpoint: str = "", sam2_config: str = "",
                     spatial_refine: bool = False) -> None:
    """Run TimeLens2 -> Molmo2 VideoTrack -> SAM2.1 canonical architecture."""
    from advanced_pipeline import run_advanced
    root = pathlib.Path(DATA_ROOT) / split; db = json.loads(_find_json(root).read_text())
    if video_id not in db: raise KeyError(f"unknown video_id: {video_id}")
    qs = [q for q in db[video_id].get("grounded_question", []) if str(q["id"]) == str(question_id)]
    if not qs: raise KeyError(f"unknown question_id: {question_id}")
    q=qs[0]; video=_video_path(root, video_id)
    tracks,diag=run_advanced(video,q["question"],backend,sam2_checkpoint,sam2_config,
                             spatial_refine=spatial_refine)
    prediction={video_id:{"grounded_question":{str(question_id):tracks}}}
    out=pathlib.Path(DATA_ROOT)/"predictions"/output; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(prediction)); volume.commit()
    print(json.dumps({"output":str(out),"video":video_id,"question":question_id,"tracks":len(tracks),"diagnostics":diag},indent=2))


@app.function(image=molmo2_image, gpu="A100-80GB", volumes={DATA_ROOT: volume},
              timeout=60 * 60 * 12)
def predict_molmo2_sam2(split: str, video_id: str, question_id: str,
                        output: str = "molmo2_sam2.json",
                        sam2_checkpoint: str = "/data/checkpoints/sam2.1_hiera_large.pt",
                        sam2_config: str = "/usr/local/lib/python3.11/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                        output_step: int = 30, tracking_stride: int = 30,
                        multi_point: bool = False) -> None:
    """Opt-in local Molmo2-8B video tracking plus bidirectional SAM2.

    multi_point defaults to False (single first-evidence-point anchor, the
    documented baseline). Pass --multi-point to additionally seed SAM2 with
    every other Molmo2-reported timestamp for the same object ID.
    """
    from advanced_pipeline import run_molmo2_sam2
    root = pathlib.Path(DATA_ROOT) / split; db = json.loads(_find_json(root).read_text())
    q = next((x for x in db[video_id].get("grounded_question", []) if str(x["id"]) == str(question_id)), None)
    if q is None: raise KeyError(f"unknown {video_id}/{question_id}")
    tracks, diag = run_molmo2_sam2(_video_path(root, video_id), q["question"],
                                   sam2_checkpoint, sam2_config, output_step, tracking_stride,
                                   multi_point=multi_point)
    pred = {video_id: {"grounded_question": {str(question_id): tracks}}}
    out = pathlib.Path(DATA_ROOT) / "predictions" / output; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pred)); volume.commit()
    manifest = out.with_suffix(out.suffix + ".manifest.json")
    manifest.write_text(json.dumps({"mode": "molmo2_8b", "split": split,
                                    "video_id": video_id, "question_id": str(question_id),
                                    "diagnostics": diag,
                                    "artifact_sha256": __import__('hashlib').sha256(out.read_bytes()).hexdigest()}, indent=2))
    volume.commit(); print(json.dumps({"output": str(out), "diagnostics": {k:v for k,v in diag.items() if k != "raw_response"}}, indent=2))


@app.function(image=molmo7_image, gpu="A100-80GB", volumes={DATA_ROOT: volume},
              secrets=[modal.Secret.from_name("gemini-api-key")], timeout=60 * 60 * 12)
def predict_direct_sam2(split: str, video_id: str, question_id: str,
                        output: str = "direct_sam2.json",
                        sam2_checkpoint: str = "/data/checkpoints/sam2.1_hiera_large.pt",
                        sam2_config: str = "/usr/local/lib/python3.11/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                        output_step: int = 30, tracking_stride: int = 30,
                        model: str = "gemini-2.5-flash") -> None:
    """Opt-in TutuAI-style direct VLM frame/box grounding plus SAM2."""
    import os, re, time
    from google import genai
    from google.genai import types
    from advanced_pipeline import run_direct_sam2
    root = pathlib.Path(DATA_ROOT) / split; db = json.loads(_find_json(root).read_text())
    q = next((x for x in db[video_id].get("grounded_question", []) if str(x["id"]) == str(question_id)), None)
    if q is None: raise KeyError(f"unknown {video_id}/{question_id}")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    uploaded = client.files.upload(file=str(_video_path(root, video_id)))
    for _ in range(120):
        state = getattr(uploaded, "state", None)
        if not state or getattr(state, "name", "ACTIVE") == "ACTIVE": break
        if getattr(state, "name", "") == "FAILED": raise RuntimeError("Gemini video processing failed")
        time.sleep(2); uploaded = client.files.get(name=uploaded.name)
    prompt = ("Find the object(s) answering the question. Return ONLY JSON with keys "
              "frame and box. frame is the first frame of appearance, zero-indexed. "
              "box is normalized [x1,y1,x2,y2]. Question: " + q["question"])
    response = client.models.generate_content(model=model, contents=[uploaded, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1))
    raw = response.text.strip(); raw = re.sub(r"^```(?:json)?|```$", "", raw).strip()
    try:
        grounding = json.loads(raw)
    except json.JSONDecodeError:
        # Thinking VLMs occasionally append commentary or emit a Python-like
        # dict; recover only the bounded object, never arbitrary code.
        import ast
        match = re.search(r"\{.*\}", raw, re.S)
        if not match: raise
        try:
            grounding = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            cleaned = re.sub(r",\s*([}\]])", r"\1", match.group(0))
            try:
                grounding = json.loads(cleaned)
            except json.JSONDecodeError:
                pair = re.search(r'"frame"\s*:\s*(\d+).*?"box"\s*:\s*\[([^]]+)\]', raw, re.S)
                if not pair: raise
                grounding = {"frame": int(pair.group(1)),
                             "box": [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", pair.group(2))]}
    if isinstance(grounding, list):
        if not grounding: raise ValueError("direct grounding returned no objects")
        grounding = grounding[0]
    grounding["frame"] = int(grounding["frame"])
    tracks, diag = run_direct_sam2(_video_path(root, video_id), grounding,
                                   sam2_checkpoint, sam2_config, output_step, tracking_stride)
    pred = {video_id: {"grounded_question": {str(question_id): tracks}}}
    out = pathlib.Path(DATA_ROOT) / "predictions" / output; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pred)); volume.commit()
    manifest = out.with_suffix(out.suffix + ".manifest.json")
    manifest.write_text(json.dumps({"mode": "direct_vlm", "model": model, "split": split,
                                    "video_id": video_id, "question_id": str(question_id),
                                    "grounding": grounding, "raw_response": response.text,
                                    "diagnostics": diag,
                                    "artifact_sha256": __import__('hashlib').sha256(out.read_bytes()).hexdigest()}, indent=2))
    volume.commit(); print(json.dumps({"output": str(out), "diagnostics": diag}, indent=2))


@app.function(image=molmo7_image, gpu="A100-80GB", volumes={DATA_ROOT: volume},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=60 * 60 * 12)
def predict_cortex_sam2(split: str, video_id: str, question_id: str,
                        output: str = "cortex_sam2_predictions.json",
                        sam2_checkpoint: str = "/data/checkpoints/sam2.1_hiera_large.pt",
                        sam2_config: str = "/usr/local/lib/python3.11/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                        output_step: int = 30, tracking_stride: int = 30,
                        reference_mode: bool = False,
                        grounding_strategy: str = "trigger",
                        proposal_grounding: bool = False) -> None:
    """Run cached CORTEX reasoning, Molmo spatial grounding, and SAM2."""
    from advanced_pipeline import run_cortex_sam2
    root = pathlib.Path(DATA_ROOT) / split
    db = json.loads(_find_json(root).read_text())
    q = next((x for x in db[video_id].get("grounded_question", []) if str(x["id"]) == str(question_id)), None)
    if q is None: raise KeyError(f"unknown question {video_id}/{question_id}")
    cache = pathlib.Path(DATA_ROOT) / "cache" / split / video_id / f"{question_id}.json"
    if not cache.exists(): raise RuntimeError(f"missing CORTEX cache: {cache}")
    reasoning = json.loads(cache.read_text()).get("reasoning", {})
    tracks, diag = run_cortex_sam2(_video_path(root, video_id), reasoning,
                                   sam2_checkpoint, sam2_config, output_step,
                                   tracking_stride=tracking_stride,
                                   reference_mode=reference_mode,
                                   grounding_strategy=grounding_strategy,
                                   proposal_grounding=proposal_grounding)
    pred = {video_id: {"grounded_question": {str(question_id): tracks}}}
    out = pathlib.Path(DATA_ROOT) / "predictions" / output; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pred)); volume.commit()
    # Emit a local reproducibility manifest alongside the serialized shard.
    manifest = out.with_suffix(out.suffix + ".manifest.json")
    manifest.write_text(json.dumps({"split": split, "video_id": video_id,
                                    "question_id": str(question_id),
                                    "output": str(out), "diagnostics": diag,
                                    "artifact_sha256": __import__('hashlib').sha256(out.read_bytes()).hexdigest()}, indent=2))
    volume.commit()
    print(json.dumps({"output": str(out), "tracks": len(tracks), "diagnostics": diag}, indent=2))


@app.function(image=molmo2_image, gpu="A100-80GB", volumes={DATA_ROOT: volume},
              timeout=60 * 60 * 24)
def batch_predict_molmo2_sam2(split: str = "test", output: str = "molmo2_multipoint_submission.json",
                              start: int = 0, limit: int = 0, output_step: int = 30,
                              tracking_stride: int = 30, multi_point: bool = True) -> None:
    """Restartable full-split batch for the standalone multi-point Molmo2+SAM2
    pipeline (MULTIPOINT_GATING_EXPERIMENT.md). Loads Molmo2-8B once and reuses
    it across all questions. A per-question failure (e.g. Molmo2 returning no
    video points) is logged and skipped rather than aborting the whole batch;
    validate.py does not require full question coverage.
    """
    from advanced_pipeline import run_molmo2_sam2
    from advanced_models import Molmo2VideoPoint
    root = pathlib.Path(DATA_ROOT) / split; db = json.loads(_find_json(root).read_text())
    jobs = [(vid, q) for vid in sorted(db) for q in db[vid].get("grounded_question", [])]
    jobs = jobs[start:start + limit] if limit else jobs[start:]
    out = pathlib.Path(DATA_ROOT) / "predictions" / output; out.parent.mkdir(parents=True, exist_ok=True)
    heartbeat = out.with_suffix(out.suffix + ".heartbeat.json")
    results = json.loads(out.read_text()) if out.exists() else {}
    sam2_checkpoint = "/data/checkpoints/sam2.1_hiera_large.pt"
    sam2_config = "/usr/local/lib/python3.11/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    track_model = Molmo2VideoPoint()
    for index, (vid, q) in enumerate(jobs, start=start):
        qid = str(q["id"])
        if qid in results.get(vid, {}).get("grounded_question", {}): continue
        heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid, "status": "running",
                                         "completed": sum(len(v.get("grounded_question", {})) for v in results.values())}))
        volume.commit()
        try:
            tracks, diag = run_molmo2_sam2(_video_path(root, vid), q["question"], sam2_checkpoint, sam2_config,
                                           output_step, tracking_stride, multi_point=multi_point,
                                           track_model=track_model)
            status = {"ok": True, "tracks": len(tracks)}
        except Exception as exc:
            tracks = []; status = {"ok": False, "error": repr(exc)}
        if not status["ok"]:
            heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid,
                                             "status": "failed", "error": status.get("error")}))
            print(json.dumps({"index": index, "video": vid, "question": qid, **status}), flush=True)
            continue
        results.setdefault(vid, {"grounded_question": {}})["grounded_question"][qid] = tracks
        out.write_text(json.dumps(results))
        heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid, "status": "completed",
                                         "completed": sum(len(v.get("grounded_question", {})) for v in results.values())}))
        volume.commit()
        print(json.dumps({"index": index, "video": vid, "question": qid, **status}), flush=True)
        # SAM2's CPU-offloaded video state and decord/opencv handles were
        # observed to accumulate host memory across iterations of this loop
        # (containers OOM-killed after ~15 questions); force collection every
        # question rather than relying on Python's own GC timing.
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"output": str(out), "jobs": len(jobs), "multi_point": multi_point}))


@app.function(image=molmo7_image, gpu="A100-80GB", volumes={DATA_ROOT: volume},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=60 * 60 * 24)
def batch_predict_cortex_sam2(split: str = "valid", output: str = "cortex_sam2_subset.json",
                              start: int = 0, limit: int = 20, output_step: int = 30,
                              grounding_strategy: str = "trigger", tracking_stride: int = 30,
                              reference_mode: bool = False,
                              proposal_grounding: bool = False) -> None:
    """Restartable CORTEX/Molmo/SAM2 validation batch."""
    from advanced_pipeline import run_cortex_sam2
    root = pathlib.Path(DATA_ROOT) / split; db = json.loads(_find_json(root).read_text())
    jobs = [(vid, q) for vid in sorted(db) for q in db[vid].get("grounded_question", [])]
    jobs = jobs[start:start + limit]
    out = pathlib.Path(DATA_ROOT) / "predictions" / output; out.parent.mkdir(parents=True, exist_ok=True)
    heartbeat = out.with_suffix(out.suffix + ".heartbeat.json")
    results = json.loads(out.read_text()) if out.exists() else {}
    for index, (vid, q) in enumerate(jobs, start=start):
        qid = str(q["id"])
        if qid in results.get(vid, {}).get("grounded_question", {}): continue
        heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid,
                                         "status": "running", "completed": sum(
                                             len(v.get("grounded_question", {})) for v in results.values())}))
        volume.commit()
        cache = pathlib.Path(DATA_ROOT) / "cache" / split / vid / f"{qid}.json"
        if not cache.exists():
            heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid,
                                             "status": "failed", "error": "missing reasoning cache"}))
            raise RuntimeError(f"missing reasoning cache for {vid}/{qid}")
        reasoning = json.loads(cache.read_text()).get("reasoning", {})
        try:
            tracks, diag = run_cortex_sam2(_video_path(root, vid), reasoning,
                "/data/checkpoints/sam2.1_hiera_large.pt",
                "/usr/local/lib/python3.11/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                output_step, grounding_strategy, tracking_stride, reference_mode,
                proposal_grounding)
            status = {"ok": True, "tracks": len(tracks), "diag": diag}
        except Exception as exc:
            tracks = []; status = {"ok": False, "error": repr(exc)}
        if not status["ok"]:
            # Failed shards must remain resumable and must never masquerade as
            # valid empty predictions in a controlled comparison.
            heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid,
                                             "status": "failed", "error": status.get("error")}))
            raise RuntimeError(f"controlled shard failed for {vid}/{qid}: {status.get('error')}")
        results.setdefault(vid, {"grounded_question": {}})["grounded_question"][qid] = tracks
        out.write_text(json.dumps(results))
        heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid,
                                         "status": "completed" if status["ok"] else "failed",
                                         "error": status.get("error"), "completed": sum(
                                             len(v.get("grounded_question", {})) for v in results.values())}))
        volume.commit()
        print(json.dumps({"index": index, "video": vid, "question": qid, **status}), flush=True)
    print(json.dumps({"output": str(out), "jobs": len(jobs),
                      "grounding_strategy": grounding_strategy,
                      "tracking_stride": tracking_stride, "reference_mode": reference_mode}))


@app.function(image=sam3_image, gpu="A100-80GB", volumes={DATA_ROOT: volume},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=60 * 60 * 8)
def predict_sam3(split: str, video_id: str, question_id: str,
                 output: str = "sam3_predictions.json", prompt: str = "",
                 anchor_frame: int = -1, output_step: int = 30) -> None:
    """Run SAM3.1 concept tracking for one video/question pair.

    `prompt` should be a short noun phrase (for example ``red umbrella``),
    preferably produced by the reasoning stage rather than the raw question.
    SAM3 itself supplies exhaustive multi-instance detections and stable IDs;
    the adapter serializes its masks as the official normalized boxes.
    """
    from sam3_backend import Sam3VideoTracker
    root = pathlib.Path(DATA_ROOT) / split
    db = json.loads(_find_json(root).read_text())
    if video_id not in db:
        raise KeyError(f"unknown video_id: {video_id}")
    qs = [q for q in db[video_id].get("grounded_question", [])
          if str(q["id"]) == str(question_id)]
    if not qs:
        raise KeyError(f"unknown question_id: {question_id}")
    question = qs[0]
    video = _video_path(root, video_id)
    prompt, anchor_frame = _sam3_prompt_anchor(
        split, video_id, str(question_id), question, prompt, anchor_frame)
    tracks = Sam3VideoTracker().track(video, prompt, anchor_frame=anchor_frame,
                                      output_step=output_step)
    prediction = {video_id: {"grounded_question": {str(question_id):
                                                     [t.to_submission() for t in tracks]}}}
    out = pathlib.Path(DATA_ROOT) / "predictions" / output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prediction))
    volume.commit()
    print(json.dumps({"output": str(out), "video": video_id,
                      "question": question_id, "prompt": prompt,
                      "tracks": len(tracks)}, indent=2))


@app.function(image=sam3_image, gpu="A100-80GB", volumes={DATA_ROOT: volume},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=60 * 60 * 24)
def batch_predict_sam3(split: str, output: str = "sam3_submission.json",
                       start: int = 0, limit: int = 0, output_step: int = 30,
                       retry_empty: bool = True) -> None:
    """Restartable SAM3.1 batch inference with one model per warm worker."""
    from sam3_backend import Sam3VideoTracker
    root = pathlib.Path(DATA_ROOT) / split
    db = json.loads(_find_json(root).read_text())
    jobs = [(vid, q) for vid in sorted(db)
            for q in db[vid].get("grounded_question", [])]
    jobs = jobs[start:(start + limit) if limit else None]
    out = pathlib.Path(DATA_ROOT) / "predictions" / output
    out.parent.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(out.read_text()) if out.exists() else {}
    tracker = Sam3VideoTracker()
    for index, (vid, question) in enumerate(jobs, start=start):
        qid = str(question["id"])
        existing = all_results.get(vid, {}).get("grounded_question", {}).get(qid)
        if existing is not None and (not retry_empty or existing):
            continue
        try:
            prompt, anchor = _sam3_prompt_anchor(split, vid, qid, question)
            tracks = tracker.track(_video_path(root, vid), prompt,
                                   anchor_frame=anchor, output_step=output_step)
            serialized = [track.to_submission() for track in tracks]
            status = {"ok": True, "tracks": len(serialized),
                      "prompt": prompt, "anchor_frame": anchor}
        except Exception as exc:
            serialized = []
            status = {"ok": False, "error": repr(exc)}
        all_results.setdefault(vid, {"grounded_question": {}})["grounded_question"][qid] = serialized
        out.write_text(json.dumps(all_results))
        volume.commit()
        print(json.dumps({"index": index, "video": vid, "question": qid, **status}), flush=True)
    print(json.dumps({"output": str(out), "jobs": len(jobs)}, indent=2))


@app.function(image=image, volumes={DATA_ROOT: volume},
              secrets=[modal.Secret.from_name("gemini-api-key"),
                       modal.Secret.from_name("gemini-api-keys"),
                       modal.Secret.from_name("gcp-vertex-service-account")],
              timeout=60 * 60 * 8)
def batch_reason_pro(split: str = "test", limit: int = 1900, start: int = 0,
                     model: str = "gemini-3.7-flash", force: bool = False,
                     cost_cap: int = 1900, use_vertex: bool = False,
                     vertex_project: str = "veotest-460812",
                     vertex_location: str = "global",
                     vertex_video_fps: float = 3.0,
                     thinking_level: str = "high",
                     quota_retries: int = 12,
                     quota_backoff_seconds: int = 15) -> None:
    """Cache Gemini CORTEX reasoning with model/auth provenance.

    The SGVR@KAIST 2025 winner used Gemini 2.5 Pro for trigger-moment reasoning;
    the Pro model's deeper video understanding produces more precise trigger frames
    and better object descriptions than the Flash variant. This function clears
    stale flash-cached entries and regenerates with the Pro model.

    Usage:
        modal run modal_app.py::batch_reason_pro --split test --limit 1900 --use-vertex
    """
    import os, time, cv2
    from google import genai
    from google.genai import types
    from gvqa_reasoning import SYSTEM_PROMPT, parse_reasoning
    root = pathlib.Path(DATA_ROOT) / split
    db = json.loads(_find_json(root).read_text())
    jobs = [(vid, q) for vid in sorted(db) for q in db[vid].get("grounded_question", [])]
    jobs = jobs[start:start + limit] if limit else jobs[start:]
    if limit > cost_cap:
        raise ValueError(f"limit {limit} exceeds cost_cap {cost_cap}")
    auth_backend = "vertex" if use_vertex else "developer-api"
    if use_vertex:
        credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not credentials_json:
            raise RuntimeError("Vertex credential JSON missing from Modal secret")
        credentials_path = pathlib.Path("/tmp/gcp_vertex_credentials.json")
        credentials_path.write_text(credentials_json)
        credentials_path.chmod(0o600)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
        clients = [genai.Client(vertexai=True, project=vertex_project,
                                location=vertex_location,
                                http_options=types.HttpOptions(timeout=180_000))]
    else:
        keys = []
        for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
            if os.environ.get(name): keys.append(os.environ[name])
        for name in sorted(os.environ):
            if name.startswith(("GOOGLE_API_KEY_", "GEMINI_API_KEY_")) and os.environ[name]:
                keys.append(os.environ[name])
        keys = list(dict.fromkeys(keys))
        if not keys:
            raise RuntimeError("No Gemini API key found in attached Modal secrets")
        clients = [genai.Client(api_key=key) for key in keys]
    done = 0; api_calls = 0; key_index = 0
    for vid, qitem in jobs:
        qid = str(qitem["id"])
        cache = pathlib.Path(DATA_ROOT) / "cache" / split / vid / f"{qid}.json"
        if cache.exists() and not force:
            try:
                cached = json.loads(cache.read_text())
                prov = cached.get("provenance", {})
                # Accept cache only if it was generated with the same Pro model
                if (prov.get("model") == model and
                        prov.get("auth_backend", "developer-api") == auth_backend and
                        prov.get("split") == split and
                        str(prov.get("question_id")) == str(qid) and
                        int(prov.get("api_calls", 0)) >= 1 and
                        int(prov.get("cost_cap", 0)) >= 1):
                    done += 1; continue
            except (OSError, ValueError, TypeError):
                pass
        video = _video_path(root, vid)
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if api_calls >= cost_cap:
            raise RuntimeError(f"Gemini cost cap exhausted after {api_calls} calls")
        api_calls += 1
        prompt = SYSTEM_PROMPT + f"\nVideo has {n} frames at {fps:.4f} FPS.\nQuestion: {qitem['question']}"
        last_error = None; response = None
        max_attempts = max(3, len(clients), quota_retries + 1)
        for attempt in range(max_attempts):
            client = clients[key_index % len(clients)]
            key_index += 1
            try:
                if use_vertex:
                    video_part = types.Part(inline_data=types.Blob(
                        data=video.read_bytes(), mime_type="video/mp4"),
                        video_metadata=types.VideoMetadata(fps=vertex_video_fps))
                    contents = [types.Content(role="user", parts=[
                        video_part, types.Part.from_text(text=prompt)])]
                else:
                    uploaded = client.files.upload(file=str(video))
                    for _ in range(120):
                        state = getattr(uploaded, "state", None)
                        if not state or getattr(state, "name", "ACTIVE") == "ACTIVE": break
                        if getattr(state, "name", "") == "FAILED":
                            raise RuntimeError("Gemini video processing failed")
                        time.sleep(2); uploaded = client.files.get(name=uploaded.name)
                    contents = [uploaded, prompt]
                if model.startswith("gemini-3"):
                    generation_config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=16384,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=thinking_level))
                else:
                    generation_config = types.GenerateContentConfig(
                        response_mime_type="application/json", temperature=0.1,
                        max_output_tokens=16384)
                response = client.models.generate_content(
                    model=model, contents=contents,
                    config=generation_config)
                candidate = response.candidates[0] if response.candidates else None
                finish_reason = getattr(candidate, "finish_reason", None)
                finish_name = getattr(finish_reason, "name", str(finish_reason or ""))
                if "MAX_TOKENS" in finish_name.upper():
                    raise RuntimeError("Gemini output was cut off at max_output_tokens")
                if not getattr(response, "text", "").strip():
                    raise RuntimeError(f"Gemini returned no text (finish_reason={finish_name})")
                break
            except Exception as exc:
                # Never allow a response rejected as truncated/empty to fall
                # through to parsing and caching after the final retry.
                response = None
                last_error = exc
                text = str(exc).upper()
                quota = "429" in text or "RESOURCE_EXHAUSTED" in text or "QUOTA" in text
                if not quota and attempt >= 2:
                    raise
                if attempt + 1 < max_attempts:
                    delay = (min(300, quota_backoff_seconds * (2 ** min(attempt, 5)))
                             if quota else 10 * (attempt + 1))
                    print(json.dumps({"retry": attempt + 1,
                                      "max_retries": max_attempts - 1,
                                      "delay_seconds": delay,
                                      "quota": quota,
                                      "video": vid, "question": qid,
                                      "error": str(exc)[:500]}), flush=True)
                    time.sleep(delay)
        if response is None:
            raise RuntimeError(f"Gemini failed after {max_attempts} attempt(s): {last_error}")
        parsed = parse_reasoning(response.text, n)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "question": qitem["question"], "raw": response.text,
            "provenance": {"model": model, "auth_backend": auth_backend,
                           "vertex_project": vertex_project if use_vertex else None,
                           "thinking_level": thinking_level if model.startswith("gemini-3") else None,
                           "api_calls": api_calls, "cost_cap": cost_cap,
                           "split": split, "question_id": str(qid)},
            "reasoning": {"objects": [{"description": o.description, "count": o.count,
                "relations": o.relations, "events": o.events,
                "anchors": [{"frame": a.frame, "confidence": a.confidence} for a in o.anchors],
                "visible_intervals": o.visible_intervals} for o in parsed.objects],
                "answer_cardinality": parsed.answer_cardinality,
                "confidence": parsed.confidence, "evidence": parsed.evidence}}, indent=2))
        done += 1; volume.commit()
        print(json.dumps({"done": done, "api_calls": api_calls,
                          "video": vid, "question": qid}), flush=True)


@app.function(image=molmo2_image, gpu="A100-80GB", volumes={DATA_ROOT: volume},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=60 * 60 * 12)
def predict_cortex_sam2_v2(split: str, video_id: str, question_id: str,
                            output: str = "cortex_v2_predictions.json",
                            sam2_checkpoint: str = "/data/checkpoints/sam2.1_hiera_large.pt",
                            sam2_config: str = "/usr/local/lib/python3.11/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                            output_step: int = 30, tracking_stride: int = 30,
                            spatial_model: str = "molmo2er",
                            grounding_strategy: str = "trigger") -> None:
    """CORTEX v2: Gemini 3.7 reasoning + Molmo2-ER/VideoPoint-4B grounding + SAM2.

    spatial_model options:
      - "molmo2er"   : Molmo2-ER (default, ~19 GB, specialized spatial pointing)
      - "molmo2vp4b" : Molmo2-VideoPoint-4B (video-specific grounding, best for clear video)
      - "molmo7b"    : Molmo-7B-O-0924 (legacy winner-aligned fallback)

    This is the v2 production endpoint. Use batch_predict_v2 for full split inference.
    """
    from advanced_pipeline import run_cortex_sam2
    from advanced_models import Molmo2ER, Molmo2VideoPoint4B, MolmoSpatial
    root = pathlib.Path(DATA_ROOT) / split
    db = json.loads(_find_json(root).read_text())
    q = next((x for x in db[video_id].get("grounded_question", [])
               if str(x["id"]) == str(question_id)), None)
    if q is None: raise KeyError(f"unknown question {video_id}/{question_id}")
    cache = pathlib.Path(DATA_ROOT) / "cache" / split / video_id / f"{question_id}.json"
    if not cache.exists():
        raise RuntimeError(f"missing CORTEX cache: {cache} — run batch_reason_pro first")
    reasoning = json.loads(cache.read_text()).get("reasoning", {})
    # Load the requested spatial grounding model
    if spatial_model == "molmo2er":
        spatial = Molmo2ER()
    elif spatial_model == "molmo2vp4b":
        spatial = Molmo2VideoPoint4B()
    else:
        spatial = MolmoSpatial()
    tracks, diag = run_cortex_sam2(_video_path(root, video_id), reasoning,
                                   sam2_checkpoint, sam2_config, output_step,
                                   grounding_strategy=grounding_strategy,
                                   tracking_stride=tracking_stride,
                                   reference_mode=False,
                                   proposal_grounding=False,
                                   spatial_model=spatial)
    pred = {video_id: {"grounded_question": {str(question_id): tracks}}}
    out = pathlib.Path(DATA_ROOT) / "predictions" / output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pred)); volume.commit()
    manifest = out.with_suffix(out.suffix + ".manifest.json")
    manifest.write_text(json.dumps({
        "split": split, "video_id": video_id, "question_id": str(question_id),
        "spatial_model": spatial_model, "grounding_strategy": grounding_strategy,
        "output": str(out), "diagnostics": diag,
        "artifact_sha256": __import__('hashlib').sha256(out.read_bytes()).hexdigest()},
        indent=2))
    volume.commit()
    print(json.dumps({"output": str(out), "tracks": len(tracks),
                      "spatial_model": spatial_model, "diagnostics": diag}, indent=2))


@app.function(image=molmo2_image, gpu="A100-80GB", volumes={DATA_ROOT: volume},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=60 * 60 * 24)
def batch_predict_v2(split: str = "test", output: str = "v2_submission.json",
                     start: int = 0, limit: int = 0, output_step: int = 30,
                     tracking_stride: int = 30, spatial_model: str = "molmo2er",
                     grounding_strategy: str = "trigger",
                     retry_empty: bool = True) -> None:
    """Restartable v2 batch: CORTEX (Gemini 3.7) + Molmo2-ER/VP4B + SAM2.

    Requires pre-cached reasoning from batch_reason_pro. Loads the spatial model
    once and reuses it across all questions to minimize cold-start overhead.

    Usage (full test split):
        modal run modal_app.py::batch_reason_pro --split test --use-vertex
        modal run modal_app.py::batch_predict_v2 --split test \\
            --output v2_test_submission.json --spatial-model molmo2er

    Usage (partial, restartable):
        modal run modal_app.py::batch_predict_v2 --split test \\
            --output v2_test_submission.json --start 0 --limit 500

    spatial_model: "molmo2er" | "molmo2vp4b" | "molmo7b"
    """
    import os, gc
    import torch
    from advanced_pipeline import run_cortex_sam2
    from advanced_models import Molmo2ER, Molmo2VideoPoint4B, MolmoSpatial
    os.environ.setdefault("HF_HOME", "/data/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/data/hf-cache/hub")
    root = pathlib.Path(DATA_ROOT) / split
    db = json.loads(_find_json(root).read_text())
    jobs = [(vid, q) for vid in sorted(db) for q in db[vid].get("grounded_question", [])]
    jobs = jobs[start:(start + limit) if limit else None]
    out = pathlib.Path(DATA_ROOT) / "predictions" / output
    out.parent.mkdir(parents=True, exist_ok=True)
    heartbeat = out.with_suffix(out.suffix + ".heartbeat.json")
    all_results = json.loads(out.read_text()) if out.exists() else {}
    sam2_checkpoint = "/data/checkpoints/sam2.1_hiera_large.pt"
    sam2_config = "/usr/local/lib/python3.11/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    # Load spatial model once for the whole batch
    if spatial_model == "molmo2er":
        spatial = Molmo2ER()
    elif spatial_model == "molmo2vp4b":
        spatial = Molmo2VideoPoint4B()
    else:
        spatial = MolmoSpatial()
    for index, (vid, q) in enumerate(jobs, start=start):
        qid = str(q["id"])
        existing = all_results.get(vid, {}).get("grounded_question", {}).get(qid)
        if existing is not None and (not retry_empty or existing):
            continue
        cache = pathlib.Path(DATA_ROOT) / "cache" / split / vid / f"{qid}.json"
        if not cache.exists():
            heartbeat.write_text(json.dumps({"index": index, "video": vid,
                                             "question": qid, "status": "skipped",
                                             "reason": "missing reasoning cache"}))
            print(json.dumps({"index": index, "video": vid, "question": qid,
                               "status": "skipped", "reason": "missing cache"}), flush=True)
            continue
        reasoning = json.loads(cache.read_text()).get("reasoning", {})
        heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid,
                                         "status": "running", "completed": sum(
                                             len(v.get("grounded_question", {}))
                                             for v in all_results.values())}))
        volume.commit()
        try:
            tracks, diag = run_cortex_sam2(
                _video_path(root, vid), reasoning,
                sam2_checkpoint, sam2_config, output_step,
                grounding_strategy=grounding_strategy,
                tracking_stride=tracking_stride,
                reference_mode=False, proposal_grounding=False,
                spatial_model=spatial)
            all_results.setdefault(vid, {"grounded_question": {}})["grounded_question"][qid] = tracks
            status = {"ok": True, "tracks": len(tracks)}
        except Exception as exc:
            tracks = []; status = {"ok": False, "error": repr(exc)}
        out.write_text(json.dumps(all_results))
        heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid,
                                         "status": "completed" if status["ok"] else "failed",
                                         "error": status.get("error"), "completed": sum(
                                             len(v.get("grounded_question", {}))
                                             for v in all_results.values())}))
        volume.commit()
        print(json.dumps({"index": index, "video": vid, "question": qid, **status}), flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"output": str(out), "jobs": len(jobs),
                      "spatial_model": spatial_model,
                      "grounding_strategy": grounding_strategy}))


@app.function(image=image, gpu="A100-80GB", volumes={DATA_ROOT: volume}, timeout=60 * 60 * 24)
def batch_predict(split: str, output: str = "submission.json", backend: str = "sam2",
                  start: int = 0, limit: int = 0, spatial_refine: bool = False,
                  retry_empty: bool = True) -> None:
    """Restartable batch inference over every video/question pair.

    `start`/`limit` partition the manifest. Each completed question is written
    immediately, so Modal interruption does not discard prior work. Run train
    or valid first; use test only after freezing the configuration.
    """
    from advanced_pipeline import run_advanced
    import os
    os.environ.setdefault("HF_HOME", "/data/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/data/hf-cache/hub")
    from advanced_models import TimeLens, MolmoVideoTrack
    temporal_model, track_model = TimeLens(), MolmoVideoTrack()
    root = pathlib.Path(DATA_ROOT) / split; db = json.loads(_find_json(root).read_text())
    jobs = [(vid, q) for vid in sorted(db) for q in db[vid].get("grounded_question", [])]
    jobs = jobs[start:(start + limit) if limit else None]
    out = pathlib.Path(DATA_ROOT) / "predictions" / output; out.parent.mkdir(parents=True, exist_ok=True)
    heartbeat = out.with_suffix(out.suffix + ".heartbeat.json")
    all_results = json.loads(out.read_text()) if out.exists() else {}
    for index, (vid, q) in enumerate(jobs, start=start):
        qid = str(q["id"])
        existing = all_results.get(vid, {}).get("grounded_question", {}).get(qid)
        if existing is not None and (not retry_empty or existing):
            continue
        try:
            tracks, diag = run_advanced(_video_path(root, vid), q["question"], backend,
                                        "/data/checkpoints/sam2.1_hiera_large.pt",
                                        "configs/sam2.1/sam2.1_hiera_l.yaml",
                                        spatial_refine=spatial_refine,
                                        temporal_model=temporal_model,
                                        track_model=track_model)
            all_results.setdefault(vid, {"grounded_question": {}})["grounded_question"][qid] = tracks
            status = {"ok": True, "tracks": len(tracks), "diagnostics": diag}
        except Exception as exc:
            # A per-question failure (e.g. this fallback model also finding
            # nothing) is logged and skipped rather than aborting the whole
            # gap-fill pass, matching batch_predict_molmo2_sam2's convention.
            status = {"ok": False, "error": repr(exc)}
            heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid,
                                             "status": "failed", "error": status["error"]}))
            print(json.dumps({"index": index, "video": vid, "question": qid, **status}), flush=True)
            continue
        out.write_text(json.dumps(all_results)); volume.commit()
        heartbeat.write_text(json.dumps({"index": index, "video": vid, "question": qid, "status": "completed"}))
        print(json.dumps({"index": index, "video": vid, "question": qid, **status}), flush=True)
    print(json.dumps({"output": str(out), "completed": len(all_results), "jobs": len(jobs)}, indent=2))


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60 * 10)
def merge_predictions(output: str, shards: str) -> None:
    """Merge comma-separated shard filenames into one submission manifest."""
    merged={}
    for name in [x.strip() for x in shards.split(",") if x.strip()]:
        path=pathlib.Path(DATA_ROOT)/"predictions"/name
        if not path.exists(): raise FileNotFoundError(path)
        part=json.loads(path.read_text())
        for vid,item in part.items():
            merged.setdefault(vid,{"grounded_question":{}})["grounded_question"].update(item.get("grounded_question",{}))
    dest=pathlib.Path(DATA_ROOT)/"predictions"/output; dest.write_text(json.dumps(merged)); volume.commit()
    print(json.dumps({"output":str(dest),"videos":len(merged),"shards":len(shards.split(","))}))


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60 * 10)
def validate_submission(split: str, output: str = "submission.json", allow_partial: bool = False) -> None:
    """Validate output shape; optionally permit an intentionally partial batch."""
    root = pathlib.Path(DATA_ROOT) / split; db = json.loads(_find_json(root).read_text())
    pred_path = pathlib.Path(DATA_ROOT) / "predictions" / output
    pred = json.loads(pred_path.read_text()); errors=[]; expected=0; found=0
    for vid,item in db.items():
        for q in item.get("grounded_question", []):
            expected += 1; tracks = pred.get(vid, {}).get("grounded_question", {}).get(str(q["id"]))
            if tracks is None:
                if not allow_partial: errors.append(f"missing {vid}/{q['id']}")
                continue
            found += 1
            if not isinstance(tracks, list): errors.append(f"not list {vid}/{q['id']}"); continue
            ids=set()
            for t in tracks:
                tid=t.get("id")
                if tid in ids: errors.append(f"duplicate track {vid}/{q['id']}")
                ids.add(tid)
                frames=t.get("frame_ids", []); boxes=t.get("bounding_boxes", [])
                if len(frames) != len(boxes):
                    errors.append(f"length mismatch {vid}/{q['id']}")
                if not isinstance(t.get("score"), (int,float)) or not 0 <= t["score"] <= 1:
                    errors.append(f"invalid score {vid}/{q['id']}")
                if any(not isinstance(f,int) or f < 0 for f in frames):
                    errors.append(f"invalid frame id {vid}/{q['id']}")
                if frames != sorted(set(frames)):
                    errors.append(f"unsorted/duplicate frames {vid}/{q['id']}")
                for box in boxes:
                    if (not isinstance(box,list) or len(box)!=4 or
                        any(not isinstance(x,(int,float)) or not 0<=x<=1 for x in box) or
                        box[0]>box[2] or box[1]>box[3]):
                        errors.append(f"invalid normalized box {vid}/{q['id']}")
    print(json.dumps({"expected": expected, "found": found, "missing": expected-found,
                      "partial": allow_partial, "errors": errors[:50]}, indent=2))
    if errors: raise RuntimeError(f"submission has {len(errors)} errors")
    if not allow_partial and found != expected: raise RuntimeError("submission is incomplete")


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60 * 30)
def evaluate_hota(split: str = "valid", output: str = "valid_submission.json", allow_partial: bool = False) -> None:
    """Run HOTA/DetA/AssA/LocA against labels available for train/valid."""
    if split == "test": raise ValueError("test labels are not available for evaluation")
    from local_eval import evaluate_files
    root=pathlib.Path(DATA_ROOT)/split
    pred=pathlib.Path(DATA_ROOT)/"predictions"/output
    result=evaluate_files(str(pred),str(_find_json(root)), only_present=allow_partial)
    print(json.dumps(result,indent=2))
    (pred.with_suffix(".metrics.json")).write_text(json.dumps(result,indent=2)); volume.commit()


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60)
def export_controlled_artifact(output: str, split: str = "valid") -> None:
    """Print artifact JSON and a separate manifest, suitable for audit capture."""
    import hashlib
    path = pathlib.Path(DATA_ROOT) / "predictions" / output
    if not path.exists(): raise FileNotFoundError(path)
    payload = path.read_bytes()
    artifact = json.loads(payload)
    manifest = {"artifact": output, "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload), "split": split,
                "videos": sorted(artifact), "payload": artifact}
    print(json.dumps(manifest, indent=2))


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60)
def inspect_cache(split: str, video_id: str, question_id: str) -> None:
    cache = pathlib.Path(DATA_ROOT) / "cache" / split / video_id / f"{question_id}.json"
    print(cache.read_text() if cache.exists() else f"missing: {cache}")


@app.local_entrypoint()
def main() -> None:
    print("Use modal run modal_app.py::download --split test")
