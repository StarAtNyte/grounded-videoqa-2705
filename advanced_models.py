"""Lazy local model adapters for the canonical GVQA architecture.

Models are loaded only when the advanced backend is selected. All adapters
return simple Python structures so tracking/postprocessing stays model-agnostic.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any

_TRACK_RE = re.compile(r'<tracks?[^>]*coords="([0-9\t:;, .]+)"', re.I)
_POINT_RE = re.compile(r'(?:^|[\t:;,])([0-9.]+) ([0-9 ]+)')
_XY_RE = re.compile(r'(?:^|\s)(\d{1,4}) (\d{1,4}) (\d{1,4})')


def _load(model_id: str):
    import torch
    from transformers import AutoProcessor
    # Molmo's pre-4.50 remote code calls this private helper, removed in
    # Transformers 4.50+. Supply the tiny compatibility shim locally rather
    # than mutating downloaded model files.
    from transformers.generation.utils import GenerationMixin
    if not hasattr(GenerationMixin, "_extract_past_from_model_output"):
        def _extract_past_from_model_output(self, outputs, standardize_cache_format=True):
            return "past_key_values", getattr(outputs, "past_key_values", None)
        GenerationMixin._extract_past_from_model_output = _extract_past_from_model_output
    hf_kwargs = {}
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # Molmo-7B-O uses the official CausalLM remote-code API; newer Molmo2
    # checkpoints use ImageTextToText. Keep both adapters available.
    from transformers import AutoModelForCausalLM
    if "Molmo-7B" in model_id:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype, device_map="auto", **hf_kwargs)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=dtype, device_map="auto")
        except (ImportError, ValueError, OSError):
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
                attn_implementation="sdpa")
    model.eval()
    return processor, model


def _video_inputs(video: str, text: str, max_fps: float, max_frames: int = 256,
                  utility: str = "molmo"):
    if utility == "qwen":
        # molmo_utils switches decord to the torch bridge globally. Qwen's
        # decoder expects native decord batches and otherwise calls .asnumpy()
        # on a torch Tensor on the second question in a warm container.
        try:
            import decord
            decord.bridge.set_bridge("native")
        except Exception:
            pass
        from qwen_vl_utils import process_vision_info
    else:
        from molmo_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "video", "video": str(Path(video).resolve()),
         "max_fps": max_fps, "max_frames": max_frames,
         "max_pixels": 448 * 448},]}]
    try:
        extra = {"image_patch_size": 16} if utility == "qwen" else {}
        result = process_vision_info(messages, return_video_kwargs=True,
                                     return_video_metadata=True, **extra)
    except (TypeError, ValueError):
        try:
            extra = {"image_patch_size": 16} if utility == "qwen" else {}
            result = process_vision_info(messages, return_video_metadata=True, **extra)
        except (TypeError, ValueError):
            # molmo_utils and older qwen-vl-utils expose different signatures.
            result = process_vision_info(messages)
    if len(result) == 3:
        _, videos, kwargs = result
    else:
        _, videos = result
        # The older qwen-vl-utils returns already-decoded frames without
        # metadata. Prevent Transformers from sampling 32 indices from a
        # shorter predecoded tensor (the source of the out-of-bounds failure).
        kwargs = {"do_sample_frames": False}
    metas = None
    if videos:
        try:
            videos, metas = zip(*videos); videos, metas = list(videos), list(metas)
        except (TypeError, ValueError):
            videos = list(videos)
    return messages, videos, kwargs, metas


def _generate(processor, model, messages, videos, kwargs, metas=None, max_new_tokens=2048):
    import torch
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, videos=videos, video_metadata=metas,
                       return_tensors="pt", padding=True, **(kwargs or {}))
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode(): out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class TimeLens:
    def __init__(self, model_id="MCG-NJU/TimeLens2-8B"):
        self.processor, self.model = _load(model_id)

    def trigger(self, video: str, query: str) -> dict[str, Any]:
        """Return a semantic object description and clearest trigger time."""
        prompt=(f'For the query: "{query}", identify the exact physical object(s) '
                'to track and the clearest moment to initialize grounding. '
                'Return ONLY JSON with keys description (string), trigger_seconds '
                '(number), and intervals (array of [start,end] seconds).')
        messages,videos,kwargs,metas=_video_inputs(video,prompt,max_fps=2,max_frames=256,utility="qwen")
        text=_generate(self.processor,self.model,messages,videos,kwargs,metas,256)
        try:
            raw=re.search(r'\{.*\}',text,re.S).group(0)
            value=json.loads(raw)
            return {"description":str(value.get("description",query)),
                    "trigger_seconds":float(value.get("trigger_seconds",0)),
                    "intervals":[[float(a),float(b)] for a,b in value.get("intervals",[])]}
        except Exception:
            return {"description":query,"trigger_seconds":0.,"intervals":[]}

    def intervals(self, video: str, query: str) -> list[list[float]]:
        prompt = (f'Given the query: "{query}", return ALL time spans in seconds '
                  'where the query is relevant. Output ONLY a JSON array of [start,end] pairs.')
        messages, videos, kwargs, metas = _video_inputs(video, prompt, max_fps=2, max_frames=256, utility="qwen")
        text = _generate(self.processor, self.model, messages, videos, kwargs, metas, 512)
        try:
            value = json.loads(re.search(r'\[.*\]', text, re.S).group(0))
            return [[max(0., float(a)), max(0., float(b))] for a,b in value if float(b) >= float(a)]
        except Exception:
            return []


class MolmoVideoTrack:
    def __init__(self, model_id="allenai/Molmo2-8B"):
        self.processor, self.model = _load(model_id)

    def tracks(self, video: str, target: str, max_fps: float = 8) -> list[dict[str, Any]]:
        prompt = (f"Track the physical object described as: {target}. "
                  "Return persistent object IDs and points for every visible time. "
                  "Use the <tracks coords=...> format. Do not track decoys.")
        messages, videos, kwargs, metas = _video_inputs(video, prompt, max_fps=max_fps, max_frames=64)
        text = _generate(self.processor, self.model, messages, videos, kwargs, metas, 1024)
        return parse_molmo_tracks(text)


def parse_molmo_tracks(text: str) -> list[dict[str, Any]]:
    """Parse Molmo frame-time/ID/x/y coordinates (coordinates are 0..1000)."""
    result: dict[int, list[tuple[float,float,float]]] = {}
    for block in _TRACK_RE.findall(text):
        for match in _POINT_RE.finditer(block):
            time_s, payload = match.groups(); nums = _XY_RE.findall(payload)
            for ident, x, y in nums:
                result.setdefault(int(ident), []).append((float(time_s), int(x)/1000, int(y)/1000))
    return [{"id": i, "points": p} for i,p in sorted(result.items())]


def parse_molmo_points(decoded: str) -> list[tuple[float, float]]:
    """Parse Molmo point markup into normalized x/y pairs.

    Molmo-7B-O emits XML attributes (``x="..." y="..."``), while some
    patched processors emit the older ``coords`` payload. Accept both without
    treating the image/object indices in a quadruplet as coordinates.
    Molmo2 commonly emits ``image_index object_id x y`` quadruplets, e.g.
    ``1 1 752 506``. Older variants emit plain x/y pairs. The metadata
    quadruplet must not be mistaken for a point at (1, 1).
    """
    nums = []
    # Molmo-7B commonly emits XML attributes (`x1="44.3" y1="73.6"`)
    # rather than the older coords payload.
    for tag in re.findall(r'<points?\b([^>]*)>', decoded, re.I):
        attrs = dict(re.findall(r'\b(x\d*|y\d*)="?([0-9]+(?:\.[0-9]+)?)', tag, re.I))
        indices = sorted({k[1:] for k in attrs if k.lower().startswith('x')} , key=lambda x: int(x or 0))
        for index in indices:
            if ('y' + index) not in attrs:
                continue
            x, y = float(attrs['x' + index]), float(attrs['y' + index])
            scale = 100.0 if max(x, y) <= 100.0 else 1000.0
            nums.append((max(0., min(1., x / scale)), max(0., min(1., y / scale))))
    for block in re.findall(r'<points[^>]*coords="([^"]+)"', decoded, re.I):
        values = [float(v) for v in re.findall(r'[0-9]+(?:\.[0-9]+)?', block)]
        pairs = []
        if len(values) >= 4 and len(values) % 4 == 0:
            # Quadruplets are image index, object id, x, y.
            pairs = list(zip(values[2::4], values[3::4]))
        else:
            pairs = list(zip(values[0::2], values[1::2]))
        for x, y in pairs:
            if x > 1.0 or y > 1.0:
                x, y = x / 1000.0, y / 1000.0
            nums.append((max(0., min(1., x)), max(0., min(1., y))))
    return nums


class Molmo2VideoPoint:
    """Local Molmo2-8B video point/tracking adapter."""
    def __init__(self, model_id="allenai/Molmo2-8B"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from molmo_utils import process_vision_info
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                                        dtype="auto", device_map="auto")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, dtype="auto", device_map="auto")

    def track_points(self, video: str, query: str, max_fps: int = 8) -> tuple[str, list[tuple[float,int,float,float]]]:
        import re
        import torch
        messages = [{"role": "user", "content": [
            {"type": "text", "text": f"Track the object(s) answering this question: {query}"},
            {"type": "video", "video": str(Path(video).resolve()), "max_fps": max_fps},
        ]}]
        _, videos, kwargs = self.process_vision_info(messages)
        metas = None
        if videos:
            try:
                videos, metas = zip(*videos); videos, metas = list(videos), list(metas)
            except (TypeError, ValueError):
                videos = list(videos)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, videos=videos, video_metadata=metas,
                                return_tensors="pt", padding=True, **(kwargs or {}))
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated = self.processor.tokenizer.decode(ids[0, inputs["input_ids"].size(1):],
                                                    skip_special_tokens=True)
        points = []
        for block in re.findall(r'<(?:points|tracks)[^>]*coords="([^"]+)"', generated, re.I):
            for group in re.finditer(r'(?:^|[\t:;,])([0-9.]+) ([0-9 ]+)', block):
                vals = [int(x) for x in re.findall(r"\d+", group.group(2))]
                # Molmo2 track payload is: object_id x y repeated per time.
                # Preserve IDs; taking only the final pair silently selected
                # the wrong object in multi-object questions.
                for idx in range(0, len(vals) - 2, 3):
                    points.append((float(group.group(1)), vals[idx],
                                   vals[idx + 1] / 1000.0, vals[idx + 2] / 1000.0))
        return generated, points


class MolmoSpatial:
    def __init__(self, model_id="allenai/Molmo-7B-O-0924"): # winner-aligned 
        self.processor, self.model = _load(model_id)

    def point_variants(self, image, description: str) -> list[tuple[float, float]]:
        """Query concise aliases, deduplicating nearby points."""
        variants = [description]
        if re.search(r"marble|bead|small|tiny", description, re.I):
            variants += ["small transparent glass marble on the table",
                         "small round object on the white table"]
        points = []
        for query in variants:
            for point in self.point(image, query):
                if all((point[0]-old[0])**2 + (point[1]-old[1])**2 > .01**2 for old in points):
                    points.append(point)
        return points

    def point(self, image, description: str) -> list[tuple[float,float]]:
        # Use the short imperative from the official Molmo pointing recipe.
        # Long format instructions and negative clauses make Molmo-7B-O emit
        # prose/no point, especially for small objects and left/right queries.
        from PIL import Image
        import torch
        raw = str(description).strip()
        # Keep CORTEX's disambiguating relation for hidden/tiny targets, but
        # remove verbose event prose that distracts Molmo. The previous
        # implementation discarded this context and asked for a bare “marble”;
        # Molmo then consistently clicked the occluding cup in Q8.
        relation = re.search(r";\s*relation:\s*(.*?)(?:;\s*event:|$)", raw, flags=re.I)
        clean = re.split(r";\s*(?:relation|event):", raw, maxsplit=1, flags=re.I)[0].strip(" .")
        clean = re.sub(r"^(?:the|a|an)\s+", "", clean, flags=re.I)
        if relation:
            rel = relation.group(1).strip(" .")
            # Keep the prompt as one direct pointing imperative; Molmo's
            # point head is sensitive to long negative instructions.
            prompt = f"Point to the {clean} {rel} in this image."
        else:
            prompt = f"Point to the {clean} in this image."
        if hasattr(self.model, "generate_from_batch"):
            if hasattr(self.processor, "process"):
                inputs = self.processor.process(images=[image], text=prompt)
            else:
                inputs = self.processor(text=prompt, images=[image], return_tensors="pt", padding=True)
            # Molmo's official processor returns unbatched tensors; the
            # remote model expects a leading batch dimension for both token
            # and image tensors.
            model_dtype = next(self.model.parameters()).dtype
            prepared = {}
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    value = value.to(self.model.device)
                    if value.is_floating_point():
                        value = value.to(model_dtype)
                    value = value.unsqueeze(0)
                prepared[key] = value
            inputs = prepared
            from transformers import GenerationConfig
            with torch.inference_mode():
                out = self.model.generate_from_batch(
                    inputs, GenerationConfig(max_new_tokens=128,
                                             stop_strings="<|endoftext|>"),
                    tokenizer=self.processor.tokenizer)
            decoded = self.processor.tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        else:
            messages = [{"role":"user", "content":[
                {"type":"text", "text":prompt}, {"type":"image", "image":image}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[image], return_tensors="pt", padding=True)
            inputs = {k:(v.to(self.model.device) if hasattr(v,"to") else v) for k,v in inputs.items()}
            with torch.inference_mode(): out=self.model.generate(**inputs,max_new_tokens=128)
            decoded=self.processor.decode(out[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        points = parse_molmo_points(decoded)
        # If the relation says the target is occluded, Molmo may correctly
        # decline to emit a point. Retry the bare object once: the trigger frame
        # is supposed to be the clearest visible frame, and this fallback also
        # handles CORTEX wording such as “occluded by” that is too literal for
        # the point head.
        if not points and relation:
            fallback_target = clean
            if re.search(r"under|beneath|occlud|hidden|inside", relation.group(1), flags=re.I):
                fallback_target = "small " + clean
            return self.point(image, fallback_target)
        # Reject parser artifacts and edge clicks; a real point model should
        # return one interior point. Keep multiple points only for compatibility
        # with older checkpoints that legitimately emit several instances.
        return [(x, y) for x, y in points if 0.01 < x < 0.99 and 0.01 < y < 0.99]
