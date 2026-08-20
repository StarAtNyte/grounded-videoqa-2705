"""Structured, cacheable reasoning contract for a video VLM."""
from __future__ import annotations
import json, re
from dataclasses import dataclass, asdict
from typing import Any

SYSTEM_PROMPT = """You are an AI assistant specializing in Grounded VideoQA.
Return ONLY valid JSON. Identify the physical object(s) that constitute the
answer and select one trigger moment: a single 0-indexed decoded frame where
the answer is most clearly and fully visible (in focus, well-lit, stationary,
and unobstructed). Reason first about why the object satisfies the question,
then why that frame is the best grounding anchor.

Answer rules:
- Use a discriminative physical description with color, material, shape,
  state, or position: e.g. 'the tilted blue book', 'the magnetic plastic
  letters', 'the blue cup on the right'.
- Resolve identical objects with spatial/relational descriptions.
- For text or letters, describe the collective physical medium, not spelling.
- Do not answer with pronouns or generic words ('one', 'object', 'things').
- Do not invent boxes or coordinates. Use decoded frame indices, zero-based.
- Output one object entry per independently tracked object; count identical
  objects when the answer is a set.

Few-shot examples:
Question: Track the slanted object on which the person launches or places another object.
Answer: 'the tilted blue book'
Reasoning: The person uses the blue book as a slanted launch surface. Frame 25 is best because the whole book is unobstructed and clearly visible.
Trigger_moment: 25
Question: Track the letters that the person interacts with.
Answer: 'the magnetic plastic letter blocks'
Reasoning: These are the physical letters handled by the person. Frame 48 is best because all blocks are stationary and visible together.
Trigger_moment: 48
Question: A person uses multiple identical objects to play an occlusion game. Track the object that occludes the smaller object.
Answer: 'the blue cup on the right'
Reasoning: The right cup is the one hiding the smaller object. Frame 16 is best after shuffling stops and the cup is unobstructed.
Trigger_moment: 16

Schema:
{"objects":[{"description":str,"count":int,"relations":[str],"events":[str],
"anchors":[{"frame":int,"confidence":number}],"visible_intervals":[[int,int]]}],
"answer_cardinality":"single|set|repeated_set","confidence":number,
"evidence":[{"frame":int,"reason":str}]}
Question: {question_text}
"""

@dataclass
class Anchor:
    frame: int
    confidence: float

@dataclass
class ObjectHypothesis:
    description: str
    count: int = 1
    relations: list[str] | None = None
    events: list[str] | None = None
    anchors: list[Anchor] | None = None
    visible_intervals: list[list[int]] | None = None

@dataclass
class Reasoning:
    objects: list[ObjectHypothesis]
    answer_cardinality: str = "single"
    confidence: float = 0.0
    evidence: list[dict[str, Any]] | None = None


def prompt(question: str, num_frames: int, fps: float, family: str | None = None) -> str:
    family_hint = f"Question family: {family}." if family else ""
    return (SYSTEM_PROMPT + "\n" + family_hint + f"\nQuestion: {question}\n"
            f"Video has {num_frames} decoded frames at measured {fps:.4f} FPS.")


def _json_blob(text: str) -> dict:
    text = text.strip().replace("```json", "").replace("```", "")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            value = json.loads(match.group(0))
        else:
            match = re.search(r"\[.*\]", text, re.S)
            if not match: raise ValueError("VLM did not return JSON")
            value = json.loads(match.group(0))
    # Flash models occasionally return the object list directly despite the
    # structured contract. Preserve it rather than discarding the response.
    if isinstance(value, list):
        return {"objects": value}
    if not isinstance(value, dict):
        raise ValueError("VLM JSON must be an object or object list")
    return value


def parse_reasoning(text: str, num_frames: int) -> Reasoning:
    raw = _json_blob(text)
    objs = []
    for item in raw.get("objects", []):
        anchors = []
        raw_anchors = item.get("anchors", [])
        if not raw_anchors and item.get("trigger_moment") is not None:
            raw_anchors = [{"frame": item.get("trigger_moment"), "confidence": .8}]
        for a in raw_anchors:
            try:
                if isinstance(a, (int, float, str)):
                    a = {"frame": a, "confidence": .8}
                f = max(0, min(num_frames - 1, int(a.get("frame", 0))))
                anchors.append(Anchor(f, float(a.get("confidence", .5))))
            except (TypeError, ValueError): pass
        if not anchors: anchors = [Anchor(num_frames // 2, .1)]
        objs.append(ObjectHypothesis(
            description=str(item.get("description", "object")),
            count=max(1, int(item.get("count", 1))),
            relations=[str(x) for x in item.get("relations", [])],
            events=[str(x) for x in item.get("events", [])],
            anchors=anchors,
            visible_intervals=item.get("visible_intervals", [])))
    if not objs: raise ValueError("VLM returned no objects")
    return Reasoning(objs, str(raw.get("answer_cardinality", "single")),
                     float(raw.get("confidence", .0)), raw.get("evidence", []))
