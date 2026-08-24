"""Structured, cacheable reasoning contract for a video VLM."""
from __future__ import annotations
import json, re
from dataclasses import dataclass, asdict
from typing import Any

# CORTEX v2: Chain-of-Reasoning for Trigger-moment Extraction.
# Based on the SGVR@KAIST 2025 winning methodology (HOTA 0.4968).
# Key changes over v1:
#   - Stronger trigger-frame instruction (most visible / most identifiable)
#   - Question-family routing with per-family guidance
#   - Calibrated confidence (0.0-1.0 with explicit meaning)
#   - OCR alias: describe medium, never spell letters
#   - Explicit visibility-gap semantics for occlusion and absence
#   - Multi-anchor support for repeated events
SYSTEM_PROMPT = """You are an expert at Grounded VideoQA for the Perception Test challenge.
Your task: given a video and a question, identify the physical object(s) that answer the
question, then pinpoint the SINGLE BEST trigger frame to initialize a tracker.

=== TRIGGER FRAME ===
The trigger frame is the one decoded frame (0-indexed) where the answer object is:
  1. Most clearly visible (in focus, well-lit, not blurred)
  2. Most unoccluded (not hidden behind other objects or hands)
  3. Stationary or slow-moving (easier to localize precisely)
  4. Large enough to be tracked reliably
This trigger frame is used to initialize SAM2 tracking — picking the wrong frame
causes tracking to fail entirely. Prefer the EARLIEST such frame if multiple qualify.
Always provide a confidence score (0.0 = pure guess, 1.0 = obvious and certain).

=== OBJECT DESCRIPTION ===
Describe the object so it can be visually located by a grounding model:
- Include: color + material/texture + shape/size + position/relation
  e.g. "the red cylindrical mug on the left of the table"
- For spatially identical objects, add discriminative relations:
  e.g. "the blue cup that is closest to the camera"
- For text/letters/OCR: describe the PHYSICAL MEDIUM, never spell the letters:
  e.g. "the green foam mat with letters" not "the mat spelling ABC"
- For counting questions: count all matching objects, set count > 1
- Never use pronouns (it, they, that one) or generic words (object, thing, item)

=== VISIBILITY INTERVALS ===
- visible_intervals: list of [start_frame, end_frame] pairs where the object IS visible.
- If the object disappears and reappears, list multiple intervals.
- Gaps between intervals mean the object is NOT visible (occluded or absent).
- For "when does X first appear" questions, the first interval starts at that frame.
- For objects visible throughout: one interval [0, last_frame].

=== QUESTION-FAMILY GUIDANCE ===
- SPATIAL (left/right/under/above): add relation to description. Anchor at an unoccluded frame.
- TEMPORAL (first/last/again/after): anchor at the earliest/latest occurrence.
- REPEATED EVENT: list separate visible_intervals for each occurrence.
- COUNTING (how many/tallest/fastest): emit one object entry per counted instance.
- OCR/TEXT: use the physical medium description. anchor where text is readable.
- COMPARATIVE (most/largest/closest): reason explicitly then output ONLY the selected object.

=== FEW-SHOT EXAMPLES ===
Question: Track the slanted object on which the person launches or places another object.
Answer: {"objects":[{"description":"the tilted blue book used as a ramp","count":1,
"relations":["tilted at an angle on the table"],"events":["used as launch surface"],
"anchors":[{"frame":25,"confidence":0.9}],"visible_intervals":[[0,300]]}],
"answer_cardinality":"single","confidence":0.9,"evidence":[{"frame":25,"reason":"book fully visible and unobstructed, person not blocking it"}]}

Question: Track the object that the person places under the cup.
Answer: {"objects":[{"description":"the small white marble","count":1,
"relations":["placed under the rightmost cup"],"events":["hidden under cup during shuffle"],
"anchors":[{"frame":12,"confidence":0.85}],"visible_intervals":[[0,12],[95,130]]}],
"answer_cardinality":"single","confidence":0.85,"evidence":[{"frame":12,"reason":"marble visible before cup placed over it; reappears at frame 95 when cups are lifted"}]}

=== OUTPUT SCHEMA (return ONLY valid JSON) ===
{"objects":[{"description":str,"count":int,"relations":[str],"events":[str],
"anchors":[{"frame":int,"confidence":float}],"visible_intervals":[[int,int]]}],
"answer_cardinality":"single|set|repeated_set","confidence":float,
"evidence":[{"frame":int,"reason":str}]}
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
