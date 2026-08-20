"""Small local validator for grounded VQA prediction JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def validate(path: str) -> int:
    data = json.loads(Path(path).read_text())
    errors = []
    if not isinstance(data, dict):
        errors.append("top-level value must be an object")
    for video_id, video in data.items() if isinstance(data, dict) else []:
        questions = video.get("grounded_question") if isinstance(video, dict) else None
        if not isinstance(questions, dict):
            errors.append(f"{video_id}: missing grounded_question object")
            continue
        for qid, tracks in questions.items():
            if not isinstance(tracks, list):
                errors.append(f"{video_id}/{qid}: tracks must be a list")
                continue
            seen = set()
            for i, track in enumerate(tracks):
                prefix = f"{video_id}/{qid}/track[{i}]"
                if not isinstance(track, dict):
                    errors.append(f"{prefix}: must be an object"); continue
                tid = track.get("id")
                key = json.dumps(tid, sort_keys=True)
                if "id" not in track: errors.append(f"{prefix}: missing id")
                if key in seen: errors.append(f"{prefix}: duplicate id {tid}")
                seen.add(key)
                score = track.get("score")
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    errors.append(f"{prefix}: invalid score")
                boxes, frames = track.get("bounding_boxes"), track.get("frame_ids")
                if not isinstance(boxes, list) or not isinstance(frames, list):
                    errors.append(f"{prefix}: missing bounding_boxes/frame_ids"); continue
                if len(boxes) != len(frames): errors.append(f"{prefix}: length mismatch")
                if any(not isinstance(f, int) or f < 0 for f in frames):
                    errors.append(f"{prefix}: invalid frame id")
                for box in boxes:
                    if (not isinstance(box, list) or len(box) != 4 or
                        any(not isinstance(x, (int, float)) or not 0 <= x <= 1 for x in box) or
                        box[0] > box[2] or box[1] > box[3]):
                        errors.append(f"{prefix}: invalid normalized box {box}")
    if errors:
        print("INVALID")
        print("\n".join(errors[:50]))
        return 1
    print(f"OK: {path} ({len(data)} videos)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python validate.py predictions.json")
    raise SystemExit(validate(sys.argv[1]))
