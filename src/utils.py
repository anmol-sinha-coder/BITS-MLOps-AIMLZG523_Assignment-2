"""Validation and output helpers shared by the pipeline, benchmark and tests."""
from __future__ import annotations

import json
import re
from typing import Any

REQUIRED_KEYS = {"answer", "confidence", "key_points"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def _strip_code_fences(text: str) -> str:
    """Remove accidental ```json ... ``` wrappers some models emit."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def extract_json_object(text: str) -> str | None:
    """Return the first balanced {...} block found in `text`, or None."""
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return None


def validate_json_response(response_text: str) -> bool:
    """True iff the response parses to the required schema (answer/confidence/key_points)."""
    payload = extract_json_object(response_text)
    if payload is None:
        return False
    try:
        data: Any = json.loads(payload)
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    if not REQUIRED_KEYS.issubset(data.keys()):
        return False
    if data.get("confidence") not in ALLOWED_CONFIDENCE:
        return False
    if not isinstance(data.get("key_points"), list):
        return False
    if not isinstance(data.get("answer"), str) or not data["answer"].strip():
        return False
    return True


def parse_json_response(response_text: str) -> dict | None:
    """Return the parsed dict if valid, else None (used by the API and tests)."""
    payload = extract_json_object(response_text)
    if payload is None:
        return None
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return None
    return data if validate_json_response(response_text) else None
