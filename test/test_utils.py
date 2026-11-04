"""Unit tests for JSON-schema validation (Task 3 valid_json metric)."""
import pytest

from src.utils import validate_json_response, parse_json_response, extract_json_object


VALID = (
    '{"answer": "Backprop updates weights using gradients.", '
    '"confidence": "high", "key_points": ["chain rule"]}'
)
WRONG_CONFIDENCE = '{"answer": "x", "confidence": "huge", "key_points": ["a"]}'
MISSING_KEY = '{"answer": "x", "confidence": "high"}'
NOT_JSON = "Backpropagation is a method to train neural networks."
FENCED = '```json\n{"answer": "x", "confidence": "low", "key_points": ["a"]}\n```'
EXTRA_PROSE = 'Sure! Here is the answer: {"answer": "x", "confidence": "low", "key_points": ["a"]}'


@pytest.mark.parametrize("payload", [VALID, FENCED, EXTRA_PROSE])
def test_valid_json_accepted(payload):
    assert validate_json_response(payload) is True


@pytest.mark.parametrize("payload", [WRONG_CONFIDENCE, MISSING_KEY, NOT_JSON, ""])
def test_invalid_json_rejected(payload):
    assert validate_json_response(payload) is False


def test_parse_returns_dict():
    parsed = parse_json_response(VALID)
    assert parsed is not None
    assert parsed["confidence"] == "high"
    assert parsed["key_points"] == ["chain rule"]


def test_parse_returns_none_on_garbage():
    assert parse_json_response("not json at all") is None


def test_extract_first_object():
    obj = extract_json_object('noise {"a": 1} trailing')
    assert obj == '{"a": 1}'
