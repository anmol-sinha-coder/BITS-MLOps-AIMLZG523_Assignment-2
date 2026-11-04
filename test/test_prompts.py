"""Unit tests for prompt versioning and rendering."""
from src.prompts import PROMPTS, render_prompt, is_json_version


def test_three_versions_present():
    assert set(PROMPTS.keys()) == {"v1", "v2", "v3"}


def test_each_has_commit_message():
    for info in PROMPTS.values():
        assert info["commit"]
        assert info["template"]
        assert "{{question}}" in info["template"]


def test_render_substitutes_question():
    out = render_prompt(PROMPTS["v1"]["template"], "What is X?")
    assert "What is X?" in out
    assert "{{question}}" not in out


def test_only_v3_is_json():
    assert is_json_version("v3") is True
    assert is_json_version("v1") is False
    assert is_json_version("v2") is False


def test_v3_has_required_schema_keys():
    tpl = PROMPTS["v3"]["template"]
    assert '"answer"' in tpl
    assert '"confidence"' in tpl
    assert '"key_points"' in tpl
