"""Unit tests for central configuration invariants."""
from src import config


def test_five_academic_questions():
    assert len(config.ACADEMIC_QUESTIONS) == 5
    for q in config.ACADEMIC_QUESTIONS:
        assert q["id"].startswith("Q")
        assert q["question"]


def test_default_model_is_qwen():
    # default tag shape: family:size
    assert ":" in config.DEFAULT_MODEL


def test_candidate_models_free_and_sized():
    assert len(config.MODEL_CANDIDATES) >= 3
    for c in config.MODEL_CANDIDATES:
        assert c["tag"]
        assert c["vram_gb"] <= 16  # fits a fair-VRAM RTX card


def test_pricing_positive():
    assert config.INPUT_PRICE_PER_M > 0
    assert config.OUTPUT_PRICE_PER_M > 0


def test_experiment_name_matches_assignment():
    assert config.EXPERIMENT_NAME == "eduassist_llmops_assignment"
