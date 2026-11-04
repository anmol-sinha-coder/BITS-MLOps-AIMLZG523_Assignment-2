"""Central configuration for the EduAssist AI LLMOps pipeline.

All tunable values (model, pricing, prompts, MLflow endpoint) live here so the
deployment script, the notebook, the FastAPI server and the benchmark all read
from one source of truth.  Secrets are never hard-coded: the Ollama host and any
future paid-API key are read from environment variables.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Ollama inference server
# ---------------------------------------------------------------------------
# Default model chosen for the widest compatibility (8 GB VRAM + CPU fallback):
#   qwen2.5:7b  -> MMLU 74.2, 4.4 GB Q4_K_M, strong JSON following.
# Override with `--model` on initial_deploy.sh or OLLAMA_MODEL env var.
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Candidate models benchmarked by `model_benchmark.py` to pick the highest
# composite eval score on the user's hardware.  Kept free / open-weight and
# sized so a "fair VRAM" RTX card (8-12 GB) can run every one of them.
MODEL_CANDIDATES = [
    {"tag": "gemma3:4b", "params_b": 4, "vram_gb": 3.0, "note": "fast fallback, CPU friendly"},
    {"tag": "qwen2.5:7b", "params_b": 7, "vram_gb": 4.4, "note": "balanced, MMLU 74.2"},
    {"tag": "qwen3:8b", "params_b": 8, "vram_gb": 5.0, "note": "newest Qwen, hybrid thinking"},
    {"tag": "llama3.1:8b", "params_b": 8, "vram_gb": 4.9, "note": "strong general baseline"},
]

# Ollama daemon. `ollama serve` listens here by default.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Keep the model warm in VRAM between requests (seconds). Default Ollama value
# is 5m; we raise it for a demo run so the first-prompt cold start does not
# repeat.  Set 0 to unload immediately (memory-constrained hosts).
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# ---------------------------------------------------------------------------
# MLflow tracking
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "eduassist_llmops_assignment"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# ---------------------------------------------------------------------------
# Cost economics (Task 2)
# ---------------------------------------------------------------------------
# The assignment permits a locally hosted open-source model when paid API
# access is unavailable.  We therefore use the *published hosted-equivalent*
# pricing as a benchmark reference (rates per 1,000,000 tokens) so the cost
# figures are comparable to a real cloud LLM bill.  Documented in the report.
# Reference: OpenAI GPT-4o-mini published pricing (input $0.15 / output $0.60
# per 1M).  Swap to your provider's rates in env vars if needed.
INPUT_PRICE_PER_M = float(os.getenv("INPUT_PRICE_PER_M", "0.15"))
OUTPUT_PRICE_PER_M = float(os.getenv("OUTPUT_PRICE_PER_M", "0.60"))
MONTHLY_REQUEST_PROJECTION = 10_000

# ---------------------------------------------------------------------------
# Task dataset: five academic questions (fixed across all prompt versions)
# ---------------------------------------------------------------------------
ACADEMIC_QUESTIONS = [
    {"id": "Q1", "question": "Explain the concept of backpropagation in neural networks."},
    {"id": "Q2", "question": "What is the difference between supervised and unsupervised learning?"},  # noqa: E501
    {"id": "Q3", "question": "Describe how the Self-Attention mechanism works in Transformers."},
    {"id": "Q4", "question": "What is overfitting in machine learning and how can it be mitigated?"},  # noqa: E501
    {"id": "Q5", "question": "Explain the trade-off between Bias and Variance in predictive models."},  # noqa: E501
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LATENCY_CSV = os.path.join(OUTPUT_DIR, "latency_tokens_cost.csv")
EVAL_CSV = os.path.join(OUTPUT_DIR, "prompt_evaluation_results.csv")
BENCHMARK_CSV = os.path.join(OUTPUT_DIR, "model_benchmark_results.csv")
SELECTED_MODEL_FILE = os.path.join(PROJECT_ROOT, "config", "selected_model.json")
