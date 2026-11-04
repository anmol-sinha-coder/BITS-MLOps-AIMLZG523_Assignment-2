"""FastAPI inference API for the EduAssist assistant.

Exposes the v3 (production) prompt as a JSON-returning endpoint, plus health,
model-info and Prometheus metrics. Mirrors the Assignment-1 serving pattern
(non-root user, HEALTHCHECK, /metrics for Prometheus).
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src import config
from src.metrics import (
    llm_cost_per_request, llm_input_tokens, llm_latency_seconds,
    llm_model_info, llm_output_tokens, llm_requests_total, llm_tokens_per_second,
    llm_valid_json, CostBreakdown,
)
from src.ollama_client import detect_device, ollama_ps, query, warmup_model
from src.prompts import PROMPTS, render_prompt
from src.utils import parse_json_response, validate_json_response

app = FastAPI(title="EduAssist AI LLMOps API", version="1.0.0")

_MODEL = os.getenv("OLLAMA_MODEL", config.DEFAULT_MODEL)
_DEVICE = detect_device()
_PROMPT = PROMPTS["v3"]
warmed = False


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    confidence: str
    key_points: list[str]
    latency_sec: float
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    model: str
    valid_json: bool


@app.on_event("startup")
def _startup() -> None:
    """Pre-warm the model so the first user request is not a cold start."""
    global warmed
    try:
        warmup_model(_MODEL)
        warmed = True
        llm_model_info.labels(model=_MODEL, device=_DEVICE).set(1)
    except Exception as exc:  # pragma: no cover - depends on Ollama running
        print(f"[!] startup warm-up failed: {exc}")
        llm_model_info.labels(model=_MODEL, device=_DEVICE).set(0)


@app.get("/health")
def health():
    return {"status": "ok", "model": _MODEL, "device": _DEVICE, "warmed": warmed}


@app.get("/model-info")
def model_info():
    return {
        "model": _MODEL, "device": _DEVICE,
        "gpu_residency": ollama_ps(),
        "ollama_host": config.OLLAMA_HOST,
        "prompt_version": "v3",
        "pricing": {
            "input_per_m": config.INPUT_PRICE_PER_M,
            "output_per_m": config.OUTPUT_PRICE_PER_M,
        },
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    prompt_text = render_prompt(_PROMPT["template"], req.question)
    try:
        res = query(_MODEL, prompt_text, json_mode=True)
    except Exception as exc:
        llm_requests_total.labels(prompt_version="v3", status="error").inc()
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    cb = CostBreakdown.from_tokens(res.input_tokens, res.output_tokens)
    valid = validate_json_response(res.content)
    parsed = parse_json_response(res.content)

    llm_requests_total.labels(prompt_version="v3", status="success").inc()
    llm_latency_seconds.labels(prompt_version="v3").observe(res.latency)
    llm_input_tokens.labels(prompt_version="v3").observe(res.input_tokens)
    llm_output_tokens.labels(prompt_version="v3").observe(res.output_tokens)
    llm_tokens_per_second.set(res.tokens_per_second)
    llm_cost_per_request.set(cb.total_cost)
    llm_valid_json.set(1.0 if valid else 0.0)

    return AskResponse(
        answer=parsed["answer"] if parsed else res.content,
        confidence=parsed["confidence"] if parsed else "low",
        key_points=parsed["key_points"] if parsed else [],
        latency_sec=round(res.latency, 4),
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
        total_cost_usd=round(cb.total_cost, 8),
        model=_MODEL,
        valid_json=valid,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
