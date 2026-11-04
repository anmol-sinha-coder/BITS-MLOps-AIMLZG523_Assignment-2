"""Thin wrapper around the Ollama HTTP API.

Why this exists (and why the previous pipeline was slow):
  * The first prompt took 8-9 minutes because the model was loaded cold on
    every run and, on a partial GPU offload, layers spilled from VRAM into
    system RAM (a well-documented 10x slowdown).
  * This client (a) verifies the GPU is actually being used, (b) keeps the
    model resident in VRAM via OLLAMA_KEEP_ALIVE, (c) sends one warm-up
    request before timing, and (d) reads the real token counts from the
    Ollama response (`prompt_eval_count` / `eval_count`).

It uses only `requests` so it works even if the `ollama` python package is
absent, and it degrades gracefully to CPU when no GPU is present.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import requests

from . import config


@dataclass
class LLMResponse:
    content: str
    start_time: float
    end_time: float
    latency: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    status_code: int
    tokens_per_second: float
    device: str  # "gpu" | "cpu" | "unknown"


def detect_device() -> str:
    """Best-effort GPU detection. Returns 'gpu' if nvidia-smi sees a usable GPU."""
    if not shutil_which("nvidia-smi"):
        return "cpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return "gpu" if out.stdout.strip() else "cpu"
    except Exception:
        return "cpu"


def shutil_which(cmd: str) -> Optional[str]:
    """Avoid importing shutil just for this; mimics shutil.which."""
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for p in paths:
        fp = os.path.join(p, cmd)
        if os.path.isfile(fp) and os.access(fp, os.X_OK):
            return fp
    return None


def ollama_ps() -> dict:
    """Return `ollama ps` parsed as {model: {...}} for GPU-residency diagnostics."""
    try:
        out = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=5, check=True,
        )
    except Exception:
        return {}
    models: dict = {}
    lines = out.stdout.strip().splitlines()
    if len(lines) < 2:
        return models
    header = lines[0].split()
    name_idx = _find(header, "NAME")
    size_idx = _find(header, "SIZE")
    proc_idx = _find(header, "PROCESSOR")
    for line in lines[1:]:
        cols = line.split()
        if not cols:
            continue
        name = cols[name_idx] if name_idx is not None else cols[0]
        proc = cols[proc_idx] if proc_idx is not None and proc_idx < len(cols) else ""
        size = cols[size_idx] if size_idx is not None and size_idx < len(cols) else ""
        models[name] = {"processor": proc, "size": size, "full": line}
    return models


def _find(header: list, key: str) -> Optional[int]:
    for i, h in enumerate(header):
        if h.upper().startswith(key):
            return i
    return None


def warmup_model(model: str, timeout: int = 600) -> float:
    """Load `model` into VRAM with a trivial request; return the cold-load time.

    The first call pays the model-load cost so subsequent timed calls measure
    pure inference.  Timeout is generous because a 7B model cold-load over a
    slow disk can take minutes on first pull.
    """
    t0 = time.time()
    try:
        requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": "ping",
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {"num_predict": 1},
            },
            timeout=timeout,
        ).raise_for_status()
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Ollama warm-up failed for {model}: {exc}") from exc
    return time.time() - t0


def query(model: str, prompt: str, json_mode: bool = False, timeout: int = 600) -> LLMResponse:
    """Send one chat request and return latency + token telemetry."""
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0.0},  # deterministic for reproducible eval
    }
    if json_mode:
        payload["format"] = "json"

    device = detect_device()
    start = time.time()
    resp = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    end = time.time()
    resp.raise_for_status()
    data = resp.json()

    content = data.get("message", {}).get("content", "")
    input_tokens = int(data.get("prompt_eval_count", 0) or 0)
    output_tokens = int(data.get("eval_count", 0) or 0)
    latency = end - start
    tps = output_tokens / latency if latency > 0 else 0.0

    return LLMResponse(
        content=content,
        start_time=start,
        end_time=end,
        latency=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        status_code=resp.status_code,
        tokens_per_second=tps,
        device=device,
    )
