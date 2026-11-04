"""Latency statistics, token-cost economics and Prometheus instrumentation.

All arithmetic lives here so it can be unit-tested without an LLM and so the
notebook, pipeline and API share one implementation.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from prometheus_client import Counter, Histogram, Gauge

from . import config


# ---------------------------------------------------------------------------
# Latency statistics (Task 1)
# ---------------------------------------------------------------------------
@dataclass
class LatencyStats:
    mean: float
    median: float
    minimum: float
    maximum: float
    std: float
    count: int

    @classmethod
    def from_latencies(cls, latencies: Sequence[float]) -> "LatencyStats":
        lat = list(latencies)
        if not lat:
            return cls(0, 0, 0, 0, 0, 0)
        return cls(
            mean=float(statistics.mean(lat)),
            median=float(statistics.median(lat)),
            minimum=float(min(lat)),
            maximum=float(max(lat)),
            std=float(statistics.pstdev(lat)) if len(lat) > 1 else 0.0,
            count=len(lat),
        )


# ---------------------------------------------------------------------------
# Token cost economics (Task 2)
# ---------------------------------------------------------------------------
@dataclass
class CostBreakdown:
    input_cost: float
    output_cost: float
    total_cost: float

    @classmethod
    def from_tokens(cls, input_tokens: int, output_tokens: int) -> "CostBreakdown":
        ic = (input_tokens / 1_000_000) * config.INPUT_PRICE_PER_M
        oc = (output_tokens / 1_000_000) * config.OUTPUT_PRICE_PER_M
        return cls(input_cost=ic, output_cost=oc, total_cost=ic + oc)


def projected_monthly_cost(avg_cost_per_request: float) -> float:
    """Projected monthly bill at the assignment's 10,000-request volume."""
    return avg_cost_per_request * config.MONTHLY_REQUEST_PROJECTION


# ---------------------------------------------------------------------------
# Prometheus metrics (served by the FastAPI app on /metrics)
# ---------------------------------------------------------------------------
PROM_NAMESPACE = "eduassist"

llm_requests_total = Counter(
    f"{PROM_NAMESPACE}_requests_total", "Total LLM requests", ["prompt_version", "status"]
)
llm_latency_seconds = Histogram(
    f"{PROM_NAMESPACE}_llm_latency_seconds", "End-to-end LLM latency",
    ["prompt_version"], buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)
llm_input_tokens = Histogram(
    f"{PROM_NAMESPACE}_input_tokens", "Prompt tokens per request",
    ["prompt_version"], buckets=(50, 100, 200, 500, 1000),
)
llm_output_tokens = Histogram(
    f"{PROM_NAMESPACE}_output_tokens", "Completion tokens per request",
    ["prompt_version"], buckets=(20, 50, 100, 200, 500),
)
llm_tokens_per_second = Gauge(
    f"{PROM_NAMESPACE}_tokens_per_second", "Generation throughput (tok/s)"
)
llm_valid_json = Gauge(
    f"{PROM_NAMESPACE}_valid_json_rate", "Share of responses that are valid JSON"
)
llm_cost_per_request = Gauge(
    f"{PROM_NAMESPACE}_cost_per_request_usd", "Average cost per request (USD)"
)
llm_model_info = Gauge(
    f"{PROM_NAMESPACE}_model_info", "1 when model is loaded", ["model", "device"]
)
