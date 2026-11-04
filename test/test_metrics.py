"""Unit tests for cost economics (Task 2) and latency statistics (Task 1)."""
import math

from src import config
from src.metrics import CostBreakdown, LatencyStats, projected_monthly_cost


def test_cost_breakdown_formula():
    cb = CostBreakdown.from_tokens(input_tokens=1_000_000, output_tokens=1_000_000)
    assert math.isclose(cb.input_cost, config.INPUT_PRICE_PER_M)
    assert math.isclose(cb.output_cost, config.OUTPUT_PRICE_PER_M)
    assert math.isclose(cb.total_cost, config.INPUT_PRICE_PER_M + config.OUTPUT_PRICE_PER_M)


def test_cost_zero_for_zero_tokens():
    cb = CostBreakdown.from_tokens(0, 0)
    assert cb.total_cost == 0.0


def test_monthly_projection_uses_10k():
    assert config.MONTHLY_REQUEST_PROJECTION == 10_000
    assert math.isclose(projected_monthly_cost(0.001), 10.0)


def test_latency_stats_basic():
    s = LatencyStats.from_latencies([2.0, 4.0, 6.0, 8.0])
    assert s.count == 4
    assert math.isclose(s.mean, 5.0)
    assert math.isclose(s.median, 5.0)
    assert s.minimum == 2.0
    assert s.maximum == 8.0
    assert s.std >= 0


def test_latency_stats_empty():
    s = LatencyStats.from_latencies([])
    assert s.count == 0
    assert s.mean == 0
