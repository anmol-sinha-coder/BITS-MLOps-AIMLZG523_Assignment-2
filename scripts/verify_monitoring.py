"""Verify the monitoring stack has data for every dashboard panel."""
from __future__ import annotations

import sys

import requests

PROM = "http://localhost:9090"
QUERIES = {
    "requests_rate": 'sum(rate(eduassist_requests_total[1m]))',
    "p95_latency": 'histogram_quantile(0.95, sum(rate(eduassist_llm_latency_seconds_bucket[5m])) by (le))',  # noqa: E501
    "throughput": 'eduassist_tokens_per_second',
    "valid_json": 'eduassist_valid_json_rate',
    "cost": 'eduassist_cost_per_request_usd',
}


def main() -> int:
    missing = []
    for name, q in QUERIES.items():
        try:
            r = requests.get(f"{PROM}/api/v1/query", params={"query": q}, timeout=10)
            r.raise_for_status()
            res = r.json().get("data", {}).get("result", [])
            val = res[0]["value"][1] if res else None
            print(f"  {name:16} = {val}")
            if not res:
                missing.append(name)
        except Exception as exc:
            print(f"  {name:16} ERROR {exc}")
            missing.append(name)
    if missing:
        print(f"\n[!] panels with no data: {missing}")
        print("    wait ~45s for Prometheus to scrape, then re-run.")
        return 1
    print("\n[+] all monitoring panels have data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
