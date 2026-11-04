"""Generate a short background load against the LLMOps API.

Used by initial_deploy.sh to populate Prometheus/Grafana panels.  Sends a mix
of the five academic questions at a gentle rate so the model stays warm.

    python scripts/generate_traffic.py --url http://localhost:8000 --duration 120
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import requests

QUESTIONS = [
    "Explain the concept of backpropagation in neural networks.",
    "What is the difference between supervised and unsupervised learning?",
    "Describe how the Self-Attention mechanism works in Transformers.",
    "What is overfitting in machine learning and how can it be mitigated?",
    "Explain the trade-off between Bias and Variance in predictive models.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--duration", type=int, default=120, help="seconds")
    ap.add_argument("--rps", type=float, default=0.2, help="requests per second")
    args = ap.parse_args()

    end = time.time() + args.duration
    sent = ok = 0
    interval = 1.0 / max(args.rps, 0.01)
    while time.time() < end:
        q = random.choice(QUESTIONS)
        try:
            r = requests.post(f"{args.url}/ask", json={"question": q}, timeout=300)
            sent += 1
            if r.status_code == 200:
                ok += 1
            print(f"[{sent}] {r.status_code}  q={q[:40]!r}", flush=True)
        except Exception as exc:
            sent += 1
            print(f"[{sent}] ERR {exc}", flush=True)
        time.sleep(interval)
    print(f"\nDone: {ok}/{sent} succeeded over {args.duration}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
