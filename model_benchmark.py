"""Model selection: benchmark free Ollama candidates and pick the best.

The assignment lets us use a locally hosted open-source model.  "Highest
evaluation metric score amongst free models deployed" is operationalised here:
for each candidate model we measure

    * valid_json_rate  - schema conformance on the v3 JSON prompt (quality)
    * tokens_per_second - throughput (speed; the inverse of the 8-9 min problem)
    * avg_latency       - end-to-end latency
    * answer_length     - proxy for completeness

composite_score = 0.5*valid_json + 0.3*norm(tps) + 0.2*norm(-latency)

The winner is written to config/selected_model.json and the pipeline picks it
up automatically.  Run via `initial_deploy.sh --benchmark` or directly:

    python model_benchmark.py
    python model_benchmark.py --candidates gemma3:4b qwen2.5:7b
"""
from __future__ import annotations

import json
import os
import sys

import mlflow
import numpy as np
import pandas as pd

from src import config
from src.ollama_client import query, warmup_model, detect_device
from src.prompts import PROMPTS, render_prompt
from src.utils import validate_json_response

BENCH_QUESTIONS = config.ACADEMIC_QUESTIONS[:3]  # subset for speed
BENCHMARK_EXPERIMENT = "eduassist_model_selection"


def _candidates_from_args() -> list[str]:
    if "--candidates" in sys.argv:
        i = sys.argv.index("--candidates")
        return sys.argv[i + 1 :]
    return [c["tag"] for c in config.MODEL_CANDIDATES]


def benchmark() -> dict:
    candidates = _candidates_from_args()
    device = detect_device()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(BENCHMARK_EXPERIMENT)

    v3 = PROMPTS["v3"]
    rows: list[dict] = []

    print(f"\n[+] Benchmarking {len(candidates)} candidate models on {device.upper()}")
    print(f"    Candidates: {candidates}\n")

    for tag in candidates:
        print(f"--- {tag} ---")
        try:
            cold = warmup_model(tag)
            print(f"    cold-load: {cold:.1f}s")
        except Exception as exc:
            print(f"    SKIP (warm-up failed): {exc}")
            rows.append({"model": tag, "error": str(exc), "composite_score": 0.0})
            continue

        lats, tps, jsons, lengths = [], [], [], []
        with mlflow.start_run(run_name=f"bench_{tag.replace(':', '_')}") as run:
            mlflow.log_param("model", tag)
            mlflow.log_param("device", device)
            mlflow.log_param("cold_load_sec", round(cold, 1))

            for item in BENCH_QUESTIONS:
                prompt_text = render_prompt(v3["template"], item["question"])
                res = query(tag, prompt_text, json_mode=True)
                valid = validate_json_response(res.content)
                lats.append(res.latency)
                tps.append(res.tokens_per_second)
                jsons.append(1.0 if valid else 0.0)
                lengths.append(res.output_tokens)

            avg_lat = float(np.mean(lats))
            avg_tps = float(np.mean(tps))
            vjson = float(np.mean(jsons))
            avg_len = float(np.mean(lengths))

            mlflow.log_metric("valid_json_rate", vjson)
            mlflow.log_metric("avg_tokens_per_second", avg_tps)
            mlflow.log_metric("avg_latency_seconds", avg_lat)
            mlflow.log_metric("avg_output_tokens", avg_len)

            rows.append({
                "model": tag, "device": device, "cold_load_sec": round(cold, 1),
                "avg_latency_sec": round(avg_lat, 3), "avg_tokens_per_second": round(avg_tps, 2),
                "valid_json_rate": vjson, "avg_output_tokens": round(avg_len, 1),
                "run_id": run.info.run_id,
            })
            print(f"    latency={avg_lat:.2f}s  tps={avg_tps:.1f}  valid_json={vjson:.2f}\n")

    # Composite score (normalise throughput and latency across candidates)
    df = pd.DataFrame([r for r in rows if "error" not in r])
    if df.empty:
        print("[!] No model completed the benchmark.")
        return {}
    tps_min, tps_max = df["avg_tokens_per_second"].min(), df["avg_tokens_per_second"].max()
    lat_min, lat_max = df["avg_latency_sec"].min(), df["avg_latency_sec"].max()

    def norm(v, lo, hi):
        return 0.5 if hi == lo else (v - lo) / (hi - lo)

    df["composite_score"] = (
        0.5 * df["valid_json_rate"]
        + 0.3 * df["avg_tokens_per_second"].apply(lambda v: norm(v, tps_min, tps_max))
        + 0.2 * df["avg_latency_sec"].apply(lambda v: 1.0 - norm(v, lat_min, lat_max))
    ).round(4)

    df.to_csv(config.BENCHMARK_CSV, index=False)
    print(f"[+] Saved {config.BENCHMARK_CSV}")

    winner = df.sort_values("composite_score", ascending=False).iloc[0]
    selected = {
        "selected_model": winner["model"],
        "composite_score": float(winner["composite_score"]),
        "valid_json_rate": float(winner["valid_json_rate"]),
        "avg_tokens_per_second": float(winner["avg_tokens_per_second"]),
        "avg_latency_sec": float(winner["avg_latency_sec"]),
        "device": device,
        "all_results": df.to_dict(orient="records"),
    }
    os.makedirs(os.path.dirname(config.SELECTED_MODEL_FILE), exist_ok=True)
    with open(config.SELECTED_MODEL_FILE, "w") as fh:
        json.dump(selected, fh, indent=2)

    print("\nBenchmark results (ranked):")
    print(df.sort_values("composite_score", ascending=False).to_string(index=False))
    print(f"\n[+] Selected model: {winner['model']} "
          f"(score={winner['composite_score']:.3f}) -> {config.SELECTED_MODEL_FILE}")
    return selected


if __name__ == "__main__":
    benchmark()
