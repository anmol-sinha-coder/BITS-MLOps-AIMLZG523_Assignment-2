"""Main LLMOps pipeline for Assignment-2.

Runs the three prompt versions against the five academic questions, logs
per-request latency/tokens/cost plus the five required aggregate metrics to
MLflow, registers each prompt in the MLflow Prompt Registry, and promotes the
best version to the `production` alias.

Designed to be called from the notebook, the deploy script (`initial_deploy.sh`)
or directly:
    python llmops_pipeline.py            # full run (3 versions x 5 questions)
    python llmops_pipeline.py --sample   # dry run (1 version x 1 question)

Output CSVs (assignment deliverables):
    outputs/latency_tokens_cost.csv         (Task 1 + Task 2 telemetry)
    outputs/prompt_evaluation_results.csv   (Task 3 responses + eval)
"""
from __future__ import annotations

import os
import sys

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src import config, metrics as M
from src.ollama_client import query, warmup_model, detect_device, ollama_ps
from src.prompts import PROMPTS, render_prompt, is_json_version
from src.utils import validate_json_response

PROMPT_REGISTRY_NAME = "eduassist_prompts"


def _resolve_model() -> str:
    """Use the benchmark-selected model if present, else the configured default."""
    if os.path.isfile(config.SELECTED_MODEL_FILE):
        import json
        try:
            with open(config.SELECTED_MODEL_FILE) as fh:
                return json.load(fh).get("selected_model", config.DEFAULT_MODEL)
        except (ValueError, OSError):
            pass
    return config.DEFAULT_MODEL


def _register_prompts(client: MlflowClient) -> dict:
    """Register / update the three prompt versions in the MLflow Prompt Registry.

    MLflow >= 2.16 exposes prompts as Registered Models of type `prompt`.
    Each version gets a meaningful commit message so the Registry shows a clean
    diff history.  Returns {version_key: registered_prompt}.
    """
    registered: dict = {}
    try:
        from mlflow.prompts import create_prompt
    except Exception:
        # Prompt Registry API name varies slightly across MLflow releases; fall
        # back to logging prompts as text artifacts so the run still succeeds.
        return registered

    for v_key, v_info in PROMPTS.items():
        try:
            prompt = create_prompt(
                name=PROMPT_REGISTRY_NAME,
                template=v_info["template"],
                commit_message=v_info["commit"],
            )
            registered[v_key] = prompt
        except Exception as exc:  # pragma: no cover - registry optional
            print(f"[!] Prompt registry skipped for {v_key}: {exc}")
    return registered


def _promote_to_production(client: MlflowClient, best_version: str) -> None:
    """Set the `production` alias on the winning prompt version's run."""
    best = SUMMARY.get(best_version)
    if not best:
        return
    try:
        client.set_registered_model_alias(
            name=PROMPT_REGISTRY_NAME, alias="production", version=best.get("prompt_version_num")
        )
    except Exception:
        pass
    # Always tag the run so promotion is visible even without the registry alias.
    try:
        client.set_tag(best["run_id"], "promoted_to", "production")
    except Exception:
        pass


SUMMARY: dict = {}  # populated during the run, used for promotion


def run_pipeline() -> dict:
    model = _resolve_model()
    device = detect_device()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)

    # --- Cold-start mitigation: warm the model once before timing -------------
    print(f"\n[+] Device: {device.upper()} | Model: {model}")
    print(f"[+] GPU residency: {ollama_ps()}")
    print("[+] Warming model into VRAM (this absorbs the 8-9 min cold start)...")
    cold = warmup_model(model)
    print(f"[+] Cold-load time: {cold:.1f}s (excluded from latency metrics)")

    _register_prompts(client)

    questions = config.ACADEMIC_QUESTIONS
    if "--sample" in sys.argv:
        questions = questions[:1]

    latency_rows: list[dict] = []
    eval_rows: list[dict] = []

    print("\nExecuting Tasks 1, 2 and 3 across prompt versions...")
    for v_key, v_info in PROMPTS.items():
        if "--sample" in sys.argv and v_key != "v3":
            continue
        print(f"\n--- Prompt {v_key} ({v_info['name']}) ---")
        with mlflow.start_run(run_name=f"prompt_{v_key}") as run:
            mlflow.log_param("prompt_version", v_key)
            mlflow.log_param("model_name", model)
            mlflow.log_param("device", device)
            mlflow.log_param("commit_message", v_info["commit"])
            mlflow.log_text(v_info["template"], f"prompts/{v_key}_template.txt")

            lats, ins, outs, costs, jsons = [], [], [], [], []

            for item in questions:
                qid = item["id"]
                prompt_text = render_prompt(v_info["template"], item["question"])
                res = query(model, prompt_text, json_mode=is_json_version(v_key))
                cb = M.CostBreakdown.from_tokens(res.input_tokens, res.output_tokens)
                valid = validate_json_response(res.content) if is_json_version(v_key) else True
                extension = "md" if v_key == "v1" else ("json" if v_key == "v3" else "txt")
                mlflow.log_text(
                    res.content, f"responses/{v_key}_{qid}_response.{extension}"
                )

                lats.append(res.latency)
                ins.append(res.input_tokens)
                outs.append(res.output_tokens)
                costs.append(cb.total_cost)
                jsons.append(1.0 if valid else 0.0)

                latency_rows.append({
                    "question_id": qid, "prompt_version": v_key, "model_name": model,
                    "device": res.device, "start_time": res.start_time, "end_time": res.end_time,
                    "latency_sec": round(res.latency, 4),
                    "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                    "total_tokens": res.total_tokens,
                    "tokens_per_second": round(res.tokens_per_second, 2),
                    "input_cost": round(cb.input_cost, 8), "output_cost": round(cb.output_cost, 8),
                    "total_cost": round(cb.total_cost, 8), "status_code": res.status_code,
                })
                eval_rows.append({
                    "question_id": qid, "prompt_version": v_key, "model_name": model,
                    "question": item["question"], "prompt_text": prompt_text,
                    "response_text": res.content, "valid_json": valid,
                    "latency_sec": round(res.latency, 4), "total_cost": round(cb.total_cost, 8),
                })

            avg_lat = float(np.mean(lats))
            avg_in = float(np.mean(ins))
            avg_out = float(np.mean(outs))
            avg_cost = float(np.mean(costs))
            vjson = float(np.mean(jsons))
            stats = M.LatencyStats.from_latencies(lats)

            mlflow.log_metric("avg_latency_seconds", avg_lat)
            mlflow.log_metric("avg_input_tokens", avg_in)
            mlflow.log_metric("avg_output_tokens", avg_out)
            mlflow.log_metric("avg_cost_per_request", avg_cost)
            mlflow.log_metric("valid_json", vjson)
            mlflow.log_metric("latency_min", stats.minimum)
            mlflow.log_metric("latency_max", stats.maximum)
            mlflow.log_metric("latency_median", stats.median)
            mlflow.log_metric("tokens_per_second", float(np.mean([
                o / l if l > 0 else 0 for o, l in zip(outs, lats)
            ])))

            SUMMARY[v_key] = {
                "run_id": run.info.run_id, "avg_latency": avg_lat,
                "avg_input_tokens": avg_in, "avg_output_tokens": avg_out,
                "avg_cost": avg_cost, "valid_json_rate": vjson,
            }
            print(f"    avg_latency={avg_lat:.2f}s  in={avg_in:.0f}  out={avg_out:.0f}  "
                  f"cost=${avg_cost:.6f}  valid_json={vjson:.2f}")

    # Persist CSVs
    pd.DataFrame(latency_rows).to_csv(config.LATENCY_CSV, index=False)
    pd.DataFrame(eval_rows).to_csv(config.EVAL_CSV, index=False)
    print(f"\n[+] Saved {config.LATENCY_CSV}")
    print(f"[+] Saved {config.EVAL_CSV}")

    # --- Promote the best version to `production` ----------------------------
    best = _select_best_version()
    if best:
        _promote_to_production(client, best)
        print(f"\n[+] Promoted {best} -> production alias "
              f"(latency={SUMMARY[best]['avg_latency']:.2f}s, "
              f"valid_json={SUMMARY[best]['valid_json_rate']:.2f})")

    _print_comparison_table()
    return SUMMARY


def _select_best_version() -> str | None:
    """Pick the winner. v3 is preferred when its JSON is valid (assignment goal);
    otherwise the lowest avg_cost_per_request with acceptable latency wins."""
    if not SUMMARY:
        return None
    v3 = SUMMARY.get("v3")
    if v3 and v3["valid_json_rate"] >= 0.8:
        return "v3"
    # Fallback: minimise a composite of latency and cost.
    return min(SUMMARY, key=lambda k: SUMMARY[k]["avg_latency"] + 100 * SUMMARY[k]["avg_cost"])


def _print_comparison_table() -> None:
    if not SUMMARY:
        return
    best = _select_best_version()
    print("\nPrompt comparison")
    print(f"{'Version':<8}{'AvgLat(s)':>10}{'AvgInTok':>9}{'AvgOutTok':>10}"
          f"{'AvgCost$':>11}{'ValidJSON':>10}{'Recommended':>12}")
    for k, s in SUMMARY.items():
        flag = "yes" if k == best else ""
        print(f"{k:<8}{s['avg_latency']:>10.2f}{s['avg_input_tokens']:>9.0f}"
              f"{s['avg_output_tokens']:>10.0f}{s['avg_cost']:>11.6f}"
              f"{s['valid_json_rate']:>10.2f}{flag:>12}")


if __name__ == "__main__":
    run_pipeline()
