import os
import sys
import json
import time
import math
import re
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import ollama

# --- Configuration & Constants ---
EXPERIMENT_NAME = "eduassist_llmops_assignment"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
MLFLOW_SERVER_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Baseline cost calculation settings (Using standard hosted endpoint rates per 1M tokens as benchmark reference)
INPUT_PRICE_PER_M = 0.30   # $0.30 per 1M tokens
OUTPUT_PRICE_PER_M = 0.60  # $0.60 per 1M tokens

ACADEMIC_QUESTIONS = [
    {"id": "Q1", "question": "Explain the concept of backpropagation in neural networks."},
    {"id": "Q2", "question": "What is the difference between supervised and unsupervised learning?"},
    {"id": "Q3", "question": "Describe how the Self-Attention mechanism works in Transformers."},
    {"id": "Q4", "question": "What is overfitting in machine learning and how can it be mitigated?"},
    {"id": "Q5", "question": "Explain the trade-off between Bias and Variance in predictive models."}
]


# Override or slice inputs for dry-run
if "--sample" in sys.argv:
    ACADEMIC_QUESTIONS = ACADEMIC_QUESTIONS[:1]

PROMPTS = {
    "v1": {
        "name": "v1_basic",
        "commit": "Basic prompt implementation for general query answering",
        "template": "Answer the following student question clearly.\nQuestion: {{question}}"
    },
    "v2": {
        "name": "v2_structured",
        "commit": "Structured domain persona prompt targeted for undergraduate students",
        "template": "You are an academic assistant.\nAnswer the following student question in a clear, concise, and accurate manner.\nUse simple language suitable for undergraduate students.\nQuestion: {{question}}"
    },
    "v3": {
        "name": "v3_json_output",
        "commit": "Strict JSON output prompt schema for structured response parsing",
        "template": (
            "You are an academic assistant.\n"
            "Answer the student question and return ONLY valid JSON.\n"
            "Do not include markdown, explanations outside JSON, or code fences.\n"
            "Use this exact JSON schema:\n"
            "{\n"
            '  "answer": "string",\n'
            '  "confidence": "high | medium | low",\n'
            '  "key_points": ["string"]\n'
            "}\n"
            "Question: {{question}}"
        )
    }
}

def render_prompt(template: str, question: str) -> str:
    return template.replace("{{question}}", question)

def validate_json_response(response_text: str) -> bool:
    cleaned = response_text.strip()
    # Strip markdown code blocks if model accidentally returns them
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return False
        required_keys = {"answer", "confidence", "key_points"}
        if not required_keys.issubset(data.keys()):
            return False
        if data["confidence"] not in ["high", "medium", "low"]:
            return False
        if not isinstance(data["key_points"], list):
            return False
        return True
    except Exception:
        return False

def calculate_costs(input_tokens: int, output_tokens: int):
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
    total_cost = input_cost + output_cost
    return input_cost, output_cost, total_cost

def query_ollama(prompt_text: str, json_format: bool = False):
    start_time = time.time()
    kwargs = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_text}]}
    if json_format:
        kwargs["format"] = "json"
        
    response = ollama.chat(**kwargs)
    end_time = time.time()
    
    latency = end_time - start_time
    content = response['message']['content']
    
    # Extract token metrics directly from Ollama response
    prompt_tokens = response.get('prompt_eval_count', len(prompt_text.split()))
    completion_tokens = response.get('eval_count', len(content.split()))
    
    return {
        "content": content,
        "start_time": start_time,
        "end_time": end_time,
        "latency": latency,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "status_code": 200
    }

def run_pipeline():
    os.makedirs("outputs", exist_ok=True)
    
    # Connect directly to running MLflow background server
    mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    client = MlflowClient(tracking_uri=MLFLOW_SERVER_URI)
    
    overall_latency_records = []
    overall_eval_records = []
    summary_metrics_per_version = {}

    print("\nExecuting Tasks 1, 2, and 3 across Prompt Versions...")

    for v_key, v_info in PROMPTS.items():
        print(f"\n--- Running Evaluation for Prompt Version: {v_key} ({v_info['name']}) ---")
        
        with mlflow.start_run(run_name=f"prompt_{v_key}") as run:
            # Log params & prompt template artifact
            mlflow.log_param("prompt_version", v_key)
            mlflow.log_param("model_name", MODEL_NAME)
            mlflow.log_param("commit_message", v_info["commit"])
            mlflow.log_text(v_info["template"], f"prompts/{v_key}_template.txt")

            run_latencies, run_inputs, run_outputs, run_costs, run_json_valid = [], [], [], [], []

            for item in ACADEMIC_QUESTIONS:
                q_id = item["id"]
                prompt_text = render_prompt(v_info["template"], item["question"])
                is_json = (v_key == "v3")
                
                res = query_ollama(prompt_text, json_format=is_json)
                inp_c, out_c, tot_c = calculate_costs(res["input_tokens"], res["output_tokens"])
                is_valid_json = validate_json_response(res["content"]) if is_json else True

                # 1. Log actual LLM text response into MLflow Artifacts
                mlflow.log_text(res["content"], f"responses/{q_id}_response.md")

                run_latencies.append(res["latency"])
                run_inputs.append(res["input_tokens"])
                run_outputs.append(res["output_tokens"])
                run_costs.append(tot_c)
                run_json_valid.append(1.0 if is_valid_json else 0.0)

                # 2. Latency & Token Metrics Record
                overall_latency_records.append({
                    "question_id": q_id,
                    "prompt_version": v_key,
                    "model_name": MODEL_NAME,
                    "start_time": res["start_time"],
                    "end_time": res["end_time"],
                    "latency_sec": res["latency"],
                    "input_tokens": res["input_tokens"],
                    "output_tokens": res["output_tokens"],
                    "total_tokens": res["total_tokens"],
                    "input_cost": inp_c,
                    "output_cost": out_c,
                    "total_cost": tot_c,
                    "status_code": res["status_code"]
                })

                # 3. Model Evaluation & Text Content Record
                overall_eval_records.append({
                    "question_id": q_id,
                    "prompt_version": v_key,
                    "model_name": MODEL_NAME,
                    "prompt_text": prompt_text,
                    "response_text": res["content"],  # Contains generated LLM answer
                    "valid_json": is_valid_json,
                    "latency_sec": res["latency"],
                    "total_cost": tot_c
                })

            # Compute aggregated metrics for MLflow
            avg_lat = float(np.mean(run_latencies))
            avg_inp = float(np.mean(run_inputs))
            avg_out = float(np.mean(run_outputs))
            avg_cost = float(np.mean(run_costs))
            valid_json_rate = float(np.mean(run_json_valid))

            mlflow.log_metric("avg_latency_seconds", avg_lat)
            mlflow.log_metric("avg_input_tokens", avg_inp)
            mlflow.log_metric("avg_output_tokens", avg_out)
            mlflow.log_metric("avg_cost_per_request", avg_cost)
            mlflow.log_metric("valid_json", valid_json_rate)

            summary_metrics_per_version[v_key] = {
                "run_id": run.info.run_id,
                "avg_latency": avg_lat,
                "avg_input_tokens": avg_inp,
                "avg_output_tokens": avg_out,
                "avg_cost": avg_cost,
                "valid_json_rate": valid_json_rate
            }

    # Save distinct CSV files
    df_latency = pd.DataFrame(overall_latency_records)
    df_latency.to_csv("outputs/latency_tokens_cost.csv", index=False)

    df_eval = pd.DataFrame(overall_eval_records)
    df_eval.to_csv("outputs/prompt_evaluation_results.csv", index=False)
    
    print("\n[+] Saved outputs/latency_tokens_cost.csv (Telemetry)")
    print("[+] Saved outputs/prompt_evaluation_results.csv (Responses & Eval)")

if __name__ == "__main__":
    run_pipeline()

