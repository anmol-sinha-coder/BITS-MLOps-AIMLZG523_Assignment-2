import os
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


os.makedirs("outputs", exist_ok=True)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

overall_latency_records = []
overall_cost_records = []
summary_metrics_per_version = {}

print("\nExecuting Tasks 1, 2, and 3 across Prompt Versions...")

for v_key, v_info in PROMPTS.items():
    print(f"\n--- Running Evaluation for Prompt Version: {v_key} ({v_info['name']}) ---")
    
    with mlflow.start_run(run_name=f"prompt_{v_key}") as run:
        # Log params
        mlflow.log_param("prompt_version", v_key)
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("commit_message", v_info["commit"])
        mlflow.log_text(v_info["template"], f"prompts/{v_key}_template.txt")

        run_latencies = []
        run_inputs = []
        run_outputs = []
        run_costs = []
        run_json_valid = []

        for item in ACADEMIC_QUESTIONS:
            q_id = item["id"]
            prompt_text = render_prompt(v_info["template"], item["question"])
            is_json = (v_key == "v3")
            
            res = query_ollama(prompt_text, json_format=is_json)
            
            inp_c, out_c, tot_c = calculate_costs(res["input_tokens"], res["output_tokens"])
            is_valid_json = validate_json_response(res["content"]) if is_json else True

            run_latencies.append(res["latency"])
            run_inputs.append(res["input_tokens"])
            run_outputs.append(res["output_tokens"])
            run_costs.append(tot_c)
            run_json_valid.append(1.0 if is_valid_json else 0.0)

            record = {
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
                "valid_json": is_valid_json,
                "status_code": res["status_code"]
            }
            overall_latency_records.append(record)

        # Compute aggregated metrics for MLflow logging
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

# Save details to CSV
df_results = pd.DataFrame(overall_latency_records)
df_results.to_csv("outputs/latency_tokens_cost.csv", index=False)
df_results.to_csv("outputs/prompt_evaluation_results.csv", index=False)
print("\n[+] Saved latency_tokens_cost.csv & prompt_evaluation_results.csv")

# --- Prompt Comparison & Promotion to Production Alias ---
print("\n==========================================")
print(" Prompt Version Performance Comparison ")
print("==========================================")

comp_data = []
best_version = "v3" # Default recommendation based on structural and JSON requirements

for v_key, metrics in summary_metrics_per_version.items():
    recommended = "YES (Promoted)" if v_key == best_version else "NO"
    comp_data.append({
        "Prompt Version": v_key,
        "Avg Latency (s)": round(metrics["avg_latency"], 4),
        "Avg Input Tokens": round(metrics["avg_input_tokens"], 2),
        "Avg Output Tokens": round(metrics["avg_output_tokens"], 2),
        "Avg Cost ($)": f"${metrics['avg_cost']:.6f}",
        "Valid JSON Rate": f"{metrics['valid_json_rate']*100:.1f}%",
        "Recommended?": recommended
    })

df_comp = pd.DataFrame(comp_data)
print(df_comp.to_string(index=False))

# Promote Best Run to Production Alias in MLflow
best_run_id = summary_metrics_per_version[best_version]["run_id"]
model_uri = f"runs:/{best_run_id}/model"
reg_model_name = "eduassist_academic_qa_prompt"

try:
    model_version = mlflow.register_model(model_uri, reg_model_name)
    client.set_registered_model_alias(
        name=reg_model_name,
        alias="production",
        version=model_version.version
    )
    print(f"\n[+] Successfully tagged Model Version {model_version.version} of '{reg_model_name}' with alias 'production'!")
except Exception as e:
    print(f"\n[!] Note on MLflow Registration: {e}")
