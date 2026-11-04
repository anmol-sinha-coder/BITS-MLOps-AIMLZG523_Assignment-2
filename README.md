# EduAssist AI — LLMOps Assignment-2

**API Cost, Latency, and Prompt Version Management** · Course: MLOps (AIMLZG523)

An end-to-end LLMOps pipeline for an academic question-answering assistant. It
calls a **free, locally hosted Ollama model** (`qwen2.5:7b` by default),
measures response latency, extracts token usage and computes request-level and
projected-monthly costs, then registers three prompt versions in the **MLflow
Prompt Registry**, evaluates them on five academic questions, and promotes the
best version to the `production` alias. The whole stack is deployed by a single
shell script that auto-detects Debian/RedHat Linux, NVIDIA RTX GPU vs CPU, and
runs entirely inside a **Python 3.12 virtual environment**.

> This Assignment-2 extends the Assignment-1 MLOps pattern
> (logistic-regression / random-forest comparison + Dockerised CI/CD +
> monitoring). Assignment-1 reused the same `initial_deploy.sh` shape; here the
> "model" is an LLM, the "features" are prompts, and the "metrics" are latency,
> token cost and JSON-validity. See `MODEL_SELECTION.md` for why the first
> prompt took 8–9 minutes in Assignment-1 and how this stack fixes it.

---

## Repository layout

```
.
├── initial_deploy.sh            # one-shot deploy: OS detect, venv 3.12, Ollama, MLflow, monitoring
├── llmops_pipeline.py           # Tasks 1+2+3: latency, tokens, cost, MLflow eval, promotion
├── model_benchmark.py           # benchmark free Ollama candidates, pick highest composite score
├── MLOps_Assignment-2.ipynb     # executable notebook (all 3 tasks, ships with rendered outputs)
├── src/
│   ├── config.py                # single source of truth: model, pricing, questions, endpoints
│   ├── prompts.py               # the three prompt versions (v1 basic / v2 structured / v3 JSON)
│   ├── ollama_client.py         # Ollama HTTP wrapper, GPU detect, warm-up, real token telemetry
│   ├── metrics.py               # latency stats, cost economics, Prometheus instrumentation
│   ├── utils.py                 # JSON-schema validation (answer/confidence/key_points)
│   └── app.py                   # FastAPI serving the v3 production prompt + /metrics
├── config/model_candidates.json # candidate free models for the benchmark
├── outputs/
│   ├── latency_tokens_cost.csv          # Task 1 + Task 2 telemetry (15 rows: 5 Q × 3 versions)
│   └── prompt_evaluation_results.csv    # Task 3 responses + valid_json (15 rows)
├── monitoring/                  # prometheus.yml, alert_rules.yml, Grafana dashboard + provisioning
├── scripts/                     # smoke_test, generate_traffic, verify_monitoring, sample outputs
├── test/                        # pytest unit tests (config, metrics, prompts, utils)
├── .github/workflows/ci.yml     # lint + tests + docker build + compose/prometheus validation
├── Dockerfile                   # lean serving image for the FastAPI API
├── docker-compose.yml           # MLflow + API + Prometheus + Grafana
├── requirements.txt             # full dev/test deps (Python 3.12)
├── requirements-serve.txt       # lean runtime deps for the serving container
└── pytest.ini
```

---

## Python version and required libraries

- **Python:** 3.12 (the venv is created with `python3.12` only; see
  `DEPLOYMENT_GUIDE.md` for installing it if your distro ships an older one).
- **Key libraries** (full list in `requirements.txt`):
  - Inference: `requests` (talks to the Ollama HTTP API; no paid SDK needed).
  - Tracking & prompt registry: `mlflow>=2.16`.
  - Data: `pandas`, `numpy`.
  - Validation: `pydantic`, `jinja2`.
  - Tokenisation reference: `tiktoken`.
  - Serving & monitoring: `fastapi`, `uvicorn`, `prometheus-client`,
    `prometheus-fastapi-instrumentator`.
  - Tests & lint: `pytest`, `pytest-cov`, `flake8`, `httpx`.
  - Notebook: `jupyterlab`, `notebook`.

---

## Environment variables

No secret is ever hard-coded. All tunables are read from the environment with
sensible defaults in `src/config.py`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Free model tag (override with `--model`). |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama daemon URL. |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keep weights resident in VRAM between requests. |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow tracking server. |
| `INPUT_PRICE_PER_M` | `0.15` | Hosted-equivalent input price per 1M tokens (USD). |
| `OUTPUT_PRICE_PER_M` | `0.60` | Hosted-equivalent output price per 1M tokens (USD). |

The hosted-equivalent pricing uses OpenAI GPT-4o-mini published rates so the
cost figures are comparable to a real cloud bill, as permitted by the
assignment's academic-integrity clause when a local open-source model is used.

---

## Installation and one-shot deployment

From the project root, on Ubuntu/WSL or any Debian/RedHat Linux with an NVIDIA
RTX GPU (CPU fallback is automatic):

```bash
chmod +x initial_deploy.sh
./initial_deploy.sh                      # full deploy: deps, venv 3.12, Ollama,
                                         # MLflow, tests, warm model, pipeline, API,
                                         # Prometheus/Grafana, traffic
```

The script detects the OS (`apt` vs `dnf/yum`), detects the GPU
(`nvidia-smi`), creates a Python 3.12 venv at `.venv`, installs Ollama and
pulls the model, starts MLflow, runs the pipeline, and brings up the
monitoring stack. Useful flags:

```bash
./initial_deploy.sh --model qwen3:8b          # override the Ollama model
./initial_deploy.sh --benchmark               # benchmark candidates & auto-pick best
./initial_deploy.sh --cpu-only                # force CPU (ignore GPU)
./initial_deploy.sh --no-monitoring           # skip Prometheus/Grafana/API
./initial_deploy.sh --mlflow-mode local       # local sqlite MLflow (no Docker)
./initial_deploy.sh --skip-install            # do not apt/dnf install anything
./initial_deploy.sh --reset                   # wipe MLflow/Ollama model cache
./initial_deploy.sh -h | --help               # full help
```

---

## MLflow UI and service endpoints

After deployment the services are reachable at:

| Service | URL | Credentials |
| --- | --- | --- |
| MLflow tracking UI | http://localhost:5000 | — |
| FastAPI docs (Swagger) | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

The MLflow experiment is **`eduassist_llmops_assignment`**; the Prompt Registry
entry is **`eduassist_prompts`** with versions `v1`, `v2`, `v3`. The winning
version (`v3`) carries the `production` alias. To launch the MLflow UI manually
(local mode):

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

---

## Reproducing the experiment

```bash
# 1. Deploy the stack (creates .venv, starts Ollama + MLflow + monitoring)
./initial_deploy.sh --model qwen2.5:7b

# 2. Activate the venv and open the notebook
source .venv/bin/activate
jupyter lab MLOps_Assignment-2.ipynb
#   -> Run All Cells. The notebook executes Tasks 1, 2, 3 end-to-end and
#      writes outputs/latency_tokens_cost.csv and outputs/prompt_evaluation_results.csv.
#   -> If Ollama/MLflow are not running, it loads the shipped sample CSVs so
#      every table still renders (headless / CI friendly).

# 3. Re-run the benchmark to pick the best free model for your hardware
python model_benchmark.py            # writes config/selected_model.json

# 4. Export the notebook for submission
jupyter nbconvert --to pdf MLOps_Assignment-2.ipynb
jupyter nbconvert --to html MLOps_Assignment-2.ipynb

# 5. Run the test suite and lint
pytest -q
flake8 .
```

The five academic questions and the three prompt templates are fixed in
`src/config.py` and `src/prompts.py` so every run is reproducible
(`temperature=0.0` is enforced for deterministic evaluation).

---

## Deliverables checklist (per the assignment)

- [x] Jupyter notebook with executable code — `MLOps_Assignment-2.ipynb`
- [x] `requirements.txt`
- [x] `outputs/latency_tokens_cost.csv` (Task 1 + Task 2)
- [x] `outputs/prompt_evaluation_results.csv` (Task 3)
- [x] MLflow experiment `eduassist_llmops_assignment` with metrics logged
- [x] Prompt Registry `eduassist_prompts` with v1, v2, v3 + `production` alias
- [x] Final report PDF (500–700 words)
- [x] README with setup and reproduction steps (this file)
- [x] One-shot deployment script (`initial_deploy.sh`)
- [x] Dockerised serving + monitoring (`Dockerfile`, `docker-compose.yml`)
- [x] CI workflow (`.github/workflows/ci.yml`)

See `DEPLOYMENT_GUIDE.md` for the detailed step-by-step Linux+NVIDIA RTX
procedure, `MODEL_SELECTION.md` for the free-model evaluation, and
`REQUIREMENTS.md` for the consolidated functional/non-functional requirements.
