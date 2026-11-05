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
For the consolidated functional/non-functional requirements.
# Requirements — EduAssist LLMOps Assignment-2

Consolidated functional and non-functional requirements, derived from the
assignment PDF (`MLOps Assignment-2 Instructions.pdf`) and the user's
deployment constraints. Each requirement is traceable to a deliverable in the
repository.

---

## 1. Functional requirements

### FR-1 — Free, locally hosted LLM (no paid API)
- **R1.1** Inference must use a **free, open-weight Ollama model** (Qwen /
  Gemma / Llama family); no paid API key is required or hard-coded.
- **R1.2** The model must be **configurable** via CLI (`--model`), environment
  variable (`OLLAMA_MODEL`), or the benchmark-selected
  `config/selected_model.json`.
- **R1.3** The default model (`qwen2.5:7b`) must have sufficiently high
  evaluation standing amongst free models and fit an 8–12 GB RTX card;
  `MODEL_SELECTION.md` documents the choice and the composite eval formula.

### FR-2 — Task 1: latency measurement (2 marks)
- **R2.1** Call the LLM for **five academic questions** and record, per
  request: question id, prompt version, model name, start time, end time,
  latency, and response status.
- **R2.2** Compute mean, median, minimum, maximum, and standard deviation of
  latency.
- **R2.3** Explain the slowest request and possible reasons.
- **R2.4** Output table: `outputs/latency_tokens_cost.csv`.

### FR-3 — Task 2: token counting and cost economics (3 marks)
- **R3.1** Extract input and output token counts from each API response
  (Ollama `prompt_eval_count` / `eval_count`).
- **R3.2** Compute per-request input cost, output cost, and total cost from
  published hosted-equivalent pricing.
- **R3.3** Project the monthly cost at 10,000 requests.
- **R3.4** Provide at least one cost-optimisation recommendation.
- **R3.5** Output table: `outputs/latency_tokens_cost.csv` (token + cost
  columns) and the notebook's cost cell.

### FR-4 — Task 3: prompt versioning and MLflow evaluation (5 marks)
- **R4.1** Create MLflow experiment `eduassist_llmops_assignment`.
- **R4.2** Register three prompt versions (v1 basic, v2 structured, v3 JSON)
  in the MLflow Prompt Registry (`eduassist_prompts`) with meaningful commit
  messages.
- **R4.3** Run each version on the same five questions.
- **R4.4** Log, per version: `avg_latency_seconds`, `avg_input_tokens`,
  `avg_output_tokens`, `avg_cost_per_request`, `valid_json`.
- **R4.5** For v3, compute `valid_json = valid JSON responses / total
  responses` against the schema `{answer, confidence, key_points}`.
- **R4.6** Compare all versions in a table and promote the best to the
  `production` alias.
- **R4.7** Provide a promotion justification of ≥ 150 words covering quality,
  JSON validity, latency, tokens, and cost.
- **R4.8** Output table: `outputs/prompt_evaluation_results.csv`.

### FR-5 — One-shot deployment script
- **R5.1** A single shell script (`initial_deploy.sh`) must perform the entire
  setup from a clean state.
- **R5.2** It must **auto-detect Debian (`apt`) vs RedHat (`dnf`/`yum`)**
  Linux and use the correct package manager.
- **R5.3** It must **auto-detect NVIDIA GPU vs CPU** (`nvidia-smi`) and run on
  either, with a `--cpu-only` override.
- **R5.4** It must create a **Python 3.12 virtual environment** (only 3.12) at
  `.venv` and install all dependencies there.
- **R5.5** It must install/start Ollama, pull the model, start MLflow, run the
  pipeline, and bring up monitoring.
- **R5.6** It must accept **argument options** (`--model`, `--benchmark`,
  `--cpu-only`, `--no-monitoring`, `--no-traffic`, `--no-pipeline`, `--reset`,
  `--skip-install`, `--mlflow-mode`, `--venv`, `--python`, `-h/--help`) with
  both space and `=` forms.
- **R5.7** It must be **idempotent and safe to re-run**.

### FR-6 — Dockerised deployment
- **R6.1** A `Dockerfile` builds a lean serving image for the FastAPI API
  (non-root user, healthcheck).
- **R6.2** A `docker-compose.yml` orchestrates MLflow, the API, Prometheus,
  and Grafana with named volumes and healthchecks.

### FR-7 — Monitoring and CI/CD
- **R7.1** Prometheus scrapes the API `/metrics`; alert rules cover high
  latency, low valid-JSON rate, and downtime.
- **R7.2** A Grafana dashboard is auto-provisioned.
- **R7.3** A GitHub Actions CI workflow runs flake8, pytest, validates
  `docker-compose config` and Prometheus config, and builds the serving image.

### FR-8 — Reproducibility artefacts
- **R8.1** A Jupyter notebook (`MLOps_Assignment-2.ipynb`) executes all three
  tasks end-to-end and is exportable to PDF/HTML.
- **R8.2** A `requirements.txt` (Python 3.12) and a lean
  `requirements-serve.txt` for the container.
- **R8.3** A `README.md` with Python version, libraries, environment
  variables, install commands, MLflow UI command, and reproduction steps.
- **R8.4** A final report PDF (500–700 words).

---

## 2. Non-functional requirements

### NFR-1 — Performance (fixes the 8–9 min first-prompt symptom)
- **N1.1** A warm-up call must load the model into VRAM **before** timing, so
  timed latencies exclude the cold load.
- **N1.2** `OLLAMA_KEEP_ALIVE=30m` keeps weights resident between requests.
- **N1.3** GPU residency is verified (`ollama ps`, `nvidia-smi`) and logged.
- **N1.4** Warm inference latency target: ≤ 5 s for a ~100-token JSON answer
  on an RTX class GPU.

### NFR-2 — Portability
- **N2.1** Works on Debian and RedHat Linux families.
- **N2.2** Works with an NVIDIA RTX GPU **and** on CPU-only systems
  (automatic fallback; `--cpu-only` override).
- **N2.3** Python 3.12 only; no system Python is modified.

### NFR-3 — Security and integrity
- **N3.1** No API key, secret, or credential is hard-coded; all are read from
  environment variables.
- **N3.2** The serving container runs as a non-root user.
- **N3.3** Deterministic evaluation (`temperature=0.0`).

### NFR-4 — Maintainability
- **N4.1** Single source of truth for configuration (`src/config.py`).
- **N4.2** Unit tests for config, metrics, prompts, and utils (`pytest`).
- **N4.3** Lint-clean (`flake8`, max-line-length 100).
- **N4.4** Modular code: `config`, `prompts`, `ollama_client`, `metrics`,
  `utils`, `app`, plus the pipeline and benchmark entry points.

### NFR-5 — Observability
- **N5.1** Prometheus metrics: request count, latency histogram, input/output
  token histograms, tokens-per-second, valid-JSON rate, cost-per-request,
  model-info gauge.
- **N5.2** MLflow tracks every run's params, metrics, and prompt artefacts.

---

## 3. Traceability matrix

| Requirement | Deliverable |
| --- | --- |
| FR-1 (free model) | `src/config.py`, `config/model_candidates.json`, `MODEL_SELECTION.md` |
| FR-2 (latency) | `llmops_pipeline.py`, `outputs/latency_tokens_cost.csv`, notebook Task 1 |
| FR-3 (tokens/cost) | `src/metrics.py`, notebook Task 2 |
| FR-4 (MLflow eval) | `llmops_pipeline.py`, `src/prompts.py`, notebook Task 3 |
| FR-5 (deploy script) | `initial_deploy.sh` |
| FR-6 (Docker) | `Dockerfile`, `docker-compose.yml` |
| FR-7 (monitoring/CI) | `monitoring/`, `.github/workflows/ci.yml` |
| FR-8 (artefacts) | `MLOps_Assignment-2.ipynb`, `requirements*.txt`, `README.md`, report PDF |
| NFR-1 (perf) | `src/ollama_client.py` (warmup, keep_alive, detect_device) |
| NFR-5 (observability) | `src/metrics.py`, `src/app.py` |
