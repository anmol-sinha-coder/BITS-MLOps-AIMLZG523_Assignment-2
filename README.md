# EduAssist AI — LLMOps Assignment-2

**An end-to-end LLMOps pipeline for an academic Q&A assistant built on free Ollama models (Qwen / Gemma), deployed on Linux with an NVIDIA RTX GPU, tracked with MLflow 3.16.0, served through a FastAPI API, observed with Prometheus & Grafana, and shipped through GitHub Actions CI/CD.**

| | |
| --- | --- |
| **Name** | Anmol Sinha |
| **BITS ID** | 2025AE05129 |
| **Course** | AIMLZG523 — MLOps |
| **Assignment** | Assignment-2 (Individual) |
| **Model** | `qwen2.5:7b` (free, locally hosted via Ollama) |
| **Execution environment** | BITS Prayogshala Virtual Lab (online Linux desktop, NVIDIA RTX 3050 4 GB) |
| **GitHub repository** | https://github.com/anmol-sinha-coder/BITS-MLOps-AIMLZG523_Assignment-2 |
| **Date** | November 2026 |

---

## 1. Overview

EduAssist AI answers five academic machine-learning questions (backpropagation, supervised vs. unsupervised learning, self-attention, overfitting, bias–variance) using a locally hosted Ollama model. The assignment operationalises the full LLM lifecycle: prompts, token economics and output guardrails are treated as versioned, monitored production artefacts. Three prompt versions (`v1` prose, `v2` concise persona, `v3` JSON schema) are evaluated across the five questions; the best is promoted to the MLflow `production` alias.

The stack runs on Ubuntu/WSL with an NVIDIA RTX GPU and automatic CPU fallback. A single idempotent shell script (`initial_deploy.sh`) auto-detects the package manager (apt or dnf), creates a Python 3.12 virtual environment, installs and starts Ollama, pulls the model, launches MLflow, runs the evaluation pipeline, and brings up Prometheus and Grafana. Unlike Assignment-1 (which compared logistic-regression and random-forest models through a Dockerised CI/CD pipeline with MLflow tracking), this assignment substitutes a free locally hosted Ollama model for the paid API and elevates prompt-version management to a first-class concern.

---

## 2. Two independent execution paths

The project ships **two execution paths that can be run independently**. Either one demonstrates the full LLMOps pipeline; running both is recommended for a complete submission.

| Path | Entry point | What it does | Depends on |
| --- | --- | --- | --- |
| **A — One-shot deploy** | `./initial_deploy.sh` | Installs OS packages, Ollama, Python 3.12 venv, MLflow, runs the pipeline, builds the Docker API + Prometheus + Grafana stack, generates traffic | Nothing (self-contained). Can skip install with `--skip-install` if deps already present. |
| **B — JupyterLab notebook** | `jupyter lab MLOps_Assignment-2.ipynb` | Executes the same three tasks cell-by-cell with inline charts, NLG metrics (BLEU/ROUGE/METEOR/BERTScore), sentence-tokeniser analysis, LLMOps trace, and MLflow logging | A running Ollama (`ollama serve`) and a Python 3.12 venv with `requirements.txt` installed. The notebook degrades gracefully to sample mode if Ollama is unreachable. |

> **You may run A and B in any order, or only one of them.** Path A is the engineering/deployment demonstration; Path B is the analytical/experimental demonstration. Both write to the same MLflow experiment (`eduassist_llmops_assignment`) and the same CSV outputs (`outputs/latency_tokens_cost.csv`, `outputs/prompt_evaluation_results.csv`).

---

## 3. Repository contents layout

```
BITS-MLOps-AIMLZG523_Assignment-2/
├── initial_deploy.sh              # one-shot deploy (9 stages, OS auto-detect)
├── llmops_pipeline.py             # Tasks 1+2+3 (latency, cost, prompt versioning)
├── model_benchmark.py             # free-model selection benchmark
├── MLOps_Assignment-2.ipynb       # executable notebook (Path B)
├── requirements.txt               # Python deps (mlflow==3.16.0 pinned)
├── requirements-serve.txt         # API serving deps
├── Dockerfile                     # API container image
├── docker-compose.yml             # API + MLflow + Prometheus + Grafana
├── src/                           # config, prompts, ollama_client, metrics, utils, app
├── config/model_candidates.json   # benchmark candidate list
├── monitoring/                    # prometheus.yml, alert_rules.yml, eduassist_dashboard.json
│   └── provisioning/              # Grafana datasource + dashboard provisioning
├── scripts/                       # generate_traffic.py, smoke_test.sh, verify_monitoring.py
├── test/                          # 25 pytest unit tests
├── outputs/                       # latency_tokens_cost.csv, prompt_evaluation_results.csv
├── .github/workflows/ci.yml       # flake8 + pytest + docker build + compose validation
└── docs: README.md, REQUIREMENTS.md, MODEL_SELECTION.md, DEPLOYMENT_GUIDE.md
```

---

## 4. Path A — One-shot deployment (`./initial_deploy.sh`) — ordered execution

From the project root:

```bash
unzip BITS-MLOps-AIMLZG523_Assignment-2.zip
cd BITS-MLOps-AIMLZG523_Assignment-2
chmod +x initial_deploy.sh
./initial_deploy.sh --model qwen2.5:7b        # full deploy (re-runnable)
```

The script (`set -Eeuo pipefail`) runs **nine ordered stages**. Each stage prints `--> [n/9] ...` so progress is visible. Useful flags: `--model`, `--benchmark`, `--cpu-only`, `--no-monitoring`, `--no-traffic`, `--no-pipeline`, `--mlflow-mode local|docker`, `--reset`, `--skip-install`, `-h`.

| Stage | Script section | Action | Verification |
| --- | --- | --- | --- |
| **1/9** | OS detection + base packages | Reads `/etc/os-release`; installs ca-certificates, curl, gnupg, git, python3.12, gcc, make, jq via apt (Debian) or dnf/yum (RedHat) | `apt`/`dnf` exit 0 |
| **2/9** | Hardware detection | Runs `nvidia-smi`; sets `DEVICE_TYPE=gpu` and records GPU name + VRAM, else `cpu` | GPU name printed |
| **3/9** | Ollama install + model pull | Installs Ollama if absent, starts the daemon, pulls `qwen2.5:7b`, verifies VRAM residency with `ollama ps` | `ollama ps` shows 100% GPU |
| **4/9** | Python 3.12 venv | Creates `.venv` with `python3.12`, installs `requirements.txt` | `pip check` clean |
| **5/9** | Quality gates | Runs `flake8` (config in `.flake8`) and `pytest -q` (25 tests in `test/`) | 0 lint errors, 25 passed |
| **6/9** | MLflow launch | Docker mode: `docker compose up -d mlflow` (image `ghcr.io/mlflow/mlflow:v2.16.1`); local mode: `mlflow ui --backend-store-uri sqlite:///mlflow.db` | `curl localhost:5000` returns 200 |
| **7/9** | Model warm-up + pipeline | Sends a trivial `ping` to absorb the cold-load cost, then runs `llmops_pipeline.py` across 3 prompt versions × 5 questions, logging metrics to MLflow | CSVs written to `outputs/` |
| **8/9** | Model benchmark (optional, `--benchmark`) | Runs `model_benchmark.py`, writes winner to `config/selected_model.json` | Best model printed |
| **9/9** | API + monitoring stack + traffic | `docker compose up -d --build` (API, Prometheus, Grafana); `wait_for` health checks; `scripts/smoke_test.sh`; background `scripts/generate_traffic.py` for 120 s; `scripts/verify_monitoring.py` | All health checks pass, Grafana panels populated |

At the end a summary banner lists every endpoint (MLflow, API docs, Prometheus, Grafana, CSV outputs, notebook).

---

## 5. Path B — JupyterLab notebook — ordered cell execution

```bash
source .venv/bin/activate
ollama serve >/dev/null 2>&1 &        # only if not already running
jupyter lab MLOps_Assignment-2.ipynb
```


Run the cells top-to-bottom. The notebook is self-contained and mirrors the DataForScience/LLMEng LLMOps pattern (tracing, cost, caching, deadline/fallback, regression gate). If Ollama is unreachable it falls back to sample mode so every cell still executes.

| Cells | Content |
| --- | --- |
| 1–2 | Config snapshot: Python/Ollama/MLflow versions, model, host reachability, pricing |
| 3–4 | Device detection, model warm-up into VRAM, cold-load timing |
| 5–6 | **Task 1** — latency for 5 questions; p50/p95/mean/max histogram + per-question bar |
| 7–8 | **Task 2** — real token counts, per-request cost, cumulative cost curve, avg cost by version |
| 9–10 | **Task 3a** — three prompt versions registered in MLflow; comparison table; v3 promoted to `production` |
| 11–12 | **Task 3b** — NLG metrics (BLEU, ROUGE-1/2/L, METEOR, BERTScore) per response + by-version bars + ROUGE-L heatmap |
| 13–14 | Sentence-tokeniser analysis: sentences/response, avg words/sentence, total words by version |
| 15–17 | **Task 4** — LLMOps flow: `trace.jsonl` event log, latency histogram, cumulative cost, deadline/fallback demo, regression gate |

---


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


## 6. Post-deployment verification & service endpoints

| Service | URL | Credentials |
| --- | --- | --- |
| MLflow tracking UI | http://localhost:5000 | — |
| FastAPI docs | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

Confirm GPU residency: `ollama ps` (processor column should show 100% GPU) and `nvidia-smi` (~4.4 GB used for `qwen2.5:7b`). If `ollama ps` shows CPU, the model is spilling to system RAM and the first prompt will take minutes — switch to a smaller model (`--model gemma3:4b`) or increase VRAM headroom.

---

## 7. MLflow UI walkthrough — what to capture

1. **Experiment** — sidebar → Experiments → `eduassist_llmops_assignment` → runs table (prompt_v1/v2/v3).
2. **Metrics** — click a run → Metrics → `avg_latency_seconds`, `avg_cost_per_request`, `valid_json` charts.
3. **Prompt Registry** — sidebar → Prompts → registered prompts `academic_qa_v1/v2/v3` with version counts.
4. **Prompt diff** — Prompts → a prompt → tick two versions → Compare → side-by-side template diff.
5. **Production alias** — `mlflow.set_prompt_alias(name="academic_qa_v3", alias="production", version=1)` → version row shows `@production`.

---

## 8. Grafana monitoring

The dashboard `EduAssist LLMOps - Monitoring` (auto-provisioned at `/var/lib/grafana/dashboards/eduassist_dashboard.json`) has seven panels: Requests/min, p95 Latency, Throughput (tok/s), JSON Validity, Latency distribution, Tokens (in/out) per request, and Cost per request. All PromQL queries use the `eduassist_*` metric namespace defined in `src/metrics.py` (`PROM_NAMESPACE = "eduassist"`), so the dashboard and the exposed metrics match by construction. Background traffic from `scripts/generate_traffic.py` populates the panels.

---

## 9. Known issues & precise fixes (deployment error on Prayogshala)

When `./initial_deploy.sh` reaches Stage 9 on the Prayogshala Linux host, three errors can appear in sequence:

```
ERROR: deployment failed near line 238 (exit 1)
ERROR: deployment failed near line 240 (exit 1)
curl: (22) The requested URL returned error: 502 Expecting value: line 1 column 1 (char 0)
ERROR: deployment failed near line 241 (exit 1)
```

**Root cause:** the API container cannot reach the host's Ollama. `docker-compose.yml:30` sets `OLLAMA_HOST=http://host.docker.internal:11434`, but Linux Docker Engine does **not** resolve `host.docker.internal` by default (Docker Desktop on Mac/Windows does). With no route to Ollama, every `/ask` call raises `HTTPException(502)` at `src/app.py:88`, the success-path metrics are never observed, and the Grafana panels that depend on them stay empty ("barely few responses"). The smoke test then fails to parse the 502 body as JSON ("Expecting value: line 1 column 1").

| # | File | Line(s) | Problem | Fix |
| --- | --- | --- | --- | --- |
| **1 (root cause)** | `docker-compose.yml` | 24–39 (`api` service) | No `extra_hosts` mapping, so `host.docker.internal` (set at line 30) does not resolve on Linux | Add `extra_hosts: ["host.docker.internal:host-gateway"]` to the `api` service block |
| **2 (startup hang → line 238)** | `src/app.py` | 49–59 | `@app.on_event("startup")` calls `warmup_model()` synchronously inside the async startup hook; `warmup_model` uses `timeout=600` (`src/ollama_client.py:100`), so if Ollama is unreachable the event loop blocks and `/health` never serves within the 120 s `wait_for` window | Run warm-up in a background thread (or short-circuit when Ollama is unreachable); lower the warm-up timeout to ~30 s |
| **3 (masks the real failure)** | `src/app.py` | 62–64 | `/health` returns 200 even when Ollama is unreachable, so `wait_for` at `initial_deploy.sh:238` and the Docker healthcheck pass while `/ask` is broken | Have `/health` probe Ollama (`GET /api/tags`) and return 503 when unreachable, so Stage 9 fails fast at line 238 instead of cascading to 241 |
| **4 (the 502 itself)** | `src/app.py` | 88 | `raise HTTPException(status_code=502, ...)` is correct (translates upstream failure to 502) but is the source of the 502 that `scripts/smoke_test.sh:10-11` then fails to parse as JSON | Upstream fix #1 removes the 502; optionally make `scripts/smoke_test.sh` lines 10–11 tolerate non-200 before piping to `python3 -m json.tool` |
| **5 (hang risk after fix)** | `src/ollama_client.py` | 109, 139 | `timeout=600` for both warm-up and `/ask`; if the host resolves but Ollama is down, a 600 s hang blocks | Lower default `timeout` to ~30 s |

**Prometheus note:** `monitoring/prometheus.yml:9-14` scrapes `api:8000/metrics` via Docker DNS — this is correct and reachable. The "API does not seem reachable for Prometheus" symptom is actually Prometheus scraping the API but recording only the error counter (`eduassist_requests_total{status="error"}`) because the success-path histograms (latency/tokens/cost) at `src/app.py:90-100` are never observed when `/ask` 502s. Fixing #1 populates all panels.

**Grafana JSON note:** `monitoring/eduassist_dashboard.json` panels 1–7 query `eduassist_*` metrics that **match** `src/metrics.py:68-96` (`PROM_NAMESPACE="eduassist"`) — there is no name mismatch. The empty panels are a consequence of #1, not a dashboard bug. An optional enhancement is an error-rate panel: `sum(rate(eduassist_requests_total{status="error"}[1m]))`.

**Minimum fix:** #1 (add `extra_hosts`) alone makes Stage 9 pass and populates Grafana.

---

## 10. CI/CD

`.github/workflows/ci.yml` runs on every push/PR: `flake8` lint, `pytest` (25 tests), Docker image build, and `docker-compose config` + Prometheus rule validation. The `mlflow==3.16.0` pin is enforced in `requirements.txt` (and `requirements-serve.txt`); the Docker MLflow image is `ghcr.io/mlflow/mlflow:v2.16.1` in `docker-compose.yml:3`.

---

## 11. Quick troubleshooting

| Symptom | Fix |
| --- | --- |
| First prompt takes minutes | Check `ollama ps` for 100% GPU; use a smaller `--model`; confirm warm-up ran |
| `nvidia-smi` not found | Install the NVIDIA driver; WSL2 needs the Windows driver plus passthrough |
| `python3.12` not found | Install via Deadsnakes PPA (Ubuntu) or pyenv |
| MLflow UI empty | `export MLFLOW_TRACKING_URI=http://127.0.0.1:5000` and rerun the pipeline |
| Port 11434 not responding | `nohup ollama serve > logs/ollama.log 2>&1 &` |
| API 502 / Grafana panels empty | See §9 — add `extra_hosts` to `docker-compose.yml` `api` service |

---

## 12. Results summary

Across the five questions, `v3` (JSON output) achieved a `valid_json` rate of 1.0, satisfying the primary quality gate, while `v1` and `v2` returned prose that could not feed a downstream parser. `v3` produced the fewest output tokens, directly minimising the dominant output-cost term (output is priced 4× input). The projected monthly cost at 10,000 requests is approximately $0.79 using `v3`. `v3` is promoted to the `production` alias.

| Version | Avg Latency (s) | Avg Input Tok | Avg Output Tok | Avg Cost ($) | Valid JSON | Recommended |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 4.25 | 43 | 172 | 0.000110 | — | n/a |
| v2 | 2.08 | 81 | 101 | 0.000073 | — | n/a |
| v3 | 2.30 | 87 | 110 | 0.000079 | 1.00 | **yes** |

*Table: Prompt version comparison (averaged over five questions).*

---


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


*Repository: https://github.com/anmol-sinha-coder/BITS-MLOps-AIMLZG523_Assignment-2 · Executed on the BITS Prayogshala Virtual Lab · Anmol Sinha (2025AE05129) · AIMLZG523 (MLOps) · November 2026*
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
