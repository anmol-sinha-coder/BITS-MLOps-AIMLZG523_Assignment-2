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
