# Deployment Guide — Linux + NVIDIA RTX (CPU fallback)

A detailed, step-by-step procedure for deploying the EduAssist LLMOps stack on
Ubuntu/WSL or any Debian/RedHat Linux with an NVIDIA RTX GPU. Every step is
also performed automatically by `initial_deploy.sh`; this guide explains what
each step does so the deployment is reproducible and debuggable.

---

## Prerequisites

- **OS:** Ubuntu 22.04+/WSL2, Debian 12+, Fedora 39+, Rocky 9+, or equivalent.
  The script auto-detects `apt` vs `dnf/yum`.
- **GPU (optional but recommended):** an NVIDIA RTX card with ≥ 8 GB VRAM and
  the NVIDIA driver installed (`nvidia-smi` must work). CUDA is bundled with
  the Ollama install; you do not need a separate CUDA toolkit.
- **RAM/CPU:** ≥ 16 GB RAM, ≥ 4 cores. The 7B model needs ~4.4 GB VRAM or
  ~8 GB system RAM on CPU.
- **Disk:** ~6 GB for the model + ~1 GB for venv and images.
- **Sudo** for package installs (the script uses `sudo` only where needed).
- **Python 3.12** — if your distro ships an older Python, install 3.12 first
  (Deadsnakes PPA on Ubuntu: `sudo add-apt-repository ppa:deadsnakes/ppa &&
  sudo apt-get install python3.12 python3.12-venv python3.12-dev`).

If any prerequisite is missing the script reports it and stops; rerun after
fixing.

---

## Step 1 — OS detection and base packages

`initial_deploy.sh` reads `/etc/os-release`, picks `dnf`/`yum` (RedHat family)
or `apt` (Debian family), and installs: `ca-certificates`, `curl`, `gnupg`,
`git`, `python3.12` (+ `-venv`/`-dev`), `build-essential`/`gcc-c++`, `make`,
`jq`. Pass `--skip-install` to skip this on a locked-down host.

Manual equivalent (Debian):

```bash
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release git \
  python3.12 python3.12-venv python3.12-dev build-essential jq
```

---

## Step 2 — Hardware detection (GPU vs CPU)

The script runs `nvidia-smi`; if it succeeds, `DEVICE_TYPE=gpu` and the GPU
name + VRAM are printed. Otherwise `DEVICE_TYPE=cpu`. Pass `--cpu-only` to
force CPU even when a GPU is present (useful for benchmarking CPU fallback).

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

---

## Step 3 — Install and start Ollama, pull the model

```bash
curl -fsSL https://ollama.com/install.sh | sh      # auto-detects CUDA
nohup ollama serve > logs/ollama.log 2>&1 &        # if not already running
ollama pull qwen2.5:7b                             # ~4.4 GB, first pull is slow
```

The script waits for `http://127.0.0.1:11434/api/tags` to respond, then pulls
the configured model. `ollama pull` is idempotent — rerunning only fetches
missing layers.

---

## Step 4 — Create the Python 3.12 virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The venv is created at `.venv` (override with `--venv`). Everything — pipeline,
notebook, tests, FastAPI, MLflow client — runs inside this venv only. No system
Python is touched.

---

## Step 5 — Run the test suite and lint

```bash
pytest -q                    # 25 unit tests: config, metrics, prompts, utils
flake8 .                     # max-line-length 100, hard fail in CI
```

CI runs the same gates (`.github/workflows/ci.yml`).

---

## Step 6 — Start MLflow tracking

Two modes, selected with `--mlflow-mode`:

- **docker (default):** `docker compose up -d mlflow` starts
  `ghcr.io/mlflow/mlflow:v2.16.1` with a sqlite backend at `http://localhost:5000`.
- **local:** `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`
  runs MLflow in the venv (no Docker needed).

The experiment `eduassist_llmops_assignment` is created automatically on first
run. Open `http://localhost:5000` to see runs, metrics, and the Prompt Registry.

---

## Step 7 — Warm the model and run the pipeline

```bash
python llmops_pipeline.py        # Tasks 1+2+3: warm-up, 5 Q × 3 versions, MLflow
```

This is the cold-start mitigation step: `warmup_model()` loads the weights into
VRAM with a trivial `ping` request (excluded from latency metrics), then the
three prompt versions are run against the five questions with real token
telemetry, and the five required metrics are logged to MLflow. The best version
is promoted to the `production` alias. Outputs are written to `outputs/`.

For a quick smoke run: `python llmops_pipeline.py --sample` (1 version × 1 Q).

---

## Step 8 — Benchmark and auto-select the best free model (optional)

```bash
./initial_deploy.sh --benchmark
# or
python model_benchmark.py --candidates gemma3:4b qwen2.5:7b qwen3:8b
```

Each candidate is measured on `valid_json_rate`, `tokens_per_second`,
`avg_latency`; the composite-score winner is written to
`config/selected_model.json` and the pipeline uses it on the next run. See
`MODEL_SELECTION.md` for the formula and weights.

---

## Step 9 — Bring up the serving + monitoring stack

```bash
docker compose up -d          # MLflow + API + Prometheus + Grafana
```

| Service | URL | Use |
| --- | --- | --- |
| FastAPI | http://localhost:8000/docs | `/ask` (v3 JSON), `/health`, `/model-info`, `/metrics` |
| MLflow | http://localhost:5000 | runs, metrics, Prompt Registry |
| Prometheus | http://localhost:9090 | scrape `/metrics`, alert rules |
| Grafana | http://localhost:3000 | dashboard (admin/admin), auto-provisioned |

The API container is built from `Dockerfile` (lean `python:3.12-slim`,
non-root user, healthcheck) and points at Ollama on the host via
`host.docker.internal`.

---

## Step 10 — Verify and generate traffic

```bash
./scripts/smoke_test.sh                 # health + one /ask round-trip
python scripts/generate_traffic.py 50   # 50 synthetic requests for dashboards
python scripts/verify_monitoring.py     # assert Prometheus targets are up
```

Prometheus alert rules (`monitoring/alert_rules.yml`) fire on high latency
(p95 > 30s), low valid-JSON rate (< 0.8), and API downtime.

---

## Re-running and resetting

```bash
./initial_deploy.sh                    # idempotent; safe to rerun
./initial_deploy.sh --reset            # wipe MLflow DB + Ollama model cache
./initial_deploy.sh --no-monitoring    # deploy infra + pipeline only
./initial_deploy.sh --no-traffic       # skip background load
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| First prompt takes minutes | Model spilled to CPU / cold load | `ollama ps` (check 100% GPU); use a smaller `--model`; confirm warm-up ran |
| `nvidia-smi` not found | Driver not installed | Install the NVIDIA driver; WSL2 needs the Windows driver + `nvidia-smi` passthrough |
| `python3.12` not found | Distros ships older Python | Install via Deadsnakes PPA (Ubuntu) or `pyenv` |
| MLflow UI empty | Wrong tracking URI | `export MLFLOW_TRACKING_URI=http://127.0.0.1:5000`; rerun pipeline |
| Port 11434 not responding | Ollama not running | `nohup ollama serve > logs/ollama.log 2>&1 &` |
| `host.docker.internal` unresolved | Old Docker | Add `extra_hosts: ["host.docker.internal:host-gateway"]` or use Docker Desktop |
