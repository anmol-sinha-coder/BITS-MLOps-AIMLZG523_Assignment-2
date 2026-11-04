#!/usr/bin/env bash
# =============================================================================
# EduAssist AI - LLMOps one-shot (re)deployment
#   Linux (Debian/Ubuntu or RedHat/Fedora/Rocky) + NVIDIA RTX GPU (CPU fallback)
#   Free Ollama model (Qwen / Gemma / Llama) - selectable & benchmarked
#   Python 3.12 venv only - MLflow tracking - FastAPI API - Prometheus/Grafana
#
#   ./initial_deploy.sh                         # full: deps, venv, tests, MLflow,
#                                               #       warm model, run pipeline, API,
#                                               #       monitoring, traffic
#   ./initial_deploy.sh --model qwen3:8b        # override the Ollama model
#   ./initial_deploy.sh --benchmark             # benchmark candidates & pick best
#   ./initial_deploy.sh --cpu-only              # force CPU (ignore GPU)
#   ./initial_deploy.sh --no-monitoring         # skip Prometheus/Grafana/API stack
#   ./initial_deploy.sh --no-traffic            # skip background traffic load
#   ./initial_deploy.sh --no-pipeline           # deploy infra only, do not eval
#   ./initial_deploy.sh --reset                 # wipe MLflow/Grafana/Ollama model cache
#   ./initial_deploy.sh --skip-install          # do not apt/dnf install anything
#   ./initial_deploy.sh --mlflow-mode local     # local sqlite MLflow (no Docker) | docker
#   ./initial_deploy.sh -h | --help
#
# Safe to re-run. Run as a normal user (sudo is used only where needed).
# =============================================================================
# set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---- defaults --------------------------------------------------------------
MODEL_NAME="${OLLAMA_MODEL:-qwen2.5:7b}"
VENV_DIR=".venv"
PYTHON_BIN="python3.12"
BENCHMARK=0; CPU_ONLY=0; MONITORING=1; TRAFFIC=1; RUN_PIPELINE=1
RESET=0; SKIP_INSTALL=0; MLFLOW_MODE="docker"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)        MODEL_NAME="${2:-$MODEL_NAME}"; shift 2 ;;
    --venv)         VENV_DIR="${2:-$VENV_DIR}"; shift 2 ;;
    --python)       PYTHON_BIN="${2:-$PYTHON_BIN}"; shift 2 ;;
    --mlflow-mode)  MLFLOW_MODE="${2:-docker}"; shift 2 ;;
    --model=*)      MODEL_NAME="${1#*=}"; shift ;;
    --venv=*)       VENV_DIR="${1#*=}"; shift ;;
    --python=*)     PYTHON_BIN="${1#*=}"; shift ;;
    --mlflow-mode=*) MLFLOW_MODE="${1#*=}"; shift ;;
    --benchmark)    BENCHMARK=1; shift ;;
    --cpu-only)     CPU_ONLY=1; shift ;;
    --no-monitoring) MONITORING=0; shift ;;
    --no-traffic)   TRAFFIC=0; shift ;;
    --no-pipeline)  RUN_PIPELINE=0; shift ;;
    --reset)        RESET=1; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help)      sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

step() { echo; echo "--> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
trap 'echo "ERROR: deployment failed near line ${LINENO} (exit $?)" >&2' ERR

wait_for() {  # wait_for <desc> <timeout-s> <cmd...>
  local what="$1" limit="$2" w=0; shift 2
  until "$@" >/dev/null 2>&1; do
    sleep 2; w=$((w + 2))
    [ "$w" -ge "$limit" ] && die "timed out after ${limit}s waiting for ${what}"
  done
}

echo "=================================================="
echo " EduAssist AI - LLMOps stack deployment"
echo "   model=${MODEL_NAME}  mlflow=${MLFLOW_MODE}  monitoring=${MONITORING}"
echo "=================================================="

# ---- [1/9] OS detection + system packages ---------------------------------
if [ "$SKIP_INSTALL" -eq 0 ]; then
  step "[1/9] Detecting OS and installing base packages"
  [ -f /etc/os-release ] || die "/etc/os-release not found; cannot detect distribution"
  . /etc/os-release
  OS_ID="${ID:-}"; OS_LIKE="${ID_LIKE:-}"

  if command -v dnf >/dev/null 2>&1; then PKGMGR="dnf"
  elif command -v yum >/dev/null 2>&1; then PKGMGR="yum"
  elif command -v apt-get >/dev/null 2>&1 || command -v apt >/dev/null 2>&1; then PKGMGR="apt"
  else die "No supported package manager (dnf/yum/apt). Use --skip-install."; fi

  if [ "$PKGMGR" = "dnf" ] || [ "$PKGMGR" = "yum" ]; then
    echo "RedHat-family OS detected ($OS_ID). Using $PKGMGR."
    sudo "$PKGMGR" install -y ca-certificates curl gnupg2 git \
      "$PYTHON_BIN" "${PYTHON_BIN}-devel" gcc gcc-c++ make jq
  else
    echo "Debian/Ubuntu-family OS detected ($OS_ID). Using apt."
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl gnupg lsb-release git \
      "$PYTHON_BIN" "${PYTHON_BIN}-venv" "${PYTHON_BIN}-dev" build-essential jq
  fi
else
  step "[1/9] Skipping system package installation (--skip-install)"
fi

# ---- [2/9] Hardware detection (GPU vs CPU) --------------------------------
step "[2/9] Detecting hardware (NVIDIA GPU vs CPU)"
DEVICE_TYPE="cpu"; GPU_INFO=""
if [ "$CPU_ONLY" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  DEVICE_TYPE="gpu"
  GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  echo "[+] NVIDIA GPU detected: $GPU_INFO"
else
  echo "[!] No NVIDIA GPU / CUDA (--cpu-only or none found). CPU execution mode."
fi
export DEVICE_TYPE

# ---- [3/9] Ollama install + model pull ------------------------------------
step "[3/9] Installing / starting Ollama and pulling model ${MODEL_NAME}"
if ! command -v ollama >/dev/null 2>&1; then
  echo "[+] Installing Ollama ..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "[+] Ollama already installed: $(ollama --version 2>&1 | head -1)"
fi

# GPU runtime for Ollama (only needed if Ollama runs in a container later; the
# bare-metal install above auto-detects CUDA). Ensure the service is up.
if ! pgrep -x ollama >/dev/null 2>&1; then
  echo "[+] Starting ollama serve in background ..."
  mkdir -p logs
  nohup ollama serve > logs/ollama.log 2>&1 &
  echo $! > logs/ollama.pid
  sleep 4
fi
wait_for "Ollama API (port 11434)" 60 curl -fsS http://127.0.0.1:11434/api/tags

if [ "$RESET" -eq 1 ]; then
  echo "[+] --reset: removing cached model $MODEL_NAME"
  ollama rm "$MODEL_NAME" 2>/dev/null || true
fi
echo "[+] Pulling model ${MODEL_NAME} (first pull may take several minutes) ..."
ollama pull "$MODEL_NAME"

# Verify GPU residency after load (the #1 cause of the 8-9 min first-prompt
# latency is the model spilling from VRAM into system RAM).
echo "[+] ollama ps (verify GPU residency):"
ollama ps || true
export OLLAMA_MODEL="$MODEL_NAME"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"   # keep model warm in VRAM

# ---- [4/9] Docker (for MLflow + monitoring stack) -------------------------
DOCKER="docker"
if [ "$MONITORING" -eq 1 ] || [ "$MLFLOW_MODE" = "docker" ]; then
  step "[4/9] Ensuring Docker Engine + Compose plugin"
  if ! command -v docker >/dev/null 2>&1; then
    if [ "$SKIP_INSTALL" -eq 1 ]; then
      echo "[!] Docker missing and --skip-install set; switching MLflow to local mode."
      MLFLOW_MODE="local"; MONITORING=0
    else
      if [ "$PKGMGR" = "dnf" ] || [ "$PKGMGR" = "yum" ]; then
        case "$OS_ID" in fedora) D=docker ;; rhel|centos|rocky|almalinux|ol) D=rhel ;; *) D=centos ;; esac
        sudo "$PKGMGR" install -y yum-utils
        sudo yum-config-manager --add-repo "https://download.docker.com/linux/${D}/docker-ce.repo" \
          || sudo dnf config-manager --add-repo "https://download.docker.com/linux/${D}/docker-ce.repo"
        sudo "$PKGMGR" install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
      else
        case "$OS_ID" in ubuntu|debian) D="$OS_ID" ;; *) D=ubuntu ;; esac
        sudo install -m0755 -d /etc/apt/keyrings
        curl -fsSL "https://download.docker.com/linux/${D}/gpg" | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${D} $(lsb_release -cs) stable" \
          | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
        sudo apt-get update -y
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
      fi
      sudo usermod -aG docker "$USER" || true
      sudo systemctl enable --now docker || true
    fi
  else
    echo "[+] Docker present: $(docker --version)"
  fi
  if ! docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
  fi
  $DOCKER info >/dev/null 2>&1 || { echo "[!] Docker daemon down; MLflow -> local mode."; MLFLOW_MODE="local"; MONITORING=0; }
else
  step "[4/9] Docker not required (--no-monitoring + local MLflow)"
fi

# ---- [5/9] Python 3.12 venv + dependencies --------------------------------
step "[5/9] Python 3.12 virtual environment + dependencies"
"$PYTHON_BIN" --version 2>&1 | grep -q "Python 3.12" || die "$PYTHON_BIN is not Python 3.12 (required)."
[ -d "$VENV_DIR" ] || "$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet -r requirements.txt

# ---- [6/9] Quality gate ---------------------------------------------------
step "[6/9] Lint + unit tests (deployment aborts on failure)"
flake8 . || die "flake8 failed"
pytest -q || die "pytest failed"

# ---- [7/9] MLflow tracking server -----------------------------------------
step "[7/9] Starting MLflow tracking server (mode=${MLFLOW_MODE})"
mkdir -p logs outputs
if [ "$MLFLOW_MODE" = "docker" ] && [ "$DOCKER" != "" ] && $DOCKER info >/dev/null 2>&1; then
  if [ "$RESET" -eq 1 ]; then $DOCKER compose down -v --remove-orphans 2>/dev/null || true; fi
  $DOCKER compose up -d mlflow
  wait_for "MLflow (port 5000)" 120 curl -fsS http://localhost:5000/health
  export MLFLOW_TRACKING_URI="http://localhost:5000"
else
  echo "[+] Running MLflow locally (sqlite backend, no Docker)."
  if [ "$RESET" -eq 1 ]; then rm -f mlflow.db; fi
  nohup mlflow ui --backend-store-uri sqlite:///mlflow.db \
       --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 \
       > logs/mlflow.log 2>&1 &
  echo $! > logs/mlflow.pid
  wait_for "MLflow (port 5000)" 90 curl -fsS http://localhost:5000
  export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
fi

# ---- [8/9] Model benchmark (optional) + pipeline --------------------------
if [ "$BENCHMARK" -eq 1 ]; then
  step "[8/9] Benchmarking candidate models and selecting the best"
  python model_benchmark.py || die "model benchmark failed"
  SELECTED="$(python -c "import json;print(json.load(open('config/selected_model.json'))['selected_model'])" 2>/dev/null || echo "$MODEL_NAME")"
  echo "[+] Benchmark selected: $SELECTED"
  export OLLAMA_MODEL="$SELECTED"; MODEL_NAME="$SELECTED"
fi

if [ "$RUN_PIPELINE" -eq 1 ]; then
  step "[8/9] Running the LLMOps pipeline (Tasks 1, 2, 3)"
  python llmops_pipeline.py || die "LLMOps pipeline failed"
  echo "[+] Outputs: outputs/latency_tokens_cost.csv, outputs/prompt_evaluation_results.csv"
else
  step "[8/9] Pipeline skipped (--no-pipeline)"
fi

# ---- [9/9] API + monitoring stack + traffic -------------------------------
if [ "$MONITORING" -eq 1 ] && $DOCKER info >/dev/null 2>&1; then
  step "[9/9] Building image and starting API, Prometheus, Grafana"
  export OLLAMA_HOST_FOR_API="http://host.docker.internal:11434"
  $DOCKER compose up -d --build
  wait_for "API (port 8000)" 120 curl -fsS http://localhost:8000/health
  wait_for "Prometheus (port 9090)" 90 curl -fsS http://localhost:9090/-/ready
  wait_for "Grafana (port 3000)" 120 curl -fsS http://localhost:3000/api/health
  scripts/smoke_test.sh http://localhost:8000

  if [ "$TRAFFIC" -eq 1 ]; then
    echo "[+] Starting background traffic (LLM load) ..."
    nohup python scripts/generate_traffic.py --url http://localhost:8000 --duration 120 \
      > logs/traffic.log 2>&1 &
    echo $! > logs/traffic.pid
    echo "    traffic PID $(cat logs/traffic.pid); sleeping 45s for Prometheus scrape ..."; sleep 45
    python scripts/verify_monitoring.py || echo "WARNING: some panels empty - see output"
  fi
else
  step "[9/9] Monitoring stack skipped (--no-monitoring or Docker unavailable)"
fi

cat <<SUMMARY

==================================================
 LLMOps stack deployed
==================================================
 Device       : ${DEVICE_TYPE} ${GPU_INFO}
 Ollama model : ${MODEL_NAME}    (OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE})
 MLflow UI    : ${MLFLOW_TRACKING_URI}
 API docs     : http://localhost:8000/docs           (if monitoring enabled)
 Prometheus   : http://localhost:9090                (if monitoring enabled)
 Grafana      : http://localhost:3000  (admin/admin) (if monitoring enabled)
 CSV outputs  : outputs/latency_tokens_cost.csv
               outputs/prompt_evaluation_results.csv
 Notebook     : jupyter lab MLOps_Assignment-2.ipynb
 Stop         : docker compose down (add -v to wipe MLflow/Grafana data)
==================================================
SUMMARY
