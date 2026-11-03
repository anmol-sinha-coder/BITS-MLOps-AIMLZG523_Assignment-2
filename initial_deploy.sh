#!/usr/bin/env bash
set -e

# Default settings
MODEL_NAME="qwen2.5:7b"
VENV_DIR=".venv"
PYTHON_BIN="python3.12"

# Parse CLI Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift ;;
        --venv) VENV_DIR="$2"; shift ;;
        --python) PYTHON_BIN="$2"; shift ;;
        -h|--help)
            echo "Usage: ./initial_deploy.sh [--model MODEL_NAME] [--venv VENV_PATH] [--python PYTHON_BIN]"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo " Starting Automated LLMOps Environment Setup"
echo "=========================================="

# 1. Package Manager Detection & System Dependencies
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt"
    echo "[+] Detected Debian/Ubuntu based OS (apt)"
    sudo apt-get update -y
    sudo apt-get install -y curl git python3.12 python3.12-venv python3.12-dev build-essential
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    echo "[+] Detected RedHat/Fedora/CentOS based OS (dnf)"
    sudo dnf check-update || true
    sudo dnf install -y curl git python3.12 python3.12-devel gcc gcc-c++ make
else
    echo "[-] Unsupported package manager. Please install dependencies manually."
    exit 1
fi

# 2. Hardware Detection (GPU vs CPU)
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "[+] NVIDIA GPU detected with CUDA drivers:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DEVICE_TYPE="gpu"
else
    echo "[!] No NVIDIA GPU/CUDA detected. Falling back to CPU execution mode."
    DEVICE_TYPE="cpu"
fi

# 3. Ollama Installation & Service Initialization
if ! command -v ollama &> /dev/null; then
    echo "[+] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[+] Ollama is already installed."
fi

# Start Ollama service in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "[+] Starting Ollama service in background..."
    ollama serve > /dev/null 2>&1 &
    sleep 5
fi

# Pull configured model
echo "[+] Pulling Ollama Model: ${MODEL_NAME}..."
ollama pull "${MODEL_NAME}"

# 4. Python 3.12 Virtual Environment Setup
if [ ! -d "$VENV_DIR" ]; then
    echo "[+] Creating Python 3.12 Virtual Environment at ${VENV_DIR}..."
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

echo "[+] Activating Virtual Environment and installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "=========================================="
echo "[✓] Environment Setup Complete!"
echo "[✓] Hardware Mode : ${DEVICE_TYPE}"
echo "[✓] Ollama Model  : ${MODEL_NAME}"
echo "[✓] Python Env    : ${VENV_DIR} (Python 3.12)"
echo "=========================================="
