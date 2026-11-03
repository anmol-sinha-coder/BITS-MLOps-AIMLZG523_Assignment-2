# EduAssist AI - LLMOps Prompt Engineering & Evaluation Pipeline

## 1. System Requirements & Setup
- **Operating System:** Linux (Debian/Ubuntu or RedHat/Fedora)
- **Python Version:** 3.12
- **Hardware:** NVIDIA GPU with CUDA drivers (Auto-fallback to CPU supported)
- **Inference Server:** Ollama (`qwen2.5:7b` / `gemma2:9b`)
- **Tracking & Governance:** MLflow 2.16+

### Quickstart Execution
```bash
chmod +x initial_deploy.sh
./initial_deploy.sh --model qwen2.5:7b
source .venv/bin/activate
python llmops_pipeline.py
mlflow ui --host 0.0.0.0 --port 5000
