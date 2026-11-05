#!/usr/bin/env bash
# Quick smoke test against the running LLMOps API.
#   scripts/smoke_test.sh [http://localhost:8000]
set -euo pipefail
URL="${1:-http://localhost:8000}"

echo "--> health";        curl -fsS "$URL/health" | python3 -m json.tool
echo "--> model-info";    curl -fsS "$URL/model-info" | python3 -m json.tool
echo "--> /ask (valid JSON question)"
curl -fsS -X POST "$URL/ask" -H 'Content-Type: application/json' \
  -d '{"question":"Explain backpropagation in one sentence."}' | python3 -m json.tool
echo "--> /metrics (first 5 lines)"; curl -fsS "$URL/metrics" | head -5
echo "[+] smoke test passed"
