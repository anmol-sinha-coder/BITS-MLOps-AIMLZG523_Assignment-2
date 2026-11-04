# Model Selection — Free Ollama Models for an NVIDIA RTX Stack

This document records how the default free model was chosen, why Qwen2.5:7B is
the recommended default on a "fair VRAM" RTX card, how to override it, and how
the benchmark in `model_benchmark.py` operationalises "highest evaluation
metric score amongst free models deployed".

---

## 1. The problem this solves

In Assignment-1 the MLflow screenshot showed the **first prompt taking 8–9
minutes** and subsequent prompts 2–4 minutes. The root cause was not the model
itself but three LLMOps anti-patterns:

1. **Cold model load on every run.** The weights were paged from disk into
   VRAM each time the pipeline started, so the first request paid the full
   load cost.
2. **Partial GPU offload.** On a card with insufficient VRAM, Ollama spills
   layers from VRAM into system RAM; the spilled layers run on CPU and incur
   a roughly **10× slowdown**.
3. **No warm-up or keep-alive.** The model was unloaded between requests, so
   every prompt re-paid the residency cost.

This stack fixes all three: a one-shot `warmup_model()` call before timing,
`OLLAMA_KEEP_ALIVE=30m` to keep weights resident, and `ollama_ps()` /
`detect_device()` checks that surface GPU residency in the deploy log.

---

## 2. Candidate free models

All candidates are open-weight, free, and small enough to fit an 8–12 GB VRAM
RTX card in their Q4_K_M quantisation. Edit `config/model_candidates.json` to
add or remove entries.

| Tag | Params | VRAM (Q4_K_M) | Strength | Eval note |
| --- | --- | --- | --- | --- |
| `gemma3:4b` | 4B | ~3.0 GB | Fast fallback, CPU friendly | Fits 8 GB cards |
| `qwen2.5:7b` | 7B | ~4.4 GB | Balanced, strong JSON | MMLU 74.2 |
| `qwen3:8b` | 8B | ~5.0 GB | Newest Qwen, hybrid thinking | Top open evals |
| `llama3.1:8b` | 8B | ~4.9 GB | Strong general baseline | MMLU ~73 |
| `gemma3:12b` | 12B | ~7.3 GB | Excellent instruction-following | Needs 12 GB VRAM |

> **Eval-metric disclaimer.** Public MMLU / MT-Bench / Arena scores are
> community-reported and shift between quantisations and releases; treat them
> as directional, not authoritative. The benchmark below measures the metrics
> that actually matter for this assignment **on your hardware**.

---

## 3. Why `qwen2.5:7b` is the default

`qwen2.5:7b` is the default (`config.DEFAULT_MODEL`) because it is the best
balance of quality, JSON-instruction-following and footprint for the widest
range of RTX cards:

- **Quality.** Qwen2.5-7B-Instruct posts an MMLU of ~74.2, competitive with
  Llama-3.1-8B, and is specifically tuned for structured/JSON output — directly
  relevant to the v3 prompt's strict schema.
- **VRAM.** At ~4.4 GB (Q4_K_M) it fully resides on an 8 GB RTX card with room
  for the KV cache, avoiding the CPU-spill slowdown that caused the 8–9 minute
  first prompt.
- **Speed.** On an RTX class GPU it generates ~40–50 tokens/s warm, so a
  typical ~100-token JSON answer completes in 2–3 seconds.
- **Fallback.** On a machine with no GPU it still runs on CPU (slowly), and the
  4B Gemma fallback (`gemma3:4b`) is available via `--model gemma3:4b`.

For a 12 GB+ card, `gemma3:12b` or `qwen3:8b` will score higher on quality;
run the benchmark to confirm.

---

## 4. The benchmark (`model_benchmark.py`)

"Highest evaluation metric score amongst free models deployed" is operationalised
as a composite measured **on your hardware**, not a leaderboard number. For each
candidate the benchmark measures:

- `valid_json_rate` — schema conformance on the v3 JSON prompt (quality gate).
- `tokens_per_second` — throughput (the inverse of the 8–9 min problem).
- `avg_latency` — end-to-end latency.
- `answer_length` — proxy for completeness.

The composite score is:

```
composite = 0.5 * valid_json_rate + 0.3 * norm(tokens_per_second) + 0.2 * norm(-avg_latency)
```

Weights live in `config/model_candidates.json` (`composite_weights`) and reflect
the assignment's priorities: JSON validity is the hard requirement (0.5),
throughput matters for user experience (0.3), and raw latency (0.2) is already
correlated with throughput. The winner is written to
`config/selected_model.json` and `llmops_pipeline.py` picks it up automatically.

Run it:

```bash
./initial_deploy.sh --benchmark                 # via the deploy script
# or directly:
python model_benchmark.py
python model_benchmark.py --candidates gemma3:4b qwen2.5:7b qwen3:8b
```

Results are also logged to a separate MLflow experiment
`eduassist_model_selection` so the model choice is itself a tracked artefact.

---

## 5. Overriding the model

The model is configurable in three places, in priority order:

1. `--model <tag>` on `initial_deploy.sh` (highest priority).
2. `OLLAMA_MODEL` environment variable.
3. `config/selected_model.json` written by the benchmark.
4. `config.DEFAULT_MODEL` in `src/config.py` (lowest priority, `qwen2.5:7b`).

Examples:

```bash
./initial_deploy.sh --model qwen3:8b          # newest Qwen, needs ~5 GB VRAM
./initial_deploy.sh --model gemma3:12b        # best quality, needs ~7.3 GB VRAM
./initial_deploy.sh --model gemma3:4b --cpu-only   # tiny model, CPU only
OLLAMA_MODEL=qwen2.5:7b ./initial_deploy.sh
```

---

## 6. GPU residency verification

After deploy, confirm the model is actually in VRAM (not spilled to CPU):

```bash
ollama ps                                    # PROCESSOR column should show 100% GPU
nvidia-smi                                   # VRAM should show ~4.4 GB used for qwen2.5:7b
curl -s http://localhost:8000/model-info | jq   # gpu_residency from the API
```

If `ollama ps` shows `CPU` in the processor column, the model is spilling and
you will reproduce the 8–9 minute symptom. Switch to a smaller model
(`--model gemma3:4b`) or increase VRAM headroom.
