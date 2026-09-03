# Qwen3-Coder-30B with pipeline parallelism (vLLM)

One replica split into **2 pipeline stages across 2 single-GPU
workers** — the Stage-0 shape from
`reson_docs/docs/inference/vllm/plan16-jul-2026.md`. Use PP when the
model doesn't fit one GPU and your GPUs live in **different machines**
(the standard shape of our worker fleet).

Full knob reference: `reson_docs/docs/inference/vllm/llmconfig.md`.
Shared vLLM contract & defaults policy: [../README.md](../README.md).

## Resources

| VRAM min→rec (total) | CPU | RAM | Min free disk per worker | Workers |
|---|---|---|---|---|
| 20 → 24 GB | 4 | 48 GB | ~40 GB (image 15 + 1.3×18 GB checkpoint — **each** PP worker downloads the full snapshot) | 2 (one per stage) |

Suggested GPUs (fleet matrix, `dc-hardware-requirements.md §2`): 2× 12 GB
cards (RTX 5070 / 4070 / 3060-12G) as built here; a single 24 GB card
(RTX 3090/4090, L4) runs it with PP=1 and a bigger KV pool; 2×24 GB in one
box → prefer TP=2 (`STRICT_PACK`) for lower latency.

## Agentic use / tool calling

`enable_auto_tool_choice=True` + `tool_call_parser="hermes"` are on in
`engine_kwargs` — Qwen coder models speak hermes-style tool calls, so
Aider / Cline / Continue / OpenHands function calling works against this
endpoint out of the box (`python predict.py --tools` for a smoke test).

## How vLLM examples differ from the predictor examples

This example doesn't follow the `predict(data: bytes) -> dict` contract.
It builds a Ray Serve app via `ray.serve.llm.build_openai_app` directly,
so the deployed cluster speaks the OpenAI API natively:

| Route                          | What it does                                      |
|--------------------------------|---------------------------------------------------|
| `POST /v1/chat/completions`    | chat completions with system + user messages      |
| `POST /v1/completions`         | classic completions (no chat template)            |
| `GET  /v1/models`              | lists `model_id`s the cluster currently serves    |

Any OpenAI-compatible client (openai-python, curl, langchain, Aider,
Cline, Continue) targets this cluster without code changes — point
`base_url` at the deployment endpoint + `/v1`.

## PP vs TP in 20 seconds

| Scenario                                                       | Right tool |
|----------------------------------------------------------------|------------|
| Model too big for one GPU; GPUs in **different machines**       | **PP — this example** |
| Model too big for one GPU; **one machine** has ≥2 GPUs          | TP (`tensor_parallel_size=2`, `STRICT_PACK`) |
| Model fits one GPU; you want throughput                        | DP — more replicas, one per worker |

- **PP** = layers split into stages, one stage per worker; each token
  crosses the network **once per stage boundary** (a few KB) — fine
  over Ethernet/WireGuard.
- **TP** = every layer's weights split across GPUs; they exchange data
  ~100× per token — dies over any network, PCIe/NVLink only.

With PP the platform invariant changes shape: **1 worker = 1 stage**
(not 1 replica). This replica consumes `pipeline_parallel_size = 2`
workers; the yaml declares that so the scheduler reserves a pair.

## The yaml ↔ Python sync rule

`inference.yaml`'s `vllm:` block and `serve_module.py`'s
`engine_kwargs` carry the **same two keys with the same names**:

```
vllm.tensor_parallel_size    == engine_kwargs.tensor_parallel_size
vllm.pipeline_parallel_size  == engine_kwargs.pipeline_parallel_size
```

yaml is what the scheduler reserves hardware from; Python is what vLLM
enforces at boot. The SDK refuses to submit on mismatch. Both are
**integers ≥ 1** — no fractions (0.5 GPUs don't exist), and TP must
divide the model's attention-head count (stick to 1, 2, 4, 8).

## Use it

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
# Deploy through the platform (from this folder):
#   inference.yaml declares: PP=2, 2 workers, ≥11 GB free VRAM each
cp .env.example .env        # RESON_API_KEY (rsk_…) + S3 keys
python submit.py            # prints RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY

# Talk to it (openai client, streamed):
python predict.py "Write a binary search in Python."
```

Or paste `scripts/serve_module.py` + `inference.yaml` into the web wizard at
<https://beta.reson.tech/dashboard/inference/submit> — same result.

## What to change

| where (`serve_module.py`)                    | for what                                              |
|----------------------------------------------|-------------------------------------------------------|
| `model_loading_config.model_source`          | any HF causal-LM repo (AWQ/4-bit builds for 12 GB cards) |
| `engine_kwargs.pipeline_parallel_size`       | stage count = workers per replica (update yaml too!)  |
| `engine_kwargs.max_model_len`                | context window — the #1 KV-cache lever                |
| `engine_kwargs.max_num_seqs`                 | concurrent sequences; keeps the pipeline fed          |
| `deployment_config.autoscaling.max_replicas` | DP scale-out; each replica needs PP more workers      |

## Memory budget

VRAM per stage ≈ `(weights ÷ PP) + KV share + ~1 GB overhead`.

Qwen3-Coder-30B-A3B AWQ (~18 GB) at PP=2: ~9 GB weights per worker
+ ~1.5 GB KV (fp8, `max_num_seqs=8`, `max_model_len=16384`) + overhead
≈ 11 GB → fits a 12 GB card at `gpu_memory_utilization=0.88`. That's
where the yaml's `min_vram_gb_per_worker: 11` comes from.

CUDA OOM at boot? Drop `max_num_seqs` first, then `max_model_len` —
both shrink KV linearly. Single-GPU fallback (no second worker): see
the comment in `serve_module.py` — PP=1 + Qwen2.5-Coder-7B-AWQ.
