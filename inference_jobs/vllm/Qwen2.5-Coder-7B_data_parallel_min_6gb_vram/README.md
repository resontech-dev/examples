# Qwen2.5-Coder-7B — data parallel ×2 (vLLM)

**Two independent replicas of a 7B AWQ coder, one per single-GPU worker.**
Use DP when the model fits one GPU and you want throughput and availability —
each replica is a full copy, there is zero inter-worker traffic, and one
replica survives a worker loss. This is the simplest vLLM shape on the fleet
and the one that exercises the platform's own scaling story
(replica-per-worker, router fan-out, per-replica health).

Shared vLLM contract (yaml ↔ Python sync rule, requirements policy, auth,
OpenAI routes): see [../README.md](../README.md). Full knob reference:
`reson_docs/docs/inference/vllm/llmconfig.md`.

## Resources

| VRAM min→rec (per worker) | CPU | RAM | Min free disk | Workers |
|---|---|---|---|---|
| 6 → 8 GB | 4 | 16 GB | ~21 GB (image 15 + 1.3×4.9 GB checkpoint) | 2 (one per replica) |

Suggested GPUs (fleet matrix, `dc-hardware-requirements.md §2`): any 12 GB+
card — RTX 5070 / 4070 / 3060-12G, T4 16 GB, L4. On 16–24 GB cards raise
`max_model_len` and `max_num_seqs` — the KV pool is the only thing that grows.

## Folder layout

```
.
├── inference.yaml          # scheduling declaration (TP=1/PP=1, 2 workers)
├── scripts/
│   └── serve_module.py     # LLMConfig + build_openai_app — exports `app`
├── submit.py               # deploy via SDK
├── predict.py              # talk to it with the openai client
└── .env.example            # copy to .env, fill in credentials
```

## Deploy

```bash
cp .env.example .env        # RESON_API_KEY (rsk_…) + S3 keys
python submit.py            # prints RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY
python predict.py "Write a binary search in Python."
```

Or paste `scripts/serve_module.py` + `inference.yaml` into the web wizard at
<https://beta.reson.tech/dashboard/inference/submit> — same result.

## What to change

| where (`serve_module.py`)                      | for what                                            |
|------------------------------------------------|-----------------------------------------------------|
| `model_loading_config.model_source`            | any HF causal-LM that fits one GPU (AWQ for 12 GB)  |
| `autoscaling_config.min/max_replicas`          | DP width — must stay ≤ `cluster.num_workers`        |
| `engine_kwargs.max_model_len`                  | context window — the #1 KV-cache lever              |
| `engine_kwargs.max_num_seqs`                   | concurrent sequences per replica                    |

## Memory budget

Per replica: ~4.9 GB AWQ weights + KV + ~1 GB overhead. On a 12 GB card at
`gpu_memory_utilization=0.88`: 10.6 − 4.9 − ~0.7 ≈ **5 GB KV pool**. Qwen2.5-7B
KV is ~28.5 KB/token at fp8 → ~175k cached tokens — `max_model_len=16384` ×
`max_num_seqs=16` schedules comfortably. `min_vram_gb_per_worker: 7` in the
yaml is that math at the minimum (6 GB usable + headroom).

## Agentic use / tool calling

`enable_auto_tool_choice=True` + `tool_call_parser="hermes"` are on — the
Qwen family speaks hermes-style tool calls. Point Aider / Cline / Continue /
LangChain at the endpoint and function calling works out of the box
(`python predict.py --tools` for a smoke test).

## Scale-up path

Same folder shape, bigger model: see the sibling
`Qwen3-Coder-30B-A3B…pipeline_parallel` example (30B MoE split across 2
workers with PP) — swapping between DP and PP is a yaml + engine_kwargs
change, not a rewrite.
