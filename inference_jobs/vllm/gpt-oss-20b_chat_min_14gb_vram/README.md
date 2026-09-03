# gpt-oss-20b — business chat (vLLM)

OpenAI's open-weights 21B MoE (3.6B active), shipped natively in MXFP4
(~13 GB). The catalog's **default business assistant** (`chat-gptoss-20b`,
priority v1): o-mini-class reasoning with an adjustable effort knob and
128k context, with full data residency. Apache-2.0.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 14 → 16 GB | 4 | 32 GB | ~39 GB (image 15 + 1.3×13 GB checkpoint) | 24gb / 2x12gb |

Suggested GPUs (`dc-hardware-requirements.md §2`): RTX 3090 / 4090 / L4 /
A10G (24 GB — as configured here, 64k context); RTX 4080 / 5080 / 5070 Ti
(16 GB — fits at reduced `max_model_len` ≈ 8–16k and `gpu_memory_utilization=0.92`);
2× 12 GB cards → PP=2 (copy the yaml/`engine_kwargs` shape from the sibling
`Qwen3-Coder-30B…pipeline_parallel` example).

## Deploy

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env        # RESON_API_KEY (rsk_…) + S3 keys
python submit.py            # prints RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY
python predict.py "Draft a polite payment reminder for a 30-day overdue invoice."
```

Or paste `scripts/serve_module.py` + `inference.yaml` into the web wizard at
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `engine_kwargs.max_model_len` | 65536 shipped; up to 131072 (native) on a 24 GB card |
| `engine_kwargs.max_num_seqs` | concurrent sequences; KV is cheap on this model |
| system prompt (client-side) | `Reasoning: low\|medium\|high` — gpt-oss effort knob |
| `autoscaling_config` | DP scale-out — raise together with `cluster.num_workers` |

## Tool calling / agentic use

gpt-oss speaks its own **harmony** format and vLLM handles it natively —
but whether the pinned image wants an explicit `tool_call_parser` flag is
**unverified**, so the flags ship commented out in `serve_module.py`. Run
the check from [../README.md](../README.md) ("Tool calling"), then
uncomment. `python predict.py --tools` smoke-tests the round trip.

## Notes

- **Alignment**: strongest of the open models we list — went through
  OpenAI's preparedness process; respects system > developer > user
  instruction hierarchy. Still no moderation layer — pair with the guard
  bundle for user-facing endpoints.
- Prefix caching is on: multi-turn chats only pay for the new tail, and
  the cache is ephemeral GPU memory — the stateless privacy story holds.
