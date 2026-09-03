# GLM-4.5-Air 106B-A12B (AWQ) — premium chat, TP=4 (vLLM)

The premium agentic/tool-use tier (`chat-glm45-air`, **MIT**): 106B MoE
with 12B active — big-model intelligence at small-model decode cost,
agentic benchmarks in genuine Sonnet-4 territory. AWQ ~60 GB sharded
**TP=4 across 4×24 GB GPUs in one machine**.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 62 → 96 GB | 8 | 192 GB | ~98 GB (image 15 + 1.3×60 GB checkpoint) | 4x24gb |

Suggested GPUs: a single box with 4× RTX 3090 / 4090 (consumer) or 4× L4 /
A10G (datacenter). A single 80 GB card (A100-80G / H100) also fits it at
TP=1 — simpler and faster if the fleet has one. **TP never spans machines**
— the all-reduce per layer dies on Ethernet; that's what `STRICT_PACK`
enforces.

## Deploy

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py            # first boot downloads ~60 GB — mirror to S3 for production
python predict.py "Plan the steps to migrate a monolith's billing module to a service."
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Tool calling / agentic use

This model's *raison d'être* is agentic tool use — but the dedicated
`glm45` parser exists only in newer vLLM builds, so the flags ship
**commented out**. Run the parser check from [../README.md](../README.md),
then uncomment `enable_auto_tool_choice` + `tool_call_parser="glm45"`.
`python predict.py --tools` smoke-tests the round trip once enabled.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `model_source` | ⚠️ community AWQ — **vet before selling**; official fp8 needs 2×80 GB |
| `max_model_len` | 65536 shipped; the pooled KV affords 128k if `max_num_seqs` drops |
| `tensor_parallel_size` | must divide attention heads — 1, 2, 4, 8 (update yaml too!) |

## Notes

- MoE decode = ~12B active: expect near-24B-class token latency despite
  the 106B brain. Prefill is where the 4-way shard earns its keep.
- Each PP/TP layout change is a yaml + engine_kwargs edit — the folder
  shape never changes.
