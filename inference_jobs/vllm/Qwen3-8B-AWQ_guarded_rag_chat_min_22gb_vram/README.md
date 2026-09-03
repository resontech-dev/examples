# Guarded RAG chat — Qwen3-8B + Qwen3Guard + embed + rerank (vLLM)

The **full user-facing stack on one GPU**, Apache-clean end to end:
Qwen3-8B chat, Qwen3Guard-Gen-8B moderation (no Llama license), and the
RAG sidecars — four vLLM engines behind one OpenAI endpoint. This is what
you put in front of *the customer's own users*: every message is guarded
in, retrieval-grounded, and guarded out.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 22 → 24 GB | 4 | 48 GB | ~46 GB (image 15 + 1.3×24 GB checkpoints) | 24gb |

Suggested GPUs: RTX 3090 / 4090 / L4 / A10G (24 GB — as configured, sum
of engine fractions 0.84); RTX 5090 (32 GB) relaxes every budget.

## The flow (`predict.py` wires it end to end)

```
user msg → guard("user")   → Unsafe? refuse before spending chat tokens
         → embed + rerank  → top-3 context chunks
         → assistant       → grounded answer
         → guard("assistant") → Unsafe? suppress before returning
```

The guard is a generative classifier: post the content as a chat message
to model `"guard"`, parse the Safe/Unsafe verdict. Exact output format is
defined by the Qwen3Guard model card — `predict.py` parses leniently.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py "When are production deployments frozen?"
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Memory rule

Four engines pre-allocate at boot — fractions must sum to ~0.85. Shipped:
chat 0.32 / guard 0.40 / embed 0.06 / rerank 0.06 = 0.84. The guard is
fp16→fp8-quantized at load (`quantization="fp8"`; Marlin path on Ampere);
swap in a vetted AWQ checkpoint if one appears and drop that kwarg.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| the four `gpu_memory_utilization` values | rebalance — keep the sum ≤ 0.85 |
| chat `hf_overrides` (commented YaRN) | 131k context — only with a bigger card / fewer engines |
| guard threshold behavior (client-side) | strictness lives in how you act on verdicts |
| chat `model_source` | any small chat model; the guard stays |

## Why this bundle exists

OpenAI's moderation API is free but requires sending data out — a
non-starter for the privacy buyers this platform targets. In-weights
refusals are brittle (jailbreaks work); a separate guard model in front
is the layer that actually holds, and at 0.4 of a shared GPU it's a
near-zero marginal cost platform add-on.
