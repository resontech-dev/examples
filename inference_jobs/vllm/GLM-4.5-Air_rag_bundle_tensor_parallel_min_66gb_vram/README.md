# GLM-4.5-Air RAG bundle — TP=4 chat + embed + rerank (vLLM)

The premium private-RAG deployment: GLM-4.5-Air (106B-A12B, MIT) sharded
TP=4 across a 4-GPU machine, with Qwen3-Embedding-0.6B and
Qwen3-Reranker-0.6B riding along on the same GPUs. One endpoint, three
engines — the top-tier version of the gpt-oss/Mistral bundles.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 66 → 96 GB | 8 | 192 GB | ~98 GB (image 15 + 1.3×62 GB checkpoints) | 4x24gb |

Suggested GPUs: one box with 4× RTX 3090 / 4090, or 4× L4 / A10G. A single
A100-80G / H100 fits the whole bundle at TP=1 with simpler placement.

## Memory model (read before editing)

The chat engine claims its `gpu_memory_utilization` fraction on **all 4
GPUs**; each 0.6B sidecar is TP=1 and lands on **one** of them — and
placement does not guarantee the two sidecars spread. Budget for both on
one GPU: shipped worst case 0.70 + 0.06 + 0.06 = 0.82. Keep any edit
under ~0.85 per GPU.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py            # ~60 GB first-boot download — mirror to S3 for production
python predict.py "Where is customer data stored?"
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| chat `model_source` | ⚠️ community AWQ — **vet before selling** |
| chat `gpu_memory_utilization` | 0.70 shipped; raise only if sidecars move off-box |
| `tool_call_parser="glm45"` | commented — VERIFY on the pinned image, then enable |
| rerank `hf_overrides` | required as-is for Qwen3-Reranker |
