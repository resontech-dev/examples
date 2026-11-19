# ⚠️ TEMPLATE: GLM-5.3, TP=8 on one 8× H200 node (text)

**Needs the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Not
deployable on the current fleet.** Nothing here has booted; every number is
napkin math.

Z.ai's 743B mixture-of-experts (39B active), sparse attention, 1M context,
text only. Terminal-Bench 2.1 88.2: built for long agentic and coding work.
Thinking is always on. Licence "glm-5.3": MIT-like, with one condition for
model-as-a-service providers above $10B yearly revenue.

**GLM-5.3 runs on vLLM 0.29.0.** The `glm5_next` tokenizer that needs vLLM
0.34 belongs to GLM-5.3-Flash, a different model; GLM-5.3 uses the GLM-5.2
architecture, which 0.29 registers.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources (planning numbers, unvalidated)

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 1,128 GB (8× H200) → 1,440 GB (8× B200) | 32 | 1 TB | ~1 TB local NVMe (image + 1.3×755.7 GB) | 8xh200 / 8xb200 |

| Node | Build | KV after weights |
|---|---|---|
| 8× H200, NVLink | FP8, as shipped | ~23 GB per GPU; capped at 262k context |
| 8× B200 | FP8, `kv_cache_dtype="fp8_e4m3"` | ~62 GB per GPU, full 1M |
| 8× H100 or 4× H200 | community `gpustack/GLM-5.3-W4A8` (400 GB, Hopper only) | estimate ~300k / ~400k tokens; VERIFY the build |
| Blackwell, smaller | `Inferact/GLM-5.3-NVFP4` (465 GB) | |

## Before first deploy

1. **DeepGEMM in the image**: `python -c "import vllm.third_party.deep_gemm"`
   must succeed; the indexer raises at init without it.
2. Pre-stage the checkpoint on NVMe with checksums.
3. `vllm serve --help | grep -A3 -E 'tool-call-parser|reasoning-parser'`
   lists `glm47`.

## Client notes

Model id `glm-5.3`. Thinking is always on, so budget output tokens for it.
Per request through `chat_template_kwargs`: `{"reasoning_effort": "low"}` or
`"high"` (default `max`), `{"clear_thinking": true}` for plain chat. The
platform cuts a request after 600 s, counted from its start, streams included (set on the Ray head node; the central Caddy proxy has no timeout of its own);
the longest `max`-effort runs can still hit it.

## Deploy (once the node exists)

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                       # effort low
python predict.py --think               # effort high
python predict.py --tools
```

## Variants

| Variant | Edits |
|---|---|
| MTP | `speculative_config={"method": "mtp", "num_speculative_tokens": 5}` (recipe); test tool calls with it on |
| GLM-5.2 `zai-org/GLM-5.2-FP8` (755.6 GB, MIT) | same flags and hardware; needs vLLM ≥ 0.24; `{"enable_thinking": false}` works there. The bare `zai-org/GLM-5.2` id is BF16 at 1.5 TB |
