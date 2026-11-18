# ⚠️ TEMPLATE: DeepSeek V4-Flash, TP=4 + expert parallel on one 4-GPU node

**Needs the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Not
deployable on the consumer fleet.** Prod runs Ray 2.55.1 / vLLM 0.18.0
until the switch described in
`reson_docs/docs/big_models_experiments/ray2.60.0-upgrade.md`. Nothing here has
booted; every number is napkin math.

284B mixture-of-experts, 13B active per token, 1M context, text only, MIT.
GPQA 88.1, HLE 34.8 (45.1 with tools), SWE-bench Verified 79.0,
LiveCodeBench 91.6. Top-tier reasoning and code on half a node, with a KV
cache so small (~3-5 KB per token) that context is never the limit.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources (planning numbers, unvalidated)

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 320 GB (4× H100) → 564 GB (4× H200) | 32 | 256 GB | ~250 GB local NVMe (image + 1.3×167 GB) | 4xh200 / 4x80gb / 8x80gb |

| Node | Shape | KV per GPU after weights |
|---|---|---|
| 4× H200, NVLink | TP=4 + EP, as shipped | ~80 GB |
| 8× H100, NVLink | TP=8 + EP (`tensor_parallel_size=8` in both files, `gpus_per_worker: 8`, `min_vram_gb_per_worker: 600`) | ~47 GB |
| 4× H100, NVLink | TP=4 + EP | ~26 GB; reported working by the community (vLLM #49101) |
| 4× B200 | TP=4 + EP plus `attention_config={"indexer_kv_dtype": "mxfp4"}` (recipe, Blackwell) | VERIFY |

The vLLM recipe's throughput layout is data-parallel 4 with expert
parallelism. Through Ray Serve LLM that is a data-parallel deployment, not
a TP change; measure it after TP=4 + EP works.

## Before first deploy

1. **DeepGEMM in the image.** The sparse-attention indexer raises at init
   without it. `docker run --rm --gpus all <ray-llm-image> python -c "import vllm.third_party.deep_gemm"`
   must succeed. The vLLM 0.29.0 wheel ships its own copy, so the standard
   image probably has it; `ep-kernels.md` says otherwise, so check.
2. **Engine arguments** against the image:
   `vllm serve --help | grep -A3 -E 'tokenizer-mode|tool-call-parser|reasoning-parser|block-size'`.
   `deepseek_v4` must be listed for all three.
3. **Prepare the node**: 4 GPUs in one worker, `nccl-tests`, the checkpoint
   pre-staged on NVMe with checksums, the image pre-pulled.
4. `trust_remote_code` is left out on purpose (vLLM ships the config). If
   the boot asks for it, read the repo code before adding it.

## Client notes

Model id `deepseek-v4-flash`. Thinking is off by default; per request send
`chat_template_kwargs: {"thinking": true, "reasoning_effort": "high"}`
(`low`, `high` or `max`; `max` needs the 393k window shipped here). Agent
sampling per the card: temperature 1.0, top_p 0.95. The key goes as both
`Authorization: Bearer` and `X-API-Key`. The platform path caps a request
at 300 s today, which cuts the longest Think Max answers.

## Deploy (once the node exists)

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                       # answer without thinking
python predict.py --think               # thinking, effort high
python predict.py --tools               # tool-calling round trip
```

## Variants

| Variant | Edits |
|---|---|
| DSpark speculative decoding | uncomment `speculative_config` (`dspark`, 7 tokens); compare tokens/s and answers |
| Preview checkpoint `deepseek-ai/DeepSeek-V4-Flash` (159.6 GB) | `model_source` and MTP instead of DSpark: `{"method": "mtp", "num_speculative_tokens": 1}` on Hopper, 2 on Blackwell |
| NVFP4 `nvidia/DeepSeek-V4-Flash-NVFP4` (168 GB) | Blackwell only; not smaller than the official mix |
| 1M context | `max_model_len=1048576`; KV allows it on 4× H200 |

## Benchmark

Chat 2K in / 512 out; reasoning at `high` and `max`; document Q&A at 128k
and 393k; tool calls; TP=4 + EP against DP=4 + EP; 2 hours at 70 % load.
