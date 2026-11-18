# ⚠️ TEMPLATE: DeepSeek V4-Pro, TP=8 + expert parallel on one 8× H200 node

**Needs the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Not
deployable on the current fleet.** Prod runs Ray 2.55.1 / vLLM 0.18.0
until the switch described in
`reson_docs/docs/big_models_experiments/ray2.60.0-upgrade.md`. Nothing here has
booted; every number is napkin math.

1.6T mixture-of-experts, 49B active per token, 1M context, text only, MIT.
GPQA 90.1, HLE 37.7 (48.2 with tools), SWE-bench Verified 80.6. The same
family and flags as `../DeepSeek-V4-Flash_reasoning_template_min_180gb_vram`;
run that one first, it proves the family on half a node.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources (planning numbers, unvalidated)

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 1,128 GB (8× H200) → 1,440 GB (8× B200) | 32 | 1 TB | ~1.2 TB local NVMe (image + 1.3×893 GB) | 8xh200 / 8xb200 |

| Node | Fits? | KV per GPU after weights |
|---|---|---|
| 8× H200, NVLink | yes, the vLLM recipe's verified shape at `gpu_memory_utilization=0.95` | ~18-22 GB |
| 8× B200 | yes, recipe-verified | ~55 GB: the full 1M window |
| 8× H100 | no: 640 GB < 893 GB | |
| 2× (8× H100) | only with EP all-to-all over RDMA, or TP=8 × PP=2 (unverified, not in the recipe); our WireGuard links carry neither | |

## Before first deploy

1. Boot `../DeepSeek-V4-Flash_reasoning_template_min_180gb_vram` on the
   same image: DeepGEMM, the `deepseek_v4` tokenizer mode and parsers are
   checked there at a fifth of the size.
2. Pre-stage the 893 GB checkpoint on NVMe with checksums. A cold HF pull
   at first boot is not a plan.
3. Check the recipe-only arguments in the image:
   `vllm serve --help | grep -A3 -E 'compilation-config|enable-flashinfer-autotune'`.

## Client notes

Model id `deepseek-v4-pro`. Same request contract as V4-Flash: thinking off
by default, `chat_template_kwargs: {"thinking": true, "reasoning_effort": "high"}`
per request. The platform path caps a request at 300 s today.

## Deploy (once the node exists)

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py
python predict.py --think
python predict.py --tools
```

## Variants

| Variant | Edits |
|---|---|
| DSpark speculative decoding | uncomment `speculative_config` |
| 800k context (recipe) | `max_model_len=800000`; KV allows it on 8× H200 at 0.95 |
| 1M context | 8× B200, `max_model_len=1048576` |
| Preview `deepseek-ai/DeepSeek-V4-Pro` (864.7 GB) | MTP instead of DSpark |
