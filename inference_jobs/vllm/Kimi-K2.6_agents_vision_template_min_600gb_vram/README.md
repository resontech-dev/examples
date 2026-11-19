# ⚠️ TEMPLATE: Kimi K2.6, TP=8 on one 8× H200 node (text + image)

**Needs the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Not
deployable on the current fleet.** Nothing here has booted; every number is
napkin math.

Moonshot's 1T mixture-of-experts, 32B active per token, 262k context,
text + image. GPQA 90.5, HLE 34.7 (54.0 with tools), SWE-bench Verified
80.2, AIME 2026 96.4. One of the strongest open agents on one node.

**Licence: Modified MIT.** Products above 100 million monthly active users
or $20 million monthly revenue must display "Kimi K2.6".

**`trust_remote_code=True` is required** (tokenizer and image processor
come from the repo). It is the only template in this folder that executes
repository code: review it and pin the commit first.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources (planning numbers, unvalidated)

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 1,128 GB (8× H200) → 1,440 GB (8× B200) | 32 | 1 TB | ~800 GB local NVMe (image + 1.3×595 GB) | 8xh200 / 8xb200 |

| Node | KV after weights | Notes |
|---|---|---|
| 8× H200, NVLink | ~48 GB per GPU, ~680k tokens per node (MLA KV copied on each rank) | target |
| 8× B200 | ~83 GB per GPU, ~1.2M tokens | |
| 4× B300, NVFP4 `nvidia/Kimi-K2.6-NVFP4`, TP=4 | | recipe-verified; NVIDIA licence on that build |
| 8× H100, 4× H200 | does not fit | |

## Before first deploy

1. **Review the repo code** (`tokenization_kimi.py`, the processor files),
   then set `revision`, `code_revision` and `tokenizer_revision` to that
   commit in `serve_module.py`. Check it imports under transformers 5.17.
2. **Pre-stage** the 595 GB checkpoint on NVMe with checksums.
3. **Engine arguments** against the image:
   `vllm serve --help | grep -A3 -E 'tool-call-parser|reasoning-parser|mm-encoder-tp-mode'`
   must list `kimi_k2`.
4. `python predict.py --think`: watch for a reasoning field of `!!!!`
   (vLLM #42426).

## Client notes

Model id `kimi-k2.6`. Thinking is on by default; `chat_template_kwargs:
{"thinking": false}` gives Instant mode, `{"thinking": true,
"preserve_thinking": true}` keeps earlier thinking for agents. Sampling:
temperature 1.0 with thinking, 0.6 Instant, top_p 0.95. Up to 4 images per
request; video only on Moonshot's own API.

## Deploy (once the node exists)

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                       # Instant mode
python predict.py --think               # thinking
python predict.py --tools               # tool-calling round trip
```

## Variants

| Variant | Edits |
|---|---|
| Decode context parallel | `decode_context_parallel_size=8`: shards the MLA KV instead of copying it on each rank |
| Eagle3 draft | `lightseekorg/kimi-k2.6-eagle3-mla` (third-party, 6 GB); vLLM #51660: structured outputs + EAGLE3 killed the engine on 0.26 |
| Text only | `language_model_only=True` |
