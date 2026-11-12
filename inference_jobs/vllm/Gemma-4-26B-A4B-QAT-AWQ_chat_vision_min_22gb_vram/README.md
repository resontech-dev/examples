# Gemma 4 26B A4B (QAT, AWQ INT4): text + image on one 24 GB card (vLLM)

> **BLOCKED on the pinned image: needs vLLM >= 0.19 (ray-llm image re-pin, reson_docs spec item J). Written ahead; nothing has booted.**
> This size needs vLLM >= 0.25.0 per the vLLM recipe (Gemma 4 MoE; the RTX 5090 config there runs Red Hat's FP8 build).

Google's Gemma 4 26B A4B (2026), Apache-2.0: a Mixture-of-Experts model with
128 experts and about 4B parameters active per token. Text and images,
thinking mode, native tool calling, 140+ languages. Each token reads about
a sixth of the weight bytes the dense 31B reads, so decode is several times
faster per card at the same quality class (GPQA 82.3 vs 84.3 on the card).
All 26B stay in memory: it needs the memory of a big model and runs at the
speed of a small one.

Shared vLLM contract and defaults: [../README.md](../README.md).

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 22 -> 24 GB | 8 | 48 GB | ~38 GB (image 15 + 1.3x17.2 GB checkpoint) | 24gb |

Suggested GPUs: RTX 4090 / 3090 (24 GB, as configured); RTX 5090 / PRO 4500
(32 GB) for 64k+ context or more parallel chats.

Budget (estimate, nothing measured yet): 24 GB x 0.92 = 22.1 GB: 17.2 GB weights + ~1.5 GB activations -> ~3.4 GB
KV. Full KV on 5 of 30 layers with two shared K=V heads, a 1,024-token window on
the rest: a full 32k sequence costs under 1 GB, so about four full-length chats
at once.

## Checkpoint

`cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`: a community AWQ INT4 of Google's
quantization-aware-trained weights (`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`),
compressed-tensors, 17.2 GB, vision tower unquantized. Google's own w4a16 line
covers E2B, E4B, 12B and 31B but not this model; Red Hat AI publishes FP8 and
NVFP4 only. The same publisher's AWQ of Qwen3-Coder already backs a catalog job.
**VERIFY before selling**: hash-check, run the quality set against the FP8 build.

Alternatives with vendor provenance:

| Checkpoint | Size | Runs on |
|---|---|---|
| `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` | 28.6 GB | 48 GB card; a 32 GB card only as the recipe's RTX 5090 config does it (`max_num_seqs=1`, fp8 KV) |
| `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | 16.4 GB | Blackwell only (RTX 5090, RTX PRO 4500 / 6000) |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 18.8 GB | Blackwell only, ModelOpt format |
| `google/gemma-4-26B-A4B-it` | 51.6 GB bf16 | one 80 GB card |

Published numbers from Google's Gemma 4 card (instruction-tuned, vendor numbers):

| | Gemma 4 31B | Gemma 4 26B A4B | Gemma 4 12B | Gemma 3 27B (no think) |
|---|---|---|---|---|
| GPQA Diamond | 84.3 | 82.3 | 78.8 | 42.4 |
| MMMLU | 88.4 | 86.3 | 83.4 | 70.7 |
| LiveCodeBench v6 | 80.0 | 77.1 | 72.0 | 29.1 |
| MMMU Pro | 76.9 | 73.8 | 69.1 | 49.7 |

Ukrainian quality is unmeasured: there is no Gemma 4 entry on the lang-uk
leaderboard yet. Compare with the Gemma 3 based MamayLM / Lapa jobs on your own
prompts before using it for Ukrainian work.

## Client notes

- One model id, `assistant`, for text and images.
- **Thinking** is off by default. Send
  `extra_body={"chat_template_kwargs": {"enable_thinking": true}}` to turn it on;
  the `gemma4` reasoning parser moves the thought into the `reasoning` field.
  In multi-turn chats keep only final answers in the history, except on
  tool-call turns (model card rule).
- **Tool calling** is on (`tool_call_parser="gemma4"`), standard OpenAI `tools`.
- **Sampling** from the card: temperature 1.0, top_p 0.95, top_k 64.
- **Images** go before the text. Each image costs 70 to 1,120 tokens depending
  on the budget (default 280); raise it for OCR, lower it for classification.
- Auth as every job here: the key as both `Authorization: Bearer` and `X-API-Key`.

## Deploy & test (after the image re-pin)

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                                   # English business reply
python predict.py --uk                              # Ukrainian business reply
python predict.py --think "Plan a 3-step migration from Gemma 3 to Gemma 4"
python predict.py --image                           # sample_data/invoice.png -> JSON
python predict.py --tools                           # tool-calling round trip
```

## What to change

| where | for what |
|---|---|
| `max_model_len`, `max_num_seqs` | 65536 / 8 on a 32 GB card |
| `model_source` | one of the vendor checkpoints in the table above |
| `limit_mm_per_prompt` | `{"image": 0}` for text-only (skips multimodal profiling) |
| `mm_processor_kwargs` | `{"max_soft_tokens": 560}` or 1120 for OCR |

## VERIFY before listing

1. Image re-pin to vLLM >= 0.25.0 (spec item J): the recipe's floor for this MoE.
2. The community checkpoint: hash-check, quality set against `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`.
3. Marlin MoE kernels for the W4A16 experts on the target card (Ampere 3090 included).
4. The vendored `tool_chat_template_gemma4.jinja` reaches the replica and matches the image's vLLM version.
5. KV numbers above are estimates; read the KV pool size from the boot log.

License: **Apache-2.0** (the first Gemma generation without the Gemma Terms of Use).
