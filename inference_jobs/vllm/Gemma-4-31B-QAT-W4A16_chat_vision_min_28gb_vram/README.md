# Gemma 4 31B (QAT W4A16): text + image on one 32 GB card (vLLM)

> **BLOCKED on the pinned image: needs vLLM >= 0.19 (ray-llm image re-pin, reson_docs spec item J). Written ahead; nothing has booted.**
> This size needs vLLM >= 0.19.1 per the vLLM recipe (the first Gemma 4 release in vLLM was 0.19.0).

Google's Gemma 4 31B (2026), Apache-2.0: the largest dense Gemma 4. Text and
images, thinking mode, native tool calling, 140+ languages, in Google's own
4-bit QAT build for one 32 GB card. On Google's card it scores GPQA Diamond
84.3 against 42.4 for Gemma 3 27B, the model MamayLM v2.0 27B is built on.

Shared vLLM contract and defaults: [../README.md](../README.md).

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 28 -> 32 GB | 8 | 64 GB | ~46 GB (image 15 + 1.3x23.3 GB checkpoint) | 32gb |

Suggested GPUs: RTX 5090 / RTX PRO 4500 (32 GB, as configured). 24 GB cards
cannot hold it: 23.3 GB of weights leave no KV.

Budget (estimate, nothing measured yet): 32 GB x 0.92 = 29.4 GB: 23.3 GB weights + ~1.5 GB activations -> ~4.6 GB
KV. Full KV on 10 of 60 layers with four shared K=V heads, a 1,024-token window
on the rest (~0.84 GB per sequence): a full 16k sequence takes 1.5 to 2 GB,
so two full-length chats at once, more when chats are short.

## Checkpoint

`google/gemma-4-31B-it-qat-w4a16-ct`: Google's official quantization-aware
trained 4-bit build in compressed-tensors format, 23.3 GB on disk, Apache-2.0,
not gated; the vLLM recipe serves it as its `w4a16` variant. The vision tower
stays unquantized. At 23.3 GB it does not fit a 24 GB card with any KV left:
this job needs 32 GB.

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
| `max_model_len`, `max_num_seqs` | 65536 / 8 on a 48 GB card |
| `model_source` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` (33.3 GB) on one 48 GB card; `RedHatAI/gemma-4-31B-it-NVFP4` on Blackwell; `google/gemma-4-31B-it` bf16 (62.5 GB) on one 80 GB card |
| `limit_mm_per_prompt` | `{"image": 0}` for text-only (skips multimodal profiling) |
| `mm_processor_kwargs` | `{"max_soft_tokens": 560}` or 1120 for OCR |
| `speculative_config` (commented) | Red Hat's EAGLE-3 speculator; DFlash and Google's MTP `google/gemma-4-31B-it-assistant` also exist |

## VERIFY before listing

1. Image re-pin to vLLM >= 0.19.1 (spec item J).
2. The vendored `tool_chat_template_gemma4.jinja` reaches the replica and matches the image's vLLM version.
3. KV numbers above are estimates; read the KV pool size from the boot log, then raise `max_model_len` if it allows.
4. Speculative decoding (commented) only after a clean baseline benchmark.

License: **Apache-2.0** (the first Gemma generation without the Gemma Terms of Use).
