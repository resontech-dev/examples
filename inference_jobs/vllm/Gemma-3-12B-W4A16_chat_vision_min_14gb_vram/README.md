# Gemma 3 12B (W4A16): chat + vision on one 16 GB card (vLLM)

Google's Gemma 3 12B instruct, text and images, 140+ languages, as an INT4 build
that fits one 16 GB card. The base the Ukrainian MamayLM and Lapa models are
trained from, and a strong multilingual small model on its own.

Shared vLLM contract and defaults: [../README.md](../README.md).
Gemma 3 engine notes (bf16 only, no on-the-fly fp8, no tool parser):
[../ukrainian_models/README.md](../ukrainian_models/README.md).

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 14 -> 16 GB | 8 | 32 GB | ~29 GB (image 15 + 1.3x10.4 GB checkpoint) | 16gb |

Suggested GPUs: RTX 5070 Ti / 4080 / 5080 (16 GB, as configured); any 24 GB card
for 32k context or more parallel chats.

Budget (estimate, nothing measured yet): 16 GB x 0.92 = 14.7 GB: 10.4 GB weights + ~1 GB activations -> ~3 GB
KV. One full 16k sequence takes ~1.4 GB (full KV on 8 of 48 layers, a 1,024-token
window on the rest), so two full-length chats at once, more when chats are short.

## Checkpoint

`RedHatAI/gemma-3-12b-it-quantized.w4a16`: Red Hat AI's INT4 weight-only build (llm-compressor, W4A16,
compressed-tensors), 10.4 GB on disk, vision tower and embeddings kept bf16.
The card reports recovery against the bf16 original on OpenLLM and vision
sets; it was made for vLLM >= 0.8, so our pinned 0.18.0 runs it.

VERIFY: the repo's tokenizer.json carries a right-side truncation default of
8,192 tokens (HF discussion "Tokenizer limited to 8K context"). Transformers
disables backend truncation unless a caller asks for it, so vLLM should take
16k prompts. Run `python predict.py --long` once after the first boot: it
expects the answer 4719 and more than 8,192 prompt tokens. If it fails, set
`max_model_len=8192` or load the tokenizer from the original Google repo
(`tokenizer="google/gemma-3-12b-it"`, gated: needs an HF token on the worker).

## Client notes

- One model id, `assistant`, for text and images. Images go before the text in
  the same message; Gemma 3 normalizes each one to 896x896 = 256 tokens.
- No tool calling and no thinking mode: plain chat completions. Agents that
  need tools: the Qwen jobs or the Gemma 4 jobs (newer image).
- Auth as every job here: the key as both `Authorization: Bearer` and `X-API-Key`.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                                   # English business reply
python predict.py --uk                              # Ukrainian business reply
python predict.py --image                           # sample_data/invoice.png -> JSON
python predict.py --long                            # 8k tokenizer default check
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where | for what |
|---|---|
| `max_model_len`, `max_num_seqs` | 32768 / 8 on a 24 GB card |
| `model_source` | `google/gemma-3-12b-it` bf16 (~24 GB, gated) on a 32 GB card; `RedHatAI/gemma-3-12b-it-FP8-dynamic` on a 24 GB card |
| `limit_mm_per_prompt` | more images per request; each costs 256 tokens of context |

## Notes

- The Ukrainian-tuned versions of this exact model are already in the catalog:
  MamayLM v2.0 12B (INSAIT) and Lapa 12B (lang-uk), in `../ukrainian_models/`.
  Use the base Gemma 3 12B when you need the original Google behaviour or a
  baseline to compare them against.

License: **Gemma Terms of Use**. Free, commercial use allowed, with a
prohibited-use policy that must be passed on to end users. Not Apache;
say so in any offer.
