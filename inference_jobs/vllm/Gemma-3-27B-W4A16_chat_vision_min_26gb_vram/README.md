# Gemma 3 27B (W4A16): chat + vision on one 32 GB card (vLLM)

Google's Gemma 3 27B instruct, text and images, 140+ languages, as an INT4 build
for one 32 GB card. MamayLM v2.0 27B (INSAIT) is this model continued-pretrained
on Ukrainian; this job is its base and the baseline to compare it against.

Shared vLLM contract and defaults: [../README.md](../README.md).
Gemma 3 engine notes (bf16 only, no on-the-fly fp8, no tool parser):
[../ukrainian_models/README.md](../ukrainian_models/README.md).

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 26 -> 32 GB | 8 | 64 GB | ~41 GB (image 15 + 1.3x19.7 GB checkpoint) | 32gb |

Suggested GPUs: RTX 5090 / RTX PRO 4500 (32 GB, as configured). A 24 GB card
(4090 / 3090) boots only with `max_model_len=4096` and `max_num_seqs=1`: 19.7 GB
of weights leave about 1 GB of KV there, so we do not recommend it.

Budget (estimate, nothing measured yet): 32 GB x 0.92 = 29.4 GB: 19.7 GB weights + ~1.5 GB activations -> ~8 GB
KV. One full 16k sequence takes ~1.75 GB (full KV on 10 of 62 layers, a 1,024-token
window on the rest), so about four full-length chats at once.

## Checkpoint

`RedHatAI/gemma-3-27b-it-quantized.w4a16`: Red Hat AI's INT4 weight-only build (llm-compressor, W4A16,
compressed-tensors), 19.7 GB on disk, vision tower and embeddings kept bf16.
The card reports recovery against the bf16 original on OpenLLM and vision
sets; it was made for vLLM >= 0.8, so our pinned 0.18.0 runs it.

VERIFY: the repo's tokenizer.json carries a right-side truncation default of
8,192 tokens (HF discussion "Tokenizer limited to 8K context"). Transformers
disables backend truncation unless a caller asks for it, so vLLM should take
16k prompts. Run `python predict.py --long` once after the first boot: it
expects the answer 4719 and more than 8,192 prompt tokens. If it fails, set
`max_model_len=8192` or load the tokenizer from the original Google repo
(`tokenizer="google/gemma-3-27b-it"`, gated: needs an HF token on the worker).

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
| `max_model_len`, `max_num_seqs` | 4096 / 1 on a 24 GB card (not recommended); 32768 / 8 on 48 GB |
| `model_source` | `RedHatAI/gemma-3-27b-it-FP8-dynamic` (~29 GB) on one 48 GB card; `google/gemma-3-27b-it` bf16 (~55 GB, gated) on 80 GB |
| `limit_mm_per_prompt` | more images per request; each costs 256 tokens of context |

## Notes

- MamayLM v2.0 27B (INSAIT, June 2026) is this model continued-pretrained on
  Ukrainian: `../ukrainian_models/MamayLM-27B-v2-FP8_chat_vision_tensor_parallel_min_40gb_vram`.
  Run both on the same prompts before choosing one for Ukrainian work.
- Gemma 4 31B (Apache-2.0, needs a newer image) is the successor at the same
  size class: `../Gemma-4-31B-QAT-W4A16_chat_vision_min_28gb_vram`.

License: **Gemma Terms of Use**. Free, commercial use allowed, with a
prohibited-use policy that must be passed on to end users. Not Apache;
say so in any offer.
