# Gemma 4 12B (QAT W4A16): text + image + audio on one 16 GB card (vLLM)

> **BLOCKED on the pinned image: needs vLLM >= 0.19 (ray-llm image re-pin, reson_docs spec item J). Written ahead; nothing has booted.**
> This size needs vLLM >= 0.23.0 per the vLLM recipe (the encoder-free 12B Unified class; the recipe names the `vllm/vllm-openai:gemma4-unified` image).

Google's Gemma 4 12B Unified (2026), Apache-2.0: an encoder-free model that
projects image patches and audio straight into the decoder. Text, images,
audio up to 30 s, thinking mode, native tool calling, 140+ languages, in
Google's own 4-bit QAT build for one 16 GB card. The direct successor to
Gemma 3 12B (the base of MamayLM and Lapa).

Shared vLLM contract and defaults: [../README.md](../README.md).

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 14 -> 16 GB | 8 | 32 GB | ~29 GB (image 15 + 1.3x10.3 GB checkpoint) | 16gb |

Suggested GPUs: RTX 5070 Ti / 4080 / 5080 (16 GB, as configured).

Budget (estimate, nothing measured yet): 16 GB x 0.92 = 14.7 GB: 10.3 GB weights + ~1.2 GB activations -> ~3 GB
KV. Gemma 4 keeps full KV on 8 of 48 layers with one shared K=V head, and a
1,024-token window on the rest, so a full 32k sequence costs well under 1 GB:
three to five full-length chats at once.

## Checkpoint

`google/gemma-4-12B-it-qat-w4a16-ct`: Google's official quantization-aware
trained 4-bit build in compressed-tensors format, 10.3 GB on disk, Apache-2.0,
not gated. Architecture `Gemma4UnifiedForConditionalGeneration`. The vision and
audio projections stay unquantized.

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

- One model id, `assistant`, for text, images and audio.
- **Thinking** is off by default. Send
  `extra_body={"chat_template_kwargs": {"enable_thinking": true}}` to turn it on;
  the `gemma4` reasoning parser moves the thought into the `reasoning` field.
  In multi-turn chats keep only final answers in the history, except on
  tool-call turns (model card rule).
- **Tool calling** is on (`tool_call_parser="gemma4"`), standard OpenAI `tools`.
- **Sampling** from the card: temperature 1.0, top_p 0.95, top_k 64.
- **Images** go before the text. Each image costs 70 to 1,120 tokens depending
  on the budget (default 280); raise it for OCR, lower it for classification.
- **Audio** (this size only among the 12B+ models): an `input_audio` part after
  the text, 30 s maximum per clip. The image needs vLLM's audio extras
  (`vllm[audio]`); VERIFY they are in the new image. For long recordings use the
  Whisper jobs.
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
python predict.py --audio                           # sample_data/hello_reson.wav -> transcript
python predict.py --tools                           # tool-calling round trip
```

## What to change

| where | for what |
|---|---|
| `max_model_len`, `max_num_seqs` | 65536 / 8 on a 24 GB card |
| `model_source` | `google/gemma-4-12B-it` bf16 (23.9 GB) on a 32 GB card; `RedHatAI/gemma-4-12B-it-FP8-Dynamic` (15.0 GB) on a 24 GB card |
| `limit_mm_per_prompt` | `{"image": 4, "audio": 0}` for image-only, `{"image": 0, "audio": 0}` for text-only (skips multimodal profiling) |
| `mm_processor_kwargs` | `{"max_soft_tokens": 560}` or 1120 for OCR |
| `speculative_config` (commented) | MTP drafting with `google/gemma-4-12B-it-assistant` |

## VERIFY before listing

1. Image re-pin to vLLM >= 0.23.0 (spec item J); the 12B Unified class is the newest of the three.
2. The vendored `tool_chat_template_gemma4.jinja` reaches the replica and matches the image's vLLM version.
3. `vllm[audio]` extras present in the image; `python predict.py --audio` returns a transcript.
4. Ray accepts `default_chat_template_kwargs` on the new image before anyone sets a thinking default.
5. KV numbers above are estimates; read the KV pool size from the boot log.

License: **Apache-2.0** (the first Gemma generation without the Gemma Terms of Use).
