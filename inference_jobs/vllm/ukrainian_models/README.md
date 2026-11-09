# Ukrainian-tuned models (`engine: vllm_openai`)

Open models with Ukrainian continued pretraining, ready to deploy with the
same folder shape as every job in [`../`](../README.md). Both families are
Gemma 3 derivatives and sit at the top of the open
[Ukrainian LLM leaderboard](https://huggingface.co/spaces/lang-uk/ukrainian-llm-leaderboard)
(translation, summarization, QA, ZNO, MMLU, GSM8K, IFEval), ahead of the
generic Qwen / Llama entries. For an Apache-2.0 model with decent Ukrainian
see `../Qwen3.5-9B-AWQ_assistant_vision_min_14gb_vram`.

| Folder | Model | Checkpoint | Shape | Min total VRAM | Status |
|---|---|---|---|---|---|
| `MamayLM-12B-W4A16_chat_min_10gb_vram` | MamayLM 12B (INSAIT), int4 | community `shportix/…-W4A16` | single, 16 GB card | 10 GB | ⚠️ verify (no model card) |
| `Lapa-12B-w4g128_chat_min_10gb_vram` | Lapa LLM 12B (lang-uk), 4-bit | community `Yehor/lapa-v0.1.2-it-w4g128` | single, 16 GB card | 10 GB | ⚠️ verify (community quant) |
| `MamayLM-12B-v2_chat_vision_tensor_parallel_min_26gb_vram` | MamayLM 12B v2.0, bf16, text + images | official `INSAIT-Institute/…-12B-IT-v2.0` | TP=2 on 2×16 GB, or 1×24 GB | 26 GB | official |
| `MamayLM-27B-v2-FP8_chat_vision_tensor_parallel_min_40gb_vram` | MamayLM 27B v2.0, FP8, text + images | official `INSAIT-Institute/…-27B-IT-v2.0-FP8-dynamic` | TP=2 on 2×24 GB, or 1×48 GB | 40 GB | official; FP8 needs Ada/Blackwell |

Which one first: on **one 16 GB card** the two int4 jobs; on **two 16 GB
cards in one machine** the official 12B v2.0 (the quality reference); the
27B is the quality tier and needs 24 GB cards.

## Who makes them

- **MamayLM** (INSAIT, Sofia): Gemma 3 4B/12B (v1.0, 2025) and 12B/27B
  (v2.0, June 2026) continued-pretrained on Ukrainian + English, text and
  images. Official FP8 of the 27B and GGUF of both v2.0 sizes exist; no
  official int4.
- **Lapa LLM** (lang-uk): Gemma 3 12B with 80k of 250k vocabulary tokens
  re-cut for Ukrainian → ~1.5× fewer tokens per Ukrainian text (faster,
  cheaper, longer documents in the same context). Training data, code and
  benchmarks are open. v0.1.3 instruct is bf16; one community 4-bit of
  v0.1.2 exists.

License for all of them: **Gemma Terms of Use** — free, commercial use
allowed, with a prohibited-use policy that must be passed on to end users.
Not Apache; say so in any offer.

## Gemma 3 engine notes (every job here)

- `dtype="bfloat16"` — fp16 overflows in Gemma 3 and produces garbage.
- Never pass `quantization="fp8"` on a Gemma 3 checkpoint: it silently
  quantizes the vision tower and breaks image inputs (INSAIT's warning).
  Use INSAIT's `-FP8-dynamic` build instead.
- No tool-call parser and no reasoning parser for the Gemma family in
  vLLM — plain chat completions. Agents that need tool calling: Qwen jobs.
- `kv_cache_dtype="fp8"` is left off (sliding-window + fp8 KV untested
  here); prefix caching stays at vLLM's default.
- Images: Gemma 3 normalizes every image to 896×896 = 256 tokens, so pages
  are cheap; resize photos to ~1 000 px client-side.

## TP=2 on consumer cards

The two `tensor_parallel` jobs need both GPUs inside **one** worker
container (start the worker with both cards visible, not one container
per card); `cluster.gpus_per_worker: 2` and `vllm.tensor_parallel_size: 2`
must match `tensor_parallel_size=2` in `serve_module.py`. TP over Ethernet
does not work — `STRICT_PACK` in `placement_group_config` enforces one node.

## Deploy (same for every job)

```bash
cd <job-folder>
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                      # Ukrainian business reply
python predict.py --image              # sample invoice → JSON (vision jobs)
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.
