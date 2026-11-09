# MamayLM 12B (Gemma 3, Ukrainian) — int4 on one 16 GB card (vLLM)

The quickest way to try a **Ukrainian-tuned** open model on a single
consumer card. MamayLM (INSAIT) is Gemma 3 continued-pretrained on
Ukrainian; with Lapa and Gemma it tops the open
[Ukrainian LLM leaderboard](https://huggingface.co/spaces/lang-uk/ukrainian-llm-leaderboard)
(translation, summarization, QA, ZNO, MMLU, IFEval). This job serves a
community int4 build (~8 GB).

Shared vLLM contract & defaults: [../../README.md](../../README.md).
Folder index and Gemma 3 engine notes: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 10 → 16 GB | 4 | 24 GB | ~26 GB (image 15 + 1.3×8 GB checkpoint) | 12gb / 16gb |

Suggested GPUs: RTX 5070 Ti / 4080 / 5080 (16 GB — as configured, 16k
context); 12 GB cards with `max_model_len=8192`.

## ⚠️ Checkpoint — verify

`shportix/MamayLM-Gemma-3-12B-IT-W4A16` has no model card (base version
not stated, naming suggests v1.0). Fallbacks are listed in
`serve_module.py`; the official one-card choice is
`INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0` (bf16, ~8.6 GB, vision),
weaker but documented. The official 12B v2.0 needs 24 GB or 2×16 GB —
see `../MamayLM-12B-v2_chat_vision_tensor_parallel_min_26gb_vram`.

License: Gemma Terms of Use (free, commercial use allowed, prohibited-use
policy applies).

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                                   # Ukrainian business reply
python predict.py --stream "Стисло перекажи: …"
python predict.py --image                           # sample_data/invoice.png → JSON (if vision loaded)
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

Client notes: the key goes as both `Authorization: Bearer` and
`X-API-Key`; strip `/predict` from the printed endpoint, the OpenAI
surface is `<endpoint>/v1`. No thinking mode, no tool-call parser —
plain chat completions.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `model_source` | the fallbacks in the banner |
| `max_model_len`, `max_num_seqs` | context vs concurrent chats — the KV levers |
| `limit_mm_per_prompt` | delete for text-only if the vision tower fails to load |
| `dtype` | keep `bfloat16` — Gemma 3 breaks in fp16 |
