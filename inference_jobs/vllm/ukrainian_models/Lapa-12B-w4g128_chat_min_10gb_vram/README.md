# Lapa LLM 12B (Gemma 3, Ukrainian tokenizer) — 4-bit on one 16 GB card (vLLM)

Lapa (lang-uk) is Gemma 3 12B continued-pretrained on Ukrainian with a
**re-cut vocabulary**: 80k of 250k tokens replaced by Ukrainian ones, so
Ukrainian text costs ~1.5× fewer tokens than on Gemma or MamayLM. Same
card, more users, longer documents. Training data and code are open;
with MamayLM and Gemma it tops the
[Ukrainian LLM leaderboard](https://huggingface.co/spaces/lang-uk/ukrainian-llm-leaderboard).

Shared vLLM contract & defaults: [../../README.md](../../README.md).
Folder index and Gemma 3 engine notes: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 10 → 16 GB | 4 | 24 GB | ~26 GB (image 15 + 1.3×8 GB checkpoint) | 12gb / 16gb |

## ⚠️ Checkpoint — verify

`Yehor/lapa-v0.1.2-it-w4g128` is a community 4-bit (auto-round) build of
v0.1.2; the official instruct is `lapa-llm/lapa-v0.1.3-instruct` (bf16,
~24 GB, vision, 128k context) — run it with the TP=2 sibling
`../MamayLM-12B-v2_chat_vision_tensor_parallel_min_26gb_vram` by swapping
`model_source`. License: Gemma Terms of Use.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py
python predict.py --stream "Склади довідку про стан виконання рішення № 12 від 03.03.2026: …"
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

Compare token counts with the MamayLM job on the same Ukrainian prompt
(`predict.py` prints `prompt + completion` tokens): Lapa should show
~1.5× fewer.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `model_source` | official bf16 on 24 GB+: `lapa-llm/lapa-v0.1.3-instruct` |
| `max_model_len`, `max_num_seqs` | context vs concurrent chats |
| `limit_mm_per_prompt` | delete for text-only |
