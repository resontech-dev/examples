# Qwen2.5-VL-7B (AWQ) — BFSI document ingest, Stack A (vLLM)

The **vision half of the banking_and_insurance pilot**
([sectors/banking_and_insurance/bfsi_pilot](../../../sectors/banking_and_insurance/bfsi_pilot/README.md)):
repair invoices, damage acts, and police reports — clean PDFs and phone
photos — in as images, schema-shaped JSON out. AWQ 4-bit (~6 GB) so it
fits the pilot's single 16 GB card; the fp16 sibling
([Qwen2.5-VL-7B_vision…](../Qwen2.5-VL-7B_vision_min_16gb_vram/)) is the
quality pick when a dedicated 24 GB worker exists.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 8 → 12 GB | 4 | 24 GB | ~23 GB (image 15 + 1.3×6 GB checkpoint) | 12gb / 16 GB card |

Suggested GPUs: RTX 5070 Ti / 4080 (16 GB — the pilot's target, roomy);
any 12 GB card fits at this context. **One-GPU pilot rule:** this job
(Stack A) and the chat+embed bundle (Stack B) run **sequentially** on a
16 GB card — stop one before submitting the other.

## Deploy

```bash
cp .env.example .env        # RESON_API_KEY (rsk_…) + S3 keys
python submit.py            # prints VL_BASE_URL + VL_API_KEY for the pilot's .env
python predict.py           # smoke: sample invoice → JSON
```

Expected smoke output: valid JSON with `NORDWIND SUPPLIES GmbH` /
`INV-2026-0142` / `2026-07-28` and 4 positions. `unit_price_uah` and
`total_uah` come back **null — that's correct**: the sample invoice is in
EUR and the prompt forbids inventing UAH values. The pilot's own seed
invoices are UAH, so these fields fill in on the real run.

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `max_model_len` | 8192 shipped — single-page extraction; raise for multi-page acts |
| `limit_mm_per_prompt` | 2 shipped — pages per request |
| `model_source` | VERIFY the AWQ repo id on HF before first deploy |

## Notes

- Extraction determinism comes from the **client** (temperature=0, schema
  prompt, pydantic validation + one retry) — see the pilot's `process_documents.py`.
- **This job deliberately ships WITHOUT the repo-wide fp8-KV / prefix-caching
  defaults** — with multimodal models on quantized checkpoints they are the
  two most common causes of garbled output (broken tokens from position 1,
  deterministic at temperature=0). If your smoke test prints garbage JSON:
  make sure you deployed THIS conservative config, then follow the
  escalation comments in `scripts/serve_module.py` (enforce_eager → swap to
  `Qwen/Qwen2.5-VL-3B-Instruct` fp16, which fits 16 GB unquantized).
- Quick fault isolation against a live endpoint: send a **text-only**
  message first (`"Напиши слово 'тест'"`). Text garbled → engine/quant
  problem; text clean but image garbled → vision path (usually the same
  two flags, or the AWQ build).
