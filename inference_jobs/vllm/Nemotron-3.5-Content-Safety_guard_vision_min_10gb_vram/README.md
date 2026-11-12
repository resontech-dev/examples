# Nemotron 3.5 Content Safety: multimodal guard (vLLM) ⚠️ non-Apache license

NVIDIA's 4B guard (June 2026, Gemma-3-4B-it base, LoRA merged): moderates
a user prompt, an optional image and an optional assistant response, either
against NVIDIA's standard safety taxonomy or against **a policy you write in
plain text**, with a reasoning trace on request. One engine, model id
`guard`, ~8.6 GB bf16. Fits a 12-16 GB card alone or rides along a chat
engine as a sidecar.

Shared vLLM contract & defaults: [../README.md](../README.md).
Guard models compared (what each one catches, where it sits in the stack):
`reson_docs/docs/inference/models/safety.md`.

## ⚠️ License: read before selling

**Not Apache.** Use is governed by three documents at once: the
[OpenMDW License Agreement 1.1](https://raw.githubusercontent.com/OpenMDW/OpenMDW/refs/heads/main/1.1/LICENSE.OpenMDW-1.1),
the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the Gemma
Prohibited Use Policy (the base model is Gemma 3). Commercial use is
allowed; the prohibited-use list binds the customer. Same handling as the
`chat-gemma3-27b` row in `reson_docs/docs/inference/vllm/llms-catalog.md`:
hand the terms to the customer's legal before the endpoint goes live. The
training dataset is published (`nvidia/Nemotron-3.5-Content-Safety-Dataset`).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 10 → 12 GB | 4 | 32 GB | ~27 GB (image 15 + 1.3×8.6 GB checkpoint) | 12gb / 16gb (shares) |

Suggested GPUs: RTX 5070 Ti / 5080 / 4080 (16 GB: as configured,
`gpu_memory_utilization=0.85`, ~4 GB KV, 8 concurrent checks); RTX 5070 /
4070 (12 GB: fits with `max_num_seqs=2`). Vision tower is on by default
(`limit_mm_per_prompt={"image": 1}`).

Model card pins `vllm>=0.11.0,<=0.20.2`; the pinned image (vLLM 0.18.0)
is inside the range. Gemma 3 is a vLLM-native architecture: no
`trust_remote_code`.

## What it adds over Qwen3Guard, and what it lacks

| | Qwen3Guard-Gen (guarded bundle) | Nemotron 3.5 Content Safety |
|---|---|---|
| Input | text | text **+ one image** |
| Policy | fixed taxonomy (safe / controversial / unsafe) | fixed taxonomy **or your written policy** per request |
| Reasoning | no | optional `<think>` trace |
| Languages | 119, Ukrainian included | 12 (en, ar, de, es, fr, hi, ja, th, nl, it, ko, zh): **no Ukrainian** |
| License | Apache-2.0 | OpenMDW-1.1 + Gemma terms |
| Sizes | 0.6B / 4B / 8B | 4B |

Pick it when the customer wants screenshots or scanned pages moderated,
or has a written usage policy to enforce verbatim. Keep Qwen3Guard for
Ukrainian traffic and Apache-only procurement.

## Vendor numbers (model card, own evaluation, not reproduced here)

Custom-policy mode on Dynaguardrail: prompt injection F1 **0.90** without
thinking, 0.88 with thinking; safety 0.91 / 0.86. Taxonomy mode: Aegis 2.0
prompt harmful-F1 0.86, Wildguard 0.85, MultiJail 0.95, LinguaSafe average
F1 0.71. Multimodal: VLGUARD 0.90, MM-SafetyBench 0.71. Measure on the
customer's own traffic before promising a false-positive rate.

## The protocol (`predict.py`)

Send the content as a normal chat; select the mode with
`chat_template_kwargs` (keys verbatim from the model card):

```python
extra_body={"chat_template_kwargs": {
    "request_categories": "/categories",   # always
    "enable_thinking": False,              # True → <think>…</think> trace first
    "custom_policy": POLICY_TEXT,          # optional → replaces the taxonomy
}}
```

Messages: `{"role": "user", "content": [image_part?, {"type": "text", "text": prompt}]}`,
plus `{"role": "assistant", "content": [{"type": "text", "text": response}]}`
when a response is moderated. Output:

```
User Safety: safe | unsafe
Response Safety: safe | unsafe      # only when a response was given
Safety Categories: Illegal Activity, Fraud/Deception   # only when unsafe
```

`predict.py` strips the `<think>` block, parses the three lines leniently
and prints the raw text as well. A missing `User Safety` line is reported;
treat it as unsafe in production. Thinking mode costs 300-600 output
tokens per check (seconds, not milliseconds): use it for audits and
flagged items, not inline on every message.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                                            # benign prompt, taxonomy
python predict.py "How do I pick the lock on my neighbour's door?"
python predict.py "…" --response "Sure, first you …"         # prompt + response
python predict.py "…" --policy sample_data/policy.txt        # written policy
python predict.py "…" --policy sample_data/policy.txt --think
python predict.py "What is this document?" --image sample_data/invoice.png
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Sidecar: replace the guard in the guarded bundle

In `../Qwen3-8B-AWQ_guarded_rag_chat_min_22gb_vram/scripts/serve_module.py`
swap the `guard` LLMConfig for this one with `gpu_memory_utilization=0.30`,
`kv_cache_memory_bytes=int(0.5 * 2**30)`, `**card_share(0.30)`, and drop
`quantization="fp8"` (bf16 weights are 8.6 GB, so the 24 GB budget still
closes: chat 0.32 + guard 0.30 + embed 0.06 + rerank 0.06 = 0.74). Then
replace the bundle's `_guard` helper with `moderate()` from this
`predict.py`. The model id stays `guard`.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `max_num_seqs` | 8 on 16 GB; 2 on 12 GB; raise on 24 GB+ |
| `max_model_len` | 8192 shipped; the card allows 128K (long responses, long policies) |
| `limit_mm_per_prompt` | `{"image": 0}` + `language_model_only=True` for a text-only, lighter engine (VERIFY the kwarg on the pinned image) |
| `kv_cache_dtype="fp8"`, `enable_prefix_caching=True` | add one at a time on a shared card; not set by default on this Gemma 3 model |
| `reasoning_parser="deepseek_r1"` | VERIFY: moves the `<think>` trace into the `reasoning` field |
| policy text (client side) | the whole moderation rule set: no redeploy needed |

## Notes

- Verdict semantics in custom-policy mode follow the policy text literally;
  the card's own example shows the model omitting `Response Safety` when no
  response was sent. Always keep a human-readable policy under version control.
- The guard is itself a language model and can be argued with. Run it with
  no tools, no network and a fixed template, as here.
- Ukrainian is not in the supported list: for Ukrainian traffic use
  Qwen3Guard (119 languages) and keep this one for images and written policies.
