# BrowseSafe: prompt-injection detector for fetched web pages (vLLM)

Perplexity's open-weight classifier (December 2025, **MIT**), fine-tuned
from Qwen3-30B-A3B-Instruct-2507 on BrowseSafe-Bench. Input: the raw HTML
of a page. Output: one token, `yes` / `no`: does the page contain
instructions aimed at the agent? Model id `web-guard`. 30B MoE with 3B
active, so a verdict on a full 16k-token page is a single prefill.

**Only for setups where web search / page fetch is enabled** (SearXNG, URL
fetch in the chat server). Without web access there is nothing for it to
check: do not add it to the bill. For user text and documents use
Qwen3Guard, Nemotron 3.5 Content Safety or gpt-oss-safeguard instead.

Shared vLLM contract & defaults: [../README.md](../README.md).
Guard models compared: `reson_docs/docs/inference/models/safety.md`.

## Where it sits

```
user asks → chat server runs web search → page fetched
         → web-guard(raw HTML)  → "yes" ? drop the page, log, tell the user
         → page goes into the context → assistant
```

Chat-server side (an Open WebUI filter / tool hook, or the fetch service),
**not** LiteLLM: the page never reaches the chat model if it is flagged.
Pass the **full HTML**, not extracted text: the card says extraction
degrades detection. Pages above 16,384 tokens are chunked and the verdicts
OR-ed (`predict.py` does this client-side; the card's reference
implementation does the same).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 34 → 48 GB | 8 | 96 GB | ~95 GB (image 15 + 1.3×61 GB bf16 download) | 48gb / 2x24gb / 80gb |

Shipped shape: **one 48 GB card** (L40S / RTX PRO 5000) with on-the-fly
fp8 weight quantization, ~31 GB weights, ~11 GB fp8 KV ≈ 8 full pages in
flight. Alternatives, three edits each (listed in `serve_module.py`):
2× 24 GB TP=2 (RTX 4090 / 3090, ~15.5 GB weights per card); 80 GB+ in
bf16 (H100 / A100-80G / RTX PRO 6000 96 GB: drop `quantization`).

Download is the bf16 checkpoint (~61 GB) regardless of shape: first boot
on a cold host takes a while; pre-mirror where the fleet allows.

## ⚠️ Verify before selling

- **fp8 on the fly for a Qwen3 MoE on vLLM 0.18.0** is expected to work
  (official Qwen3-30B-A3B-FP8 checkpoints run on it, and
  `../Qwen3-Coder-30B-A3B…pipeline_parallel` is the same architecture in
  AWQ) but this exact combination has not been booted. If a vetted community
  FP8 / AWQ re-upload of `browsesafe` appears, switch `model_source` and drop
  the `quantization` kwarg.
- **Verdict polarity.** The card documents `yes` / `no` and calls the
  positive class VIOLATES; `predict.py` reads `yes` as "instructions for the
  agent present". Confirm on the two sample pages at first run
  (`injected.html` must come back `yes`, `benign.html` `no`).
- The card gives no system prompt: the HTML is the whole user message and
  the chat template adds the generation prompt. `predict.py` sends exactly that.

## Vendor numbers (model card, own benchmark, not reproduced here)

BrowseSafe-Bench test set, 3,680 HTML pages: **F1 0.904, precision 0.978,
recall 0.841**, balanced accuracy 0.912, zero refusals. Same table: Llama
Prompt Guard 2 86M F1 0.36, gpt-oss-safeguard-20b (low effort) F1 0.79,
GPT-5 (medium) 0.855. One vendor's benchmark of its own model: measure on
the pages your customer's users actually open.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv httpx   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                              # sample_data/injected.html → expect yes
python predict.py sample_data/benign.html      # expect no
python predict.py https://example.com          # live page, raw HTML
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

`sample_data/injected.html` is a test fixture: a normal product page with a
hidden `div` carrying an instruction aimed at an AI agent (exfiltrate the
session cookie, lie about stock). `benign.html` is the same page without it.

## Latency and cost

One verdict = prefill of the page (up to 16k tokens) + 1 output token. On a
48 GB card expect well under a second per average page and a few seconds
for a 16k chunk; 3B active parameters keep the prefill cheap. Run it
asynchronously in the fetch path, as the card suggests, so search latency
is unchanged for clean pages.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `quantization="fp8"` | drop on 80 GB+ cards (bf16) or when switching to a pre-quantized checkpoint |
| `vllm.tensor_parallel_size: 2` + `gpus_per_worker: 2` (yaml) + `tensor_parallel_size=2` + `placement_group_config` (py) | 2×24 GB shape |
| `max_num_seqs` | 8 on 48 GB fp8; 16 on 80 GB bf16 |
| `max_model_len` | keep 16384: the model's trained window; chunk client-side instead |
| `CHARS_PER_TOKEN`, `--chunk-tokens` (`predict.py`) | chunk sizing; keep chunks ≤ ~12k tokens so the template fits |

## Notes

- A detector, not a boundary: the page can still be dropped only if the
  fetch path honours the verdict. Keep tool sandboxes without network and
  allow-lists in place regardless.
- The guard is itself a language model; it sees untrusted HTML by design.
  It has no tools and its answer is one token, so there is nothing an
  injection can make it do except give a wrong verdict.
- English-only training data (card); pages in other languages are
  untested: sample them in the acceptance run.
