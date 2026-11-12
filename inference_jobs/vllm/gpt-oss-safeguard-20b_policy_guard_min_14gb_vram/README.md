# gpt-oss-safeguard-20b: policy-as-prompt safety classifier (vLLM)

OpenAI's open-weight safety reasoning model (October 2025, Apache-2.0):
the gpt-oss-20b shape (21B MoE, 3.6B active, MXFP4 ~13 GB) fine-tuned to
read a **written policy** in the system message and classify the user
message against it, with its reasoning exposed. The policy is text, so a
rule change is a prompt change: no retraining, no new checkpoint. This is
the model to enforce rules a fixed-taxonomy guard cannot express:
"no credentials in chat", "no third-party personal data", "no instructions
to move data outside". Pair it with the fast guards for the hot path.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 14 → 16 GB | 4 | 32 GB | ~32 GB (image 15 + 1.3×13 GB checkpoint) | 16gb / 24gb |

Suggested GPUs: RTX 4080 / 5080 / 5070 Ti (16 GB, as configured: ~0.9 GB
KV, two full 16k checks in flight); RTX 3090 / 4090 / L4 (24 GB: ~7.8 GB
KV, raise `max_num_seqs` to 8). The 120b sibling
(`openai/gpt-oss-safeguard-120b`, ~65 GB) has the same shape as
[`../gpt-oss-120b_chat_min_66gb_vram`](../gpt-oss-120b_chat_min_66gb_vram):
copy that job and change `model_source`.

## Where it sits

| Use | Fit |
|---|---|
| Background audit of sampled or flagged conversations against the customer's policy | yes, this is the job |
| Pre-release policy testing: run a new policy over last month's chats, read the rationales | yes |
| Inline gate in front of 80 concurrent users | no: every verdict is preceded by reasoning and takes seconds; use Qwen3Guard-0.6B / Nemotron Content Safety inline and send only their hits here |
| High-volume, well-defined categories | no: OpenAI's own guide says classifiers trained on thousands of examples do that better and cheaper |

Caveat from OpenAI's technical report: the chain of thought can contain
content that does not reflect the policy as written. Treat the `reasoning`
field as a hint for the human auditor, never as evidence.

## Writing the policy

`predict.py` ships a ~450-token research data-handling policy in the layout
the OpenAI / ROOST guide recommends: `INSTRUCTIONS`, `DEFINITIONS`,
`VIOLATES (1)`, `SAFE (0)`, `EXAMPLES`, with the output instruction repeated
at the top and the bottom. The guide's measured sweet spot is 400–600
tokens per policy; up to ~10k tokens still works. Put the reasoning effort
in the system message (`Reasoning: low|medium|high`, medium if absent) and
do not cap `max_tokens` tightly: the reasoning channel needs room. Output
formats that work: one line (`VIOLATION R2` / `COMPLIANT`, shipped), a
`0`/`1` character, or a JSON object with `violation`, `policy_category`,
`rule_ids`, `rationale` when you want the rationale returned.

Prefix caching does the heavy lifting here: the policy is the same system
message in every check, so after the first request only the content is
prefilled. The cache is ephemeral GPU memory and never touches disk.

## Deploy

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env        # RESON_API_KEY (rsk_…) + S3 keys
python submit.py            # prints RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY
python predict.py           # three samples: compliant / violation / borderline
python predict.py --policy customer_policy.md "text to classify"
REASONING=high python predict.py "text to classify"
```

Or paste `scripts/serve_module.py` + `inference.yaml` into the web wizard at
<https://beta.reson.tech/dashboard/inference/submit>.

## Sidecar in a bundle

To run it next to a chat model on a 32 GB card, copy the `LLMConfig` into
the bundle's `llm_configs` list with the `card_share(0.45)` pattern from
[`../gpt-oss-20b_rag_bundle_min_18gb_vram`](../gpt-oss-20b_rag_bundle_min_18gb_vram),
`gpu_memory_utilization=0.45` and a fixed `kv_cache_memory_bytes` of 1 GiB
(~40k tokens at fp8), and lower the chat share by the same amount.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `engine_kwargs.max_model_len` | 16384 shipped; raise on a 24 GB card for policies near the 10k-token bound plus long content |
| `engine_kwargs.max_num_seqs` | 4 on 16 GB; 8 on 24 GB |
| system prompt (client-side) | the policy text and `Reasoning: low\|medium\|high` |
| `model_source` | `openai/gpt-oss-safeguard-120b` for the 65 GB sibling, with the 120b chat job's resources |

## Notes

- No tool parser and no `reasoning_parser`: vLLM handles harmony natively
  for gpt-oss; `predict.py` reads the verdict from `content` and reports
  whether a `reasoning` field came back, so a missing field on the pinned
  image shows up in the first run.
- Benchmarks from OpenAI's card: 120b beats GPT-5 on their internal
  multi-policy set (46.3 % vs 43.2 % accuracy); the 20b is close behind on
  single-policy tasks. Vendor numbers on vendor data: measure on the
  customer's policy and 100–200 real messages before promising a hit rate.
