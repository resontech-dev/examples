# gpt-oss-120b: premium general assistant (vLLM)

OpenAI's open-weights 117B MoE (5.1B active), shipped natively in MXFP4
(~65 GB). One level above the 20b on every card number (GPQA ~80 vs ~72,
SWE-bench Verified ~62 vs ~61, HLE ~15 vs ~11), same harmony format,
effort knob and 128k context, Apache-2.0. The llms-catalog premium
general tier (`chat-gptoss-120b`) and the **phase-0 model for the Nick
pilot** (`reson_docs/docs/clients/Nick`).

**Status: written, not booted.** Every number below is an estimate from the
checkpoint size, the KV-cache rule and card bandwidth. Measure on the first
rented card before quoting anything.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 66 → 80 GB (96 GB or H200 for long context) | 16 | 128 GB | ~100 GB (image 15 + 1.3×65 GB checkpoint) | 80gb / 96gb / 4x24gb |

| Card, TP=1 | KV pool (fp8) | What that buys, estimate |
|---|---|---|
| H100 / A100 80 GB | ~6.6 GB ≈ 350k tokens | ~10 sessions at 30k context; 64k shipped |
| RTX PRO 6000 96 GB | ~21 GB ≈ 1.1M tokens | ~35 sessions at 30k; raise `max_model_len` to 131072 |
| H200 141 GB | ~62 GB ≈ 3.3M tokens | ~100 sessions at 30k; 131072 and `max_num_seqs` 32 |

Decode speed per stream follows memory bandwidth over active weights:
H200 (4.8 TB/s) fastest, H100 (3.4) next, PRO 6000 (1.8) about half of
H100. One copy per card (data parallel) beats splitting one copy over
cards whenever the model fits.

## Multi-card shapes

| Shape | Cards | Edits (yaml + `serve_module.py`) |
|---|---|---|
| TP=2 | 2× 48 GB (L40S, RTX PRO 5000) or 2× 80 GB | `vllm.tensor_parallel_size: 2`, `cluster.gpus_per_worker: 2`, `tensor_parallel_size=2`, uncomment `placement_group_config` (STRICT_PACK) |
| TP=4 | 4× 24 GB (RTX 3090 / 4090 / L4), the `4x24gb` tier, ~16 GB weights per card | the same three edits with 4 |

TP spans cards **inside one worker** over PCIe or NVLink; consumer cards
without peer-to-peer run it, slower. It never crosses workers.

## Deploy

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env        # RESON_API_KEY (rsk_…) + S3 keys
python submit.py            # prints RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY
python predict.py "Compare two experimental designs for …"
REASONING=low python predict.py "…"   # faster, shallower
python predict.py --tools             # once the parser flag is verified
```

Or paste `scripts/serve_module.py` + `inference.yaml` into the web wizard at
<https://beta.reson.tech/dashboard/inference/submit>.

First boot downloads 65 GB. Until the fleet has a weight cache, that is
tens of minutes on a cold host and happens on every redeploy; pre-stage the
checkpoint on the host when you can.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `engine_kwargs.max_model_len` | 65536 shipped; 131072 (native) on 96 GB / H200 |
| `engine_kwargs.max_num_seqs` | 16 shipped; 32 on H200 |
| system prompt (client-side) | `Reasoning: low\|medium\|high` — gpt-oss effort knob |
| `autoscaling_config` + `cluster.num_workers` | DP scale-out: one copy per card |
| `tensor_parallel_size` + yaml + placement group | the TP=2 / TP=4 shapes above |

## Tool calling / agentic use

gpt-oss speaks its own **harmony** format and vLLM handles it natively,
but whether the pinned image wants an explicit `tool_call_parser` flag is
**unverified**, so the flags ship commented out in `serve_module.py`. Run
the check from [../README.md](../README.md) ("Tool calling"), then
uncomment. `python predict.py --tools` smoke-tests the round trip.

## Notes

- Kernels: MXFP4 native on Blackwell, Marlin path on Hopper / Ampere since
  vLLM 0.10.1; the pinned 0.18.0 covers every card above.
- Same alignment story as the 20b (OpenAI preparedness process, instruction
  hierarchy); still no moderation layer. For user-facing endpoints pair it
  with [`../gpt-oss-120b_rag_bundle_min_70gb_vram`](../gpt-oss-120b_rag_bundle_min_70gb_vram)
  and a guard (Qwen3Guard-0.6B inline, `gpt-oss-safeguard-20b` for policy audits).
- Prefix caching is on: multi-turn chats only pay for the new tail, and the
  cache is ephemeral GPU memory, so the stateless privacy story holds.
