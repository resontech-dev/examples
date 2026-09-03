# Devstral Small 24B (AWQ) — agentic coding (vLLM)

Mistral's **agentic SWE specialist** (`code-devstral-24b`, Apache-2.0):
built for issue→patch loops on private repos (OpenHands / SWE-bench
lineage). Strong at multi-step agent work; the Qwen coders beat it on raw
completion. One 24 GB worker, AWQ 4-bit.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 16 → 24 GB | 4 | 48 GB | ~39 GB (image 15 + 1.3×13 GB checkpoint) | 24gb / 2x12gb |

Suggested GPUs: RTX 3090 / 4090 / L4 / A10G (24 GB — as configured, 32k
context); 2× 12 GB cards → PP=2 (copy the shape from the sibling
`Qwen3-Coder-30B…pipeline_parallel` example, ~7 GB weights/stage).

## Deploy

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py "Write a Python function that merges overlapping intervals, with a doctest."
python predict.py --tools        # function-calling smoke test
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Wire it into agents

Point OpenHands / Cline / Aider at `<endpoint>/v1` with model
`team-coder`. Tool calling is **on** (`tool_call_parser="mistral"`,
long-stable in vLLM). Prefix caching is on — agent loops that resend
20k-token contexts only pay for the new tail.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `model_source` | ⚠️ community AWQ — **vet before selling**; official bf16 needs 2×24 TP=2 |
| `max_model_len` | 32k shipped — matches agent-scaffold reality on this VRAM |
| `max_num_seqs` / `max_ongoing_requests` | per-seat capacity; agent-heavy teams: ~5–10 devs/worker |

## Notes

- Sizing (catalog): one 24 GB worker ≈ 5–10 devs on agent-heavy loops —
  an agent occupies a concurrency slot for minutes. Offload inline
  autocomplete to the 7B/3B sibling job (two-tier story).
- Alignment: Mistral's minimal-refusal posture — complies more on
  dual-use security code. Endpoint keys are the real guardrail.
