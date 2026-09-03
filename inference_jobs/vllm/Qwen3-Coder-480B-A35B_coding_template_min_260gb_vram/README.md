# ⚠️ TEMPLATE — Qwen3-Coder-480B-A35B (AWQ), TP=4 frontier coding

**Not deployable on the current fleet.** This is the ready-to-test config
for the frontier coding agent (`code-qwen3-coder-480b`, Apache-2.0):
480B MoE / 35B active, the open-weights answer to Claude/GPT coding API
tiers. It ships now so the day a 4×80 GB host joins the fleet, testing
starts immediately — every number inside is a planning estimate.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources (catalog planning numbers — unvalidated)

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 260 → 320 GB | 16 | 640 GB | ~358 GB (image 15 + 1.3×260 GB checkpoint) | frontier (4x80gb) |

Suggested GPUs: one node with 4× H100 / A100-80G (as configured, TP=4,
`STRICT_PACK`); 2× H200 (141 GB) also fits at TP=2; B200 single-card is
close but check real headroom. Consumer fleet: no.

## Before first deploy (checklist)

1. **Pre-mirror the ~260 GB checkpoint** into Garage/S3
   (`CloudMirrorConfig`) — an HF cold pull at this size is not a plan.
2. **Vet `model_source`** — community AWQ re-upload; hash-check + evals.
3. Confirm the host's driver ≥ R570 and free disk ≥ 358 GB.
4. `vllm bench serve` → tune `max_model_len` / `max_num_seqs` from
   measurements, not from this template's estimates.
5. If fleet nodes carry Ray accelerator labels, set
   `accelerator_type="H100"` to pin placement; otherwise keep `None`.
6. Drop the TEMPLATE banners once validated.

## Deploy (once hardware exists)

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py "Refactor a 2k-line God class into cohesive services."
python predict.py --tools        # hermes tool calling is enabled
```

## Notes

- MoE 35B-active decode ≈ 35B-class token latency with a 480B brain —
  plan capacity around total VRAM, not FLOPs (catalog 5-year bet #1).
- Official FP8 (~490 GB) is the quality-ceiling alternative: 8×80 GB,
  TP=8 — same folder shape, different numbers.
