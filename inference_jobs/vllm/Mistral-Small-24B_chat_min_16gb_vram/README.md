# Mistral Small 24B (AWQ) — business chat (vLLM)

Dense 24B, the popular EU-deployment pick (`chat-mistral-small-24b`):
fast, low-refusal business assistant with first-class function calling.
Apache-2.0. One 24 GB worker, AWQ 4-bit.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 16 → 24 GB | 4 | 48 GB | ~39 GB (image 15 + 1.3×13 GB checkpoint) | 24gb |

Suggested GPUs: RTX 3090 / 4090 / L4 / A10G (24 GB — as configured, 32k
context); 2×24 GB one node → official bf16 checkpoint with TP=2
(`STRICT_PACK`) for a measurable quality bump.

## Deploy

```bash
cp .env.example .env
python submit.py
python predict.py "Summarize the key obligations of a data processor under GDPR in 5 bullets."
python predict.py --tools        # function-calling smoke test
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Tool calling / agentic use

`enable_auto_tool_choice=True` + `tool_call_parser="mistral"` are **on** —
the Mistral parser is long-stable in vLLM. The endpoint works as an agent
backend (LangChain, function-calling clients) out of the box.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `model_source` | ⚠️ community AWQ re-upload — **vet before selling**, or swap a newer 3.x AWQ |
| `max_model_len` | 32768 shipped; KV affords more on this card if you drop `max_num_seqs` |
| `autoscaling_config` | DP scale-out — raise together with `cluster.num_workers` |

## Notes

- **Alignment posture**: lightest touch of the catalog's chat models —
  minimal moralizing, refuses less. Good for business drafting; pair with
  the guard bundle when exposing to end users.
- The official bf16 checkpoint is gated on HF — community AWQ re-uploads
  usually aren't, but check; gated sources need `HF_TOKEN` via
  `runtime_env.env_vars`.
