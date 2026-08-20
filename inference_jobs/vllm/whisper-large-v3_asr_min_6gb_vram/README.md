# Whisper large-v3 — speech-to-text (vLLM) ⚠️ validate before listing

Meeting/call transcription on-prem (`asr-whisper-large-v3`, MIT, priority
v1): one 12 GB worker runs 10–30 parallel streams; breakeven vs the
OpenAI Whisper API (~$0.36/audio-hour) at a few hundred audio-hours/month
— and recordings never leave the customer.

**⚠️ The catalog's own caveat applies: validate Whisper on the pinned
vLLM before listing.** Encoder-decoder + the `/v1/audio/transcriptions`
route through Ray Serve LLM's router is the one path in this repo we
haven't proven on the exact pinned image. `predict.py` fails with a clear
message if the route is absent; plan B is a predictor-mode job wrapping
faster-whisper.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 6 → 10 GB | 4 | 20 GB | ~24 GB (image 15 + 1.3×3 GB checkpoint) | 12gb |

Suggested GPUs: any 12 GB card (RTX 5070 / 4070 / 3060-12G); it also
*shares* a bigger worker — see the catalog's `(shares)` tiers (e.g. ride
along a chat worker at a small `gpu_memory_utilization`).

## Deploy & test

```bash
cp .env.example .env
python submit.py
python predict.py                    # transcribes sample_data/hello_reson.wav
python predict.py ./meeting.wav
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

`sample_data/hello_reson.wav` is a short synthesized speech clip for the
smoke test.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `max_num_seqs` / `max_ongoing_requests` | parallel stream count |
| `model_source` | `openai/whisper-large-v3-turbo` → 5–8× faster decode, slight quality cost |
| deliberately absent KV knobs | fp8 KV / prefix caching are version-sensitive for encoder-decoder — leave defaults |

## Notes

- Whisper's decode budget is ≤ 448 tokens per 30 s window — `max_model_len=448`
  is the model's shape, not a squeeze.
- Long recordings are chunked client-side (the OpenAI API contract);
  `predict.py` sends one file per request.
