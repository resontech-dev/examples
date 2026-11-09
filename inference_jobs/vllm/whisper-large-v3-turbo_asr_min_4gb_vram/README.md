# Whisper large-v3-turbo — speech-to-text, smallest rung (vLLM) ⚠️ validate before listing

Speech-to-text at the lowest cost: same encoder as large-v3,
decoder pruned 32 → 4 layers, **5–8× faster decode at ~1.6 GB**, MIT.
Rung 1 of `reson_docs/docs/inference/using-small-over-big-models/speech-to-text.md`.
English and Italian: use it as is. Ukrainian: measure WER against
`../whisper-large-v3_asr_min_6gb_vram` on an hour of your own calls first —
turbo loses more on less common languages, and that is where fine-tuning
(`job_speech`) pays off.

**The large-v3 caveat applies unchanged**: the `/v1/audio/transcriptions`
route through the Ray Serve LLM router is unproven on the pinned image.
`predict.py` fails with a clear message if the route is absent; plan B is a
predictor-mode job wrapping faster-whisper.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 4 → 6 GB | 4 | 16 GB | ~18 GB (image 15 + 1.3×1.6 GB checkpoint) | 12gb (shares) |

Suggested GPUs: any card; built to **share** — copy the `LLMConfig` into the
Qwen3-8B agents bundle at `gpu_memory_utilization=0.15` for one-card
text + voice (see `serve_module.py`).

## Client notes

Standard OpenAI transcription call (`client.audio.transcriptions.create`).
Pass `language: uk` (or `en`, `it`) when the caller's language is known —
auto-detect on short Ukrainian clips sometimes flips to Russian. Auth as
every job here: the key as both `Authorization: Bearer` and `X-API-Key`.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                              # sample_data/hello_reson.wav
python predict.py ./dzvinok.mp3 --language uk
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Limits (same as large-v3)

- 30 s windows, ≤ 448 decode tokens each (`max_model_len=448` is the model's shape).
- Central nginx: 32 MB and 300 s per request → ~17 min of 16 kHz WAV or hours of MP3; chunk longer files client-side.
- Speaker diarization and streaming-during-the-call are separate services, not this route.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `model_source` | `openai/whisper-large-v3` — quality tier, 3 GB (the sibling job) |
| `max_num_seqs` / `max_ongoing_requests` | parallel streams per card |
| `gpu_memory_utilization` | 0.85 alone; 0.15 when riding along a chat bundle |
