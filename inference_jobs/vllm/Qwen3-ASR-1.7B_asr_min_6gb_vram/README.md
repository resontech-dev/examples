# Qwen3-ASR-1.7B: speech-to-text, mid rung (vLLM). VALIDATE before listing

Alibaba's 1.7B audio-language model (2026, Apache-2.0): a Qwen3 decoder
behind an audio encoder, ~4 GB bf16. Language identification plus
transcription for 30 languages and 22 Chinese dialects, singing voice and
songs with background music. The rung between Whisper large-v3-turbo and
Whisper large-v3 in
`reson_docs/docs/inference/using-small-over-big-models/speech-to-text.md`:
about twice the weights of turbo, and card numbers above large-v3.

Card numbers (vendor, Open ASR leaderboard WER): 4.94 for Qwen3-ASR-1.7B
against 7.44 for Whisper large-v3. Not ours; our measured numbers live in
`reson_docs/docs/benchmarks/inference/` once this job has run.

**Ukrainian is not a supported language of this model.** The card lists
zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl,
sv, da, fi, pl, cs, fil, fa, el, hu, mk, ro. So: English, Italian, Polish,
German go here; Ukrainian stays on `../whisper-large-v3_asr_min_6gb_vram`
(and on `job_speech` fine-tuning). Before choosing for any language,
measure WER on an hour of your own recordings; leaderboard audio is clean.

**Routes.** The model card and the vLLM recipe serve it through
chat-completions with an `audio_url` content part, and vLLM also exposes it
on the OpenAI transcription API. The transcription route through the Ray
Serve LLM router is unproven on the pinned image (same caveat as the
Whisper jobs). `predict.py` uses the chat route by default and the
transcription route with `--transcribe`; a 404 there is reported with a
clear message.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 6 -> 8 GB | 4 | 16 GB | ~20 GB (image 15 + 1.3x4 GB checkpoint) | 12gb (shares) |

Suggested GPUs: any card from 8 GB. Built to **share**: copy the
`LLMConfig` into a chat bundle at `gpu_memory_utilization=0.20` (heavier
than Whisper turbo because of the audio encoder) and lower the chat
fraction by the same amount; the model id `asr` stays.

## What the vLLM recipe specifies

The recipe (`Qwen/Qwen3-ASR` usage guide, 2026-01-29) is `vllm serve
Qwen/Qwen3-ASR-1.7B` with no flags, plus the `vllm[audio]` extra. Day-0
support landed in the January 2026 nightlies, so 0.18.0 is expected to
carry the architecture. VERIFY list on first boot:

- engine starts on the pinned image (unknown architecture = wait for the image re-pin);
- the audio decoders are in the image (the Whisper jobs need them too);
- usable minutes of audio per request at `max_model_len=8192`;
- the language-tag format of the chat reply (`predict.py` splits leniently);
- whether `/v1/audio/transcriptions` answers through the router.

## Client notes

Chat route: send the clip as a base64 `data:audio/wav;base64,...` URL in an
`audio_url` content part, `temperature=0`, no text prompt needed. The reply
is a language tag followed by the transcript; the `qwen-asr` package parses
it with `parse_asr_output`. Transcription route: standard
`client.audio.transcriptions.create`; pass `language` only as a hint, the
model identifies the language itself. Auth as every job here: the key as
both `Authorization: Bearer` and `X-API-Key`.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                                  # sample_data/hello_reson.wav, chat route
python predict.py ./meeting.wav --transcribe       # OpenAI transcription API
python predict.py ./call.mp3 --transcribe --language pl
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Limits

- Central nginx: 32 MB and 300 s per request, so about 17 minutes of 16 kHz
  WAV or hours of MP3; chunk longer files client-side.
- One clip per request (`limit_mm_per_prompt={"audio": 1}`).
- Streaming transcription and word timestamps exist in the `qwen-asr`
  toolkit (vLLM backend, forced aligner sibling), not on this route.
- Translation (`/v1/audio/translations`) is not supported by the model.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `model_source` | `Qwen/Qwen3-ASR-0.6B`: the smaller sibling, ~1.5 GB, same languages |
| `max_model_len` | longer clips per request on a bigger card |
| `max_num_seqs` / `max_ongoing_requests` | parallel streams per card |
| `gpu_memory_utilization` | 0.85 alone; 0.20 when riding along a chat bundle |

## License

Apache-2.0 (model and weights). The `qwen-asr` client toolkit is optional
and not part of this job.
