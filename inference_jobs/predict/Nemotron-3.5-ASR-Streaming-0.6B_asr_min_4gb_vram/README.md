# Nemotron 3.5 ASR Streaming 0.6B: low-latency speech-to-text, Ukrainian included (NeMo predictor). VALIDATE before listing

NVIDIA's cache-aware streaming model (released 2026-06-04, OpenMDW-1.1,
ready for commercial use): a FastConformer encoder with an RNN-T decoder
that transcribes live audio in chunks of 80 ms to 1.12 s, with punctuation
and capitals. 40 language locales in three tiers; the 19
transcription-ready ones are en-US, en-GB, es-US, es-ES, fr-FR, fr-CA,
it-IT, pt-BR, pt-PT, nl-NL, de-DE, tr-TR, ru-RU, ar-AR, hi-IN, ja-JP, ko-KR,
vi-VN and **uk-UA**. It is the multilingual successor of
`nvidia/nemotron-speech-streaming-en-0.6b`; NVIDIA recommends that English
model for English-only use.

**What this job does and does not do.** The platform has no WebSocket route
for a live session, so this predictor transcribes whole files per request.
It answers the question that matters before building a live mode: how good
and how fast is this model on our audio, at the latency it is set to
(`att_context_size`). A live session (audio chunks in, partial text out,
the encoder cache kept per session) is the next step if the numbers are
good; NeMo's `speech_to_text_cache_aware_streaming_infer.py` is the
reference for it.

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 4 -> 8 GB | 4 | 16 GB | ~20 GB (NeMo + torch + checkpoint) | 12gb (shares) |

The card gives no VRAM figure. Weights are ~2.4 GB in fp32; the limited
attention context (56 frames left) keeps memory linear in the audio length.

## Latency setting

`att_context_size` is `[left, right]` in 80 ms frames; the right value sets
the chunk a live session would wait for:

| right | chunk | |
|---|---|---|
| 0 | 80 ms | fastest |
| 1 | 160 ms | |
| 3 | 320 ms | |
| 6 | 560 ms | |
| 13 | 1.12 s | most accurate (this job's default) |

## VERIFY on first boot

- `nemo_toolkit[asr]>=3.0` installs from PyPI on the worker image (Python 3.11) and boot fits the timeout. NeMo main, which the model card names, needs Python >= 3.12 and does not install there; 3.0.0 has the model's class (`EncDecRNNTBPEModelWithPrompt`, saved by NeMo 2.8.0rc0);
- `pyarrow<26` stays while the platform installs `numpy<2` for predictor jobs (see the Parakeet job);
- `set_default_att_context_size([56, 13])` exists on the encoder (the call is skipped if not);
- with automatic detection the text ends with a tag such as `<uk-UA>`; the predictor cuts it off into `language`;
- `transcribe(..., target_lang="uk-UA")`: NeMo 3.0.0's signature takes it (`**prompt`); confirm the text changes with the tag;
- Ukrainian accuracy against Whisper large-v3 and Parakeet v3 on the same recordings, at right context 13 and 3.

## Client notes

Send the audio file as the body (`client.predict(bytes)`), or JSON
`{"audio_b64": "...", "language": "uk-UA", "timestamps": false}` with
`client.predict_json`. WAV, FLAC, OGG and MP3 are decoded with soundfile (no
FFmpeg on the worker). Reply:

```json
{"text": "...", "language": "uk-UA", "segments": [], "duration_s": 12.3,
 "att_context_size": [56, 13], "model_id": "...", "elapsed_ms": 640.0}
```

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                              # sample_data/hello_reson.wav
python predict.py ./call.mp3                   # language detected by the model
python predict.py ./call.mp3 --language uk-UA  # pinned
```

## Limits

- 32 MB per request (24 MB of audio with `--language`, which goes as base64). One request to a predictor job is cut off after 300 s (set on the Ray head node; the predict wrapper answers 504 at 290 s).
- Whole files only on this route; no live session yet.
- One transcription at a time per replica (a lock).

## What to change

| where | for what |
|---|---|
| `init_args.model_id` | `nvidia/nemotron-speech-streaming-en-0.6b` for English only |
| `init_args.att_context_size` | the latency/accuracy point to measure |
| `init_args.timestamps` | `true` for segment times (VERIFY for this decoder) |

## License

OpenMDW-1.1 (model card: "ready for commercial use").
