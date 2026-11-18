# Parakeet TDT 0.6B v3: speech-to-text, 25 European languages (NeMo predictor). VALIDATE before listing

NVIDIA's 600M FastConformer-TDT model (CC-BY-4.0, ready for commercial use):
transcription with punctuation, capitals and segment timestamps in 25
European languages, **Ukrainian and Russian included**, with the language
found by the model itself. A small, very fast alternative to Whisper for the
same calls: one request carries a whole call, no 30-second windows.

Not a vLLM model, so this is a predictor job: NeMo runs it inside
`predict()`. The whole file goes in one request; the reply is the text and
its segments with start and end times.

Languages (model card): bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it,
lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk.

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 4 -> 8 GB | 4 | 16 GB | ~20 GB (NeMo + torch + 2.5 GB checkpoint) | 12gb (shares) |

The card gives no VRAM figure ("at least 2 GB RAM to load"). Weights are
~2.4 GB in fp32; long audio adds encoder activations, which is why files
over 10 minutes switch to local attention (`local_attention_after_s`).

## VERIFY on first boot

- `nemo_toolkit[asr]` installs on the worker image (`rayproject/ray:2.55.1-py311-gpu`, Python 3.11) and boot fits the timeout;
- `pyarrow<26` stays in `requirements:` while the platform installs `numpy<2` for predictor jobs: the first boot (2026-11-18) died on `pyarrow requires NumPy 2.0 or newer, found 1.26.4`;
- `from_pretrained` pulls the checkpoint; `transcribe()` returns hypotheses with `.text` and `.timestamp["segment"]`;
- `change_attention_model("rel_pos", [-1, -1])` restores the checkpoint's own attention after a long file;
- VRAM and time for a 10-minute and a 60-minute file (64 kbit/s MP3 fits the 32 MB request limit);
- Ukrainian accuracy against Whisper large-v3 on the same recordings (an hour of your own calls, not leaderboard audio).

## Client notes

Send the audio file as the request body (`client.predict(bytes)`), or JSON
`{"audio_b64": "...", "timestamps": false}` with `client.predict_json`.
WAV, FLAC, OGG and MP3 are decoded with soundfile, so the worker needs no
FFmpeg; convert M4A/AAC first. Reply:

```json
{"text": "...", "segments": [{"start": 0.0, "end": 2.4, "text": "..."}],
 "duration_s": 12.3, "local_attention": false, "model_id": "...", "elapsed_ms": 812.0}
```

This route is not OpenAI-compatible: the playground's Transcription tab does
not talk to it yet.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                     # sample_data/hello_reson.wav
python predict.py ./call.mp3          # your own recording
python predict.py ./call.mp3 --json   # the whole response
```

## Limits

- 32 MB and 300 s per request at the central proxy: about an hour of 64 kbit/s MP3.
- One transcription at a time per replica (a lock); add replicas for parallel files.
- Streaming exists in NeMo (chunked inference script), not on this route.

## What to change

| where | for what |
|---|---|
| `init_args.local_attention_after_s` | lower on a small card if long files run out of memory |
| `init_args.timestamps` | `false` for text only |
| `cluster.num_workers` | parallel files |

## License

CC-BY-4.0 (model card: "ready for commercial/non-commercial use").
Attribution to NVIDIA is required when you redistribute.
