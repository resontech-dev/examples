# Voxtral Mini 4B Realtime: streaming speech-to-text (vLLM, WebSocket). VALIDATE before listing

Mistral's natively streaming ASR model (2026-02, Apache-2.0, 4B parameters,
~9 GB bf16). Unlike Whisper, which hears finished 30-second windows, it
takes audio while it is spoken and writes the text a fixed delay behind the
speaker: 480 ms by default, any multiple of 80 ms from 80 to 1200 ms, or
2400 ms. This is what real live captions look like, without the repeated
requests the playground uses with Whisper.

**Languages (13): en, fr, es, de, ru, zh, ja, it, pt, nl, ar, hi, ko. Not
Ukrainian.** For Ukrainian live audio, measure
`../../predict/Nemotron-3.5-ASR-Streaming-0.6B_asr_min_4gb_vram` (uk-UA)
instead.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Route

vLLM's realtime API over WebSocket, `/v1/realtime`:

1. server: `session.created`
2. client: `session.update` `{model}`, then `input_audio_buffer.commit`
3. client: `input_audio_buffer.append` `{audio}` per chunk: base64 of PCM16, 16 kHz mono (4096 bytes in vLLM's example)
4. client: `input_audio_buffer.commit` `{final: true}` when the audio ends
5. server: `transcription.delta` `{delta}` as text arrives, `transcription.done` `{text, usage}` at the end, `error` `{error}`

Auth as every job here: the key as both `Authorization: Bearer` and
`X-API-Key` on the handshake.

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 16 -> 24 GB | 4 | 24 GB | ~30 GB (image 15 + 1.3x9 GB checkpoint) | 16gb |

One token is about 80 ms of audio, so `max_model_len=45000` is about an
hour per session; the default 131072 (~3 h) does not fit next to the
weights on 16 GB. KV for one full-hour session is several GB; short sessions
share it.

## VERIFY on first boot

- the engine starts on the pinned image (vLLM 0.29) with `tokenizer_mode="mistral"` and `enforce_eager=True`, the setup of vLLM 0.18's realtime example; 0.29 meets the recipe's >= 0.20, so try PIECEWISE CUDA graphs once eager works;
- `mistral_common[audio]` (>= 1.9) is in the image;
- **`/v1/realtime` through Ray Serve LLM's router**: `build_openai_app` serves the REST routes; a WebSocket route there is unproven. If the handshake gets 404, the route is not exposed: plan B is `predict.py --transcribe` on the same job, and a real fix is serving vLLM's own app in a predictor-style job;
- the WebSocket upgrade through the per-job nginx and the central proxy;
- latency (first delta after speech starts) and accuracy on our recordings in the 13 languages;
- long sessions: vLLM issues #39996 and #38233 report hangs or crashes on long sessions; the recipe suggests restarting sessions periodically.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" python-dotenv websockets soundfile numpy scipy openai   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                          # sample_data/hello_reson.wav, streamed as fast as possible
python predict.py ./talk.wav --pace        # at speaking speed, like a microphone
python predict.py ./talk.wav --transcribe  # plan B, whole file
```

## What to change

| where | for what |
|---|---|
| `max_model_len` | longer sessions on a bigger card (131072 ≈ 3 h) |
| `max_num_seqs` / `max_ongoing_requests` | parallel live sessions |
| `enforce_eager` | try dropping it (the pinned vLLM 0.29 meets the recipe's >= 0.20) and set `compilation_config={"cudagraph_mode": "PIECEWISE"}` (recipe) |
| delay | `transcription_delay_ms` in the checkpoint's `tekken.json`: a fork of the checkpoint, not a serve flag |

## License

Apache-2.0 (model and weights).
