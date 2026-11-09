# Piper TTS — Ukrainian / English / Italian on CPU (predictor mode)

Text-to-speech at the lowest cost, and today the only
Apache/MIT-clean way to synthesize **Ukrainian**: ONNX voices, ~2–5× realtime
per CPU core, espeak-ng embedded in the wheel. Quality is working-grade
(IVR prompts, status read-outs), not studio. Rung 1 of
`reson_docs/docs/inference/using-small-over-big-models/voice.md`.

## Resources

| VRAM | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 0 (CPU) | 4 | 4 GB | ~9 GB (predictor image 8 + voices ~0.2 GB) | any worker |

`device: cpu` — the job lands on a GPU worker (the SDK requires
`gpus_per_worker ≥ 1`) but leaves the card to the chat job. Scale streams
with replicas, not queue depth.

## Voices

Names resolve to `rhasspy/piper-voices` on HF: `uk_UA-ukrainian_tts-medium`
→ `uk/uk_UA/ukrainian_tts/medium/`. ⚠️ Check the exact names in that tree
before deploy — a wrong name fails at replica boot, not at submit.

| Voice | Language | Note |
|---|---|---|
| `uk_UA-ukrainian_tts-medium` | uk | default; also `lada` (x_low), `mykyta`, `oleksa`, `tetiana` folders exist |
| `en_US-lessac-medium` | en | clean neutral English |
| `it_IT-paola-medium` | it | Italian; `riccardo` (x_low) is the lighter one |

## Request / response

JSON via `predict_json`: `text` (required), `voice`, `length_scale`
(2.0 = twice as slow), `noise_scale`, `noise_w_scale`, `volume`. Response:
`audio_b64` (16-bit WAV), `sample_rate` (22 050 Hz for medium voices),
`duration_s`, `voice`, `elapsed_ms`.

This is the predictor route, not OpenAI's `/v1/audio/speech` — a caller
that expects the OpenAI route needs a 10-line adapter (text →
`predict_json` → WAV bytes).

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                                              # Ukrainian, default voice
python predict.py --voice it_IT-paola-medium "Il suo ordine arriva domani."
# → out/tts_<voice>.wav
```

Or paste `scripts/piper_serve.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Next rungs

`../Orpheus-TTS-3B_tts_min_8gb_vram` — expressive English on a 12 GB card
(Llama-3.2 base, released Apache-2.0). Kokoro-82M (EN / IT,
Apache) needs `espeak-ng` from apt for Italian, which a pip-only
`requirements:` cannot install — not written yet. Your own Ukrainian
voice: train a Piper voice on 1–3 h of studio recordings (outside the
platform), drop the `.onnx` + `.onnx.json` into S3 and load from `model.path`.
