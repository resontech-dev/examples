# Orpheus-TTS 3B — expressive text-to-speech (predictor mode)

The TTS **quality tier** (`tts-orpheus-3b`, Apache-2.0): a Llama-based 3B
that generates emotive 24 kHz speech — inline emotion tags (`<laugh>`,
`<sigh>`, `<gasp>`) work mid-sentence. 8 stock voices. License-clean, which
in TTS is itself the selling point.

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 8 → 12 GB | 4 | 24 GB | ~18 GB (predictor image 8 + 1.3×7 GB weights) | 12gb |

Suggested GPUs: any 12 GB card (RTX 5070 / 4070 / 3060-12G). Generation
is LM-decode-bound (~realtime on a modern card) — scale with replicas,
not queue depth (`max_ongoing_requests: 2` on purpose).

## Folder layout

```
.
├── inference.yaml          # predictor contract + tight queue
├── scripts/
│   └── orpheus_serve.py    # OrpheusTTS.predict(data) → WAV b64
├── submit.py
├── predict.py              # text → out/tts_<voice>.wav
└── .env.example
```

## Deploy & test

```bash
cp .env.example .env
python submit.py
python predict.py "Welcome to Reson. <sigh> Deploying models used to be hard."
python predict.py --voice leo "Short status update: all systems nominal."
# → saves out/tts_<voice>.wav — listen with afplay/open
```

Or paste `scripts/orpheus_serve.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Request / response

JSON via `predict_json`: `text` (required), `voice`
(tara/leah/jess/leo/dan/mia/zac/zoe), `temperature`, `top_p`,
`max_new_tokens` (7 tokens ≈ one 85 Hz frame; 2048 ≈ 24 s). Response:
`audio_b64` (16-bit WAV), `sample_rate`, `duration_s`.

## ⚠️ Validate before selling

The special-token framing (start/end markers, 7-token SNAC frames,
`128266` offset) follows the canopylabs reference implementation for
`orpheus-3b-0.1` and is **model-revision-specific**. Smoke-test on a GPU
and *listen* to the output before listing. Cheaper/faster alternatives in
the catalog: Kokoro-82M (near-free, CPU) and Chatterbox (cloning, MIT) —
same predictor shape.
