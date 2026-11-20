# Nemotron 3.5 ASR Streaming 0.6B: live speech-to-text, Ukrainian included (NeMo predictor). VALIDATE before listing

NVIDIA's cache-aware streaming model (released 2026-06-04, OpenMDW-1.1,
ready for commercial use): a FastConformer encoder with an RNN-T decoder
that transcribes live audio in chunks of 80 ms to 1.12 s, with punctuation
and capitals. 40 language locales in three tiers; the 19
transcription-ready ones are en-US, en-GB, es-US, es-ES, fr-FR, fr-CA,
it-IT, pt-BR, pt-PT, nl-NL, de-DE, tr-TR, ru-RU, ar-AR, hi-IN, ja-JP, ko-KR,
vi-VN and **uk-UA**. It is the multilingual successor of
`nvidia/nemotron-speech-streaming-en-0.6b`; NVIDIA recommends that English
model for English-only use.

One endpoint, two routes:

- **live sessions**: the client sends the microphone's audio piece by
  piece and gets the text back while the person is still speaking, about
  one frame (560 ms by default) plus the network round trip behind them;
- **whole files**: one request per recording.

For live captions in English and 12 other languages without Ukrainian, the
WebSocket job [`../../vllm/Voxtral-Mini-4B-Realtime_asr_streaming_min_16gb_vram`](../../vllm/Voxtral-Mini-4B-Realtime_asr_streaming_min_16gb_vram/README.md)
is the other option (its WebSocket route through the platform is not
proven yet).

## Live sessions

The platform has no WebSocket route for predictor jobs, so a live session
is plain HTTP: one `predict` request per piece of audio. The state between
requests (each session's encoder cache, decoder hypothesis and the text so
far) stays on the GPU in NeMo 3.0's cache-aware streaming pipeline
(`nemo.collections.asr.inference`), which keeps one state per stream and
the language per stream.

```
client                                    job
{"action":"start","language":"uk-UA"}  →  {"session":"…","frame_ms":560,"sample_rate":16000,"encoding":"pcm_s16le"}
{"action":"append","session":"…",      →  {"text":"Добрий день,","partial":"це тест","delta":"Добрий день,","final":false,…}
 "audio_b64":"<PCM16 LE 16 kHz mono>"}
… one append per frame_ms of audio …
{"action":"stop","session":"…"}        →  {"text":"Добрий день, це тест.","final":true,…}
```

- **Audio**: raw PCM16 little-endian, 16 kHz, mono, base64. Any length
  per append, up to 30 s. The job cuts it into `frame_ms` frames and keeps
  the remainder for the next call. `append` with `"final": true` (or
  `stop`) pads the last frame and closes the session.
- **Text**: `text` holds the finalized text since the start. The model
  finalizes at the end of an utterance (800 ms of silence) and at the end
  of the session. `partial` holds the words since the last finalization;
  they can still change. `delta` is the part of `text` this call
  finalized. Show `text` + `partial` (dim); append `delta` to a document.
- **Order**: one append at a time per session; wait for the reply before
  sending the next. A slow reply just means the next append carries more
  audio.
- **Lifetime**: a session without a request for `session_idle_s` (60 s) is
  closed. Sessions do not survive a restart, and `num_workers` must stay 1:
  a second replica would not have the session.
- **Language**: `"language"` on `start`, e.g. `"uk-UA"` or `"uk"` (see
  [Languages](#languages)); without it, `init_args.live_language` applies
  (`null`: `"auto"`). The `start` reply carries `languages`, the keys the
  model takes; an unknown language is an error whose message starts with
  `unknown language`.

**NeMo 3.0.0 bug, patched here.** The streaming pipeline builds a one-hot
language vector per session and passes it to the model wrapper's
`stream_step(prompt_vectors=...)`. 3.0.0's wrapper never uses it, so the
decoder gets encoder output without the language projection it was trained
on and emits no tokens: sessions answer 200 with empty `text` and `partial`,
while whole files work. NeMo fixed it after the release (#16047). The
predictor installs the same fix at boot (`_patch_prompt_injection`: the
model's `prompt_kernel` applied right after the encoder step, in float32)
and skips it on a NeMo that has its own. `health()["live"]["prompt_patch"]`
says whether it is on. The live copy also runs in float32, not bfloat16: in
bfloat16 the one-hot prompt is rounded away inside the projection.

Latency per spoken word ≈ one frame (`live_att_context_size` right value:
6 → 560 ms) + the HTTP round trip through the platform + a few tens of ms
on the GPU. A browser client sends the same requests with `fetch`: the
JSON body goes as `Content-Type: application/octet-stream`, because the
platform's predict wrapper answers `application/json` bodies without
`url` / `data_b64` with 400.

## Two passes: Nemotron draft + Parakeet / Whisper final text

Nemotron is built for latency, not accuracy: FLEURS uk WER 13.7 % at the
560 ms frame. Parakeet-TDT-0.6B-v3 gets 6.8 % on the same test set, but it
transcribes finished audio. `init_args.final_model` puts both on one
endpoint:

```
mic ─► append ─► Nemotron (streaming) ─► partial: the draft, word by word
                     │ phrase finished (~0.8 s of silence)
                     ▼
              that phrase's audio ─► final model (background worker) ─► text: grows phrase by phrase
```

| `final_model` | checkpoint | VRAM (fp16/fp32) | notes |
|---|---|---|---|
| `null` | Nemotron only | — | the single-pass behavior above |
| `parakeet` | `nvidia/parakeet-tdt-0.6b-v3` (NeMo) | ~2.5 GB | detects the language itself (25 European languages, Ukrainian included); FLEURS uk 6.8 % |
| `whisper` | `openai/whisper-large-v3` (transformers) | ~3.1 GB | takes the session's language as an ISO code (`uk`) |
| `whisper-turbo` | `openai/whisper-large-v3-turbo` | ~1.6 GB | faster, a little less accurate |

`final_model_id` swaps the checkpoint inside a family: a NeMo checkpoint
for `parakeet`, a Whisper checkpoint (a Ukrainian fine-tune, say) for
`whisper`.

- **Same protocol.** `text` only grows, now with the final model's text
  (`delta` stays append-only). `partial` holds the drafts of phrases still in
  the second pass, then the words being spoken. A client needs no change; it
  sees the dim draft turn into the final text about a phrase later.
- **The closing call** (`final: true` / `stop`) waits up to 60 s for the last
  phrases' second pass, outside the session lock.
- **Whole files** go to the final model. The Nemotron file copy is not
  loaded.
- **Order**: one worker re-transcribes phrases in the order they end, so the
  text never jumps. A failed second pass keeps Nemotron's draft for that
  phrase (logged).
- **Phrase audio** is cut where Nemotron finalizes, in a pause. Before the
  first word only the last 2 s of silence are kept.

```bash
python submit.py --final parakeet          # or whisper / whisper-turbo; plain submit.py = the yaml as is
```

`--final` submits a temporary copy of `inference.yaml` with `final_model`
set, and names the job after it, so several variants can run side by side
for an A/B on the same recordings.

## Latency setting

`att_context_size` is `[left, right]` in 80 ms frames; the right value sets
the frame a live session waits for:

| right | frame | |
|---|---|---|
| 0 | 80 ms | fastest, least accurate; 12 requests per second per session |
| 1 | 160 ms | |
| 3 | 320 ms | |
| 6 | 560 ms | **live default**: about 2 requests per second |
| 13 | 1.12 s | most accurate (the whole-file default) |

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 8 -> 16 GB (4 GB with `live: false`) | 4 | 16 GB | ~20 GB (NeMo + torch + checkpoint) | 12gb (shares) |

The card gives no VRAM figure. Weights are ~2.4 GB in fp32; the live
route loads a second copy (bf16 compute on cards that support it) plus an
encoder cache per session slot (`max_sessions`).

## VERIFY on first boot

- `nemo_toolkit[asr]>=3.0,<3.1` installs from PyPI on the worker image (Python 3.11) and boot fits the timeout. NeMo main, which the model card names, needs Python >= 3.12 and does not install there; 3.0.0 has the model's class (`EncDecRNNTBPEModelWithPrompt`, saved by NeMo 2.8.0rc0) and the streaming pipeline;
- `pyarrow<26` stays while the platform installs `numpy<2` for predictor jobs (see the Parakeet job);
- **live**: `PipelineBuilder.build_pipeline` loads this model with the config in `_live_config` (NeMo's `cache_aware_rnnt.yaml`, trimmed, float32); `health()["live"]` shows `frame_ms: 560`, `prompt_patch: true` and the model's languages, `uk-UA` and `auto` among them;
- **live, words at all**: `predict.py --live sample_data/hello_reson.wav --language uk-UA` and `en-US` print the whole-file words (empty `text`/`partial` on every append was the NeMo 3.0.0 prompt bug above); a session with `auto`;
- **whole files, language**: `python predict.py` (`hello_reson.wav`) without a language, with `en-US`, `uk-UA` and `uk`: no errors, the right text; 20 whole-file requests in a row without a language: none fails (on the Lhotse route about half did);
- **auto = index 101**: if the checkpoint has no `auto` key, the text from requests without a language is as good as with the language pinned on clear speech (the added `auto: 101` follows NeMo's prompt-index dataset; not confirmed for this checkpoint);
- **non-Lhotse dataset**: `transcribe()` with `use_lhotse=False` builds its dataloader from the temporary manifest (`duration: 100000`) on this NeMo; if it does not, the fallback is to run the file through a live session (language already works there, but at right context 6 and holding the lock for the whole file);
- **live**: `python predict.py --live sample_data/hello_reson.wav --language uk-UA` prints the same words as the whole-file route; then `--mic` with real speech; note the round-trip median it prints;
- **live**: two sessions at once (two terminals) keep their texts apart; a session left open is closed after 60 s;
- `set_default_att_context_size([56, 13])` exists on the encoder (the call is skipped if not);
- **two passes**: `submit.py --final parakeet`, then `--mic --language uk-UA`: the dim draft turns into Parakeet's text a phrase later; `health()["final_model"]`; peak VRAM with Nemotron + Parakeet and with Nemotron + Whisper large-v3;
- **two passes, Whisper**: `transformers` as NeMo 3.0 resolves it loads `WhisperForConditionalGeneration` and generates with `language="uk"`; Whisper on a phrase that is mostly silence (hallucinated "Дякую за перегляд"-style text);
- **two passes, timing**: the second pass per phrase on the 5070 Ti (Parakeet, Whisper large-v3, turbo), and how far `text` trails `partial` while speaking;
- with automatic detection the text ends with a tag such as `<uk-UA>`; the predictor cuts it off into `language`;
- Ukrainian accuracy against Whisper large-v3 and Parakeet v3 on the same recordings, at right context 13, 6 and 3.

## Languages

Both routes take the same `language` values and map them the same way
(`resolve_language` in the predictor):

| sent | goes to the model as |
|---|---|
| nothing or `""` | `auto` (the model detects the language and tags the text; the tag comes back as `language`) |
| a key of the model's prompt dictionary: `uk-UA`, `en-US`, `en`, `de-DE`, … | as is |
| an ISO code: `uk`, `de`, `pl` | the first key that starts with it: `uk` → `uk-UA` |
| anything else | an error: `unknown language 'xx'; this model takes: …` |

The checkpoint's prompt dictionary may have no `auto` key. NeMo's own
prompt-index dataset then uses index 101 for auto, so the predictor adds
`auto: 101` to the dictionary of both model copies at boot (only when the
model has more than 101 prompts). `health()["languages"]` lists the keys.

## Whole files

Send the audio file as the body (`client.predict(bytes)`), or JSON
`{"audio_b64": "...", "language": "uk-UA", "timestamps": false}` with
`client.predict_json`. WAV, FLAC, OGG and MP3 are decoded with soundfile (no
FFmpeg on the worker).

The language reaches the model. NeMo 3.0.0's `transcribe()` uses a Lhotse
dataset by default. That dataset reads the language from the manifest's
`lang` field, which `transcribe()` never writes, and ignores `target_lang`.
About half the files then failed with `Unknown prompt key: 'None'`, and the
other half ran as auto. The predictor calls `transcribe()` with
`RNNTPromptTranscribeConfig(use_lhotse=False, target_lang=…, num_workers=0)`.
Without Lhotse the batch carries no prompt index, and the model takes
`target_lang` from its prompt dictionary. Reply:

```json
{"text": "...", "language": "uk-UA", "segments": [], "duration_s": 12.3,
 "att_context_size": [56, 13], "model_id": "...", "elapsed_ms": 640.0}
```

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" python-dotenv numpy soundfile scipy   # Python 3.11+
pip install sounddevice                          # for --mic
cp .env.example .env
python submit.py
python predict.py --mic --language uk-UA         # speak; Ctrl-C ends the session
python predict.py --live ./call.wav --language uk-UA   # a file at speaking speed, like a microphone
python predict.py                                # whole file: sample_data/hello_reson.wav
python predict.py ./call.mp3 --language uk-UA    # whole file, language pinned
```

## Limits

- 32 MB per request (24 MB of audio with `--language`, which goes as base64). One request to a predictor job is cut off after 300 s (set on the Ray head node; the predict wrapper answers 504 at 290 s); live appends take milliseconds, so only whole files come near it.
- One replica: live sessions live in its memory. Up to `max_sessions` (32) at once.
- Live requests share one lock on the GPU: each frame takes tens of ms, so dozens of sessions fit before they queue behind each other. Measure it.

## What to change

| where | for what |
|---|---|
| `init_args.model_id` | `nvidia/nemotron-speech-streaming-en-0.6b` for English only |
| `init_args.live_att_context_size` | the delay behind the speaker: `[56, 3]` for 320 ms, `[56, 13]` for the most accurate |
| `init_args.live_language` | the default when a session does not send one |
| `init_args.max_sessions` / `session_idle_s` | parallel sessions, idle timeout |
| `init_args.final_model` / `final_model_id` | two passes: `parakeet`, `whisper`, `whisper-turbo` (or `submit.py --final …`) |
| `init_args.live: false` | whole files only, one model copy (4 GB) |
| `init_args.att_context_size` | the whole-file latency/accuracy point |
| `init_args.timestamps` | `true` for segment times on files (VERIFY for this decoder) |

## License

OpenMDW-1.1 (model card: "ready for commercial use").
