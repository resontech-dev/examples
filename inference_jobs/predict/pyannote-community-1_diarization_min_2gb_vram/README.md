# pyannote Community-1: who spoke when (speaker diarization predictor). VALIDATE before listing

> **Blocked on the platform today.** pyannote.audio 4 needs numpy >= 2
> (through pyannote.core 6), and the platform installs `numpy<2` for every
> predictor job (flare-dispatcher `src/include/base_requirements.txt`), so
> the install fails before the job boots. pyannote.audio 3 has no
> Community-1 pipeline. Submit this job once predictor jobs keep numpy 2:
> Ray 2.60 images plus the dispatcher change proposed in reson_docs
> `docs/inference/using_requirements.md` ("Що зміниться").

The open pipeline of pyannote.audio 4 (code MIT, weights CC-BY-4.0):
voice activity, speaker embeddings and clustering in one call. A recording
goes in; turns come out, each with a speaker label (`SPEAKER_00`,
`SPEAKER_01`, …) and start/end times. It does not transcribe: pair it with
a speech-to-text job for "who said what". `predict.py --with-whisper` does
exactly that against a running Whisper job.

This is the offline answer to one recording with several voices: a meeting
room, a phone on speaker, an interview. A call recorded in two channels
(the playground's "Mic + call audio") already knows who is who and does not
need it.

Two outputs: `turns` may overlap (two people at once); `exclusive_turns`
never do, one speaker at a time, which is what cutting audio for
transcription needs.

## Weights: gated, shipped by you

The pipeline is gated on Hugging Face: you accept its conditions once and
download it with your own token. `pack_weights.py` does that on your
machine and packs the folder into `model/speaker-diarization-community-1.tar.gz`
(git-ignored); `submit.py` uploads that file as the job's `model.path`, and
`load_weights()` unpacks it and loads the pipeline offline. **The token is
never shipped with the job.**

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 2 -> 4 GB | 4 | 8 GB | ~10 GB (torch + pyannote; weights ~35 MB) | any GPU (shares); CPU works, slower |

## VERIFY on first boot

- `pyannote.audio>=4.0,<5` + `torchaudio` install on the worker image (`rayproject/ray:2.55.1-py311-gpu`) with a matching torch;
- the import warns about torchcodec/FFmpeg (the image has no FFmpeg) and still works: audio reaches pyannote as a waveform, which never calls torchcodec;
- `load_weights()` finds `config.yaml` in the unpacked folder and `Pipeline.from_pretrained(<folder>)` loads without network;
- time and memory for a 10-minute and a 60-minute recording on GPU and on CPU;
- quality on our own recordings: speaker count, short interjections, overlapping speech.

## Client notes

Send the audio file as the body (`client.predict(bytes)`), or JSON
`{"audio_b64": "...", "num_speakers": 2}` (or `min_speakers` /
`max_speakers`) with `client.predict_json`. WAV, FLAC, OGG and MP3 are
decoded with soundfile. Reply:

```json
{"speakers": ["SPEAKER_00", "SPEAKER_01"],
 "turns": [{"speaker": "SPEAKER_00", "start": 0.52, "end": 4.1}],
 "exclusive_turns": [{"speaker": "SPEAKER_00", "start": 0.52, "end": 4.1}],
 "duration_s": 61.2, "elapsed_ms": 2300.0}
```

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" python-dotenv huggingface_hub   # Python 3.11+
cp .env.example .env            # + HF_TOKEN, an account that accepted the conditions
python pack_weights.py          # -> model/speaker-diarization-community-1.tar.gz
python submit.py
python predict.py ./meeting.mp3
python predict.py ./meeting.mp3 --speakers 3
# who said what: a running Whisper job's URL and key in .env as WHISPER_URL / WHISPER_API_KEY
pip install openai soundfile numpy
python predict.py ./meeting.mp3 --with-whisper --language uk
```

`--with-whisper` joins neighbouring turns of one speaker (pause up to 1 s,
up to 28 s per piece), cuts the audio locally and sends each piece to
Whisper on its own.

## Limits

- 32 MB and 300 s per request at the central proxy: about an hour of 64 kbit/s MP3.
- Offline only: the whole recording per request. Live diarization (NVIDIA Sortformer streaming, up to 4 speakers) is a separate job.
- Labels are per recording: `SPEAKER_00` in one file is not the same person in the next.

## License

Weights CC-BY-4.0, code MIT. Access requires accepting the conditions on
<https://huggingface.co/pyannote/speaker-diarization-community-1>.
