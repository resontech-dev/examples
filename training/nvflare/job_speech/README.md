# job_speech — Speech Recognition (Whisper + LoRA + LibriSpeech-100)

Federated LoRA fine-tuning of OpenAI's Whisper ASR model on **LibriSpeech-100h** — the canonical ASR benchmark Whisper itself reports against on its model card.

## Model

- **Name**: `openai/whisper-small`
- **HF link**: https://huggingface.co/openai/whisper-small
- **Architecture**: Whisper encoder-decoder transformer, 244 M parameters
- **Library**: `transformers` + `peft` (LoRA on `q_proj`, `v_proj`)
- **Pretrained**: 680,000 hours of multilingual web audio

## Published metrics (verbatim from `openai/whisper-small` HF card)

| Test set | WER |
|---|---|
| **LibriSpeech (clean)** | **3.432** |
| **LibriSpeech (other)** | **7.628** |
| Common Voice 11.0 | 87.300 |
| Common Voice 13.0 | 125.698 |

Reference: see https://huggingface.co/openai/whisper-small (Evaluation section).

The clean/other LibriSpeech numbers are what matters for our setup since we're fine-tuning ON LibriSpeech.

For comparison: `whisper-base` (74M) hits LibriSpeech-clean WER 5.009 / other 12.849, and `whisper-tiny` (39M) is much worse. Whisper-small is the sweet spot for FL.

## Dataset

- **Name**: `openslr/librispeech_asr` (config: `clean`, split: `train.100`)
- **HF link**: https://huggingface.co/datasets/openslr/librispeech_asr
- **Origin**: Panayotov et al. (2015), https://www.openslr.org/12
- **Examples**: 28,539 audio clips, ~100 hours of read English speech
- **Sample rate**: 16 kHz (Whisper-native)

For larger scale: `train.360` (360h) or `train.500` (500h, full corpus).

## Validation

- ✅ Static checks pass
- ⚠️ Simulator end-to-end: not run (`peft` not in sim env; FL plumbing identical to validated job_embed)
- ⚠️ Platform: not yet run

## Training cost (your FL scenario)

| Metric | Value |
|---|---|
| Per-step time (batch=4, Whisper-small + LoRA, fp16) | ~1.5-2 s on RTX 4090 |
| Per-round wall time | ~30-60 min (7k clips / 4 batch × 2 epochs) |
| **Total FL time** | **~3-6 hours** (5 rounds × 4 workers parallel) |
| Per-round transfer | ~5-10 MB (LoRA-only, ~3 M trainable params) |
| Expected WER reduction | **0.5-2 points** below baseline 3.43 (small but real, depending on shard heterogeneity) |

## Hardware

- **Min GPU**: 12 GB VRAM (Whisper-small + LoRA + audio batches with 30-second padding)
- **Per-worker disk**: ~3 GB (Whisper-small ~1 GB cached + audio shard ~2 GB after 16-bit PCM)

## Quick start

```bash
cd training/nvflare/job_speech
pip install datasets librosa soundfile
python build_shards_hf.py   # downloads LibriSpeech-100 (~6 GB), builds 4 shards
```

For a smoke test with smaller data, set `MAX_EXAMPLES = 4000` in `build_shards_hf.py` (1000 clips per shard, ~10 min of training per round).

## Output

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
sd = torch.load("result.pt", map_location="cpu")
model.load_state_dict(sd, strict=False)

# Inference
import librosa
audio, sr = librosa.load("test.wav", sr=16000)
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
gen = model.generate(inputs.input_features)
print(processor.batch_decode(gen, skip_special_tokens=True))
```

## Why FL fits here

LibriSpeech is read speech — *not* private. But the **same FL pipeline applies to private call-center recordings, medical dictation, voice-assistant logs**. Each org's audio stays local; only the LoRA delta moves. This is the textbook FL pitch for ASR.

## Original training recipe (canonical Whisper LoRA from HF blog + PEFT examples)

Source: https://huggingface.co/blog/fine-tune-whisper, https://github.com/Vaibhavs10/fast-whisper-finetuning, PEFT `examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb`

| Param | Value | Source |
|---|---|---|
| Optimizer | **adamw_torch** (or 8-bit Adam in PEFT examples) | HF blog / PEFT |
| Learning rate | **1e-3** (LoRA — higher than full-FT 1e-5) | PEFT example |
| LR schedule | linear (HF default) | PEFT example |
| Warmup steps | **50** | PEFT example |
| Batch size | **8** per-device, grad_accum=1 | blog/PEFT |
| Epochs | **3** (LoRA) — or max_steps=5000 in full-FT blog | PEFT example |
| Audio length | 30s chunks @ 16 kHz | Whisper standard |
| Mixed precision | fp16 + gradient_checkpointing | PEFT example |
| LoRA | **r=32, alpha=64, dropout=0.05**, target_modules=["q_proj", "v_proj"], task_type=SEQ_2_SEQ_LM | PEFT example |
| Loss | cross-entropy on text tokens (Whisper Seq2Seq) | architecture |
| Preprocessing | WhisperFeatureExtractor (80-bin log-Mel) + WhisperTokenizer; resample → 16 kHz | architecture |
| generation_max_length | 225, predict_with_generate=True | PEFT example |
| Final metric (base, no FT) | **WER 3.43% on LibriSpeech test-clean**, 7.63% test-other | https://huggingface.co/openai/whisper-small |
| Final metric (LoRA target) | ~3-4% WER on test-clean (depends on data) | community runs |

**FL config matches**: `learning_rate: 1e-3`, `batch_size: 8`, `local_epochs: 1`, 5 rounds → effective ~3 epochs (close to canonical). `lora_rank=32` set in `model_def.py`.

## Caveats

- **LoRA-only**: trains attention projections only. For dramatic adaptation (new language, new domain vocab), consider full fine-tuning — but that's ~1 GB / round / direction in transfer, impractical for FL.
- **30-second clip limit**: Whisper truncates inputs to 30 seconds during training. LibriSpeech clips are typically 5-15s, so this isn't an issue here, but with longer audio you'd need windowed training.
- **Class imbalance for noisy domains**: if your private speech is heavily accented or domain-specific, a single round won't be enough — bump `num_rounds` to 10-20 in `config_fed_server.json` for real adaptation.
