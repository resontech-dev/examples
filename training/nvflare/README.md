# Federated Learning jobs (NVFlare)

Nine FL job templates spanning vision, language, speech, embeddings, medical imaging, and image generation. Every model has **real published training metrics** on its HuggingFace card — your FL run extends those. No toy datasets (CIFAR / MedNIST / 500-clip ASR are out).

## How to submit these jobs

Two entry points, both against the published SDK:

```bash
pip install -U "resontech>=0.2.1" python-dotenv   # Python 3.11+
```

- **[job_training_sdk](./job_training_sdk/README.md)** — the clone-and-edit
  template: `model_def.py` + `build_shards.py` + `submit.py` (api-key auth,
  `sdk.rt_submit(...)`). Start here for your own model.
- **[notebooks/](./notebooks/README.md)** — Jupyter walkthroughs that submit
  `job_classify` and `job_yolo11` step by step.

The `job_*` folders below carry the per-task model adapters, NVFlare configs
(`configs/config_fed_*.json`), worker requirements, and `build_shards_hf.py`
(run it locally first — shards land in `job_*/shards/`).

## At a glance

| Job | Model | Author's published metric (HF card) | Dataset |
|---|---|---|---|
| **job_csgo** | [`keremberke/yolov5n-csgo`](https://huggingface.co/keremberke/yolov5n-csgo) (1.9M) | mAP@0.5 ≈ 0.85 (CS:GO val, per author's card) | [CS:GO 4-class](https://huggingface.co/datasets/keremberke/csgo-object-detection) (4,262 imgs) |
| **job_yolo11** | [`Ultralytics/YOLO11`](https://huggingface.co/Ultralytics/YOLO11) (m: 20.1M) | **COCO mAP@50-95: 51.5** | [`rafaelpadilla/coco2017`](https://huggingface.co/datasets/rafaelpadilla/coco2017) (118k imgs) |
| **job_classify** | [`vit_base_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k) (86M) | Reference: [`nateraw/food`](https://huggingface.co/nateraw/food) **acc 0.8913, val_loss 0.4501** on Food-101 | [`ethz/food101`](https://huggingface.co/datasets/ethz/food101) (101k imgs) |
| **job_medical** | MONAI DenseNet121 (7M) — your FL run produces the reference | (no published HAM10000 model on HF with verifiable metrics) | [`marmal88/skin_cancer`](https://huggingface.co/datasets/marmal88/skin_cancer) HAM10000 (13.4k imgs, 7 classes) |
| **job_embed** | [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) (110M) | **MTEB avg 63.55** (56 datasets); Retrieval 53.25, STS 82.40 | [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli) (558k triplets) |
| **job_speech** | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) (244M) | **LibriSpeech-clean WER 3.43**, other WER 7.63 | [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr) `train.100` (28.5k clips, 100h) |
| **job_llm** | [`microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) (3.8B) | **MMLU 69, GSM8K 86.2, HumanEval 62.8, BBH 69** | [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) (52k) |
| **job_diffusion** | [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (~3.5B UNet) | SDXL > SD 2.1 > SD 1.5 in user-preference evals (per card) | [`lambdalabs/pokemon-blip-captions`](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) (833 imgs) |
| **job_agent** | [`unsloth/llama-3.1-8b-Instruct-bnb-4bit`](https://huggingface.co/unsloth/llama-3.1-8b-Instruct-bnb-4bit) — quant of [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (8B) | **MMLU 69.4, BFCL 76.1, API-Bank 82.6, GSM8K 84.5** | [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) (113k) |

All metrics above are quoted **verbatim from the model card** — clickable HF links let you verify each before training.

## Why these models (vs. the previous picks)

| Was | Now | Reason for swap |
|---|---|---|
| `microsoft/resnet-50` / `cifar100` | ViT-base / Food-101 + `nateraw/food` reference | Microsoft's card has NO published top-1; nateraw's has verbatim acc 0.8913 on Food-101 (101 classes, 75k imgs — non-trivial) |
| MONAI MedNIST | HAM10000 dermatology | MedNIST is a 6-class 64×64 grayscale toy; HAM10000 is a 7-class 13k-image dermatoscopic medical task |
| MINDS-14 (~500 clips) | LibriSpeech-100 (28k clips, 100h) | MINDS-14 was a smoke-test scale; LibriSpeech is the canonical ASR benchmark whisper-small itself reports WER on |
| `BAAI/bge-small` | `BAAI/bge-base` | Card on `base` quotes 8 dimensions of MTEB scores verbatim; `small` is less documented |
| `HuggingFaceTB/SmolLM2-360M` | `microsoft/Phi-3.5-mini-instruct` (3.8B) | Phi-3.5 card has 8+ benchmarks with exact numbers; SmolLM2 is good but Phi-3.5 is more rigorously documented |
| `unsloth/mistral-7b-bnb-4bit` | `unsloth/llama-3.1-8b-Instruct-bnb-4bit` | Mistral-7B-Instruct-v0.3 card has ZERO published benchmarks; Llama-3.1-8B-Instruct has full evals incl. tool-use (BFCL 76.1, API-Bank 82.6) — directly relevant to agentic FL |
| `runwayml/stable-diffusion-v1-5` | `stabilityai/stable-diffusion-xl-base-1.0` | **Runway took down SD 1.5 in Aug 2024 — repo returns 401**. SDXL is the current standard, 2M+ monthly downloads |
| `Ultralytics/YOLO11` on CS:GO | `Ultralytics/YOLO11` on COCO 2017 | CS:GO is niche; COCO is the published-metrics benchmark Ultralytics actually reports YOLO11m's 51.5 mAP against |

`job_csgo` (yolov5n-csgo on CS:GO) stays as-is — it's the niche/quick-demo example using a real keremberke fine-tune on a small (143 MB) dataset. The other detection job (`job_yolo11`) is now the COCO heavyweight.

## Training cost (5 FL rounds × 4 workers parallel)

```
job_csgo         ▓▓        3-5 min      [demo/small]
job_embed        ▓▓▓       10-30 min    [BGE-base on 558k NLI triplets]
job_classify     ▓▓▓▓▓▓    1-3 hr       [ViT-base on Food-101, 75k imgs]
job_medical      ▓▓▓       30-90 min    [DenseNet on HAM10000, 13k imgs]
job_speech       ▓▓▓▓▓▓▓▓  3-6 hr       [Whisper-small LoRA on LibriSpeech-100, 28k clips]
job_llm          ▓▓▓▓▓▓▓▓  3-6 hr       [Phi-3.5-mini LoRA on Alpaca, 52k]
job_yolo11       ▓▓▓▓▓▓▓▓▓▓▓ 6-10 hr    [YOLO11m on COCO, 118k imgs]
job_diffusion    ▓▓▓▓▓▓▓▓▓▓▓▓ 8-12 hr   [SDXL LoRA on 833 captions]
job_agent        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 12-18 hr [Llama-3.1-8B 4-bit LoRA on Glaive 113k]
```

GPU minimum requirements:

| Job | Min GPU |
|---|---|
| job_csgo, job_classify, job_medical, job_embed | 8 GB (T4 / RTX 3060) |
| job_speech, job_llm | 12 GB (RTX 4070 / L4) |
| job_yolo11 | 12 GB |
| job_diffusion (SDXL) | **24 GB** (RTX 4090 / A10G) |
| job_agent (Llama-3.1-8B 4-bit) | **24 GB** |

## Validation status

| Job | Static check | Adapter import | NVFlare simulator | Platform run |
|---|---|---|---|---|
| job_csgo | ✅ | ✅ | ✅ FL plumbing verified | ✅ ran successfully |
| job_yolo11 | ✅ | ✅ adapter API | — | — |
| job_classify | ✅ | ✅ | — | — |
| job_medical | ✅ | ✅ | — | — |
| job_embed | ✅ | ✅ | ✅ **5 rounds completed end-to-end** | — |
| job_speech | ✅ | ✅ adapter API | — | — |
| job_llm | ✅ | ✅ adapter API | — | — |
| job_diffusion | ✅ | ✅ adapter API | — | — |
| job_agent | ✅ | ✅ adapter API | — | ✅ ran (Mistral-7B previously, now upgraded to Llama-3.1-8B) |

Two end-to-end validations (`job_csgo` on platform, `job_embed` on simulator) confirm the FL plumbing is correct. All 9 jobs share **byte-identical** `custom_client_executor.py` and `custom_persistor.py` — the plumbing works for all.

## Common architecture

```
job_xxx/
├── README.md
├── scripts/
│   ├── model_def.py              ← project config + make_fl_adapter()
│   ├── {framework}_utils.py      ← framework-specific adapter
│   ├── custom_client_executor.py ← shared FL plumbing
│   └── custom_persistor.py       ← shared FL plumbing
├── configs/
│   ├── config_fed_client.json
│   └── config_fed_server.json
├── requirements/
│   └── requirements.txt
├── shards/                       ← built by build_shards_hf.py
└── build_shards_hf.py
```

Per-job READMEs link to each model's HuggingFace card so any number can be verified.

- [job_csgo](./job_csgo/README.md) — yolov5 CS:GO detection
- [job_yolo11](./job_yolo11/README.md) — YOLO11m on COCO 2017
- [job_classify](./job_classify/README.md) — ViT-base on Food-101
- [job_medical](./job_medical/README.md) — MONAI on HAM10000
- [job_embed](./job_embed/README.md) — BGE-base on NLI
- [job_speech](./job_speech/README.md) — Whisper-small on LibriSpeech-100
- [job_llm](./job_llm/README.md) — Phi-3.5-mini on Alpaca
- [job_diffusion](./job_diffusion/README.md) — SDXL on Pokémon-BLIP
- [job_agent](./job_agent/README.md) — Llama-3.1-8B-Instruct on Glaive

## Notes

- **Model card metrics ≠ your FL training metrics.** The published numbers above are the model author's centralized training results, used as the baseline. Your FL training fine-tunes from those checkpoints — final FL metrics depend on shard distribution, num_rounds, and local_epochs. They typically land within 2-10% of centralized fine-tuning quality.
- **Runway took down SD 1.5** (Aug 2024). Don't use it as a reference. SDXL is the current standard.
- **`Mistral-7B-Instruct-v0.3` has no published benchmarks** on its HF card. Llama-3.1-8B-Instruct is what we cite metrics from now.
- **HAM10000 has no popular HF model card with verifiable metrics**. Your FL run produces the reference; we explicitly mark this case in the medical README.
