# SDK notebooks

One notebook per job that submits the **same FL run** through the
[ResonTech SDK](../../nvflare-infra-test/sdk) instead of the platform's
prepared workspace files (`../job_<name>/{configs,scripts,requirements}/`).

Each notebook mirrors the matching `job_<name>/` directory:
- Same model + pretrained weights
- Same hyperparameters (lr / batch / local_epochs / num_rounds)
- Same dataset shards (point `shards_dir` at `../job_<name>/shards`)
- Same training loop (translated into the SDK's
  `fl_train(model, env, out_dir, logger)` contract)

| Notebook | Job | Recipe source |
|---|---|---|
| `job_csgo.ipynb` | YOLOv5n CS:GO | `keremberke/yolov5n-csgo` |
| `job_yolo11.ipynb` | YOLO11m COCO | `Ultralytics/YOLO11` |
| `job_classify.ipynb` | ViT-base Food-101 | `nateraw/food` |
| `job_medical.ipynb` | DenseNet121 HAM10000 | MONAI |
| `job_embed.ipynb` | BGE-base NLI | `BAAI/bge-base-en-v1.5` |
| `job_speech.ipynb` | Whisper-small LoRA LibriSpeech | `openai/whisper-small` + PEFT |
| `job_llm.ipynb` | Phi-3.5-mini LoRA Alpaca | `microsoft/Phi-3.5-mini-instruct` |
| `job_agent.ipynb` | Llama-3.1-8B Unsloth Glaive | `unsloth/llama-3.1-8b-Instruct-bnb-4bit` |
| `job_diffusion.ipynb` | SDXL LoRA Pokémon | `stabilityai/stable-diffusion-xl-base-1.0` |
| `job_imagenet_scratch.ipynb` | ResNet-50 ImageNet (timm A2, from scratch) | `arXiv 2110.00476` |
| `job_lm_scratch.ipynb` | Pythia-1B Pile (from scratch) | EleutherAI Pythia |

## How to run

1. Install the SDK:
   ```bash
   pip install resontech
   # or, from source: pip install -e ../../nvflare-infra-test/sdk
   ```
2. Provision an S3 bucket on the dashboard (Profile → Storage), then export:
   ```bash
   export RESONTECH_EMAIL=...
   export RESONTECH_PASSWORD=...
   export RESONTECH_S3_KEY=AKIA...
   export RESONTECH_S3_SECRET=...
   ```
3. Build shards for the matching job (only needed once per job):
   ```bash
   python ../job_<name>/build_shards_hf.py
   ```
4. Open the notebook and run all cells.

## Regenerating

The notebooks are produced by `_gen.py` — edit per-job specs there and rerun:

```bash
python _gen.py
```
