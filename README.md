# ResonTech examples

Copy-paste-ready jobs for the ResonTech platform, organized by workload.
Everything runs against the published SDK:

```bash
pip install -U "resontech>=0.2.1" python-dotenv   # Python 3.11+
```

| Directory | What's inside |
|---|---|
| [inference_jobs/vllm/](./inference_jobs/vllm/README.md) | OpenAI-compatible LLM endpoints (`engine: vllm_openai`): chat, coding, RAG bundles, multi-LoRA, vision, ASR, guarded chat — 16 jobs + shared contract |
| [inference_jobs/predict/](./inference_jobs/predict/) | Predictor-contract jobs (`predict(bytes) -> dict`): classification, segmentation, TTS, image generation |
| [training/nvflare/](./training/nvflare/README.md) | Federated-learning job catalog (NVFlare): 11 task templates + the SDK submit template + notebooks |
| [sectors/](./sectors/banking_and_insurance/bfsi_pilot/README.md) | Sector pilots built on the jobs above — currently banking & insurance (digital twin + agent) |

Each job folder is self-contained: `README.md` with resources and a
quickstart, `.env.example` for credentials, and a `submit.py` /
`predict.py` pair (or `sdk_utils/`). Deploys go through `submit.py` or the
dashboard wizard at <https://beta.reson.tech/dashboard/inference/submit>.
