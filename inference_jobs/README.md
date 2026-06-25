## What you write

| file              | what it is                                                        |
|-------------------|-------------------------------------------------------------------|
| `inference.yaml`  | which class to load, what init args to pass, what deps to install |
| `<module>.py`     | a Python class with `predict(data: bytes) -> dict`                |
| `model/...` (opt) | your trained weights (`.pt`, `.onnx`, `.gguf`, `.safetensors`...) |

You do **not** write a Dockerfile, a compose file, a Ray Serve config, an
HTTP handler, or anything about networking. The platform generates all
of that.

## The predictor contract

Your `<module>.py` exposes one class whose interface is:

```python
class YourPredictor:
    def __init__(self, **init_args):
        # init_args come from inference.yaml -> model.init_args.
        # Build the model, load tokenizer/preprocessor, pick a device.
        ...

    def load_weights(self, path: str):           # optional
        # If inference.yaml -> model.path is set, the platform mounts
        # that file at /app/model/<basename> and calls this hook.
        # If you don't define it, the model.path block is ignored.
        ...

    def predict(self, data: bytes) -> dict:      # required; may also be `async def`
        # `data` is the raw POST body. For binary inputs (image/audio/
        # video), it's the bytes verbatim. For JSON requests, decode it
        # yourself (`json.loads(data)`).
        # Return any JSON-serialisable dict.
        ...

    def warmup(self):                            # optional
        # Called once per replica after deploy so the first real
        # request doesn't pay JIT / autotune cost.
        ...

    def health(self) -> dict:                    # optional
        # Surfaced via /health for the platform's observability.
        ...
```

### vLLM / OpenAI-compatible templates (`build_openai_app`)

These bypass the Predictor contract and use Ray Serve's first-class LLM
support. The cluster exposes `POST /v1/chat/completions`,
`POST /v1/completions`, `GET /v1/models` — any OpenAI-compatible client
(openai-python, langchain, llamaindex, the IDE's OpenAI extension) works
without code changes.

## TP / PP / DP — when does each one make sense?

Short version:

| Method                  | What gets parallelised                       | Use it for                                          |
|-------------------------|----------------------------------------------|-----------------------------------------------------|
| **Data parallelism**    | N independent replicas, each a full copy     | Anything that fits on one GPU. Default for CV.      |
| **Tensor parallelism**  | One replica; every layer's weights sharded   | Models too big for one GPU; multiple GPUs same node |
| **Pipeline parallelism**| One replica; layers split into serial stages | Models too big for one GPU; one GPU per node        |

### "Should I use TP/PP for my segmentation model?"

**No, almost certainly overkill.** A few reasons:

- DeepLab v3 + ResNet-101 is ~250 MB fp32 / 125 MB fp16. It fits on any
  GPU with ≥ 1 GB. Same story for SAM, YOLO, U-Net variants, most
  Mask R-CNN configs. TP/PP solves a problem CV models don't have.
- TP adds an NCCL all-reduce on every layer. For a convnet with O(100)
  layers, the communication overhead per inference dominates the actual
  compute. You'd be slower than running on a single GPU.
- PP only helps when the model literally doesn't fit on one GPU. Adding
  it to a 200 MB model just multiplies your end-to-end latency by N for
  no memory benefit.

**Use data parallelism for any CV model that fits on one GPU.** That's
~99% of them. The only real exceptions are giant diffusion stacks (SDXL
+ refiner + ControlNet chained, or 7B+ vision-language models like
LLaVA-Next, CogVLM, InternVL) — and for those you'd use TP exactly like
the LLM templates here.

### When to combine TP + DP

Common in production LLM deployments: TP within one node (low-latency
single replica) and DP across nodes (multiple replicas for throughput).
The vLLM templates wire this up — `tensor_parallel_size=N` shards each
replica across N GPUs, and `autoscaling_config.max_replicas` scales out
the number of replicas. Template 05's defaults show TP=2 × up to 2
replicas → up to 4 GPUs used at peak traffic.
