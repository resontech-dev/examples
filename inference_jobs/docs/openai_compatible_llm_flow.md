# Integrate OpenAI-compatible LLM (vLLM mode)

Use `ray.serve.llm.build_openai_app` to serve an LLM behind the OpenAI
HTTP wire format. Existing OpenAI SDK clients (openai-python, langchain,
llamaindex, IDE extensions) point at the cluster with one line change
and talk to it unmodified.

If you want a non-OpenAI request shape, custom RAG inside `predict`, or
a non-generative model, use predictor mode instead
([custom_llm_inference_flow.md](custom_llm_inference_flow.md)).

---

## What you get

- **OpenAI-compatible routes**: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`. Streaming (`stream: true`) works.
- **vLLM under the hood**: continuous batching, PagedAttention, KV-cache sharing. Throughput is ~5–10× the transformers-based predictor mode.
- **Built-in TP/PP**: shard one model across N GPUs (template 05) or N nodes (template 06) with one `engine_kwargs` value.
- **Built-in autoscaling**: scale replicas up/down by traffic via `deployment_config.autoscaling_config`.
- **Built-in auth**: `Authorization: Bearer <key>` enforced by vLLM when `engine_kwargs.api_key` is set.

---

## Minimal example

`serve_module.py`:

```python
import os
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-7b",
        model_source="Qwen/Qwen2.5-7B-Instruct",
    ),
    accelerator_type="L4",
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=2),
        max_ongoing_requests=32,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=2,
        max_model_len=8192,
        max_num_seqs=64,
        enable_chunked_prefill=True,
        api_key=os.environ.get("API_KEY", ""),
    ),
    placement_group_config=dict(
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="PACK",
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})
```

`inference.yaml`:

```yaml
engine: vllm_openai           # platform discriminator — bypasses templates/serve_app.py

cluster:
  num_workers: 1
  mode: "tensor_parallel"
  gpus_per_worker: 1
  cpus_per_worker: 8
  max_ongoing_requests: 32

requirements:
  # DO NOT redeclare ray here. base_requirements.txt must include
  # ray[serve,llm]==<version>. See "Pip resolver conflict" below.
  - "vllm>=0.18.0"
  - "transformers>=4.45.0"
```

---

## `LLMConfig` reference (Ray 2.55+)

### `model_loading_config`

| field          | type   | meaning                                                            |
|----------------|--------|--------------------------------------------------------------------|
| `model_id`     | string | client-facing name. Clients pass this as `model="<id>"`.            |
| `model_source` | string | HF Hub repo or local path to weights.                              |

### `engine_kwargs` (passed to vLLM)

| field                       | type | meaning                                                                |
|-----------------------------|------|------------------------------------------------------------------------|
| `tensor_parallel_size`      | int  | Shard weights across N GPUs. Must divide attention head count. PACK them on one node. |
| `pipeline_parallel_size`    | int  | Split layers into N sequential stages. STRICT_SPREAD them across nodes. |
| `max_model_len`             | int  | Context window in tokens. Affects KV-cache size linearly.              |
| `max_num_seqs`              | int  | Max concurrent sequences per replica. KV-cache size scales with this.   |
| `max_num_batched_tokens`    | int  | Cap on the prefill+decode batch. Tune for tail latency.                 |
| `enable_chunked_prefill`    | bool | Splits long prefills into chunks; smoother decode tail. Leave on.       |
| `dtype`                     | str  | `"auto"` (recommended) / `"float16"` / `"bfloat16"` / `"float32"`.      |
| `api_key`                   | str  | If non-empty, vLLM requires `Authorization: Bearer <api_key>` on every /v1/* request. |
| `gpu_memory_utilization`    | float| Fraction of each GPU vLLM may use (default 0.9). Drop to 0.85 if OOM. |
| `quantization`              | str  | `"awq"` / `"gptq"` / `"fp8"` — only with quantised weights.            |
| `trust_remote_code`         | bool | Required for some HF model architectures (Qwen, Llama-3 generally don't). |

### `deployment_config`

| field                  | type     | meaning                                                                 |
|------------------------|----------|-------------------------------------------------------------------------|
| `autoscaling_config.min_replicas` | int | Floor for replica count.                                          |
| `autoscaling_config.max_replicas` | int | Ceiling. Data-parallel scale-out on top of TP/PP.                  |
| `max_ongoing_requests` | int      | Per-replica backpressure cap. vLLM handles internal batching.            |
| `ray_actor_options`    | dict     | Rarely needed — placement_group_config covers most cases.               |

### `placement_group_config`

| field                | type   | meaning                                                                |
|----------------------|--------|------------------------------------------------------------------------|
| `bundle_per_worker`  | dict   | `{"GPU": 1, "CPU": 4}`. Auto-replicated `TP × PP` times. The simple path. |
| `bundles`            | list   | Explicit list `[{"GPU": 1, ...}, ...]`. Use only if bundle_per_worker can't express what you want. |
| `strategy`           | string | `PACK` (default — co-locate), `STRICT_PACK`, `SPREAD`, `STRICT_SPREAD`. |

When to pick which strategy:

| Layout                                | Strategy        |
|---------------------------------------|-----------------|
| TP only (within one node)             | `PACK`          |
| PP across nodes                       | `STRICT_SPREAD` |
| TP × PP (TP inside node, PP across)   | `PACK` (Ray will figure it out) |
| DP scale-out across nodes for HA      | `SPREAD`        |

### `accelerator_type`

String like `"L4"`, `"A10G"`, `"A100"`, `"H100"`, `"V100"`, `"T4"`,
`"L40S"`. When set, Ray scheduling places this LLMConfig's actors
**only** on workers advertising that accelerator class. Useful for
heterogeneous clusters. Leave as `None` to allow any GPU.

### `runtime_env`

Standard Ray runtime env. The most common use is forwarding secrets:

```python
runtime_env=dict(
    env_vars={
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
        # add any other secrets your model needs at load time
    }
)
```

---

## TP vs PP vs DP — pick the right one

| Method                  | What gets parallelised                       | Best for                                            |
|-------------------------|----------------------------------------------|-----------------------------------------------------|
| **Data parallelism**    | N independent replicas                       | Anything that fits on one GPU. Default for throughput. |
| **Tensor parallelism**  | One replica; every layer's weights sharded   | Model too big for one GPU; multi-GPU on one node    |
| **Pipeline parallelism**| One replica; layers split into serial stages | Model too big for one GPU; one GPU per node         |

Practical rules:

- **TP needs NVLink or PCIe Gen4.** All-reduce per layer. Across nodes
  over Ethernet, TP collapses.
- **PP needs higher first-token latency tolerance.** Stages are serial
  per token, so first-token time roughly multiplies by `pipeline_parallel_size`.
- **Combine TP+DP for prod LLMs.** TP inside each node (low latency
  single replica), DP across nodes via `autoscaling.max_replicas` for
  throughput.
- **Don't reach for TP/PP just because you have multiple GPUs.** If the
  model fits on one GPU, plain DP (`tensor_parallel_size=1`, `max_replicas=N`)
  is faster and simpler.

---

## Pinning rules for vLLM jobs

| package         | pin?                          | reason                                                       |
|-----------------|-------------------------------|--------------------------------------------------------------|
| `vllm`          | `>=0.18.0` (within ray[llm] requirement) | tight coupling to ray[llm] version              |
| `transformers`  | `>=4.45.0` loose               | vLLM specifies the minimum                                   |
| `huggingface_hub` | range with upper bound       | breaks `from_pretrained` periodically                       |
| `torch`         | **don't declare**              | vllm pulls a specific build with CUDA bindings; let it choose |
| `flash-attn`    | **don't declare**              | vllm picks a compatible wheel                                |
| `xformers`      | **don't declare**              | same                                                          |

The reason "don't declare torch/flash-attn/xformers": vLLM ships its
own CUDA-version-specific wheels and links a torch built against a
matching CUDA. If you pin a different torch, vLLM's CUDA kernels
silently mismatch and you get import errors at runtime.

---

## When to skip vLLM mode and use predictor mode

| Reason to skip                                              | Use predictor mode |
|-------------------------------------------------------------|--------------------|
| You need inline RAG inside one /predict call                | yes                |
| You need a custom request/response shape                    | yes                |
| Your model is not a causal LM (embeddings, classifiers)     | yes                |
| You want to run the predictor on Apple Silicon via MPS      | yes (vLLM is CUDA-only) |
| Model fits on one GPU AND you don't need OpenAI clients     | usually yes — simpler |
