# 05 — Qwen 7B with tensor parallelism (vLLM)

One replica, weights sharded across N GPUs **on a single node**. Use when
the model doesn't fit on one GPU but a node with N GPUs is available.

## How vLLM templates differ from templates 01–04

These templates don't follow the `predict(data: bytes) -> dict` contract.
They build a Ray Serve app via `ray.serve.llm.build_openai_app` directly,
so the deployed cluster speaks the OpenAI API natively:

| Route                          | What it does                                      |
|--------------------------------|---------------------------------------------------|
| `POST /v1/chat/completions`    | chat completions with system + user messages      |
| `POST /v1/completions`         | classic completions (no chat template)            |
| `GET  /v1/models`              | lists `model_id`s the cluster currently serves    |

Any OpenAI-compatible client (openai-python, curl, langchain, llamaindex,
the OpenAI extension in your IDE) targets this cluster without code
changes — just point `base_url` at `http://<head>:8000/v1`.

## When tensor parallelism actually helps

| Scenario                                                          | TP?                                              |
|-------------------------------------------------------------------|--------------------------------------------------|
| Model weights too large for one GPU; one node has N small GPUs     | **YES** — exactly what TP is for                |
| Model fits on one GPU; you want throughput                         | NO — use data parallelism with multiple replicas |
| Multiple nodes, each with 1 modest GPU                             | NO — use pipeline parallelism (template 06)      |
| Need lower per-token latency on a model that already fits          | Sometimes — TP cuts single-pass time but adds NCCL |

TP introduces an NCCL all-reduce per transformer layer. Within one node
on NVLink / PCIe Gen4 the overhead is small. Across nodes over Ethernet
it dominates and pipeline parallelism is the right tool instead.

## Use it

```bash
cp -r inference_templates/05_llm_qwen_tensor_parallel jobs/qwen_tp
./scripts/setup.sh jobs/qwen_tp

# OpenAI Python SDK:
python3 -c '
import openai
c = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
r = c.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Explain attention in one sentence."}],
)
print(r.choices[0].message.content)
'

# Or curl:
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}]}'
```

## What to change

| where (`serve_module.py`)                                  | for what                                        |
|------------------------------------------------------------|-------------------------------------------------|
| `model_loading_config.model_source`                        | any HF Hub causal-LM repo                       |
| `engine_kwargs.tensor_parallel_size`                       | shard count: 1, 2, 4, 8 (must divide attention heads) |
| `engine_kwargs.max_model_len`                              | context window in tokens                        |
| `engine_kwargs.max_num_seqs`                               | concurrent sequences per replica                |
| `accelerator_type`                                         | `"L4"`, `"A10G"`, `"A100"`, `"H100"`, or `None` |
| `deployment_config.autoscaling.max_replicas`               | data-parallel scale-out cap on top of TP        |
| `placement_group_config.bundle_per_worker`                 | resources per shard worker                      |

## Memory budget

VRAM per GPU ≈ `(model_size_fp16 / TP) + KV-cache + activations`.

For Qwen 7B fp16 (~14 GB) at TP=2: ~7 GB weights per GPU + ~3 GB KV-cache
at `max_num_seqs=64, max_model_len=8192` + ~1 GB activations ≈ 11 GB per
GPU. L4 (24 GB) fits comfortably with room for batching.

If you see CUDA OOM at boot, drop `max_num_seqs` first, then
`max_model_len`. Both reduce KV-cache linearly.
