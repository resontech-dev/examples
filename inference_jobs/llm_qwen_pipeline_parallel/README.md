# Qwen 7B with pipeline parallelism (vLLM, cross-node)

One replica, transformer layers split into N **sequential stages** on N
separate nodes. Use when no single GPU is big enough and you don't have
a single node with multiple GPUs either.

## TP vs PP — pick the right one

| Property                    | Tensor parallelism (TP)                  | Pipeline parallelism (PP)              |
|-----------------------------|------------------------------------------|----------------------------------------|
| What gets split             | Every layer's weights (column/row-wise)  | Layers themselves (sequential blocks)  |
| Inter-GPU traffic           | All-reduce **every layer**                | Send activations **between stages**     |
| Bandwidth requirement       | High — needs NVLink or PCIe Gen4         | Low — works over plain network          |
| Best topology               | One node with N GPUs                     | N nodes with 1 GPU each                |
| Per-token latency           | ≈ 1× single-GPU                          | ≈ N× single-GPU (stages are serial)    |
| Throughput at high QPS      | Excellent (continuous batching)          | Good (micro-batching across stages)    |

**Rule of thumb**: TP when you have NVLink. PP across the cluster. Combine
both for very large models (TP within node + PP across nodes — template
07 shows this for Llama 70B if you want to scale that direction).

## When PP makes sense

| Setup                                          | PP?  |
|------------------------------------------------|------|
| 2 boxes each with 1× L4 (24 GB)                | YES — fits 14 GB Qwen-7B with one stage per box |
| 1 box with 2× L4 on same node                  | NO — use TP, way faster                        |
| 1 box with 1× A100 (80 GB)                     | NO — fits on one GPU, just do data parallel    |
| 4 boxes each with 1× L4, want Llama 70B        | YES — but combine with TP if any box has 2 GPUs |

## Cluster requirements

Ray's `STRICT_SPREAD` placement strategy forces each pipeline stage onto
a **different node**. The cluster must have at least
`pipeline_parallel_size` nodes registered, each with at least 1 GPU
matching `accelerator_type`. If only 1 GPU node is available, the
LLMConfig will sit in `PENDING_NEW_REPLICA` indefinitely — `ray status`
will surface the unsatisfied placement group request.

## What to change

| where (`serve_module.py`)                  | for what                                                |
|--------------------------------------------|---------------------------------------------------------|
| `pipeline_parallel_size`                   | how many stages (= how many nodes)                      |
| `tensor_parallel_size`                     | bump above 1 to TP-shard each stage too (combined TP+PP)|
| `placement.bundle_per_worker.GPU`          | GPUs per stage (raise if each stage needs more than 1)  |
| `placement.strategy`                       | `STRICT_SPREAD` (one stage per node) or `SPREAD` (best-effort) |
| `deployment.autoscaling.max_replicas`      | data-parallel scale-out on top of PP (rarely useful for PP) |

## Latency note

PP per-token latency is roughly N × single-GPU latency because stages run
serially per token. vLLM micro-batches across stages (while stage 1 emits
token T, stage 0 starts on token T+1), so steady-state throughput is
close to single-GPU. **First-token latency is the visible regression** —
clients streaming responses will feel the first chunk arrive later.

If you need both lower first-token latency AND a model that doesn't fit
on one GPU, prefer TP within a multi-GPU node. PP is for when topology
forces your hand.
