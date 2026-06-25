"""Qwen-7B with tensor parallelism on Ray Serve LLM.

The OpenAI-compatible app is exported as `app`. The platform's generate.py
detects `engine: vllm_openai` in inference.yaml and emits a
serve_config.yaml whose import_path is `serve_module:app`, bypassing the
templates/serve_app.py Predictor wrapper used by templates 01–04.

Routes exposed once deployed (all OpenAI-compatible):
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models
"""
import os

from ray.serve.llm import LLMConfig, build_openai_app


# Tensor parallelism: 1 replica, 2 GPUs, weights split column/row-wise
# across the GPUs. PACK strategy keeps both bundles on the same node so
# the per-layer NCCL all-reduce stays on NVLink / PCIe Gen4.
llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-7b",                                # client-facing model name
        model_source="Qwen/Qwen2.5-7B-Instruct",           # HF repo
    ),

    # Gate placement to a specific GPU class. Set to None to allow any GPU
    # — useful when your cluster has mixed accelerators and you want to
    # guarantee this model only lands on cards big enough to hold its
    # shard of the weights.
    accelerator_type="L4",                                  # 24 GB → fits half of Qwen-7B fp16 with room

    deployment_config=dict(
        # Data-parallel scale-out on top of TP. min=1 max=2 means: up to
        # two full TP-sharded replicas (so up to 4 GPUs used) depending
        # on traffic.
        autoscaling_config=dict(min_replicas=1, max_replicas=2),
        max_ongoing_requests=32,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=2,                             # shard weights across 2 GPUs
        pipeline_parallel_size=1,                           # no pipeline staging
        max_model_len=8192,                                 # context window
        max_num_seqs=64,                                    # concurrent sequences per replica
        enable_chunked_prefill=True,                        # prefill in chunks → better tail latency
        max_num_batched_tokens=4096,
        dtype="auto",                                       # fp16 on GPU, fp32 fallback

        # vLLM checks `Authorization: Bearer <api_key>` on every /v1/* request
        # when this is non-empty. The platform mints API_KEY per job and
        # injects it via env; clients pass the same value via the OpenAI
        # SDK's `api_key=...` argument. Empty string disables the check
        # (intended for local dev / smoke tests only).
        api_key=os.environ.get("API_KEY", ""),
    ),

    placement_group_config=dict(
        # bundle_per_worker is auto-replicated TP*PP times → here 2 bundles,
        # one per GPU worker. Each bundle reserves 1 GPU + 4 CPUs.
        bundle_per_worker={"GPU": 1, "CPU": 4},
        # PACK: try to fit all bundles on the fewest nodes possible. TP
        # needs fast interconnect, so on-node is what we want.
        strategy="PACK",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
