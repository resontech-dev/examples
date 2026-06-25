"""Qwen-7B with pipeline parallelism on Ray Serve LLM.

PP differs from TP in two important ways:
  - **Per-layer**: each pipeline stage owns a contiguous block of
    transformer layers (not a slice of every layer's weights, as in TP).
  - **Per-token**: activations move between stages serially. The first
    stage runs layers 0-N/2 on its GPU, hands the activations to the
    second stage, which runs layers N/2-N on its GPU, then emits the
    next token. This is bandwidth-light and tolerates higher latency,
    so it works across nodes via plain network sockets.

Routes exposed:
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models
"""
import os

from ray.serve.llm import LLMConfig, build_openai_app


# Pipeline parallelism: 1 replica, 2 stages on 2 different nodes.
# STRICT_SPREAD forces each bundle onto a different node so each stage
# gets its own GPU/CPU/memory budget. Without STRICT_SPREAD, Ray's PACK
# default would try to cram both stages onto one node — defeating the
# purpose, and probably running out of VRAM in the process.
llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-7b-pp",
        model_source="Qwen/Qwen2.5-7B-Instruct",
    ),

    # Pin to one GPU class so both stages get equivalent hardware.
    # Set to None if your cluster is uniform.
    accelerator_type="L4",

    deployment_config=dict(
        # PP latency is ~2× a single-GPU forward pass per token (stages
        # are serial). Throughput per replica is roughly the same as a
        # single GPU's. Scale by adding replicas, not by raising PP.
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                             # no within-stage sharding
        pipeline_parallel_size=2,                           # 2 cross-node stages
        max_model_len=8192,
        max_num_seqs=64,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
        dtype="auto",

        # `Authorization: Bearer <api_key>` enforced by vLLM on every /v1/*.
        # See template 05 for the auth model rationale.
        api_key=os.environ.get("API_KEY", ""),
    ),

    placement_group_config=dict(
        # 2 bundles (one per pipeline stage), each landing on a separate
        # node thanks to STRICT_SPREAD.
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_SPREAD",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
