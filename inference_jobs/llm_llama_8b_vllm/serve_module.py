"""Llama 3.1 8B-Instruct on Ray Serve LLM with tensor parallelism.

The reference deployment for an 8B chat model. Three placement modes
inline below — uncomment whichever matches your cluster.

Routes exposed:
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models

Auth: Llama 3.1 is gated on HF Hub. Set HUGGING_FACE_HUB_TOKEN in the
deployment env (the platform forwards it via runtime_env below). Without
it, vLLM's __init__ raises a 401 from HF Hub.
"""
import os

from ray.serve.llm import LLMConfig, build_openai_app


# ─────────────────────────────────────────────────────────────────────
# Pick ONE of the three configs below and rename it to `llm_config`.
# Default: single-node TP=2 — the cheapest realistic 8B deployment.
# ─────────────────────────────────────────────────────────────────────


# (A) Single-node TP=2 — 2× consumer GPUs (e.g. 2× L4, 2× RTX 4090) on one box.
#     ~8 GB per GPU. Lowest latency. Use this unless your topology demands otherwise.
llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="llama-3.1-8b",
        model_source="meta-llama/Meta-Llama-3.1-8B-Instruct",
    ),
    accelerator_type="L4",                                # 24 GB cards; pin to your real class
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=2),
        max_ongoing_requests=32,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
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
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="PACK",                                  # both shards on the same node
    ),
    runtime_env=dict(
        env_vars={
            # Forwarded into the vLLM worker so it can pull the gated weights.
            "HUGGING_FACE_HUB_TOKEN": os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
        }
    ),
)


# (B) Single-GPU, no parallelism — fits on a single A100 80 GB or H100.
#     Highest throughput per dollar when the GPU is big enough. Uncomment
#     to use; comment out (A) above.
#
# llm_config = LLMConfig(
#     model_loading_config=dict(
#         model_id="llama-3.1-8b",
#         model_source="meta-llama/Meta-Llama-3.1-8B-Instruct",
#     ),
#     accelerator_type="A100",
#     deployment_config=dict(
#         autoscaling_config=dict(min_replicas=1, max_replicas=4),
#         max_ongoing_requests=32,
#     ),
#     engine_kwargs=dict(
#         tensor_parallel_size=1,
#         pipeline_parallel_size=1,
#         max_model_len=8192,
#         max_num_seqs=128,
#         enable_chunked_prefill=True,
#         max_num_batched_tokens=8192,
#         dtype="auto",
#     ),
#     placement_group_config=dict(
#         bundle_per_worker={"GPU": 1, "CPU": 4},
#     ),
#     runtime_env=dict(env_vars={"HUGGING_FACE_HUB_TOKEN": os.environ.get("HUGGING_FACE_HUB_TOKEN", "")}),
# )


# (C) Cross-node TP=2 × PP=2 (4 GPUs across 2 nodes) — only useful if you
#     hit memory ceilings with (A) AND don't have a single A100. Combines
#     within-node tensor sharding with across-node pipeline staging.
#     Uncomment to use.
#
# llm_config = LLMConfig(
#     model_loading_config=dict(
#         model_id="llama-3.1-8b",
#         model_source="meta-llama/Meta-Llama-3.1-8B-Instruct",
#     ),
#     accelerator_type="L4",
#     deployment_config=dict(
#         autoscaling_config=dict(min_replicas=1, max_replicas=1),
#         max_ongoing_requests=32,
#     ),
#     engine_kwargs=dict(
#         tensor_parallel_size=2,                         # within-node TP
#         pipeline_parallel_size=2,                       # across-node PP
#         max_model_len=8192,
#         max_num_seqs=64,
#         enable_chunked_prefill=True,
#         max_num_batched_tokens=4096,
#         dtype="auto",
#     ),
#     placement_group_config=dict(
#         # TP*PP = 4 bundles. Ray spreads pipeline stages across nodes;
#         # within each stage, TP shards stay co-located thanks to PACK.
#         bundle_per_worker={"GPU": 1, "CPU": 4},
#         strategy="PACK",
#     ),
#     runtime_env=dict(env_vars={"HUGGING_FACE_HUB_TOKEN": os.environ.get("HUGGING_FACE_HUB_TOKEN", "")}),
# )


app = build_openai_app({"llm_configs": [llm_config]})
