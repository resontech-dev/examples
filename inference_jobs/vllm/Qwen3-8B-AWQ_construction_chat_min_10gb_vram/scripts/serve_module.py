"""Qwen3-8B (AWQ) via vLLM — cheap-tier chat + agent baseline.

Catalog `chat-qwen3-8b` (priority v1): the quality-per-GB workhorse —
119 languages (Ukrainian included), hybrid thinking mode, and reliable
hermes-style tool calling, which makes it the baseline for agent
experiments (SQL tools, BIM assistants — see this folder's README).

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                    # client-facing name — keep stable
        model_source="Qwen/Qwen3-8B-AWQ",        # official AWQ, ~5.5 GB
    ),

    # Consumer GPUs carry NO Ray accelerator label — non-None = PENDING forever.
    accelerator_type=None,

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,                 # ~2× max_num_seqs
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                  # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                # = yaml vllm.pipeline_parallel_size
        # KV budget: on 12 GB → 10.8 − 5.5 − 0.6 ≈ 4.7 GB ≈ 64k tokens at
        # fp8 (~74 KB/token) — 32k context fits with headroom. On a 16 GB
        # card (5070 Ti) the pool roughly doubles (~100k tokens) — same
        # config, just more concurrent long-context sessions.
        max_model_len=32768,
        max_num_seqs=8,
        kv_cache_dtype="fp8",                    # free 2× on KV — default everywhere
        enable_prefix_caching=True,              # agent loops resend the same digest/schema
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,             # 0.85 if the card drives a display
        dtype="auto",

        # ── Tool calling — REQUIRED for agent use (Level B experiments,
        # run_sql/get_element tools). Qwen speaks hermes; long-stable in vLLM.
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",

        # YaRN — stretch the native 32k window to ~131k. A 24 GB+ card
        # lever only: 128k of fp8 KV ≈ 9.5 GB for ONE conversation.
        # hf_overrides={"rope_scaling": {"rope_type": "yarn", "factor": 4.0,
        #                                "original_max_position_embeddings": 32768}},
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
