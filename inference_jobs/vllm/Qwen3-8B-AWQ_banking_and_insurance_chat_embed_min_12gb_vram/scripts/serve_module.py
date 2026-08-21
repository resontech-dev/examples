"""Qwen3-8B (AWQ) + Qwen3-Embedding-0.6B — BFSI agent bundle (Stack B).

The chat half of the banking_and_insurance pilot
(sectors/banking_and_insurance/bfsi_pilot): the claims-assistant agent
calls "assistant" with run_sql/search_docs/get_document/red_flags tools;
the ingest worker's --embed pass and the agent's search_docs both call
"embed" (1024-dim vectors → pgvector).

Two engines, one deployment, one 16 GB card. Budget:

    chat  0.60 → 9.6 GB (5.5 AWQ weights + ~3.5 GB KV ≈ 47k tokens fp8
                          → 16k context × a couple of sessions)
    embed 0.12 → 1.9 GB
    sum   0.72            (roomy on purpose — pilot card may drive a display)

⚠️ Tool calling MUST stay on (enable_auto_tool_choice + hermes) — the
agent's tool loop half-works at best without it.

⚠️ The yaml's `vllm:` block MUST mirror the *_size values (all 1 here).
"""
from ray.serve.llm import LLMConfig, build_openai_app


chat = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",
        model_source="Qwen/Qwen3-8B-AWQ",        # official AWQ, ~5.5 GB
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,
        max_queued_requests=32,
        graceful_shutdown_timeout_s=120,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=16384,         # agent turns: DDL + tool results, not novels
        max_num_seqs=4,
        gpu_memory_utilization=0.60,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,  # system prompt + DDL resent every turn
        enable_chunked_prefill=True,
        dtype="auto",
        # ── Tool calling — REQUIRED for the agent loop ──
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    ),
)

embed = LLMConfig(
    model_loading_config=dict(
        model_id="embed",
        model_source="Qwen/Qwen3-Embedding-0.6B",  # 1024-dim — matches pgvector DDL
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,                 # embeddings batch cheaply
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        task="embed",                            # → served on /v1/embeddings
        max_model_len=8192,
        gpu_memory_utilization=0.12,
        enforce_eager=True,                      # tiny model — save CUDA-graph VRAM
    ),
)


app = build_openai_app({"llm_configs": [chat, embed]})
