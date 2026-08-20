"""GLM-4.5-Air RAG bundle — TP=4 chat + embed + rerank on one 4-GPU worker.

The premium private-RAG stack: Sonnet-4-class agentic chat plus the RAG
sidecars, one deployment, one machine:

    "assistant" → GLM-4.5-Air AWQ (TP=4)   POST /v1/chat/completions
    "embed"     → Qwen3-Embedding-0.6B     POST /v1/embeddings
    "rerank"    → Qwen3-Reranker-0.6B      POST /v1/score

Memory model: the chat engine takes its gpu_memory_utilization fraction
on EVERY one of the 4 GPUs (weights shard evenly). Each 0.6B sidecar is
TP=1 and lands on ONE of those GPUs — placement does NOT guarantee the
two sidecars spread, so budget for BOTH landing on the same GPU:

    chat    0.70 → 16.8 GB/GPU (15 weights-shard + ~1.8 GB KV share)
    sidecar 0.06 →  1.4 GB (each, on whichever GPU Ray places it)
    worst-case GPU: 0.70 + 0.06 + 0.06 = 0.82  ✔ (< 0.85)

⚠️ The yaml's `vllm:` block mirrors the CHAT engine's *_size values —
the scheduler sizes hardware for the largest engine.
"""
from ray.serve.llm import LLMConfig, build_openai_app


chat = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",
        # VERIFY before selling: community AWQ re-upload (no official AWQ).
        model_source="cpatonn/GLM-4.5-Air-AWQ",
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=4,      # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,
        max_model_len=32768,
        max_num_seqs=8,
        gpu_memory_utilization=0.70, # leaves room for BOTH sidecars on one GPU
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        dtype="auto",
        # Tool calling — VERIFY the glm45 parser on the pinned image first
        # (see ../README.md "Tool calling"):
        # enable_auto_tool_choice=True,
        # tool_call_parser="glm45",
    ),
    placement_group_config=dict(
        bundle_per_worker={"GPU": 1, "CPU": 1},
        strategy="STRICT_PACK",      # TP never crosses machines
    ),
)

embed = LLMConfig(
    model_loading_config=dict(
        model_id="embed",
        model_source="Qwen/Qwen3-Embedding-0.6B",
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        task="embed",
        max_model_len=8192,
        gpu_memory_utilization=0.06,
        enforce_eager=True,
    ),
)

rerank = LLMConfig(
    model_loading_config=dict(
        model_id="rerank",
        model_source="Qwen/Qwen3-Reranker-0.6B",
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        task="score",
        max_model_len=8192,
        gpu_memory_utilization=0.06,
        enforce_eager=True,
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    ),
)


app = build_openai_app({"llm_configs": [chat, embed, rerank]})
