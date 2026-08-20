"""Mistral Small 24B RAG bundle — chat + embed + rerank, one 24 GB worker.

One deployment serves the whole private-RAG stack:

    "assistant" → Mistral Small 24B AWQ    POST /v1/chat/completions
    "embed"     → Qwen3-Embedding-0.6B     POST /v1/embeddings
    "rerank"    → Qwen3-Reranker-0.6B      POST /v1/score

⚠️ gpu_memory_utilization is per engine and PRE-ALLOCATED at boot.
Budget on a 24 GB card (sum 0.84 — never let it default to 0.9 each):

    chat   0.62 → 14.9 GB (13.2 weights + ~1.2 GB KV ≈ 14k tokens fp8)
    embed  0.11 →  2.6 GB
    rerank 0.11 →  2.6 GB

The chat KV pool is deliberately tight — RAG turns are short-context
(top-3 chunks + question). For long chats move to a 32 GB card or drop
the sidecars onto a second small worker.

⚠️ The yaml's `vllm:` block MUST mirror the *_size values (all 1 here).
"""
from ray.serve.llm import LLMConfig, build_openai_app


chat = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",
        # VERIFY before selling: community AWQ re-upload (no official
        # Mistral AWQ exists) — vet the repo + re-run evals first.
        model_source="casperhansen/mistral-small-24b-instruct-2501-awq",
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=8192,          # RAG turns are short — context comes retrieved
        max_num_seqs=4,
        gpu_memory_utilization=0.62,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        dtype="auto",
        enable_auto_tool_choice=True,
        tool_call_parser="mistral",  # stable for the Mistral family
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
        gpu_memory_utilization=0.11,
        enforce_eager=True,          # tiny model — save the CUDA-graph VRAM
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
        gpu_memory_utilization=0.11,
        enforce_eager=True,
        # vLLM's documented conversion for Qwen3-Reranker (ships as a
        # causal LM, scored as sequence classification):
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    ),
)


app = build_openai_app({"llm_configs": [chat, embed, rerank]})
