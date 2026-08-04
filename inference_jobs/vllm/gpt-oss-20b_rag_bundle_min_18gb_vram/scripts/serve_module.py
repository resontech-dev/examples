"""gpt-oss-20b RAG bundle — chat + embeddings + reranker, one 24 GB worker.

Ray Serve LLM's OpenAI router accepts a LIST of LLMConfigs, so one
deployment serves the whole private-RAG stack ("ChatGPT for our
documents"):

    "assistant" → gpt-oss-20b MXFP4        POST /v1/chat/completions
    "embed"     → Qwen3-Embedding-0.6B     POST /v1/embeddings
    "rerank"    → Qwen3-Reranker-0.6B      POST /v1/score

⚠️ THE critical knob: gpu_memory_utilization is per engine, and each
engine PRE-ALLOCATES its fraction at boot. The fractions below must sum
to ~0.85 — leaving each at the 0.9 default is an instant OOM. Budget on
a 24 GB card:

    chat   0.60 → 14.4 GB (13 GB weights + ~0.8 GB KV ≈ 32k cached tokens fp8)
    embed  0.12 →  2.9 GB (1.2 GB weights + activations)
    rerank 0.12 →  2.9 GB
    sum    0.84            (rest = CUDA context + fragmentation headroom)

⚠️ The yaml's `vllm:` block MUST mirror the *_size values (all 1 here).
"""
from ray.serve.llm import LLMConfig, build_openai_app


chat = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",
        model_source="openai/gpt-oss-20b",       # native MXFP4, ~13 GB
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # Chat gets the big share but a tight KV pool (~0.8 GB ≈ 32k
        # tokens at fp8) — hence the conservative context. For 64k+
        # context alongside the sidecars, use a 32 GB card (RTX 5090)
        # and raise this to 0.70 / max_model_len to 65536.
        max_model_len=16384,
        max_num_seqs=4,
        gpu_memory_utilization=0.60,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        dtype="auto",
        # Tool calling: harmony is native to gpt-oss — VERIFY parser flag
        # before enabling (see ../README.md "Tool calling"):
        # enable_auto_tool_choice=True,
        # tool_call_parser="openai",
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
        max_ongoing_requests=32,                 # embeddings batch cheaply
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        task="embed",                            # → served on /v1/embeddings
        max_model_len=8192,
        gpu_memory_utilization=0.12,
        enforce_eager=True,                      # tiny model — save the CUDA-graph VRAM
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
        task="score",                            # → served on /v1/score
        max_model_len=8192,
        gpu_memory_utilization=0.12,
        enforce_eager=True,
        # Qwen3-Reranker ships as a causal LM; vLLM scores it as sequence
        # classification via this documented conversion:
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    ),
)


app = build_openai_app({"llm_configs": [chat, embed, rerank]})
