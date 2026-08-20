"""Guarded RAG chat — Qwen3-8B + Qwen3Guard-8B + embed + rerank, one GPU.

The full user-facing stack on one 24 GB worker, Apache-clean end to end
(no Llama-license guard):

    "assistant" → Qwen3-8B AWQ            POST /v1/chat/completions
    "guard"     → Qwen3Guard-Gen-8B       POST /v1/chat/completions (classifier)
    "embed"     → Qwen3-Embedding-0.6B    POST /v1/embeddings
    "rerank"    → Qwen3-Reranker-0.6B     POST /v1/score

The guard is a *generative* classifier: send it the content as a chat
message, it emits a safety verdict (Safe/Unsafe + categories). Callers
run it on user input before the chat model and on the answer after —
predict.py wires the full loop.

⚠️ gpu_memory_utilization is per engine and PRE-ALLOCATED at boot.
Budget on a 24 GB card (sum 0.84):

    chat   0.32 → 7.7 GB (5.5 AWQ weights + ~1.7 GB KV ≈ 23k tokens fp8)
    guard  0.40 → 9.6 GB (~8.5 GB fp8-quantized weights + small KV)
    embed  0.06 → 1.4 GB
    rerank 0.06 → 1.4 GB

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
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=16384,
        max_num_seqs=4,
        gpu_memory_utilization=0.32,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        dtype="auto",
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",               # stable for the Qwen family
        # YaRN — stretch Qwen3-8B's 32k window to ~131k on a bigger card.
        # 24 GB+ tier lever only: 128k of fp8 KV ≈ 9.5 GB for ONE
        # conversation — this bundle's 0.32 share can't afford it.
        # hf_overrides={"rope_scaling": {"rope_type": "yarn", "factor": 4.0,
        #                                "original_max_position_embeddings": 32768}},
    ),
)

guard = LLMConfig(
    model_loading_config=dict(
        model_id="guard",
        # Apache-clean guard (the catalog's no-Llama-license story).
        # fp16 is ~16 GB — too big next to the chat model, so weights are
        # fp8-quantized at load (~8.5 GB). On Ampere cards vLLM runs fp8
        # weights via Marlin (W8A16) — fine for a classifier. If a vetted
        # community AWQ of Qwen3Guard appears, swap it and drop
        # `quantization` (auto-detected from the checkpoint).
        model_source="Qwen/Qwen3Guard-Gen-8B",
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,                 # verdicts are short — high turnover
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        quantization="fp8",                      # on-the-fly weight quant, see above
        max_model_len=4096,                      # moderation inputs are short
        max_num_seqs=4,
        gpu_memory_utilization=0.40,
        kv_cache_dtype="fp8",
        enforce_eager=True,                      # classifier — save CUDA-graph VRAM
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


app = build_openai_app({"llm_configs": [chat, guard, embed, rerank]})
