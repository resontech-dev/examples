"""Mistral Small 24B (AWQ) via vLLM — single-worker business chat.

Dense 24B, the popular EU-deployment pick (`chat-mistral-small-24b`):
fast, low-refusal business assistant with strong native function
calling. AWQ 4-bit ≈ 13 GB weights on one 24 GB card.

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                    # client-facing name — keep stable
        # VERIFY before selling: community AWQ 4-bit re-upload (Mistral
        # publishes no official AWQ). Vet the repo + re-run your evals,
        # or swap in a newer vetted Mistral-Small-3.x AWQ build. The
        # official bf16 (mistralai/Mistral-Small-3.2-24B-Instruct-2506,
        # ~47 GB) needs 2×24 GB TP=2 instead.
        model_source="casperhansen/mistral-small-24b-instruct-2501-awq",
    ),

    accelerator_type=None,                       # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,                 # ~2× max_num_seqs
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # KV budget on 24 GB: 21.6 − 13.2 weights − ~0.8 activations
        # ≈ 7.6 GB. Mistral Small KV ≈ 82 KB/token at fp8 (dense 40-layer
        # GQA) → ~92k cached tokens. 32k × a few seqs is comfortable.
        max_model_len=32768,
        max_num_seqs=8,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,
        dtype="auto",

        # ── Tool calling (stable for the Mistral family) ──
        enable_auto_tool_choice=True,
        tool_call_parser="mistral",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
