"""Qwen2.5-VL-32B (AWQ) via vLLM — TP=2 document-vision quality tier.

The quality tier for hard documents (`vision-qwen25-vl-32b`): degraded
scans, handwriting, dense tables — where the 7B starts guessing. AWQ
~19 GB sharded across 2×24 GB GPUs in ONE machine.

Same OpenAI vision protocol as the 7B sibling — clients don't change,
only the `model` quality behind the endpoint does.

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="vision",                       # same alias as the 7B — drop-in upgrade
        # Official Qwen AWQ build. VERIFY the exact repo id against HF
        # before first deploy (Qwen ships AWQ for the VL line; the 32B
        # AWQ repo id should match this pattern).
        model_source="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    ),

    accelerator_type=None,                       # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=2,                  # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                # = yaml vllm.pipeline_parallel_size
        # Per-GPU budget: 21.6 − 9.5 weights-shard − ~1 vision buffers
        # ≈ 11 GB KV/GPU → ~22 GB pooled ≈ 168k tokens at fp8. Plenty
        # for multi-page batches at 32k.
        max_model_len=32768,
        max_num_seqs=8,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,
        dtype="auto",
        limit_mm_per_prompt={"image": 6},

        # ── Tool calling (stable for the Qwen family) ──
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    ),

    placement_group_config=dict(
        # One bundle per TP shard (auto-replicated 2×), both on ONE node.
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_PACK",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
