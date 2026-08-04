"""GLM-4.5-Air 106B-A12B (AWQ) via vLLM — TP=4 premium chat.

The catalog's premium agentic tier (`chat-glm45-air`, MIT license):
106B MoE with 12B active parameters — agentic benchmarks in genuine
Sonnet-4 territory, decode cost of a 12B. AWQ ~60 GB sharded across
4×24 GB GPUs in ONE worker (TP needs PCIe/NVLink; it dies over any
network — hence STRICT_PACK below).

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                    # client-facing name — keep stable
        # VERIFY before selling: community AWQ re-upload of GLM-4.5-Air
        # (Zhipu publishes bf16 + fp8, no official AWQ). Vet the repo +
        # re-run evals. Official fp8 (~110 GB) needs 8×24 or 2×80 instead.
        model_source="cpatonn/GLM-4.5-Air-AWQ",
    ),

    accelerator_type=None,                       # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,                 # ~2× max_num_seqs
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=4,                  # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                # = yaml vllm.pipeline_parallel_size
        # Per-GPU budget: 24×0.90 = 21.6 − ~15 weights-shard − ~1
        # activations ≈ 5.5 GB KV/GPU → ~22 GB pooled. At fp8 KV that's
        # plenty for 64k contexts across several sequences.
        max_model_len=65536,
        max_num_seqs=16,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,              # agent loops resend huge prefixes
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,
        dtype="auto",

        # ── Tool calling — VERIFY before enabling ──
        # GLM-4.5 has a dedicated parser only in newer vLLM builds; check
        # the pinned image first (wrong name = crash at engine boot):
        #   docker run --rm <ray-llm-image> vllm serve --help | grep -A3 tool-call-parser
        # enable_auto_tool_choice=True,
        # tool_call_parser="glm45",
    ),

    placement_group_config=dict(
        # One bundle per TP shard (auto-replicated 4×), all on ONE node —
        # fail loudly rather than let a shard land across the network.
        bundle_per_worker={"GPU": 1, "CPU": 2},
        strategy="STRICT_PACK",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
