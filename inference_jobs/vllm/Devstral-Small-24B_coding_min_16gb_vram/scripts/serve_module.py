"""Devstral Small 24B (AWQ) via vLLM — agentic coding endpoint.

Mistral's agentic SWE specialist (`code-devstral-24b`): OpenHands /
SWE-bench lineage, punches above its size on issue-fixing loops, a bit
weaker on raw completion than the Qwen coders. Dense 24B, AWQ ≈ 13 GB
on one 24 GB card.

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="team-coder",                   # tools point at this alias — keep stable
        # VERIFY before selling: community AWQ re-upload (Mistral ships
        # bf16 only). Vet the repo + rerun evals; swap in the newest
        # vetted Devstral-Small AWQ build. Official bf16
        # (mistralai/Devstral-Small-2507, ~47 GB) needs 2×24 GB TP=2.
        model_source="stelterlab/Devstral-Small-2507-AWQ",
    ),

    accelerator_type=None,                       # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,                  # agents retry politely on fast 503s
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # Agent scaffolds (OpenHands, Cline) send 10–40k-token contexts;
        # 32k + prefix caching is the sweet spot on 24 GB. KV budget:
        # 21.6 − 13.2 − 0.8 ≈ 7.6 GB ≈ 92k tokens at fp8.
        max_model_len=32768,
        max_num_seqs=8,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,              # THE win for agent loops (70–90% hit rates)
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,
        dtype="auto",

        # ── Tool calling (stable for the Mistral family) ──
        enable_auto_tool_choice=True,
        tool_call_parser="mistral",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
