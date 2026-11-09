"""Qwen3.5-27B AWQ via vLLM — premium text + vision assistant.

Top rung of the small-over-big ladder. One engine, one model id for text
tasks and image input. Hybrid attention (16 of 64
layers keep KV) → cheap long context; native 262k window.

⚠️ CHECKPOINT — VERIFY before selling. No official 4-bit Qwen3.5-27B;
`QuantTrio/Qwen3.5-27B-AWQ` is a data-free community AWQ (~21 GiB,
Apache-2.0, vLLM ≥ 0.16). Alternatives:
    "Qwen/Qwen3.5-27B-FP8"   official FP8, ~30 GB → 2×24 GB TP=2 or 48 GB
    "Qwen/Qwen3.5-27B"       official bf16, ~54 GB → 2×32 GB TP=2 or 80 GB

Budget on a 32 GB card at 0.90 → 28.8 GB: ~22.5 GB weights + ~1.5 GB
activations/vision → ~5 GB KV ≈ 150k cached tokens (hybrid KV ≈ 32 KB
per token bf16). 32k × 8 seqs is comfortable. Sidecars on the same card:
assistant 0.78 + embed 0.06 + rerank 0.06 (copy the two LLMConfigs from
the Qwen3-8B bundle) — chat KV then drops to ~1.5 GB.

2×24 GB (4090 / 3090) instead: yaml vllm.tensor_parallel_size: 2,
cluster.gpus_per_worker: 2, and tensor_parallel_size=2 below. Per card
~11 GB weights → KV room on each; decode bandwidth doubles.

Throughput (estimate, measure before quoting): 5090 ≈ 0.6–0.8 req/s at
3k-token prompts, 160–220 generated tok/s, 6–8 users under 5 s TTFT;
≈ 3× slower per card than the 9B — memory-bandwidth-bound decode.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                     # one id for text AND images — keep stable
        model_source="QuantTrio/Qwen3.5-27B-AWQ", # VERIFY — community AWQ, see banner
    ),

    accelerator_type=None,                        # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                   # 2 for the 2×24 GB shape (= yaml)
        pipeline_parallel_size=1,
        max_model_len=32768,                      # 262144 native
        max_num_seqs=8,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        dtype="auto",

        limit_mm_per_prompt={"image": 4},         # a page ≈ 1–2.5k tokens at 1 000–1 500 px
        # language_model_only=True,               # text-only: drop the vision tower

        # ── Tool calling (Qwen3.5) ──
        # Official recipe: qwen3_coder. Community reports qwen3_xml holds
        # up better on long agentic runs — VERIFY on your scenarios.
        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder",

        # ── Thinking ── on unless the request sends
        # chat_template_kwargs={"enable_thinking": False}, as predict.py does.
        # Don't set `default_chat_template_kwargs` here: Ray 2.55.1 turns it into a
        # Namespace and every chat request then fails with a TypeError
        # (reson_docs inference/vllm/known_issues.md).
        reasoning_parser="qwen3",

        # Deliberately NOT set on this hybrid model (add one at a time,
        # verify output): kv_cache_dtype="fp8", enable_prefix_caching=True.
        # If boot dies with a `causal_conv1d_update` assertion:
        # max_cudagraph_capture_size=256,
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
