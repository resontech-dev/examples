"""Qwen2.5-VL-7B (fp16) via vLLM — document-vision endpoint.

The doc-understanding workhorse (`vision-qwen25-vl-7b`, priority v1):
invoices, scans, receipts, screenshots → structured JSON, tens of
thousands of pages/day flat-rate on one 24 GB card.

Requests use the standard OpenAI chat route with image content parts
(data URLs or fetchable URLs) — every OpenAI vision client works
unmodified. See predict.py.

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="vision",                       # client-facing name — keep stable
        # Official fp16 checkpoint, ~16.6 GB. fp16 (not AWQ) on purpose:
        # document OCR quality degrades visibly under 4-bit for small
        # text — the AWQ variant is the 32B sibling's game.
        model_source="Qwen/Qwen2.5-VL-7B-Instruct",
    ),

    accelerator_type=None,                       # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # KV budget on 24 GB: 21.6 − 16.6 weights − ~1 vision buffers
        # ≈ 4 GB ≈ 140k tokens at fp8. A document page costs ~1–2.5k
        # vision tokens, so 32k context handles multi-page batches.
        max_model_len=32768,
        max_num_seqs=8,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,              # same extraction prompt every request
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,
        dtype="auto",
        # Cap images per request — vision tokens eat KV fast. Raise for
        # multi-page-per-call workloads on bigger cards.
        limit_mm_per_prompt={"image": 4},

        # ── Tool calling (stable for the Qwen family) ──
        # Useful for agents that route extracted fields into systems.
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
