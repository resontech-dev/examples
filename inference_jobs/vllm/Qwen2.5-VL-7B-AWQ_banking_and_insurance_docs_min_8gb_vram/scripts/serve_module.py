"""Qwen2.5-VL-7B (AWQ) via vLLM — BFSI document-ingest endpoint (Stack A).

The vision engine of the banking_and_insurance pilot: repair invoices,
damage acts, and police reports (clean PDFs and phone photos) go in as
images, structured JSON per a pydantic schema comes out. The pilot's
ingest worker (sectors/banking_and_insurance/bfsi_pilot/process_documents.py) is
the production client; predict.py here is just a smoke test.

AWQ (not fp16) on purpose: the pilot runs on ONE 16 GB card and the
models are used sequentially (Stack A → Stack B). The fp16 sibling
(../Qwen2.5-VL-7B_vision_min_16gb_vram) is the quality pick for a
dedicated 24 GB worker.

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="vision",                       # process_documents.py passes model="vision"
        # Official Qwen AWQ build (~6 GB). VERIFY the repo id on HF
        # before first deploy, as with every quantized source.
        model_source="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    ),

    accelerator_type=None,                       # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,
        max_queued_requests=32,
        graceful_shutdown_timeout_s=60,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # Extraction is one page + one JSON per request — 8k is plenty
        # and keeps the KV pool tiny on the shared 16 GB card.
        max_model_len=8192,
        max_num_seqs=4,
        # ── Deliberately CONSERVATIVE for the VL engine ──
        # fp8 KV cache and prefix caching are the two most common causes of
        # garbled output (broken tokens from position 1, deterministic at
        # temperature=0) with multimodal models on quantized checkpoints —
        # both OFF here, unlike the text-only jobs. The VRAM win would be
        # negligible anyway: one page ≈ 1–2.5k tokens, KV pool is tiny.
        # kv_cache_dtype="fp8",
        # enable_prefix_caching=True,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,
        dtype="auto",
        limit_mm_per_prompt={"image": 2},        # one doc page (+1 reference) per call
        # Still garbled after a redeploy? Escalate in this order:
        #   1) enforce_eager=True   — rules out CUDA-graph issues (~15% slower)
        #   2) model_source="Qwen/Qwen2.5-VL-3B-Instruct"  — fp16, ~7 GB,
        #      fits a 16 GB card UNQUANTIZED; if 3B fp16 is clean, the AWQ
        #      build is the culprit on this stack — hunt a different quant.
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
