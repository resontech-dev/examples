"""Qwen2.5-7B + N customer LoRA adapters — multi-LoRA serving on 12 GB.

The north-star product's serving half (`lora-qwen25-7b`): one AWQ base
model in VRAM, up to 16 customer adapters (tone, vocabulary, format
compliance) resolved on demand from S3 and selected per request:

    model="qwen-7b"              → base, no adapter
    model="qwen-7b:acme-support" → base + adapter folder "acme-support"

Adapters are standard PEFT LoRA checkpoints (adapter_config.json +
weights) in folders under `dynamic_lora_loading_path`. Per-adapter
billing hangs off the model id.

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
import os

from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-7b",                      # adapter requests prefix this id
        # Official Qwen AWQ 4-bit — ~5.5 GB, the best quality-per-GB
        # dense LoRA base (vLLM LoRA-on-AWQ is supported).
        model_source="Qwen/Qwen2.5-7B-Instruct-AWQ",
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
        # KV budget on 12 GB: 10.6 − 5.5 base − ~0.6 activations − ~1 GB
        # adapter head-room ≈ 3.5 GB ≈ 120k tokens at fp8 (Qwen2.5-7B
        # GQA is light). 8k × 8 seqs is comfortable.
        max_model_len=8192,
        max_num_seqs=8,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.88,
        dtype="auto",

        # ── LoRA engine knobs ──
        enable_lora=True,
        max_loras=8,                             # adapters resident per batch
        max_lora_rank=32,                        # raise to 64 if customers train r=64
    ),

    # ── Adapter resolution ──
    # <prefix>/<adapter_name>/ must hold a PEFT checkpoint. Request
    # model="qwen-7b:<adapter_name>" loads it on first use and caches it.
    lora_config=dict(
        dynamic_lora_loading_path=os.environ.get(
            "LORA_S3_PREFIX",
            "s3://YOUR-BUCKET/loras/qwen25-7b/",  # ← point at your adapter store
        ),
        max_num_adapters_per_replica=16,          # VRAM-bounded (~8–32)
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
