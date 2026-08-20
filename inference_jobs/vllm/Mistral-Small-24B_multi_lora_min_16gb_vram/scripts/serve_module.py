"""Mistral Small 24B + N customer LoRA adapters — multi-LoRA on 24 GB.

The EU-friendly mid-size LoRA base (`lora-mistral-small-24b`): regulated
buyers get their own fine-tuned tone/format model without the data ever
leaving their deployment. One AWQ base, adapters on demand:

    model="mistral-24b"                → base, no adapter
    model="mistral-24b:acme-reports"   → base + adapter "acme-reports"

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
import os

from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="mistral-24b",                  # adapter requests prefix this id
        # VERIFY before selling: community AWQ re-upload (no official
        # Mistral AWQ). vLLM supports LoRA on AWQ bases; adapters must be
        # trained against the SAME base revision customers deploy on.
        model_source="casperhansen/mistral-small-24b-instruct-2501-awq",
    ),

    accelerator_type=None,                       # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=12,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # KV budget on 24 GB: 21.6 − 13.2 base − ~0.8 activations − ~1.5
        # adapter head-room ≈ 6 GB ≈ 73k tokens at fp8 (~82 KB/token).
        max_model_len=16384,
        max_num_seqs=6,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,
        dtype="auto",

        # ── Tool calling (stable for the Mistral family) ──
        enable_auto_tool_choice=True,
        tool_call_parser="mistral",

        # ── LoRA engine knobs ──
        enable_lora=True,
        max_loras=4,                             # 24B adapters are heavier — keep modest
        max_lora_rank=32,                        # raise to 64 if customers train r=64
    ),

    lora_config=dict(
        dynamic_lora_loading_path=os.environ.get(
            "LORA_S3_PREFIX",
            "s3://YOUR-BUCKET/loras/mistral-small-24b/",  # ← your adapter store
        ),
        max_num_adapters_per_replica=8,
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
