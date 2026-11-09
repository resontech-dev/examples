"""MamayLM 27B v2.0 official FP8 via vLLM — TP=2 across two 24 GB cards.

The Ukrainian quality tier: Gemma 3 27B continued-pretrained on Ukrainian
by INSAIT (June 2026), text + images. INSAIT ships the FP8-dynamic build
themselves (all linear layers of the LM in FP8; vision tower, projector
and lm_head kept in bf16), ~30 GB — two 24 GB cards in ONE machine, or
one 48 GB card at TP=1.

    "assistant" → INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-FP8-dynamic   POST /v1/chat/completions

⚠️ Do NOT add quantization="fp8": the checkpoint is already quantized and
the flag re-quantizes the vision tower → garbage on image inputs (INSAIT).
⚠️ FP8 needs compute capability ≥ 8.9: 4090 / 5090 / RTX PRO yes, 3090 no.

No int4 build of the 27B exists yet. For 2×16 GB or one 32 GB card you
would quantize the bf16 checkpoint yourself (llm-compressor, W4A16 with a
Ukrainian calibration set) — ~17 GB — and serve it with the same file.

Per-GPU budget at 0.90 × 24 = 21.6 GB: 15 GB weight shard + ~1 GB
activations/vision → ~5.5 GB KV per card. 32k × 8 seqs fits.

Gemma 3 engine notes: dtype bfloat16 only; no tool/reasoning parsers;
fp8 KV left off. See ../README.md.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",
        model_source="INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-FP8-dynamic",
    ),

    accelerator_type=None,                               # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=2,                          # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                        # = yaml vllm.pipeline_parallel_size
        max_model_len=32768,                             # 128k native
        max_num_seqs=8,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        dtype="bfloat16",                                # Gemma 3: never fp16
        limit_mm_per_prompt={"image": 4},
        # quantization is auto-detected from the checkpoint — never set it here.
    ),

    placement_group_config=dict(
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_PACK",                          # both shards on ONE node
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
