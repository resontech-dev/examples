"""⚠️ TEMPLATE: Mistral Large 3 675B Instruct (official FP8) via vLLM, TP=8 on one 8× H200 node. Text + image.

NEEDS the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Prod runs
Ray 2.55.1 / vLLM 0.18.0 until the switch (reson_docs
docs/big_models_experiments/ray2.60.0-upgrade.md). Written 2026-11-18; nothing
has booted, every number is napkin math.

Mistral's 675B mixture-of-experts (673B language model + 2.5B vision
encoder), 41B active per token, 256k context, text + image, Apache-2.0.
An instruct model: no reasoning variant, no thinking switch. The
Apache-licensed European option among the frontier-size models.

    "mistral-large-3" -> mistralai/Mistral-Large-3-675B-Instruct-2512    POST /v1/chat/completions

Checkpoint: the official FP8 build in Mistral's consolidated format,
681.5 GB. Memory per GPU at TP=8 (ESTIMATES): 85 GB weights.
  8x H200 at 0.90: 127 - 85 - ~4 ≈ 37 GB KV per GPU, ~530k tokens of bf16 KV
  8x B200:         ~1M tokens
  8x H100:         needs the NVFP4 build (403 GB, Marlin fallback): ~17 GB per GPU

Engine notes (model card, vLLM recipe, vLLM 0.29.0 source):
  * tokenizer_mode / config_format / load_format "mistral" set explicitly:
    auto-detection keys on the mistralai repo id and misses a pre-staged
    local path. mistral-common 1.11.7 is in the Ray 2.60 lock (>= 1.8.6).
  * The Eagle draft (`...-Instruct-2512-Eagle`) is broken on vLLM >= 0.28
    when the target is multimodal (vLLM #57630, #43298). Not shipped.
  * Card sampling: temperature below 0.1. The repo's SYSTEM_PROMPT.txt is
    meant as the system message; whether vLLM injects it is unverified, so
    clients send it.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="mistral-large-3",                   # client-facing id
        # 681.5 GB: pre-stage it, never pull at first boot:
        #   model_source="/models/local/mistral-large-3-2512",
        model_source="mistralai/Mistral-Large-3-675B-Instruct-2512",
    ),

    accelerator_type=None,                           # or "H200" once the node carries Ray's label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=64,
        max_queued_requests=128,
        health_check_period_s=30,
        health_check_timeout_s=1800,                 # 680 GB load + CUDA graphs
        graceful_shutdown_timeout_s=300,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=8,                      # = yaml
        pipeline_parallel_size=1,                    # = yaml
        max_model_len=262144,                        # native window
        max_num_seqs=32,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,

        # Mistral's own format, explicit (see banner):
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",

        limit_mm_per_prompt={"image": 10},
        mm_encoder_tp_mode="data",

        enable_auto_tool_choice=True,
        tool_call_parser="mistral",
        # No reasoning_parser: an instruct model without a thinking mode.

        # Optional after the first boot, with an output check:
        # kv_cache_dtype="fp8",
    ),

    placement_group_config=dict(
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_PACK",                      # all 8 shards on one node, or fail loudly
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
