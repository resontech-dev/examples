"""⚠️ TEMPLATE: DeepSeek V4-Pro (official 0813 release) via vLLM, TP=8 + expert parallel on one 8× H200 node.

NEEDS the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Prod runs
Ray 2.55.1 / vLLM 0.18.0 until the switch (reson_docs
docs/big_models_experiments/ray2.60.0-upgrade.md). Written 2026-11-18; nothing
has booted, every number is napkin math.

1.6T mixture-of-experts, 49B active per token, 1M context, text only, MIT.
GPQA 90.1, HLE 37.7 (48.2 with tools), SWE-bench Verified 80.6. The same
architecture and flags as V4-Flash, about 5x the weights.

    "deepseek-v4-pro" -> deepseek-ai/DeepSeek-V4-Pro-0813    POST /v1/chat/completions

Checkpoint: the official 0813 release, 892.7 GB (FP4 experts + FP8 rest)
with a built-in DSpark draft. It fits ONE node only on 8x H200 or 8x B200:
  8x H200 at 0.95: 134 - 111.6 GB weights - ~4 runtime ≈ 18-22 GB KV per GPU
  8x B200:          ≈ 55 GB KV per GPU, the full 1M window
  8x H100: does not fit (640 GB < 893 GB). Two H100 nodes need EP
  all-to-all over RDMA, which WireGuard is not; out of scope here.

Before this file runs: DeepGEMM in the image, the deepseek_v4 tokenizer
mode, no trust_remote_code: see ../DeepSeek-V4-Flash_reasoning_template_min_180gb_vram,
which proves the family on half a node first.

Thinking is off unless a request sends chat_template_kwargs
{"thinking": true, "reasoning_effort": "low" | "high" | "max"}; "max"
needs max_model_len >= 393216.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="deepseek-v4-pro",                   # client-facing id
        # 893 GB: pre-stage it, never pull at first boot:
        #   model_source="/models/local/deepseek-v4-pro-0813",
        model_source="deepseek-ai/DeepSeek-V4-Pro-0813",
    ),

    accelerator_type=None,                           # or "H200" once the node carries Ray's label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=64,
        max_queued_requests=128,
        health_check_period_s=30,
        health_check_timeout_s=1800,                 # 893 GB load + warm-up
        graceful_shutdown_timeout_s=300,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=8,                      # = yaml
        pipeline_parallel_size=1,                    # = yaml
        enable_expert_parallel=True,
        max_model_len=393216,                        # Think Max; recipe runs 800000 on 8x H200
        max_num_seqs=64,                             # recipe: 512 for throughput runs
        gpu_memory_utilization=0.95,                 # recipe: weights take 111.6 GB of each H200
        enable_chunked_prefill=True,
        enable_prefix_caching=True,

        # ── vLLM recipe for DeepSeek V4-Pro on 8x H200 ──
        tokenizer_mode="deepseek_v4",                # mandatory: renders the chat format
        kv_cache_dtype="fp8",
        block_size=256,
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
        enable_flashinfer_autotune=False,

        enable_auto_tool_choice=True,
        tool_call_parser="deepseek_v4",
        reasoning_parser="deepseek_v4",

        # DSpark speculative decoding, after the first clean boot:
        # speculative_config={"method": "dspark", "num_speculative_tokens": 7,
        #                     "draft_sample_method": "probabilistic"},
    ),

    placement_group_config=dict(
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_PACK",                      # all 8 shards on one node, or fail loudly
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
