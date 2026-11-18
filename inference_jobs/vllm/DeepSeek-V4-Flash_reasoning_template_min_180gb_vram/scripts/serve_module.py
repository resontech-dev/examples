"""⚠️ TEMPLATE: DeepSeek V4-Flash (official 0731 release) via vLLM, TP=4 + expert parallel on one 4-GPU node.

NEEDS the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Prod runs
Ray 2.55.1 / vLLM 0.18.0 until the switch (reson_docs
docs/big_models_experiments/ray2.60.0-upgrade.md). Written 2026-11-18; nothing
has booted, every number is napkin math.

284B mixture-of-experts, 13B active per token, sparse attention with a
lightning indexer, 1M context, text only, MIT. GPQA 88.1, HLE 34.8 (45.1
with tools), SWE-bench Verified 79.0. Cheap on KV: ~3-5 KB per token, so
memory is never the limit for context on the shapes below.

    "deepseek-v4-flash" -> deepseek-ai/DeepSeek-V4-Flash-0731    POST /v1/chat/completions

Checkpoint: the official 0731 release, 166.9 GB (FP4 experts + FP8 rest)
with a built-in DSpark draft for speculative decoding. The earlier preview
`deepseek-ai/DeepSeek-V4-Flash` (159.6 GB) has an MTP head instead.

Memory per GPU (ESTIMATES):
  4x H200 141 GB at 0.90: 127 - 42 GB weights - ~4 runtime ≈ 80 GB KV per GPU
  4x H100 80 GB at 0.90:   72 - 42 - ~4 ≈ 26 GB KV per GPU (community-reported shape)
  8x H100, TP=8:           72 - 21 - ~4 ≈ 47 GB KV per GPU

Before this file runs:
  * VERIFY DeepGEMM in the image. DeepSeek V4's sparse-attention indexer
    raises at init on CUDA without it. The vLLM 0.29.0 wheel ships its own
    copy; check `python -c "import vllm.third_party.deep_gemm"` in the
    ray-llm image.
  * `tokenizer_mode="deepseek_v4"` is mandatory: the repo has no Jinja chat
    template, the tokenizer mode renders the chat format.
  * The vLLM recipe adds trust_remote_code=True, but the repo has no
    auto_map and vLLM 0.29 ships its own deepseek_v4 config, so it is left
    out here. If the boot asks for it, read the repo code first.
  * The recipe's VLLM_USE_RUST_FRONTEND=1 does not apply: Ray Serve uses
    the Python path.

Thinking is off unless a request sends chat_template_kwargs
{"thinking": true, "reasoning_effort": "low" | "high" | "max"}. "max"
(Think Max) needs max_model_len >= 393216, hence the value below.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="deepseek-v4-flash",                 # client-facing id
        # 166.9 GB cold pull unless pre-staged; prefer the local path:
        #   model_source="/models/local/deepseek-v4-flash-0731",
        model_source="deepseek-ai/DeepSeek-V4-Flash-0731",
    ),

    accelerator_type=None,                           # or "H200" once the node carries Ray's label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=64,
        max_queued_requests=128,
        health_check_period_s=30,
        health_check_timeout_s=900,                  # 167 GB load + DeepGEMM warm-up + CUDA graphs
        graceful_shutdown_timeout_s=300,             # long reasoning answers finish on redeploy
    ),

    engine_kwargs=dict(
        tensor_parallel_size=4,                      # = yaml; 8 on 8x H100
        pipeline_parallel_size=1,                    # = yaml
        enable_expert_parallel=True,                 # recipe: experts spread across the 4 GPUs
        max_model_len=393216,                        # Think Max needs it; 1M native
        max_num_seqs=64,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,

        # ── vLLM recipe for DeepSeek V4 ──
        tokenizer_mode="deepseek_v4",                # mandatory: renders the chat format
        kv_cache_dtype="fp8",
        block_size=256,

        enable_auto_tool_choice=True,
        tool_call_parser="deepseek_v4",
        reasoning_parser="deepseek_v4",

        # DSpark speculative decoding (built into 0731). Add after the first
        # clean boot and compare tokens/s and answers with and without it:
        # speculative_config={"method": "dspark", "num_speculative_tokens": 7,
        #                     "draft_sample_method": "probabilistic"},
        # Preview checkpoint instead: {"method": "mtp", "num_speculative_tokens": 1} (Hopper).
    ),

    placement_group_config=dict(
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_PACK",                      # all 4 shards on one node, or fail loudly
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
