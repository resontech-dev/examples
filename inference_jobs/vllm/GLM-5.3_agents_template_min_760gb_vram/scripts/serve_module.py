"""⚠️ TEMPLATE: GLM-5.3 (official FP8) via vLLM, TP=8 on one 8× H200 node. Text only.

NEEDS the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Written
2026-11-18; nothing has booted, every number is napkin math.

Z.ai's 743B mixture-of-experts (39B active per the vLLM recipe), sparse
attention with an indexer (GlmMoeDsaForCausalLM, the GLM-5.2 base with new
post-training), 1M context, text only. Terminal-Bench 2.1 88.2. Licence
"glm-5.3": MIT-like, with one condition for model-as-a-service providers
above $10B yearly revenue.

    "glm-5.3" -> zai-org/GLM-5.3    POST /v1/chat/completions

Runs on vLLM 0.29.0: the recipe's min_vllm_version is 0.29.0 and needs no
tokenizer flag. The `glm5_next` tokenizer (vLLM 0.34) belongs to
GLM-5.3-Flash, a different model.

Checkpoint: native FP8, 755.7 GB. Memory per GPU at TP=8 (ESTIMATES):
94.5 GB weights.
  8x H200 at 0.90: 127 - 94.5 - ~4 ≈ 23 GB KV per GPU. vLLM sizes that at
                   ~450k tokens; capped at 262k below (recipe)
  8x B200:         ~62 GB per GPU, the full 1M with max_num_seqs=32
  8x H100, 4x H200: only the community W4A8 build (Hopper), see README

Before this file runs: VERIFY DeepGEMM in the image. The sparse-attention
indexer raises at init on CUDA without it; the vLLM 0.29.0 wheel ships its
own copy (`python -c "import vllm.third_party.deep_gemm"`).

Thinking is always on: the template has no enable_thinking. Per request:
chat_template_kwargs {"reasoning_effort": "low" | "high"} (default "max"),
{"clear_thinking": true} for plain chat.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="glm-5.3",                           # client-facing id
        # 755.7 GB: pre-stage it, never pull at first boot:
        #   model_source="/models/local/glm-5.3",
        model_source="zai-org/GLM-5.3",
    ),

    accelerator_type=None,                           # or "H200" once the node carries Ray's label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=64,
        max_queued_requests=128,
        health_check_period_s=30,
        health_check_timeout_s=1800,                 # 756 GB load + DeepGEMM warm-up
        graceful_shutdown_timeout_s=300,             # always-on thinking: long answers
    ),

    engine_kwargs=dict(
        tensor_parallel_size=8,                      # = yaml
        pipeline_parallel_size=1,                    # = yaml
        max_model_len=262144,                        # recipe cap on 8x H200; 1M on 8x B200
        max_num_seqs=32,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        kv_cache_dtype="fp8",                        # recipe, Hopper; "fp8_e4m3" on Blackwell

        enable_auto_tool_choice=True,
        tool_call_parser="glm47",
        reasoning_parser="glm47",

        # MTP speculative decoding from the recipe, after the first clean
        # boot. For GLM-5.2 on 0.24 the recipe warned that tool calling + MTP
        # needed vLLM main; test tool calls with it on:
        # speculative_config={"method": "mtp", "num_speculative_tokens": 5},
    ),

    placement_group_config=dict(
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_PACK",                      # all 8 shards on one node, or fail loudly
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
