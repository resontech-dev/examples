"""⚠️ TEMPLATE: Kimi K2.6 (official INT4 QAT) via vLLM, TP=8 on one 8× H200 node. Text + image.

NEEDS the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Written
2026-11-18; nothing has booted, every number is napkin math.

Moonshot's 1T mixture-of-experts, 32B active per token, MLA attention,
262k context, text + image (MoonViT 400M). GPQA 90.5, HLE 34.7 (54.0 with
tools), SWE-bench Verified 80.2. Licence: Modified MIT; products above
100M monthly active users or $20M monthly revenue must show "Kimi K2.6".

    "kimi-k2.6" -> moonshotai/Kimi-K2.6    POST /v1/chat/completions (text, images, tools)

Checkpoint: the official INT4 QAT build, 595.2 GB (compressed-tensors).
Memory per GPU at TP=8 (ESTIMATES): 74.4 GB weights.
  8x H200 at 0.90: 127 - 74.4 - ~4 ≈ 48 GB KV per GPU. MLA KV is copied on
                   every TP rank, ~70 KB per token in bf16 -> ~680k tokens per node
  8x B200:         ≈ 83 GB per GPU, ~1.2M tokens
  8x H100, 4x H200: do not fit

⚠️ trust_remote_code=True is REQUIRED: the tokenizer (tiktoken-based
tokenization_kimi.py) and the image processor come from the repo through
auto_map. This is the only template here that runs repo code. Before the
first boot: read that code, pin the reviewed commit in `revision`,
`code_revision` and `tokenizer_revision`, and check it imports under
transformers 5.17 (unverified).

Thinking is on by default. Per request: chat_template_kwargs
{"thinking": false} (Instant mode) or {"thinking": true,
"preserve_thinking": true} for agents. Card sampling: temperature 1.0 with
thinking, 0.6 in Instant mode, top_p 0.95. Open issue: vLLM #42426, the
reasoning field sometimes comes back as "!!!!".

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="kimi-k2.6",                        # client-facing id
        # 595 GB: pre-stage it, never pull at first boot:
        #   model_source="/models/local/kimi-k2.6",
        model_source="moonshotai/Kimi-K2.6",
    ),

    accelerator_type=None,                          # or "H200" once the node carries Ray's label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=64,
        max_queued_requests=128,
        health_check_period_s=30,
        health_check_timeout_s=1800,                # 595 GB load + CUDA graphs
        graceful_shutdown_timeout_s=300,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=8,                     # = yaml
        pipeline_parallel_size=1,                   # = yaml
        max_model_len=262144,                       # native window
        max_num_seqs=32,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,

        # Required by the repo (tokenizer + processor via auto_map). Pin the
        # reviewed commit before any deploy:
        trust_remote_code=True,
        # revision="<reviewed sha>", code_revision="<reviewed sha>", tokenizer_revision="<reviewed sha>",

        # ── Multimodal (model card: video works only on Moonshot's own API) ──
        limit_mm_per_prompt={"image": 4, "video": 0},
        mm_encoder_tp_mode="data",
        # language_model_only=True,                 # text only: skips the vision path

        enable_auto_tool_choice=True,
        tool_call_parser="kimi_k2",
        reasoning_parser="kimi_k2",

        # After the first clean boot, one at a time:
        # decode_context_parallel_size=8,           # recipe: shards the MLA KV instead of copying it
        # speculative_config={"model": "lightseekorg/kimi-k2.6-eagle3-mla",   # third-party draft
        #                     "method": "eagle3", "num_speculative_tokens": 3},
        #   (vLLM #51660: structured outputs + EAGLE3 killed the engine on 0.26)
    ),

    placement_group_config=dict(
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_PACK",                     # all 8 shards on one node, or fail loudly
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
