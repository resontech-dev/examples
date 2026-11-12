"""Qwen3.5-122B-A10B GPTQ-Int4 via vLLM: level-2 text + vision assistant.

122B mixture-of-experts with 10B active parameters per token, 48 layers
(36 DeltaNet + 12 full attention), native 262k window (1M with YaRN),
images and tool calls on one model id. Benchmarks put it at the Claude
Sonnet 4.6 level on GPQA Diamond and HLE (level 2 in
reson_docs/docs/inference/models/README.md). It is the switch option next
to gpt-oss-120b on the same 96 GB / H200 hardware: same card, higher
level, with vision. Int4 weights ~79 GB.

VERIFY, CHECKPOINT: "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" is the official
Int4 repo name as listed in the Qwen3.5 collection; confirm it exists
before the first submit. Alternative:
    "Qwen/Qwen3.5-122B-A10B-FP8"   official FP8, ~125 GB: TP=2 on 2x H100 / 2x H200

VERIFY, KERNEL: the Int4 expert layers run through vLLM's `moe_wna16`
fused-MoE path, unverified on the pinned image (vLLM 0.18.0, which does
ship the Qwen3.5 model classes). Boot the sibling
../Qwen3.5-35B-A3B-GPTQ-Int4_agents_vision_min_22gb_vram on a 24 GB card
first: same kernel, same answer, 4x cheaper. If that one fails with an
unsupported-quantization error for FusedMoE, use the FP8 checkpoint here.

Budget on a 96 GB card (RTX PRO 6000) at 0.92 -> 88 GB: ~79 GB weights +
~2 GB activations/vision -> ~7 GB KV. KV is about 25 KB/token in bf16
(only the 12 attention layers keep KV), so ~280k cached tokens: 64k x 4
seqs, or 32k x 8. No room for sidecars on this card; put embed/rerank on
another card. On an H200 (141 GB) the same config leaves ~49 GB KV, about
1.9M cached tokens: raise max_num_seqs to 32 and max_model_len to 262144.

2x80 GB (H100 / A100) instead: yaml vllm.tensor_parallel_size: 2,
cluster.gpus_per_worker: 2, tensor_parallel_size=2 below, and the
placement_group_config block uncommented. ~40 GB of weights and ~30 GB
KV per card.

Throughput (estimate, nothing measured yet): 10B active parameters read
per token -> on a PRO 6000 (1.8 TB/s) about 40-60 generated tok/s per
stream, 300-500 aggregate; on an H200 (4.8 TB/s) roughly 2.5x that.
Measure before quoting.

The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                     # one id for text AND images; keep stable
        model_source="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",  # VERIFY repo name, see banner
    ),

    # Datacenter GPUs may carry Ray accelerator labels (H100, H200). With
    # an unlabeled fleet keep None; pin only if the fleet labels its nodes.
    accelerator_type=None,

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=180,          # long reasoning answers finish on redeploy
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                   # 2 for the 2x80 GB shape (= yaml)
        pipeline_parallel_size=1,
        max_model_len=65536,                      # 262144 native (fits the KV pool on an H200)
        max_num_seqs=16,
        gpu_memory_utilization=0.92,
        enable_chunked_prefill=True,
        dtype="auto",

        limit_mm_per_prompt={"image": 4},         # a page is 1-2.5k tokens at 1 000-1 500 px
        # language_model_only=True,               # text-only: drop the vision tower

        # Tool calling (Qwen3.5): official recipe is qwen3_coder. Community
        # reports qwen3_xml holds up better on long agentic runs. VERIFY on
        # your scenarios.
        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder",

        # Thinking: on unless the request sends
        # chat_template_kwargs={"enable_thinking": False}, as predict.py does.
        # Don't set `default_chat_template_kwargs` here: Ray 2.55.1 turns it
        # into a Namespace and every chat request then fails with a TypeError
        # (reson_docs inference/vllm/known_issues.md).
        reasoning_parser="qwen3",

        # Speculative decoding with the model's own MTP head: the model card
        # lists `qwen3_next_mtp`. VERIFY on vLLM 0.18.0 (boot without it
        # first, then add it and compare tok/s):
        # speculative_config={"method": "qwen3_next_mtp", "num_speculative_tokens": 2},

        # Deliberately NOT set on this hybrid model (add one at a time,
        # verify output): kv_cache_dtype="fp8", enable_prefix_caching=True.
        # If boot dies with a `causal_conv1d_update` assertion:
        # max_cudagraph_capture_size=256,
    ),

    # TP=1 needs no placement group. For the 2x80 GB TP=2 shape uncomment:
    # placement_group_config=dict(
    #     bundle_per_worker={"GPU": 1, "CPU": 4},
    #     strategy="STRICT_PACK",                  # TP never crosses machines
    # ),
)


app = build_openai_app({"llm_configs": [llm_config]})
