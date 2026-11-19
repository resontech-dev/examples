"""Qwen3.8-27B (official FP8) via vLLM: text + vision assistant for agents.

NEEDS the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Written
2026-11-19; nothing has booted.

Successor of Qwen3.6-27B on the same dense 27B hybrid body (August 2026):
GPQA 89.2, HLE 30.8, SWE-bench Pro 61.7, Terminal-Bench 2.1 73.0,
OSWorld-Verified 84.3 (Qwen3.6-27B: 87.8 / 24.0 / 53.5 / 63.4 / 63.9;
model card numbers, not replicated). Text, images and video, 262k native
window (1M with YaRN), Apache-2.0. Hybrid attention: 16 of 64 layers keep
KV (4 KV heads x 256), the 48 Gated DeltaNet layers keep a fixed state per
sequence.

    "assistant" -> Qwen/Qwen3.8-27B-FP8    POST /v1/chat/completions (text, images, tools)

Checkpoint: Qwen's own FP8 (block size 128). The vLLM recipe loads
2 x 14.28 GiB at TP=2, so ~31 GB of weights; the size on disk is not on the
card (VERIFY). No official 4-bit; for a 32 GB card see "What to change" in
README (NVFP4 builds).

Budget on one 48 GB card at 0.92 -> 44.2 GB: ~31 GB weights + ~2.5 GB
activations, vision encoder and CUDA graphs -> ~10 GB KV. 64k x 8
sequences queue rather than crash. Estimates carried over from the
Qwen3.6-27B job (same layer layout): H100 80 GB ~38 GB KV, 96 GB card
~54 GB KV, H200 ~95 GB KV.

Engine notes (vLLM recipe recipes.vllm.ai/Qwen/Qwen3.8-27B + model card):
  * The recipe asks for vLLM >= 0.17.0 and transformers >= 5.8.0; the
    Ray 2.60 image has 0.29.0 and 5.17. The recipe itself runs the
    `vllm/vllm-openai:qwen38` image, so a boot on stock 0.29.0 is the
    first thing to VERIFY.
  * The chat template opens every assistant turn with <think>: without
    reasoning_parser="qwen3" the reasoning lands in `content`.
  * Thinking is on by default at reasoning_effort "xhigh"; "medium" and
    "low" exist. The card warns that lower effort does not always shorten
    multi-turn agent runs (more failed steps and retries).
  * preserve_thinking is ON by default in Qwen3.8 (it was opt-in in 3.6):
    earlier thinking stays in the history. Good for agents and the prefix
    cache, costs context. Off per request:
    chat_template_kwargs={"preserve_thinking": False}.
  * The recipe verified text serving only; the image path is ours to
    VERIFY (predict.py --image).
  * MTP and prefix caching together garbled output on the Qwen3.5 / 3.6
    hybrids (vLLM #53912). Not re-checked on 3.8: MTP stays off below; to
    try it, turn prefix caching off in the same edit.
  * FP8 KV cache: off. The recipe's own FP8 command passes
    --kv-cache-dtype fp8, but Qwen3.6-FP8 shipped no KV scales; check the
    boot log for a missing-scales warning before turning it on.
  * On RTX (sm_120) block FP8 runs through the DeepGEMM copy vendored in
    vLLM 0.29, untested by us there: if the output is garbage, first try
    runtime_env=dict(env_vars={"VLLM_USE_DEEP_GEMM": "0"}).

The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                     # one id for text AND images: keep stable
        model_source="Qwen/Qwen3.8-27B-FP8",      # official FP8, ~31 GB
    ),

    accelerator_type=None,                        # workers carry no Ray accelerator label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=300,          # long thinking answers finish on redeploy
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                   # 2 for 2x24 GB (= yaml, gpus_per_worker: 2)
        pipeline_parallel_size=1,
        max_model_len=65536,                      # 262144 native; 131072 on H100, 262144 on 96 GB / H200
        max_num_seqs=8,                           # 16 on H100 / 96 GB / H200
        gpu_memory_utilization=0.92,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,               # default in 0.29; agents resend history
        dtype="auto",

        limit_mm_per_prompt={"image": 4},         # add "video": 1 only after the image path works
        # language_model_only=True,               # text only: frees the vision tower
        # mm_encoder_tp_mode="data",              # recipe flag; matters only at TP > 1

        # Tool calling + thinking (model card, vLLM recipe).
        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        # Thinking is on unless a request sends
        # chat_template_kwargs={"enable_thinking": False} (predict.py does).
        # Server-wide defaults (work from Ray 2.58), for example:
        # default_chat_template_kwargs={"reasoning_effort": "medium"},

        # kv_cache_dtype="fp8",                   # see the FP8 KV note above
        # MTP speculative decoding (in-checkpoint head): faster decode;
        # with prefix caching OFF until #53912 is re-checked on 3.8.
        # speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        # 1M context (YaRN, only when prompts really are that long; the card
        # says it can hurt short prompts). Needs an H200-class KV pool and
        # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1:
        # hf_overrides={"text_config": {"rope_parameters": {...}}},  # values from the recipe
        # max_model_len=1000000,
        # If boot dies with a `causal_conv1d_update` assertion:
        # max_cudagraph_capture_size=256,
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
