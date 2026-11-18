"""Qwen3.6-35B-A3B (NVIDIA NVFP4) via vLLM: fast MoE text + vision assistant. One RTX 5090.

NEEDS the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+) and a
Blackwell card: NVFP4 is a Blackwell format. Prod runs Ray 2.55.1 / vLLM
0.18.0 until the switch (reson_docs docs/big_models_experiments/
ray2.60.0-upgrade.md). Written 2026-11-18; nothing has booted.

The speed rung of the Qwen3.6 family: 35B mixture-of-experts with 3B active
per token, GPQA 86.0, HLE 21.4, SWE-bench Verified 73.4 (Qwen3.5-35B-A3B:
84.2 / 19.7 / 69.2). Text + images, 262k native window, Apache-2.0. Decodes
roughly 3x faster per card than the dense 27B; a notch below it on hard
reasoning.

    "assistant" -> nvidia/Qwen3.6-35B-A3B-NVFP4    POST /v1/chat/completions

Checkpoint: NVIDIA's ModelOpt NVFP4 build, 23.4 GB, needs vLLM >= 0.28 on
sm_120. It ships FP8 KV scales, so fp8 KV is safe here (it is not on
community AWQ builds). Engine flags are the vLLM recipe for one RTX 5090.

Budget at 0.85 x 32 GB = 27.2 GB: 23.4 GB weights + ~1.5 GB activations
-> ~2.3 GB KV. Hybrid attention keeps ~20 KB per token in bf16 (half in
fp8) plus a fixed ~63 MB DeltaNet state per sequence: the recipe's 64k
window fits, several 16k chats run at once.

No 16 GB option exists on stock vLLM 0.29.0: the 2-bit builds need a
patched vLLM or leave no KV. On 24 GB only Intel's int4-mixed build fits
(see README), with reports of endless loops.

The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                          # one id for text AND images: keep stable
        model_source="nvidia/Qwen3.6-35B-A3B-NVFP4",   # 23.4 GB, Blackwell only
    ),

    accelerator_type=None,                             # consumer cards carry no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=300,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                        # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                      # = yaml vllm.pipeline_parallel_size
        max_model_len=65536,                           # recipe value for a 5090; 262144 native
        max_num_seqs=8,
        gpu_memory_utilization=0.85,                   # recipe: headroom for the FlashInfer workspace
        enable_chunked_prefill=True,
        enable_prefix_caching=True,                    # default in 0.29; agents resend history

        # ── vLLM recipe for NVFP4 on an RTX 5090 ──
        quantization="modelopt_fp4",
        kv_cache_dtype="fp8",                          # the checkpoint ships KV scales
        block_size=128,
        moe_backend="marlin",
        attention_backend="flashinfer",
        attention_config={"use_trtllm_attention": True},

        limit_mm_per_prompt={"image": 4},
        # language_model_only=True,                    # text only: frees the vision tower

        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        # Thinking is on unless a request sends
        # chat_template_kwargs={"enable_thinking": False} (predict.py does).
        # Server default from Ray 2.58:
        # default_chat_template_kwargs={"enable_thinking": False},

        # MTP stays off: with prefix caching it garbles hybrid models
        # (vLLM #53912), and on 4-bit MoE builds it can fail to load (#63598,
        # fixed after 0.29.0).
    ),

    runtime_env=dict(env_vars={
        "VLLM_HAS_FLASHINFER_CUBIN": "1",              # vLLM recipe: prebuilt FlashInfer kernels
    }),
)


app = build_openai_app({"llm_configs": [llm_config]})
