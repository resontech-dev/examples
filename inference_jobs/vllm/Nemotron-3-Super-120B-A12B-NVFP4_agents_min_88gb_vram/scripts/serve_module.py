"""Nemotron 3 Super 120B-A12B (NVIDIA NVFP4) via vLLM: text reasoning + agents on one 96 GB card.

NEEDS the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+) and a
Blackwell card for NVFP4. Prod runs Ray 2.55.1 / vLLM 0.18.0 until the
switch (reson_docs docs/big_models_experiments/ray2.60.0-upgrade.md). Written
2026-11-18; nothing has booted.

NVIDIA's open 120B mixture-of-experts, 12B active per token, hybrid
Mamba-2 + MoE + attention (88 layers: 40 Mamba-2, 40 MoE, 8 attention),
1M context (256k default), text only. GPQA 79.2, HLE 18.3, SWE-bench 60.5
(OpenHands) per the card. Licence: NVIDIA Nemotron Open Model License.

    "assistant" -> nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4    POST /v1/chat/completions

Checkpoint: NVIDIA's NVFP4 build, 80.3 GB on disk, quantization detected
from the checkpoint. The vLLM recipe verifies it on ONE RTX PRO 6000
Blackwell (sm_120) at TP=1 with vLLM >= 0.24; it is not B200-only.

Budget at 0.92 x 96 GB = 88.3 GB: 80.3 GB weights + activations -> ~6-9 GB
for KV and Mamba state. Only 8 attention layers keep KV: ~4 KB per token in
fp8. Each sequence also holds a fixed fp32 Mamba state of ~170 MB. That is
why the recipe runs 262k x 8 sequences on one card.

Engine flags are the RTX PRO 6000 recipe, with two changes:
  * MTP off. The recipe pairs MTP with prefix caching, and that pair
    garbles output on Mamba hybrids in 0.29.0 (vLLM #53912). Agents gain
    more from prefix caching; MTP is one commented line below.
  * MAX_JOBS=4: the first boot compiles kernels and can use up all host
    RAM on a PRO 6000 (vLLM #54237). The compiled cache persists.

Per-request switches (chat_template_kwargs): enable_thinking (default
True), low_effort, force_nonempty_content (True for coding agents: the
answer always lands in `content`), truncate_history_thinking (default
True). Card sampling: temperature 1.0, top_p 0.95.

The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",
        model_source="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",   # 80.3 GB, Blackwell
    ),

    accelerator_type=None,                       # workers carry no Ray accelerator label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        health_check_timeout_s=600,              # an 80 GB load plus the first kernel compile
        graceful_shutdown_timeout_s=300,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                  # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                # = yaml vllm.pipeline_parallel_size
        max_model_len=262144,                    # recipe; up to 1048576 with VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        max_num_seqs=8,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.92,
        enable_chunked_prefill=True,
        kv_cache_dtype="fp8",                    # only 8 attention layers keep KV
        mamba_ssm_cache_dtype="float32",         # vLLM: only fp32 Mamba state is known accurate
        mamba_cache_mode="align",
        enable_prefix_caching=True,
        max_cudagraph_capture_size=128,          # model card

        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder",          # the card's tool format
        reasoning_parser="nemotron_v3",          # in 0.29.0 it already does what the super_v3 plugin did

        # MTP: faster decode, but only with enable_prefix_caching=False here (vLLM #53912).
        # speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    ),

    runtime_env=dict(env_vars={
        "VLLM_FLOAT32_MATMUL_PRECISION": "high",  # vLLM recipe
        "MAX_JOBS": "4",                          # first-boot kernel compile, vLLM #54237
    }),
)


app = build_openai_app({"llm_configs": [llm_config]})
