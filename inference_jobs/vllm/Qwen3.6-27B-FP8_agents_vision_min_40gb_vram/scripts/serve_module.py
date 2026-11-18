"""Qwen3.6-27B (official FP8) via vLLM: text + vision assistant for agents.

NEEDS the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Prod runs
Ray 2.55.1 / vLLM 0.18.0 until the switch (reson_docs
docs/big_models_experiments/ray2.60.0-upgrade.md); on dev the head takes
RAY_LLM_IMAGE=resontechtdev/ray-llm:2.60.0-py312-cu130. Written 2026-11-18;
nothing has booted.

Successor of Qwen3.5-27B on the same dense 27B body: GPQA 87.8, HLE 24.0,
SWE-bench Verified 77.2 (Qwen3.5-27B: 85.5 / 24.3 / 72.4), text + images,
262k native window, Apache-2.0. Hybrid attention: 16 of 64 layers keep KV
(64 KB per token in bf16), the DeltaNet layers keep a fixed fp32 state
(~150 MB per sequence).

    "assistant" -> Qwen/Qwen3.6-27B-FP8    POST /v1/chat/completions (text, images, tools)

Checkpoint: Qwen's own FP8 (block quantization), 30.9 GB on disk with the
vision tower and the MTP head. No official 4-bit exists; for a 32 GB card
see "What to change" in README (community int4 or NVIDIA NVFP4).

Budget on one 48 GB card at 0.92 -> 44.2 GB: 30.9 GB weights + ~2.5 GB
activations, vision encoder and CUDA graphs -> ~10 GB KV, about 160k cached
tokens across all sequences. 64k x 8 sequences queue rather than crash.
On one 96 GB card (RTX PRO 6000): ~54 GB KV, 262k context x 16 sequences.

Engine notes (vLLM recipe + model card, vLLM 0.29.0 source):
  * qwen3_coder and qwen3_xml are one parser class in 0.29.0; the old
    "xml holds up better" advice no longer applies.
  * Prefix caching is on by default in 0.29.0; on hybrid models vLLM sets
    mamba_cache_mode="align" itself. Agents resend the whole history, so
    keep it on.
  * MTP and prefix caching together garble output on hybrid models
    (vLLM #53912, open). MTP stays off below; to try it, turn prefix
    caching off in the same edit.
  * FP8 KV cache: off. The FP8 checkpoint ships no KV scales.
  * VERIFY on the first boot: vLLM #55291 reports FP8 Qwen3.6 output
    collapsing into "!!!". Read one real answer (predict.py) before
    anything else. On RTX (sm_120) block FP8 runs through the DeepGEMM copy
    vendored in vLLM 0.29, untested by us there: if the output is garbage,
    first try runtime_env=dict(env_vars={"VLLM_USE_DEEP_GEMM": "0"}), then
    the community int4 checkpoint.

The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                     # one id for text AND images: keep stable
        model_source="Qwen/Qwen3.6-27B-FP8",      # official FP8, 30.9 GB
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
        max_model_len=65536,                      # 262144 native; 262144 on 96 GB
        max_num_seqs=8,                           # 16 on 96 GB
        gpu_memory_utilization=0.92,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,               # default in 0.29; agents resend history
        dtype="auto",

        limit_mm_per_prompt={"image": 4},         # a page is ~1-2.5k tokens at 1000-1500 px
        # language_model_only=True,               # text only: frees the vision tower

        # Tool calling + thinking (model card, vLLM recipe).
        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        # Thinking is on unless a request sends
        # chat_template_kwargs={"enable_thinking": False} (predict.py does).
        # {"preserve_thinking": True} keeps earlier thinking in the history,
        # useful for long agent loops. A server default works from Ray 2.58:
        # default_chat_template_kwargs={"enable_thinking": False},

        # MTP speculative decoding: faster decode, but only with prefix
        # caching OFF on this hybrid model (vLLM #53912). Measure both.
        # speculative_config={"method": "mtp", "num_speculative_tokens": 2},
        # If boot dies with a `causal_conv1d_update` assertion:
        # max_cudagraph_capture_size=256,
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
