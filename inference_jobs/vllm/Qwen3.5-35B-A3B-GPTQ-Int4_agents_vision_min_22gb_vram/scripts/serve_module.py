"""Qwen3.5-35B-A3B GPTQ-Int4 via vLLM: fast MoE text + vision assistant.

35B mixture-of-experts with 3B active parameters per token. Same contract
as the dense 27B job: one model id for text, images and tool calls,
hybrid attention, native 262k window. Decode is memory-bandwidth-bound and
reads only the active parameters, so this model generates roughly 3x
faster per card than the 27B at a similar quality tier. Int4 weights
~20 GB.

VERIFY, CHECKPOINT: "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" is the official
checkpoint named in the vLLM recipe (recipes.vllm.ai/Qwen/Qwen3.5-35B-A3B).
Confirm the repo exists before the first submit. Alternatives:
    "Qwen/Qwen3.5-35B-A3B-FP8"             official FP8, ~36 GB: 48 GB card or 2x24 GB TP=2
    "liming22/Qwen3.5-35B-A3B-GPTQ-Int4"   community GPTQ of the same shape; vet the repo

VERIFY, KERNEL: the Int4 expert layers run through vLLM's `moe_wna16`
fused-MoE path. On the pinned image (vLLM 0.18.0) that path is unverified:
the dense 27B and 9B jobs prove the Qwen3.5 hybrid architecture, not the
quantized MoE kernel. If boot fails with an unsupported-quantization error
for FusedMoE, switch model_source to the FP8 checkpoint and the card size
with it. This job is the cheap test: the same kernel question decides the
Qwen3.5-122B-A10B GPTQ-Int4 job, so one boot here answers both.

Budget on a 24 GB card at 0.92 -> 22.1 GB: ~20 GB weights + ~1.2 GB
activations/vision -> ~0.8 GB KV. Hybrid attention keeps KV only in the
attention layers, so that is about 16k context x 2 sequences. Thin, but
it boots. A 32 GB card (RTX 5090 / PRO 4500) at 0.90 leaves ~9 GB KV:
32k x 8 seqs is comfortable there.

2x16 GB (our RTX 5070 Ti bench) instead: yaml vllm.tensor_parallel_size: 2,
cluster.gpus_per_worker: 2, and tensor_parallel_size=2 below. ~10 GB of
weights and ~3 GB KV per card.

Throughput (estimate, nothing measured yet): on a 5090 roughly 3x the
27B's decode rate, 400-600 generated tok/s aggregate at 3k-token prompts.
Measure before quoting.

The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                     # one id for text AND images; keep stable
        model_source="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",  # VERIFY repo name, see banner
    ),

    accelerator_type=None,                        # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                   # 2 for the 2x16 GB shape (= yaml)
        pipeline_parallel_size=1,
        max_model_len=32768,                      # 262144 native; drop to 16384 if a 24 GB card refuses to boot
        max_num_seqs=8,
        gpu_memory_utilization=0.92,              # 0.90 on a 32 GB card
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

        # Deliberately NOT set on this hybrid model (add one at a time,
        # verify output): kv_cache_dtype="fp8", enable_prefix_caching=True.
        # If boot dies with a `causal_conv1d_update` assertion:
        # max_cudagraph_capture_size=256,
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
