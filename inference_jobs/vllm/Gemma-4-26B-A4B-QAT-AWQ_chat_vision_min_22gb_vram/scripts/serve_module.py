"""Gemma 4 26B A4B (QAT, AWQ INT4) via vLLM: text + image, tool calling, thinking. One 24 GB card.

BLOCKED on the pinned image: needs vLLM >= 0.19 (ray-llm image re-pin, reson_docs spec item J). Written ahead; nothing has booted.
This size: vLLM >= 0.25.0 per the vLLM recipe (Gemma 4 MoE; the RTX 5090 config there runs Red Hat's FP8 build).

Google's Gemma 4 26B A4B (2026), Apache-2.0: a Mixture-of-Experts model with
128 experts and about 4B parameters active per token. Text and images,
thinking mode, native tool calling, 140+ languages. Each token reads about
a sixth of the weight bytes the dense 31B reads, so decode is several times
faster per card at the same quality class (GPQA 82.3 vs 84.3 on the card).
All 26B stay in memory: it needs the memory of a big model and runs at the
speed of a small one.

    "assistant" -> cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4    POST /v1/chat/completions

Checkpoint: community AWQ INT4 of Google's QAT weights
(google/gemma-4-26B-A4B-it-qat-q4_0-unquantized), compressed-tensors, 17.2 GB,
vision tower unquantized. Google's own w4a16 line skips this model, and Red Hat
publishes only FP8 and NVFP4 for it. VERIFY the repo before selling.

Budget at 0.92 x 24 GB = 22.1 GB: 17.2 GB weights + ~1.5 GB activations
-> ~3.4 GB KV cache (estimate). Gemma 4 keeps full KV on 5 of 30 layers
(two shared K=V heads of 512 dims, ~10 to 20 KB per token) and a 1,024-token window on the rest (~0.21 GB per sequence), so
one full 32k sequence takes ~0.5 to 0.9 GB: about four full-length sequences at once.

Engine notes (vLLM recipe + model card, not booted here):
  * tool_call_parser and reasoning_parser are both "gemma4"; the recipe pairs
    them with vLLM's examples/tool_chat_template_gemma4.jinja, vendored next
    to this file (Apache-2.0, vLLM main, fetched 2026-11-12). VERIFY: replace
    it with the copy from the vLLM version in the new image.
  * Thinking is off unless a request sends chat_template_kwargs
    {"enable_thinking": true}. No server default here: Ray 2.55.1 breaks
    `default_chat_template_kwargs`; check the new Ray before using it.
  * Card sampling: temperature 1.0, top_p 0.95, top_k 64 (predict.py sends them).
  * Image budget per image: 70 / 140 / 280 / 560 / 1120 tokens, default 280;
    raise it with mm_processor_kwargs for OCR and small print.
  * dtype bfloat16 as for every Gemma; no on-the-fly fp8.

The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from pathlib import Path

from ray.serve.llm import LLMConfig, build_openai_app


# vLLM's Gemma 4 tool-call chat template, vendored in scripts/. VERIFY that the
# file reaches the replica next to this module (it ships with scripts/).
TOOL_CHAT_TEMPLATE = (Path(__file__).resolve().parent / "tool_chat_template_gemma4.jinja").read_text()


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                            # one id for every modality: keep stable
        model_source="cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4",       # VERIFY: community AWQ of Google's QAT weights
    ),

    accelerator_type=None,                               # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                          # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                        # = yaml vllm.pipeline_parallel_size
        max_model_len=32768,                             # 262144 in config.json; KV decides
        max_num_seqs=8,
        gpu_memory_utilization=0.92,
        enable_chunked_prefill=True,
        dtype="bfloat16",                                # Gemma: never fp16
        limit_mm_per_prompt={"image": 4},
        # mm_processor_kwargs={"max_soft_tokens": 560},  # image budget: 70/140/280/560/1120

        # Tool calling + thinking (vLLM recipe for Gemma 4).
        enable_auto_tool_choice=True,
        tool_call_parser="gemma4",
        chat_template=TOOL_CHAT_TEMPLATE,
        reasoning_parser="gemma4",

        # Add one at a time after the first boot, verify output each time:
        # kv_cache_dtype="fp8",                          # recipe: ~50% less KV
        # async_scheduling=True,                         # recipe: higher throughput
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
