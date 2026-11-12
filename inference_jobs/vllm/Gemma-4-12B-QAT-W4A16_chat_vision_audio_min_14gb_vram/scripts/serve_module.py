"""Gemma 4 12B (QAT W4A16) via vLLM: text + image + audio, tool calling, thinking. One 16 GB card.

BLOCKED on the pinned image: needs vLLM >= 0.19 (ray-llm image re-pin, reson_docs spec item J). Written ahead; nothing has booted.
This size: vLLM >= 0.23.0 per the vLLM recipe (the encoder-free 12B Unified class; the recipe names the `vllm/vllm-openai:gemma4-unified` image).

Google's Gemma 4 12B Unified (2026), Apache-2.0: an encoder-free model that
projects image patches and audio straight into the decoder. Text, images,
audio up to 30 s, thinking mode, native tool calling, 140+ languages, in
Google's own 4-bit QAT build for one 16 GB card. The direct successor to
Gemma 3 12B (the base of MamayLM and Lapa).

    "assistant" -> google/gemma-4-12B-it-qat-w4a16-ct    POST /v1/chat/completions

Checkpoint: Google's official QAT checkpoint, w4a16 in compressed-tensors format
for vLLM (quantization-aware training keeps quality close to bf16). Vision and
audio projections stay unquantized.

Budget at 0.92 x 16 GB = 14.7 GB: 10.3 GB weights + ~1.2 GB activations
-> ~3.2 GB KV cache (estimate). Gemma 4 keeps full KV on 8 of 48 layers
(one shared K=V head of 512 dims, ~8 to 16 KB per token) and a 1,024-token window on the rest (~0.33 GB per sequence), so
one full 32k sequence takes ~0.6 to 0.9 GB: three to five full-length sequences at once.

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
        model_source="google/gemma-4-12B-it-qat-w4a16-ct",       # official QAT, compressed-tensors
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
        limit_mm_per_prompt={"image": 4, "audio": 1},
        # mm_processor_kwargs={"max_soft_tokens": 560},  # image budget: 70/140/280/560/1120

        # Tool calling + thinking (vLLM recipe for Gemma 4).
        enable_auto_tool_choice=True,
        tool_call_parser="gemma4",
        chat_template=TOOL_CHAT_TEMPLATE,
        reasoning_parser="gemma4",

        # Add one at a time after the first boot, verify output each time:
        # kv_cache_dtype="fp8",                          # recipe: ~50% less KV
        # async_scheduling=True,                         # recipe: higher throughput
        # speculative_config={"method": "mtp", "model": "google/gemma-4-12B-it-assistant",
        #                     "num_speculative_tokens": 4},  # VERIFY: recipe says nightly only
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
