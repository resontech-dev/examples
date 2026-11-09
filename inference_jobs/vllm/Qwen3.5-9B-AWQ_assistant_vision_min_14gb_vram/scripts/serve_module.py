"""Qwen3.5-9B AWQ via vLLM — text + vision assistant on one 16 GB card.

Rung 2 of the small-over-big ladder. One engine, one model id, two jobs
at once:

    "assistant" → text tasks (classification, Q&A, drafting, checks)
                  AND image input (image content parts on the same
                  /v1/chat/completions route)

Qwen3.5 is a hybrid (Gated DeltaNet ×3 : full attention ×1): only 8 of
32 layers keep a KV cache, so long dialogs and multi-page requests are
cheap. Native window 262k; we cap at 16k on 16 GB.

⚠️ CHECKPOINT — VERIFY before selling. There is no official 4-bit
Qwen3.5-9B; `QuantTrio/Qwen3.5-9B-AWQ` is a data-free community AWQ
(~12 GiB: vision tower and linear-attention layers stay bf16, Apache-2.0,
needs vLLM ≥ 0.16 → fine on the pinned 0.18). Alternatives, same file:
    "axteam/Qwen3.5-9B-AWQ-4bit"       W4A16 compressed-tensors, vision bf16
    "RedHatAI/Qwen3.5-9B-FP8-dynamic"  14 GB on disk → needs a 24 GB card
    "Qwen/Qwen3.5-9B"                  official bf16, 19 GB → 24 GB card
    "Qwen/Qwen3.5-4B"                  official bf16, ~8 GB — fits 16 GB with
                                       no quantization risk; weaker model

Budget on 16 GB at gpu_memory_utilization 0.92 → 14.7 GB: ~12 GiB weights
+ ~1 GB activations/vision → ~1.5 GB for KV + Mamba state. Hybrid KV is
~16 KB/token bf16 → ~60k cached tokens; 16k × 4 seqs is comfortable.

Sidecars (embed / rerank from the Qwen3-8B bundle) do NOT fit next to
this on 16 GB. On 24 GB: assistant 0.68 + embed 0.085 + rerank 0.085.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                    # one id for text AND images — keep stable
        model_source="QuantTrio/Qwen3.5-9B-AWQ", # VERIFY — community AWQ, see banner
    ),

    accelerator_type=None,                       # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=16384,                     # 262144 native; raise on 24 GB+
        max_num_seqs=4,
        gpu_memory_utilization=0.92,             # single engine on the card — take it all
        enable_chunked_prefill=True,
        dtype="auto",

        # Images per request. A page at 1 000–1 500 px ≈ 1–2.5k tokens;
        # 2 pages + prompt + JSON fit 16k. Raise on 24 GB.
        limit_mm_per_prompt={"image": 2},
        # Text-only deployments: drop the vision tower (~1 GB) and its
        # processor. Use it if the engine OOMs at boot on 16 GB.
        # language_model_only=True,

        # ── Tool calling (Qwen3.5) ──
        # Official recipe: qwen3_coder. Community reports qwen3_xml holds
        # up better on long agentic runs with complex schemas — VERIFY on
        # your scenarios (wrong parser name = crash at boot).
        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder",

        # ── Thinking ──
        # Hybrid-thinking model: thinks unless the request sends
        # chat_template_kwargs={"enable_thinking": False}, as predict.py does.
        # Don't set `default_chat_template_kwargs` here: Ray 2.55.1 turns it into a
        # Namespace and every chat request then fails with a TypeError
        # (reson_docs inference/vllm/known_issues.md).
        reasoning_parser="qwen3",

        # Deliberately NOT set on this hybrid model (add one at a time,
        # verify output): kv_cache_dtype="fp8", enable_prefix_caching=True
        # (prefix caching over the Mamba cache is experimental in vLLM 0.18).
        # If boot dies with a `causal_conv1d_update` assertion:
        # max_cudagraph_capture_size=256,
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
