"""Nemotron 3.5 Content Safety via vLLM: multimodal guard endpoint.

    "guard" → nvidia/Nemotron-3.5-Content-Safety   POST /v1/chat/completions

A 4B guard (Gemma-3-4B-it base, LoRA merged by NVIDIA, June 2026). It
moderates a user prompt, an optional image and an optional assistant
response, either against NVIDIA's standard safety taxonomy or against a
policy you write in plain text, with a reasoning trace on request. The
verdict is plain text:

    User Safety: safe | unsafe
    Response Safety: safe | unsafe        (only when a response was given)
    Safety Categories: A, B              (only when unsafe)

Everything is selected per request through `chat_template_kwargs`
(predict.py shows the exact keys from the model card):
    request_categories="/categories"   always
    enable_thinking=False|True          reasoning trace in <think>…</think>
    custom_policy="…"                   your policy instead of the taxonomy

Do NOT put these in `default_chat_template_kwargs` here: Ray 2.55.1 turns
that into a Namespace and every chat request fails with a TypeError
(reson_docs inference/vllm/known_issues.md).

Memory (estimate, 16 GB card): 16×0.85 = 13.6 − 8.6 weights − ~0.9
vision tower / activations ≈ 4 GB KV. Gemma 3 keeps full KV only on its
global-attention layers when vLLM's hybrid allocator is active, so 8k ×
8 seqs is comfortable; without it ~140 KB/token bf16 still gives ~3 full
8k sequences. On a 12 GB card the KV pool shrinks to ~0.7 GB: lower
max_num_seqs to 2.

Sidecar: to run this next to a chat engine, copy this LLMConfig into
that serve_module's list with gpu_memory_utilization=0.30 (24 GB card),
kv_cache_memory_bytes=int(0.5 * 2**30), and the `card_share(0.30)` helper
from ../Qwen3-8B-AWQ_guarded_rag_chat_min_22gb_vram/scripts/serve_module.py;
lower the chat fraction by the same amount. The model id "guard" matches
the guarded bundle, so predict scripts written for Qwen3Guard only need
the verdict parser swapped.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="guard",                        # client-facing name: keep stable
        # Official NVIDIA checkpoint, bf16, ~8.6 GB. Gemma3ForConditionalGeneration
        # is a vLLM-native architecture: no trust_remote_code.
        # ⚠️ OpenMDW-1.1 + Gemma Terms of Use + Gemma Prohibited Use Policy.
        model_source="nvidia/Nemotron-3.5-Content-Safety",
    ),

    # Consumer GPUs carry NO Ray accelerator label: non-None = PENDING forever.
    accelerator_type=None,

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,                 # verdicts are short: high turnover
        max_queued_requests=64,                  # fast 503 beats a silent queue
        graceful_shutdown_timeout_s=60,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                  # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                # = yaml vllm.pipeline_parallel_size
        # The card allows 128K; moderation inputs are a prompt, a response and
        # at most one image (~260 tokens for Gemma 3). 8k also bounds the
        # reasoning trace: predict.py caps max_tokens at 600 in think mode.
        max_model_len=8192,
        max_num_seqs=8,
        gpu_memory_utilization=0.85,             # alone on the card; 0.30 as a sidecar
        limit_mm_per_prompt={"image": 1},        # the model is defined for one image
        dtype="auto",

        # Deliberately NOT set on this Gemma 3 model (sliding-window layers):
        # kv_cache_dtype="fp8", enable_prefix_caching=True. Add one at a time
        # and re-run predict.py; both are wins only on a shared card.

        # No reasoning_parser: the model wraps its trace in <think>…</think>
        # inside `content` and predict.py strips it. VERIFY on the pinned image
        # whether reasoning_parser="deepseek_r1" (generic <think> tags) moves
        # the trace into the `reasoning` field; it is optional either way.

        # No tool parser: this is a classifier, not an agent.
    ),
    # TP=1/PP=1 → no placement_group_config needed.
)


app = build_openai_app({"llm_configs": [llm_config]})
