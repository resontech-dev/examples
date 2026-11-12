"""gpt-oss-safeguard-20b via vLLM: policy-as-prompt safety classifier.

OpenAI's open-weight safety reasoning model (October 2025, Apache-2.0).
Same architecture, MXFP4 checkpoint (~13 GB) and harmony format as
gpt-oss-20b, so the pinned image runs it unchanged. It does not answer
users: it reads a written policy from the system message, reasons about
the user message, and returns the verdict format the policy asks for.

    "policy-guard" → POST /v1/chat/completions
        system = your policy (+ "Reasoning: low|medium|high")
        user   = the content to classify

Why it exists next to Qwen3Guard: a guard with a fixed taxonomy cannot
enforce "no credentials in chat" or "no third-party personal data". This
one can, from the policy text, and explains which rule fired. The price
is latency: reasoning runs before the verdict, so a check takes seconds.
Use it in the background on flagged or sampled conversations, not inline
in front of 80 concurrent users.

Exports `app`; the platform detects `engine: vllm_openai` and points
serve_config's import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="policy-guard",                 # client-facing name, keep stable
        # Official OpenAI release, natively MXFP4-quantized (~13 GB).
        # 120b sibling: "openai/gpt-oss-safeguard-120b" (~65 GB), same
        # shape as ../gpt-oss-120b_chat_min_66gb_vram.
        model_source="openai/gpt-oss-safeguard-20b",
    ),

    # Consumer GPUs carry NO Ray accelerator label — non-None = PENDING forever.
    accelerator_type=None,

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,          # ~2× max_num_seqs, batcher never starves
        max_queued_requests=64,          # fast 503 beats a silent queue
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,          # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,        # = yaml vllm.pipeline_parallel_size
        # KV budget on a 16 GB card: 16×0.92 = 14.7 − 13 weights − ~0.8
        # activations ≈ 0.9 GB. gpt-oss KV is light (~24.5 KB/token at
        # fp8) → ~37k cached tokens: two full 16k checks in flight, more
        # via preemption. A 24 GB card gives ~7.8 GB ≈ 320k tokens.
        # 16k = a 10k-token policy (the guide's upper bound) + content +
        # room for the reasoning channel; the model's native window is 128k.
        max_model_len=16384,
        max_num_seqs=4,
        kv_cache_dtype="fp8",            # halves KV — default in every template
        # Prefix caching matters more here than in chat: the policy is the
        # same system message in every check, so after the first request
        # only the content to classify is prefilled. Ephemeral GPU memory,
        # never written to disk.
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.92,     # drop to 0.85 if the card drives a display
        dtype="auto",

        # No tool parser: this is a classifier, it never calls tools.
        # Reasoning effort is read from the system prompt ("Reasoning: low"
        # keeps a check in the low seconds; "high" for audits). The guide
        # says not to cap max_tokens tightly: the reasoning channel needs
        # room. predict.py uses 1024 and "Reasoning: low".
    ),
    # TP=1/PP=1 → no placement_group_config needed.
)


app = build_openai_app({"llm_configs": [llm_config]})
