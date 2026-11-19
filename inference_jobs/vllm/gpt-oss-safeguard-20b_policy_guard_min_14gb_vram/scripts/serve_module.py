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
        # KV budget: under vLLM 0.29 a 16 GB card keeps only a few hundred MB
        # for KV (see gpu_memory_utilization below). If boot says max_model_len
        # is larger than the KV cache holds, use 11264 there or a 24 GB card.
        # gpt-oss KV is light (~24.5 KB/token at fp8): a 24 GB card gives
        # ~7.8 GB ≈ 320k tokens.
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
        # Same architecture as gpt-oss-20b. Under vLLM 0.29 (Ray 2.60) on a
        # 16 GB RTX 5070 Ti (2026-11-19) gpt-oss-20b took 13.8 GiB of weights
        # and 0.46 GiB of activations, and since vLLM 0.21 CUDA graphs come
        # out of this budget: KV was -0.64 GiB at 0.92 and -0.03 GiB at 0.95
        # with graphs up to 8 tokens. 0.95 still leaves ~0.8 GiB for the CUDA
        # context; on a 16 GB card it must not drive a display.
        gpu_memory_utilization=0.95,
        # Hence no CUDA graphs (torch.compile stays on; decode is slower):
        # ~0.44 GiB of KV. On a 24 GB card drop this, set
        # max_cudagraph_capture_size=8 (unset, vLLM captures gpt-oss up to
        # 1024 tokens) and gpu_memory_utilization=0.90.
        compilation_config=dict(cudagraph_mode="NONE"),
        # FlashInfer (vLLM 0.29's pick on sm_120) allocates a 394 MiB
        # workspace (VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE) at the first
        # decode step, after the KV cache has taken the rest: OOM on 16 GB
        # (2026-11-19). Triton attention has no such buffer and supports the
        # gpt-oss sinks. Its fp8 KV needs SM 8.9+, so on RTX 30xx / A10G /
        # A100 also set kv_cache_dtype="auto".
        attention_config=dict(backend="TRITON_ATTN"),
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
