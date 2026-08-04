"""gpt-oss-20b via vLLM — single-worker business chat endpoint.

21B MoE with 3.6B active parameters, shipped natively in MXFP4 (~13 GB).
OpenAI-lineage general assistant: strong reasoning with an adjustable
effort knob, robust alignment, 128k native context. The default business
assistant in the catalog (`chat-gptoss-20b`, priority v1).

Exports `app`; the platform detects `engine: vllm_openai` and points
serve_config's import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                    # client-facing name — keep stable
        # Official OpenAI release, natively MXFP4-quantized (~13 GB).
        model_source="openai/gpt-oss-20b",
    ),

    # Consumer GPUs carry NO Ray accelerator label — non-None = PENDING forever.
    accelerator_type=None,

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,         # ~2× max_num_seqs — batcher never starves
        max_queued_requests=64,          # fast 503 beats a silent queue
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,          # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,        # = yaml vllm.pipeline_parallel_size
        # KV budget on a 24 GB card: 24×0.90 = 21.6 − 13 weights − ~0.8
        # activations ≈ 7.8 GB. gpt-oss KV is light (~24.5 KB/token at
        # fp8) → ~320k cached tokens. 64k context × several seqs is
        # comfortable; the model's native window is 128k — raise
        # max_model_len to 131072 on this card if your clients need it.
        max_model_len=65536,
        max_num_seqs=16,
        kv_cache_dtype="fp8",            # halves KV — default in every template
        enable_prefix_caching=True,      # chat clients resend history every turn
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,     # drop to 0.85 if the card drives a display
        dtype="auto",

        # ── Tool calling — VERIFY before enabling ──
        # gpt-oss speaks its own `harmony` format; vLLM handles it natively,
        # but whether the pinned image wants an explicit parser flag must be
        # checked first (wrong parser name = crash at engine boot):
        #   docker run --rm <ray-llm-image> vllm serve --help | grep -A3 tool-call-parser
        # enable_auto_tool_choice=True,
        # tool_call_parser="openai",     # ← only if the help output lists it

        # Reasoning effort: gpt-oss reads it from the system prompt
        # ("Reasoning: low|medium|high") — a client-side knob, not an
        # engine kwarg. predict.py's system prompt shows the pattern.
    ),
    # TP=1/PP=1 → no placement_group_config needed.
)


app = build_openai_app({"llm_configs": [llm_config]})
