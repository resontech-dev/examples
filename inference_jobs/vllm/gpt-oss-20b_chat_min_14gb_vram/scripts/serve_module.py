"""gpt-oss-20b via vLLM — single-worker business chat endpoint.

21B MoE with 3.6B active parameters, shipped natively in MXFP4 (~13 GB).
OpenAI-lineage general assistant: strong reasoning with an adjustable
effort knob, robust alignment, 128k native context. A solid default
business assistant.

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
        max_ongoing_requests=8,         # ~2× max_num_seqs — batcher never starves
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
        max_model_len=16384,
        max_num_seqs=4,
        kv_cache_dtype="fp8",            # halves KV — default in every template
        enable_prefix_caching=True,      # chat clients resend history every turn
        enable_chunked_prefill=True,
        # vLLM 0.29 (Ray 2.60) on a 16 GB RTX 5070 Ti, 2026-11-19: weights
        # 13.8 GiB (Marlin MXFP4 on sm_120), activations 0.46 GiB, and since
        # vLLM 0.21 CUDA graphs are paid out of this budget too. KV left:
        #   0.92, graphs up to 256 tokens (0.61 GiB) -> -0.64 GiB
        #   0.95, graphs up to 8 tokens (0.47 GiB)   -> -0.03 GiB
        # 0.95 still leaves ~0.8 GiB for the CUDA context; on a 16 GB card it
        # must not drive a display.
        gpu_memory_utilization=0.95,
        # Hence no CUDA graphs, so the job also fits the 16 GB cards its
        # min_vram_gb_per_worker lets the platform pick (torch.compile stays
        # on; decode is slower). On a 24 GB card drop this, set
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

        # ── Tool calling ──
        # gpt-oss writes calls in its `harmony` format; the `openai` parser
        # turns them into `tool_calls`. Present in vLLM 0.18.0 and 0.29.0
        # (checked in source, 2026-11-18). On 0.29 a request with tools runs
        # under a strict tool-call grammar by default.
        enable_auto_tool_choice=True,
        tool_call_parser="openai",

        # Reasoning effort: gpt-oss reads it from the system prompt
        # ("Reasoning: low|medium|high") — a client-side knob, not an
        # engine kwarg. predict.py's system prompt shows the pattern.
    ),
    # TP=1/PP=1 → no placement_group_config needed.
)


app = build_openai_app({"llm_configs": [llm_config]})
