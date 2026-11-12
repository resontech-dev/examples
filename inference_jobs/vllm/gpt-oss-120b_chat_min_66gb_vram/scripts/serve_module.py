"""gpt-oss-120b via vLLM: premium general assistant on one big card.

117B MoE with 5.1B active parameters, shipped natively in MXFP4 (~65 GB).
OpenAI-lineage assistant one level above the 20b: GPQA ~80, HLE ~15,
SWE-bench Verified ~62 (card numbers), the same harmony format, effort
knob and 128k window. Apache-2.0. The llms-catalog premium general tier
(`chat-gptoss-120b`).

Configured TP=1 for ONE card of 80 GB or more. Every number below is an
estimate from the memory-bandwidth rule and the card; the job has not
booted on real hardware yet. Measure before quoting.

Kernels: MXFP4 runs natively on Blackwell (RTX PRO 6000, B200) and
through Marlin kernels on Hopper / Ampere (H100, H200, A100) since vLLM
0.10.1, so the pinned 0.18.0 image covers every card listed.

Exports `app`; the platform detects `engine: vllm_openai` and points
serve_config's import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                    # client-facing name — keep stable
        # Official OpenAI release, natively MXFP4-quantized (~65 GB).
        model_source="openai/gpt-oss-120b",
    ),

    # Datacenter GPUs DO carry Ray accelerator labels (H100, A100-80G, H200).
    # If the fleet labels its nodes, pin here to keep this replica off
    # smaller cards; with an unlabeled fleet (and every consumer card) keep
    # None, or the replica stays PENDING forever.
    accelerator_type=None,

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,         # ~2× max_num_seqs, batcher never starves
        max_queued_requests=64,          # fast 503 beats a silent queue
        graceful_shutdown_timeout_s=180, # long reasoning answers finish on redeploy
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,          # = yaml vllm.tensor_parallel_size (2 / 4 for multi-card)
        pipeline_parallel_size=1,        # = yaml vllm.pipeline_parallel_size
        # KV budget, TP=1 (gpt-oss-120b KV ≈ 37 KB/token bf16, ≈ 18.5 KB at fp8):
        #   H100 80 GB:  80×0.92 = 73.6 − 65 weights − ~2 activations ≈ 6.6 GB → ~350k tokens
        #   PRO 6000 96 GB:                                            ≈ 21 GB  → ~1.1M tokens
        #   H200 141 GB:                                               ≈ 62 GB  → ~3.3M tokens
        # 64k shipped so an 80 GB card holds several full contexts; on 96 GB
        # or H200 raise max_model_len to 131072 (native) and max_num_seqs.
        max_model_len=65536,
        max_num_seqs=16,
        kv_cache_dtype="fp8",            # halves KV — default in every template
        enable_prefix_caching=True,      # chat clients resend history every turn
        enable_chunked_prefill=True,     # one long prompt never stalls the others
        gpu_memory_utilization=0.92,
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

    # TP=1 → no placement group needed. For the TP=2 / TP=4 shapes
    # uncomment (bundles are auto-replicated TP times; STRICT_PACK keeps
    # every shard on one machine; TP never crosses workers):
    # placement_group_config=dict(
    #     bundle_per_worker={"GPU": 1, "CPU": 4},
    #     strategy="STRICT_PACK",
    # ),
)


app = build_openai_app({"llm_configs": [llm_config]})
