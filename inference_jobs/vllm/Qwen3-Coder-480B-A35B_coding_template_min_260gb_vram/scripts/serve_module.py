"""⚠️ TEMPLATE — Qwen3-Coder-480B-A35B (AWQ) via vLLM, TP=4 frontier coding.

The frontier coding agent on-prem: 480B MoE, 35B active — the open
answer to Claude/GPT coding API tiers. This is a READY-TO-TEST template,
not a deployable job on today's fleet:

    * needs ONE machine with 4× 80 GB GPUs (H100 / A100-80G class)
    * every number below is napkin math (llmconfig §4), unvalidated
    * first boot downloads ~260 GB — pre-mirror to S3/Garage, always

When frontier hosts join the fleet: vet the checkpoint, run
`vllm bench serve`, tune max_model_len/max_num_seqs from measurements,
then drop the TEMPLATE banners.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="frontier-coder",
        # VERIFY before ANY deploy: community AWQ re-upload (~260 GB).
        # No official Qwen AWQ at this size; official FP8 (~490 GB) needs
        # 8×80 GB instead. Vet the repo, hash-check, re-run evals.
        model_source="QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ",
    ),

    # Frontier datacenter GPUs DO carry Ray accelerator labels (H100,
    # A100-80G…). If the fleet labels its nodes, pin here to keep this
    # replica off smaller cards; with an unlabeled fleet keep None.
    accelerator_type=None,

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=180,         # frontier generations run long
    ),

    engine_kwargs=dict(
        tensor_parallel_size=4,                  # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                # = yaml vllm.pipeline_parallel_size
        # Per-GPU budget (80 GB): 80×0.92 = 73.6 − ~65 weights-shard
        # − ~2 activations ≈ 6.6 GB KV/GPU → ~26 GB pooled ≈ enough for
        # 128k contexts × several agent sessions at fp8. ESTIMATES ONLY.
        max_model_len=131072,
        max_num_seqs=16,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,              # agent loops live and die by this
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.92,
        dtype="auto",

        # ── Tool calling (stable for the Qwen family) ──
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    ),

    placement_group_config=dict(
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_PACK",                  # TP never crosses machines
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
