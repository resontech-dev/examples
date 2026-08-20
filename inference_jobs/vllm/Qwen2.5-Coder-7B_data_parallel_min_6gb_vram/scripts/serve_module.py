"""Qwen2.5-Coder-7B-AWQ via vLLM — data parallel, 2 replicas.

Each replica is a full copy of the model on its own single-GPU worker
(12 GB is plenty: ~4.9 GB AWQ weights + ~5 GB KV pool). Two replicas =
2× throughput + survives a worker loss. Zero inter-worker traffic.

The OpenAI-compatible app is exported as `app`. The platform's generate.py
detects `engine: vllm_openai` in inference.yaml and emits a
serve_config.yaml whose import_path is `serve_module:app`.

Routes exposed once deployed (all OpenAI-compatible):
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below —
the scheduler reserves workers from the yaml, vLLM enforces from here.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="coder-7b",                     # client-facing model name — keep stable
        # Official Qwen AWQ 4-bit — ~4.9 GB weights, fits any 12 GB card
        # with a healthy KV pool.
        model_source="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        # Scale-up path: the 30B MoE coder needs pipeline parallelism on
        # this fleet — see the sibling example
        # Qwen3-Coder-30B-A3B…pipeline_parallel_min_20gb_vram.
    ),

    # Consumer GPUs (5070/4090/3090) carry NO Ray accelerator label —
    # any non-None value here leaves the replica PENDING forever.
    accelerator_type=None,

    deployment_config=dict(
        # DP width. Each replica lands on its own worker — keep
        # max_replicas ≤ cluster.num_workers or the extra replica sits
        # PENDING with no error.
        autoscaling_config=dict(min_replicas=2, max_replicas=2),
        max_ongoing_requests=32,         # ~2× max_num_seqs — batcher never starves
        max_queued_requests=64,          # fast 503 beats a silent queue
        graceful_shutdown_timeout_s=120, # let generations finish on redeploy
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,          # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,        # = yaml vllm.pipeline_parallel_size
        # KV budget on a 12 GB card: 12×0.88 = 10.6 − 4.9 weights − ~0.7
        # activations ≈ 5 GB. Qwen2.5-7B KV ≈ 28.5 KB/token at fp8 →
        # ~175k cached tokens. 16k × 16 seqs schedules comfortably; on
        # 16–24 GB cards raise max_model_len to 32768 (native ceiling).
        max_model_len=16384,
        max_num_seqs=16,
        kv_cache_dtype="fp8",            # halves KV — default in every template
        enable_prefix_caching=True,      # editors/agents resend identical prefixes
        enable_chunked_prefill=True,     # long prompts don't stall other streams
        gpu_memory_utilization=0.88,     # 0.90 if the cards don't drive a display
        dtype="auto",

        # ── Tool calling (agentic editors: Aider, Cline, Continue…) ──
        # Qwen chat/coder models speak hermes-style tool calls.
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",

        # No engine-level `api_key` on purpose — auth is edge-only (nginx
        # serve-proxy validates X-API-Key). vLLM 0.18's FrontendArgs.api_key
        # wants list[str]; passing a string crashes engine startup.
    ),
    # TP=1/PP=1 → no placement_group_config needed; one {"GPU": 1}
    # bundle per replica is inferred.
)


app = build_openai_app({"llm_configs": [llm_config]})
