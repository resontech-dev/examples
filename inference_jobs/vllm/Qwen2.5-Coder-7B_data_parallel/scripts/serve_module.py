"""Qwen3-Coder-30B-A3B via vLLM, pipeline-parallel across 2 workers.

Stage-0 shape from reson_docs/docs/inference/vllm/plan16-jul-2026.md:
one replica of a 30B MoE coder split into 2 pipeline stages, each stage
on its own single-GPU worker (2× RTX 5070, 12 GB each). ~9 GB
weights-stage per GPU + KV.

The OpenAI-compatible app is exported as `app`. The platform's generate.py
detects `engine: vllm_openai` in inference.yaml and emits a
serve_config.yaml whose import_path is `serve_module:app`.

Routes exposed once deployed (all OpenAI-compatible):
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models

How PP works here, in one line: worker A holds the first half of the
layers, worker B the second half; every token flows A → B with ONE
network hop per token — that's why this survives ordinary
Ethernet/WireGuard between workers, unlike tensor parallelism.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below —
the scheduler reserves workers from the yaml, vLLM enforces from here.
"""
import os

from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="team-coder",                             # client-facing model name
        # 7B dense AWQ — ~4.5 GB weights, fits ONE ~12-16 GB card with room for
        # KV. Matches TP=1/PP=1/num_workers=1 below (single worker, no pipeline).
        model_source="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        # SCALE UP to the 30B MoE (Qwen/Qwen3-Coder-30B-A3B-Instruct) only on much
        # bigger hardware: unquantized BF16 is ~56 GB of weights — it needs ~4×
        # 24 GB (TP) or an 80 GB card. It does NOT fit 1-2× 12-16 GB even with PP
        # (~28 GB/stage), and no official AWQ build of it exists. When you have the
        # hardware, raise pipeline_parallel_size + inference.yaml vllm/num_workers.
    ),

    # Consumer GPUs (5070/4090/3090) carry NO Ray accelerator label —
    # any non-None value here leaves the replica PENDING forever.
    accelerator_type=None,

    deployment_config=dict(
        # One replica spanning two workers. Bump replicas together with
        # cluster.num_workers — each replica consumes
        # pipeline_parallel_size workers (worker PAIRS here).
        autoscaling_config=dict(min_replicas=2, max_replicas=2),
        max_ongoing_requests=32,
        max_queued_requests=64,          # fast 503 beats a silent queue
        graceful_shutdown_timeout_s=120, # let generations finish on redeploy
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,          # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,        # = yaml vllm.pipeline_parallel_size
        max_model_len=16384,             # honest on 12 GB stages; raise on bigger cards
        max_num_seqs=16,                  # keeps the pipeline fed without drowning KV
        kv_cache_dtype="fp8",            # halves KV — Blackwell handles it natively
        enable_prefix_caching=True,      # agent tools resend identical prefixes
        enable_chunked_prefill=True,     # long prompts don't stall other streams
        gpu_memory_utilization=0.88,
        dtype="auto",

        # No engine-level `api_key` on purpose. The platform edge (nginx + the
        # per-job predict key) is the auth boundary — see llmconfig.md §3.3 — and
        # the platform does NOT inject API_KEY into replicas. vLLM's
        # FrontendArgs.api_key is also version-flaky (0.18.1 wants list[str], not a
        # string), so passing it crashes engine startup. Leave auth to the edge.
    ),

    placement_group_config=dict(
        # One bundle per PP stage (auto-replicated TP×PP = 2 times).
        bundle_per_worker={"GPU": 1, "CPU": 4},
        # SPREAD: put the two stages on two different workers — the
        # whole point of PP. (TP would want STRICT_PACK instead.)
        strategy="SPREAD",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
