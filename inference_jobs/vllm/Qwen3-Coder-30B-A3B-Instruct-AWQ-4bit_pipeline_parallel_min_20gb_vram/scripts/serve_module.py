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
        # Community 4-bit AWQ quant of Qwen3-Coder-30B-A3B (compressed-tensors
        # format, vLLM-native). 18.1 GB weights → ~9 GB per PP stage — fits
        # 2× 12 GB cards. Verified 2026-07-31: repo exists, 550k+ downloads.
        # (There is NO official Qwen-org AWQ of this model; the official
        # alternatives don't fit 12 GB stages: BF16 ~61 GB → 30 GB/stage,
        # official FP8 ~31 GB → 15.5 GB/stage.)
        model_source="cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
    ),

    # Consumer GPUs (5070/4090/3090) carry NO Ray accelerator label —
    # any non-None value here leaves the replica PENDING forever.
    accelerator_type=None,

    deployment_config=dict(
        # One replica spanning two workers. Bump replicas together with
        # cluster.num_workers — each replica consumes
        # pipeline_parallel_size workers (worker PAIRS here).
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,         # ~2× max_num_seqs — batcher never starves
        max_queued_requests=64,          # fast 503 beats a silent queue
        graceful_shutdown_timeout_s=120, # let generations finish on redeploy
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,          # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=2,        # = yaml vllm.pipeline_parallel_size
        # KV budget per 12 GB stage: 12×0.88 = 10.56 GB − 9.05 GB weights
        # − ~0.6 GB activations/graphs ≈ 0.9 GB for KV. At fp8 that's
        # ~24.5 KB/token/stage → ~37k tokens capacity. 16k context boots
        # with headroom (one full seq needs 0.4 GB); 32k would need
        # 0.8 GB just for one sequence — right at the cliff, don't.
        # Raise to 32k only on 16 GB+ cards.
        max_model_len=16384,
        max_num_seqs=8,                  # ~2 full-context seqs guaranteed, more via preemption
        kv_cache_dtype="fp8",            # halves KV — Blackwell handles it natively
        enable_prefix_caching=True,      # agent tools resend identical prefixes
        enable_chunked_prefill=True,     # long prompts don't stall other streams
        gpu_memory_utilization=0.88,     # 0.90 if the cards don't drive a display
        dtype="auto",

        # ── Tool calling (agentic editors: Aider, Cline, Continue…) ──
        # Qwen chat/coder models speak hermes-style tool calls; this is
        # what turns the endpoint into a usable agent backend.
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",

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
