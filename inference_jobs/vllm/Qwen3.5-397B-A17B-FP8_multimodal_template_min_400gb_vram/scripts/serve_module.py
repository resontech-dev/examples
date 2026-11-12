"""⚠️ TEMPLATE: Qwen3.5-397B-A17B (official FP8) via vLLM, TP=8, text + image.

The first large model on the platform
(reson_docs/docs/inference/run_qwen_3.5-397B-A17B_plan.md): 397B MoE with
17B active per token, hybrid attention (Gated DeltaNet layers mixed with
full-attention layers, so only a fraction of the layers keep a growing KV
cache), native vision encoder, 262K context, Apache-2.0. The only top-tier
open model that is multimodal, Apache-licensed and fits a single 8-GPU node.

This is a READY-TO-TEST template, not a deployable job on today's fleet:

    * needs ONE worker with 8 datacenter GPUs on NVLink/NVSwitch:
      8× H200 (target, ~690 GB left for KV) or 8× H100 80 GB (fallback,
      ~200 GB left for KV); `nvidia-smi topo -m` must show NV# between
      every pair
    * every number below is napkin math (plan §3 and §7), unvalidated
    * the ~400 GB FP8 checkpoint must be PRE-STAGED on the node's local
      NVMe (plan §6 step 3, spec item B); a cold HF pull at first boot is
      not a plan

Order of work before this file runs: plan steps 1 to 4. Step 1 proves the
family (parsers, hybrid kernels, vision path) on a small Qwen3.5 member on
the pinned image; `../Qwen3.5-27B-AWQ_agents_vision_min_28gb_vram` is that
job. This file is the same shape with model_source and TP changed.

Model ids and parsers come from the model card and the vLLM recipe
(`vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --tensor-parallel-size 8
--max-model-len 262144 --reasoning-parser qwen3`, tool parser qwen3_coder).
Verify every argument name against `vllm serve --help` inside the image.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen3.5-397b",                 # the plan's client-facing id (§7)
        # Official FP8 build, ~400 GB (card: 403B params at FP8). Apache-2.0.
        # The HF id resolves from the node's model cache once spec item B
        # mounts it; until then it is a 400 GB cold pull. The plan's target
        # is the pre-staged local path (plan §6 step 3):
        #   model_source="/models/local/qwen3.5-397b-fp8",
        model_source="Qwen/Qwen3.5-397B-A17B-FP8",
        # Official GPTQ-Int4 (~220 GB) is the small-node variant, see the
        # module docstring in README "Variants". VERIFY the repo id
        # Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 and the moe_wna16 kernel path on
        # vLLM 0.18.0 before any deploy.
    ),

    # Datacenter nodes MAY carry Ray accelerator labels (H200, H100).
    # None until the node is in the console and its label is known;
    # a wrong non-None value leaves the replica PENDING forever.
    accelerator_type=None,                       # or "H200" once the node carries Ray's label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=64,
        max_queued_requests=128,
        health_check_period_s=30,
        health_check_timeout_s=600,              # a 400 GB load + CUDA graph capture takes minutes
        graceful_shutdown_timeout_s=300,         # multi-minute reasoning answers finish on redeploy
    ),

    engine_kwargs=dict(
        tensor_parallel_size=8,                  # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                # = yaml vllm.pipeline_parallel_size

        # Memory per GPU (ESTIMATES, plan §3): weights ≈ 50 GB per shard.
        #   8× H100 80 GB: 80×0.90 = 72 − 50 − ~3 activations/vision/graphs ≈ 19 GB KV per GPU
        #   8× H200 141 GB: 141×0.90 = 127 − 50 − ~3 ≈ 74 GB KV per GPU
        # Hybrid attention keeps KV only in the full-attention layers, so a
        # 262K session costs a fraction of a classic transformer's. VERIFY
        # bytes per token from config.json (full-attention layers × 2 ×
        # kv_heads × head_dim × 2 bytes) before quoting sessions per node.
        max_model_len=262144,                    # decision 2 default: 262K; 131072 if KV is short on 8× H100
        max_num_seqs=64,                         # tune from the benchmark (plan §8)
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,              # repeated system prompts and long chats (plan §7)
        enable_chunked_prefill=True,             # long prompts don't stall other users
        dtype="auto",

        # FP8 KV cache: test before enabling. On AWQ models it produced
        # garbage output (llmconfig.md §4); it would roughly double the number
        # of long sessions per node, so it gets its own quality check in §8.
        # kv_cache_dtype="fp8",

        # ── Multimodal ──
        limit_mm_per_prompt={"image": 8, "video": 0},   # video off until decision 3
        # vLLM recipe settings for multimodal throughput. VERIFY both names
        # exist in the pinned image's `vllm serve --help`; drop if absent.
        mm_encoder_tp_mode="data",
        mm_processor_cache_type="shm",

        # ── Reasoning and tool calling (Qwen3.5 family, from the card) ──
        # Thinking is on unless the request sends
        # chat_template_kwargs={"enable_thinking": False}, as predict.py does.
        # Don't set `default_chat_template_kwargs` here: Ray 2.55.1 turns it
        # into a Namespace and every chat request then fails with a TypeError
        # (reson_docs inference/vllm/known_issues.md).
        reasoning_parser="qwen3",
        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder",

        # ── Throughput variant (benchmark after TP=8 works, plan §7 note 1) ──
        # The vLLM recipe's throughput layout on 8× H200 is data-parallel 8
        # with expert parallelism: each GPU holds the attention weights and
        # one eighth of the experts. Through Ray Serve LLM that is a
        # data-parallel deployment, not a TP change; this flag alone with
        # TP=8 is the intermediate step to measure first:
        # enable_expert_parallel=True,

        # 1M context (YaRN past 262K) stays off until decision 2 and its own
        # benchmark: it changes quality and memory.
    ),

    placement_group_config=dict(
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_PACK",                  # all 8 shards on one node, or fail loudly
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
