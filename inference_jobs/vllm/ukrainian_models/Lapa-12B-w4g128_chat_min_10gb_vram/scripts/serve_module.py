"""Lapa LLM 12B (Gemma 3, Ukrainian tokenizer) 4-bit via vLLM — one 16 GB card.

Lapa (lang-uk, fully open training data) is Gemma 3 12B adapted to
Ukrainian with a re-cut vocabulary: Ukrainian text costs ~1.5× fewer
tokens than on Gemma or MamayLM, so the same card serves more users and
longer documents. Instruct version v0.1.3 is bf16 (~24 GB); this job
serves a community 4-bit build of v0.1.2 (auto-round, w4 g128):

    "assistant" → Yehor/lapa-v0.1.2-it-w4g128      POST /v1/chat/completions

⚠️ CHECKPOINT — VERIFY: community quantization of v0.1.2 (v0.1.3 differs
only by alignment fixes). auto-round checkpoints load through vLLM's
GPTQ/AWQ kernels; if boot fails on the quantization config, use the
official bf16 "lapa-llm/lapa-v0.1.3-instruct" in the TP=2 sibling job
(2×16 GB) or on one 24 GB card.

Gemma 3 engine notes: dtype bfloat16 only; never quantization="fp8";
no tool/reasoning parsers; fp8 KV left off. See ../README.md.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",
        model_source="Yehor/lapa-v0.1.2-it-w4g128",     # VERIFY — see banner
    ),

    accelerator_type=None,                               # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=16384,                             # 128k native
        max_num_seqs=4,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        dtype="bfloat16",                                # Gemma 3: never fp16
        limit_mm_per_prompt={"image": 1},                # delete if the vision tower fails to load
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
