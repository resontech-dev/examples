"""MamayLM 12B v2.0 (official bf16) via vLLM — TP=2 across two 16 GB cards.

The documented INSAIT checkpoint (June 2026): Gemma 3 12B adapted to
Ukrainian, text + images, Ukrainian/English. bf16 is ~24 GB, so one
16 GB card cannot hold it; two cards in ONE machine shard it (TP=2).

    "assistant" → INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0    POST /v1/chat/completions

Same shape serves the other official 12B Ukrainian checkpoints — swap
`model_source` only:
    "lapa-llm/lapa-v0.1.3-instruct"                 Lapa 12B (Ukrainian tokenizer, 1.5× fewer tokens)
    "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"  previous MamayLM

Per-GPU budget at 0.92 × 16 = 14.7 GB: 12.2 GB weight shard + ~0.8 GB
activations/vision → ~1.7 GB KV per card (KV heads split by TP, so ~3.4 GB
pooled). Gemma 3 keeps full KV only on every 6th layer → 16k × 4 seqs fits.
On 2×24 GB raise max_model_len to 32768 and max_num_seqs to 8.

Gemma 3 engine notes: dtype bfloat16 only; never quantization="fp8"
(breaks the vision tower — INSAIT's warning); no tool/reasoning parsers;
fp8 KV left off. See ../README.md.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",
        model_source="INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0",
    ),

    accelerator_type=None,                               # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=2,                          # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                        # = yaml vllm.pipeline_parallel_size
        max_model_len=16384,                             # 32768 on 2×24 GB
        max_num_seqs=4,
        gpu_memory_utilization=0.92,
        enable_chunked_prefill=True,
        dtype="bfloat16",                                # Gemma 3: never fp16
        limit_mm_per_prompt={"image": 2},                # a page at 896 px = 256 tokens
    ),

    placement_group_config=dict(
        # One bundle per TP shard (auto-replicated 2×), both on ONE node —
        # TP over Ethernet does not work.
        bundle_per_worker={"GPU": 1, "CPU": 4},
        strategy="STRICT_PACK",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
