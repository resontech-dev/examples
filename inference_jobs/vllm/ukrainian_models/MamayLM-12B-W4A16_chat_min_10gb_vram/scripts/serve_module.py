"""MamayLM 12B (Gemma 3, Ukrainian-adapted) int4 via vLLM — one 16 GB card.

MamayLM (INSAIT) is Gemma 3 continued-pretrained on Ukrainian: top of the
open Ukrainian LLM leaderboard (lang-uk) together with Lapa and Gemma.
This job serves a community int4 build so it fits a single consumer card:

    "assistant" → shportix/MamayLM-Gemma-3-12B-IT-W4A16    POST /v1/chat/completions

⚠️ CHECKPOINT — VERIFY before relying on it: no model card, base version
not stated (naming suggests v1.0), compressed-tensors W4A16. If it fails
to load or answers badly, swap `model_source` for one of:
    "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0"   official bf16, ~8.6 GB, vision — fits 16 GB
    "Yehor/lapa-v0.1.2-it-w4g128"                   Lapa 12B 4-bit (auto-round) — the sibling job
    "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0"  official bf16, ~24 GB — 2×16 GB TP=2 (sibling job)

Gemma 3 engine notes (apply to every job in this folder):
- dtype MUST be bfloat16: fp16 overflows in Gemma 3 and produces NaNs/garbage.
- Never pass quantization="fp8" on a Gemma 3 checkpoint — it silently
  quantizes the vision tower and breaks image inputs (INSAIT's warning).
- No tool-call parser for the Gemma family in vLLM; no reasoning parser.
- kv_cache_dtype="fp8" and sliding-window attention are an untested combo
  here — left at default. Prefix caching stays at vLLM's default (on).

Budget on 16 GB at 0.90 → 14.4 GB: ~8 GB weights (int4 LM + bf16 vision
tower) + ~0.8 GB activations → ~5 GB KV. Gemma 3 keeps a full KV only on
every 6th (global) layer, the rest is a 1k sliding window → 16k × 4 seqs
is comfortable.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                            # keep stable; clients never see the repo name
        model_source="shportix/MamayLM-Gemma-3-12B-IT-W4A16",   # VERIFY — see banner
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
        max_model_len=16384,                             # 128k native; raise on 24 GB
        max_num_seqs=4,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        dtype="bfloat16",                                # Gemma 3: never fp16
        # One image per request. If the engine fails at load on the vision
        # tower (quantized checkpoints sometimes drop it), delete this line
        # and expect text-only — or switch to an official checkpoint above.
        limit_mm_per_prompt={"image": 1},
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
