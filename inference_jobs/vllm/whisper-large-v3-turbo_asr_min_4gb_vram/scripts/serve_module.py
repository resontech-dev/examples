"""Whisper large-v3-turbo via vLLM — speech-to-text endpoint, smallest rung.

    POST /v1/audio/transcriptions   (multipart file + model [+ language])

Same encoder as large-v3, decoder pruned 32 → 4 layers: 5–8× faster
decode at ~1.6 GB. MIT. EN / IT: use it. UK: measure WER against
large-v3 on your recordings first (turbo loses more on less common
languages).

⚠️ VALIDATE BEFORE LISTING: Whisper rides vLLM's transcription support
through the Ray Serve LLM router — confirm the route on the pinned image
(ray 2.55.1 + vllm 0.18.0) before selling. Plan B: predictor-mode job
with faster-whisper.

Sharing a card: to run this next to the Qwen3-8B agents bundle, copy
this LLMConfig into that serve_module's `llm_configs` list with
gpu_memory_utilization=0.15 and lower the chat fraction by the same
amount (model id "whisper" stays).

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="whisper",                      # clients pass this as model=
        model_source="openai/whisper-large-v3-turbo",  # official, ~1.6 GB fp16
    ),

    accelerator_type=None,                       # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,                 # parallel transcription streams
        max_queued_requests=64,
        graceful_shutdown_timeout_s=60,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # Whisper decodes ≤ 448 tokens per 30 s window — this is the
        # model's shape, not a squeeze. fp8 KV / prefix caching deliberately
        # NOT set (encoder-decoder support is version-sensitive; the win
        # on a 1.6 GB model is nil).
        max_model_len=448,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,             # alone on the card; 0.15 when sharing
        dtype="auto",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
