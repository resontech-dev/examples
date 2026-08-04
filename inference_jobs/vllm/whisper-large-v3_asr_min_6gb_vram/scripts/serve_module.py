"""Whisper large-v3 via vLLM — speech-to-text endpoint.

⚠️ VALIDATE BEFORE LISTING (llms-catalog §3.8): Whisper rides vLLM's
transcription support and the OpenAI audio route:

    POST /v1/audio/transcriptions   (multipart file + model)

Confirm the pinned ray-llm image (ray 2.55.1 + vllm 0.18.0) exposes the
route end-to-end before selling. If it doesn't, the fallback is a
predictor-mode job (faster-whisper inside predict()) — same platform
contract, different folder.

Economics: one 12 GB worker transcribes 10–30 parallel streams;
breakeven vs the OpenAI Whisper API (~$0.36/audio-hour) at a few hundred
audio-hours/month — and recordings never leave the customer.

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="whisper",                      # clients pass this as model=
        model_source="openai/whisper-large-v3",  # official checkpoint, ~3 GB fp16
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
        # Whisper decodes ≤ 448 tokens per 30 s audio window — the KV
        # levers that dominate text LLMs barely matter here. Deliberately
        # NOT set: kv_cache_dtype fp8 / prefix caching (encoder-decoder
        # support for both is version-sensitive; the defaults are safe
        # and the VRAM win would be negligible on a 3 GB model).
        max_model_len=448,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        dtype="auto",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
