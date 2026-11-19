"""Voxtral Mini 4B Realtime via vLLM: streaming speech-to-text.

A natively streaming ASR model (Mistral, Apache-2.0, 2026-02): audio goes in
while it is spoken and text comes back after a fixed delay, 480 ms by
default (80 ms to 2.4 s, set by `transcription_delay_ms` in the
checkpoint's tekken.json). 13 languages: en, fr, es, de, ru, zh, ja, it,
pt, nl, ar, hi, ko. **Not Ukrainian.**

Route: vLLM's WebSocket realtime API.

    ws(s)://<endpoint>/v1/realtime
      -> session.created
      <- session.update {model}, input_audio_buffer.commit
      <- input_audio_buffer.append {audio: base64 PCM16, 16 kHz mono} ...
      <- input_audio_buffer.commit {final: true}
      -> transcription.delta {delta} ... transcription.done {text}

⚠️ VALIDATE BEFORE LISTING. vLLM ships the realtime endpoint since 0.18,
whose example serves this model with --enforce-eager (the recipe asks for
>= 0.20 with PIECEWISE CUDA graphs, which the pinned 0.29 meets). Unproven: whether Ray Serve LLM's
OpenAI router (build_openai_app) exposes a WebSocket route at all, and
whether the per-job nginx passes the Upgrade. Plan B on the same
deployment: POST /v1/audio/transcriptions with whole files.

Exports `app`; the platform points import_path at `serve_module:app`.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="voxtral",                                    # clients pass this as model=
        model_source="mistralai/Voxtral-Mini-4B-Realtime-2602",  # ~9 GB bf16
    ),

    accelerator_type=None,                       # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,                  # live sessions + a few file requests
        max_queued_requests=16,
        graceful_shutdown_timeout_s=60,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # Required: the tokenizer loads only through mistral_common.
        tokenizer_mode="mistral",
        # vLLM 0.18's realtime example serves this model eager; the recipe's
        # cudagraph_mode PIECEWISE wants vLLM >= 0.20; the pinned 0.29 meets it, try without this.
        enforce_eager=True,
        # One token is ~80 ms of audio: 45k tokens ≈ one hour per session.
        # The default (131072, ~3 h) does not fit next to the weights on 16 GB.
        max_model_len=45_000,
        max_num_seqs=4,                          # parallel live sessions
        gpu_memory_utilization=0.90,
        dtype="auto",
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
