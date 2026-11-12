"""Qwen3-ASR-1.7B via vLLM: speech-to-text endpoint, mid rung of the speech ladder.

    POST /v1/chat/completions         (audio as an `audio_url` content part; primary)
    POST /v1/audio/transcriptions     (multipart file + model; secondary, VALIDATE)

1.7B audio-language model from Alibaba (2026, Apache-2.0): a Qwen3 decoder
behind an audio encoder. Language identification and transcription for 30
languages plus 22 Chinese dialects, singing voice, songs with background
music. Card numbers (vendor): Open ASR leaderboard WER 4.94 against 7.44
for Whisper large-v3 (see the registry, docs/inference/models/speech.md).

Languages: zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr,
hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, hu, mk, ro. Ukrainian is NOT
in the list: keep the Whisper jobs for Ukrainian, use this one for English,
Italian, Polish, German.

Stack facts (vLLM recipe, Qwen/Qwen3-ASR usage guide, 2026-01-29): plain
`vllm serve Qwen/Qwen3-ASR-1.7B`, no extra flags; `vllm[audio]` extra for
the audio decoders; day-0 support landed in the vLLM nightlies of January
2026, so the pinned 0.18.0 is expected to carry the architecture. VERIFY
on first boot: an unknown-architecture error at engine start means the
class is missing from 0.18.0 and this job waits for the image re-pin.

Output format on the chat route: the model emits a language tag followed
by the transcript (the `qwen-asr` package parses it with
`parse_asr_output`); predict.py splits it leniently. The transcription
route returns plain text.

Sharing a card: copy this LLMConfig into a chat bundle's `llm_configs`
list at gpu_memory_utilization=0.20 and lower the chat fraction by the
same amount (model id "asr" stays). The audio encoder makes this one
heavier than Whisper turbo at the same share.

Exports `app`; the platform points import_path at `serve_module:app`.

The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="asr",                          # clients pass this as model=
        model_source="Qwen/Qwen3-ASR-1.7B",      # official, ~4 GB bf16, Apache-2.0
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
        # The recipe sets nothing here; these are our conservative picks.
        # Audio enters as encoder tokens plus the transcript as decoder
        # tokens: 8k covers several minutes of speech per request. The
        # card gives no hard per-request audio limit (the forced aligner
        # sibling is rated for 5 minutes); chunk longer files client-side
        # as with Whisper. VERIFY the usable minutes per 8k on first boot.
        max_model_len=8192,
        max_num_seqs=8,
        limit_mm_per_prompt={"audio": 1},        # one clip per request
        gpu_memory_utilization=0.85,             # alone on the card; 0.20 when sharing
        dtype="auto",
        # fp8 KV / prefix caching deliberately NOT set, as in the Whisper
        # jobs: multimodal encoder paths are version-sensitive and the win
        # on a 4 GB model with short transcripts is nil.
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
