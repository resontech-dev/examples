"""Gemma 3 27B (W4A16) via vLLM: text + vision, one 32 GB card.

Google's Gemma 3 27B instruct, text and images, 140+ languages, as an INT4 build
for one 32 GB card. MamayLM v2.0 27B (INSAIT) is this model continued-pretrained
on Ukrainian; this job is its base and the baseline to compare it against.

    "assistant" -> RedHatAI/gemma-3-27b-it-quantized.w4a16    POST /v1/chat/completions (text + images)

Checkpoint: Red Hat AI's INT4 weight-only quantization (llm-compressor,
compressed-tensors W4A16, vLLM-native). The vision tower, the multimodal
projector and the embeddings stay bf16, so images behave like the original.
Variant: Red Hat's pre-quantized `RedHatAI/gemma-3-27b-it-FP8-dynamic` (~29 GB,
vision tower kept bf16) on one 48 GB card. That is not the forbidden on-the-fly
fp8: the checkpoint itself is FP8, the vision tower is untouched.

Budget at 0.92 x 32 GB = 29.4 GB: 19.7 GB weights + ~1.5 GB
activations and vision profiling -> ~8 GB KV cache. Gemma 3 keeps full KV
only on every 6th layer (10 of 62, 80 KB per token in bf16) and a
1,024-token window on the rest (~0.44 GB per sequence), so one full 16k
sequence takes ~1.75 GB: 4 full-length sequences at once, more when
chats are short.

VERIFY: Red Hat's tokenizer.json sets a right-side truncation default of
8,192 tokens (HF discussion "Tokenizer limited to 8K context"). Transformers
turns backend truncation off unless a caller asks for it, so vLLM should
accept 16k prompts; `python predict.py --long` checks it (expects the
answer 4719 and more than 8,192 prompt tokens).

Gemma 3 engine notes (../ukrainian_models/README.md): dtype bfloat16 only;
never quantization="fp8" on the fly (breaks the vision tower); no tool or
reasoning parser in vLLM for Gemma 3; fp8 KV left off.

The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                            # one id for text and images: keep stable
        model_source="RedHatAI/gemma-3-27b-it-quantized.w4a16",
    ),

    accelerator_type=None,                               # consumer cards have no Ray label

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                          # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                        # = yaml vllm.pipeline_parallel_size
        max_model_len=16384,                             # 131072 native (config); KV decides
        max_num_seqs=8,
        gpu_memory_utilization=0.92,
        enable_chunked_prefill=True,
        dtype="bfloat16",                                # Gemma: never fp16
        limit_mm_per_prompt={"image": 4},              # 896 px per image = 256 tokens
    ),
)


app = build_openai_app({"llm_configs": [llm_config]})
