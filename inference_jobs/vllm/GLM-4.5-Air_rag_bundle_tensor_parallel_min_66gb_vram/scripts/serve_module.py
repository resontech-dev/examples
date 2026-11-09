"""GLM-4.5-Air RAG bundle — TP=4 chat + embed + rerank on one 4-GPU worker.

The premium private-RAG stack: Sonnet-4-class agentic chat plus the RAG
sidecars, one deployment, one machine:

    "assistant" → GLM-4.5-Air AWQ (TP=4)   POST /v1/chat/completions
    "embed"     → Qwen3-Embedding-0.6B     POST /v1/embeddings
    "rerank"    → Qwen3-Reranker-0.6B      POST /v1/score

Memory model: the chat engine takes its share on EVERY one of the 4 GPUs
(weights shard evenly), so it must not ask Ray for whole GPUs: the sidecars
would then wait forever. Each 0.6B sidecar is TP=1 and lands on ONE of those
GPUs — placement does NOT guarantee the two sidecars spread, so budget for BOTH
landing on the same GPU. The share is Ray's accounting and a free-VRAM check at
boot, so real use differs (each engine also takes ~0.85 GB of CUDA context):

            share   real use
    chat    0.70    16.8 GB/GPU (14.8 weights-shard + 1.2 GB KV/GPU ≈ 54k tokens)
    sidecar 0.06     2.4 GB each (1.1 weights + 0.4 GB KV ≈ 3.7k tokens bf16)
    worst-case GPU: 16.8 + 2.4 + 2.4 ≈ 21.6 GB of 24

⚠️ The yaml's `vllm:` block mirrors the CHAT engine's *_size values —
the scheduler sizes hardware for the largest engine.
"""
from ray.serve.llm import LLMConfig, build_openai_app


# Fixed KV cache per engine. Without it vLLM sizes the cache from how much VRAM
# was used while it loaded, and that count includes other processes on the card.
# Ray Serve starts these engines together, so one loading next to another would
# see its share as used up and fail with "No available memory for the cache
# blocks". With a fixed size gpu_memory_utilization only checks free VRAM at boot.
CHAT_KV_BYTES = int(1.2 * 2**30)                # per GPU
SIDECAR_KV_BYTES = int(0.4 * 2**30)


def card_share(fraction, gpus=1):
    """One engine's share of each GPU it runs on.

    Ray schedules GPUs separately from gpu_memory_utilization: by default every
    engine asks for a whole GPU, the worker advertises one, and every engine
    after the first waits forever. The bundle reserves the fraction, and
    VLLM_RAY_PER_WORKER_GPUS makes vLLM's worker request the same fraction. Ray
    2.55.1 infers that value only for its helper tasks, not for the replica that
    starts the engine, so it must be set here or the worker requests a whole GPU
    and fails. VLLM_DISABLE_COMPILE_CACHE avoids torch compile-cache contention
    between engines on one card.
    """
    return dict(
        placement_group_config=dict(bundles=[dict(GPU=fraction)] * gpus, strategy="STRICT_PACK"),
        runtime_env=dict(env_vars={
            "VLLM_RAY_PER_WORKER_GPUS": str(fraction),
            "VLLM_DISABLE_COMPILE_CACHE": "1",
        }),
    )


# vLLM applies a score template only when given one explicitly. Without it the
# reranker sees bare "query + document" instead of the prompt it was trained on,
# and scores come back with no error but no meaning. From vLLM v0.18.0
# examples/pooling/score/template/qwen3_reranker.jinja.
RERANK_TEMPLATE = """<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query") }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


chat = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",
        # VERIFY before selling: community AWQ re-upload (no official AWQ).
        model_source="cpatonn/GLM-4.5-Air-AWQ",
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=4,      # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,
        max_model_len=32768,
        max_num_seqs=8,
        gpu_memory_utilization=0.70, # leaves room for BOTH sidecars on one GPU
        kv_cache_memory_bytes=CHAT_KV_BYTES,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        dtype="auto",
        # Tool calling — VERIFY the glm45 parser on the pinned image first
        # (see ../README.md "Tool calling"):
        # enable_auto_tool_choice=True,
        # tool_call_parser="glm45",
    ),
    **card_share(0.70, gpus=4),
)

embed = LLMConfig(
    model_loading_config=dict(
        model_id="embed",
        model_source="Qwen/Qwen3-Embedding-0.6B",
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        runner="pooling",                        # vLLM 0.18 has no `task`; → /v1/embeddings
        convert="embed",
        max_model_len=2048,          # RAG chunks are ≤ 512 tokens; keeps the worst-case GPU under budget
        gpu_memory_utilization=0.06,
        kv_cache_memory_bytes=SIDECAR_KV_BYTES,
        enforce_eager=True,
    ),
    **card_share(0.06),
)

rerank = LLMConfig(
    model_loading_config=dict(
        model_id="rerank",
        model_source="Qwen/Qwen3-Reranker-0.6B",
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        runner="pooling",                        # vLLM 0.18 has no `task`; → /v1/score
        convert="classify",
        chat_template=RERANK_TEMPLATE,
        max_model_len=2048,          # RAG chunks are ≤ 512 tokens; keeps the worst-case GPU under budget
        gpu_memory_utilization=0.06,
        kv_cache_memory_bytes=SIDECAR_KV_BYTES,
        enforce_eager=True,
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    ),
    **card_share(0.06),
)


app = build_openai_app({"llm_configs": [chat, embed, rerank]})
