"""Guarded RAG chat — Qwen3-8B + Qwen3Guard-8B + embed + rerank, one GPU.

The full user-facing stack on one 24 GB worker, Apache-clean end to end
(no Llama-license guard):

    "assistant" → Qwen3-8B AWQ            POST /v1/chat/completions
    "guard"     → Qwen3Guard-Gen-8B       POST /v1/chat/completions (classifier)
    "embed"     → Qwen3-Embedding-0.6B    POST /v1/embeddings
    "rerank"    → Qwen3-Reranker-0.6B     POST /v1/score

The guard is a *generative* classifier: send it the content as a chat
message, it emits a safety verdict (Safe/Unsafe + categories). Callers
run it on user input before the chat model and on the answer after —
predict.py wires the full loop.

⚠️ All engines share one card. Each declares its share (`card_share`, sum
0.84) and a fixed KV cache. The share is Ray's accounting and a free-VRAM check
at boot, so real use differs. On a 24 GB card, about 22 GB in total, including
~0.85 GB of CUDA context and activations per engine:

            share   real use
    chat    0.32    7.8 GB (5.7 AWQ weights + 1.3 GB KV ≈ 19k tokens fp8)
    guard   0.40    9.2 GB (~7.9 GB fp8-quantized weights + 0.4 GB KV)
    embed   0.06    2.4 GB (1.1 weights + 0.4 GB KV ≈ 3.7k tokens bf16)
    rerank  0.06    2.4 GB

⚠️ The yaml's `vllm:` block MUST mirror the *_size values (all 1 here).
"""
from ray.serve.llm import LLMConfig, build_openai_app


# Fixed KV cache per engine. Without it vLLM sizes the cache from how much VRAM
# was used while it loaded, and that count includes other processes on the card.
# Ray Serve starts these engines together, so one loading next to another would
# see its share as used up and fail with "No available memory for the cache
# blocks". With a fixed size gpu_memory_utilization only checks free VRAM at boot.
CHAT_KV_BYTES = int(1.3 * 2**30)
GUARD_KV_BYTES = int(0.4 * 2**30)
SIDECAR_KV_BYTES = int(0.4 * 2**30)


def card_share(fraction):
    """One engine's share of the single GPU.

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
        placement_group_config=dict(bundles=[dict(GPU=fraction)]),
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
        model_source="Qwen/Qwen3-8B-AWQ",        # official AWQ, ~5.5 GB
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=16384,
        max_num_seqs=4,
        gpu_memory_utilization=0.32,
        kv_cache_memory_bytes=CHAT_KV_BYTES,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        dtype="auto",
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",               # stable for the Qwen family
        # Qwen3 thinks unless the request turns it off (predict.py does). The
        # parser keeps the thought out of `content`, in the `reasoning` field.
        reasoning_parser="qwen3",
        # YaRN — stretch Qwen3-8B's 32k window to ~131k on a bigger card.
        # 24 GB+ tier lever only: 128k of fp8 KV ≈ 9.5 GB for ONE
        # conversation — this bundle's 0.32 share can't afford it.
        # hf_overrides={"rope_scaling": {"rope_type": "yarn", "factor": 4.0,
        #                                "original_max_position_embeddings": 32768}},
    ),
    **card_share(0.32),
)

guard = LLMConfig(
    model_loading_config=dict(
        model_id="guard",
        # Apache-clean guard (the catalog's no-Llama-license story).
        # fp16 is ~16 GB — too big next to the chat model, so weights are
        # fp8-quantized at load (~8.5 GB). On Ampere cards vLLM runs fp8
        # weights via Marlin (W8A16) — fine for a classifier. If a vetted
        # community AWQ of Qwen3Guard appears, swap it and drop
        # `quantization` (auto-detected from the checkpoint).
        model_source="Qwen/Qwen3Guard-Gen-8B",
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,                 # verdicts are short — high turnover
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        quantization="fp8",                      # on-the-fly weight quant, see above
        max_model_len=4096,                      # moderation inputs are short
        max_num_seqs=4,
        gpu_memory_utilization=0.40,
        kv_cache_memory_bytes=GUARD_KV_BYTES,
        kv_cache_dtype="fp8",
        enforce_eager=True,                      # classifier — save CUDA-graph VRAM
    ),
    **card_share(0.40),
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
        max_model_len=2048,                      # RAG chunks are ≤ 512 tokens; 8k won't fit next to the guard
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
        max_model_len=2048,                      # RAG chunks are ≤ 512 tokens; 8k won't fit next to the guard
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


app = build_openai_app({"llm_configs": [chat, guard, embed, rerank]})
