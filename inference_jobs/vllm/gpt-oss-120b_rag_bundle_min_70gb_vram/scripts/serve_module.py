"""gpt-oss-120b RAG bundle: chat + embeddings + reranker, one 96 GB worker.

Ray Serve LLM's OpenAI router accepts a LIST of LLMConfigs, so one
deployment serves the whole private-RAG stack ("ChatGPT for our
documents") on the premium chat model:

    "assistant" → gpt-oss-120b MXFP4       POST /v1/chat/completions
    "embed"     → Qwen3-Embedding-0.6B     POST /v1/embeddings
    "rerank"    → Qwen3-Reranker-0.6B      POST /v1/score

⚠️ All engines share one card. Each declares its share (`card_share`, sum
0.86) and a fixed KV cache. Budget on a 96 GB card (RTX PRO 6000), about
82 GB in total (each engine also takes ~1 GB of CUDA context and
activations). ESTIMATES: this bundle has not booted on real hardware.

    chat   0.78 → 74.9 GB (65 GB weights + 6 GiB KV ≈ 340k tokens fp8)
    embed  0.04 →  3.8 GB (1.1 GB weights + 0.9 GB KV ≈ 8k tokens bf16)
    rerank 0.04 →  3.8 GB

H200 141 GB: chat 0.60 → 84.6 GB, CHAT_KV_BYTES 16 GiB (~900k tokens),
max_model_len 131072; sidecars 0.03 each.

An 80 GB card does not fit this bundle: 80×0.78 = 62 GB is below the
weights alone. Use ../gpt-oss-120b_chat_min_66gb_vram there and put the
sidecars on another card (the agents bundle's LLMConfigs).

⚠️ The yaml's `vllm:` block MUST mirror the *_size values (all 1 here).
"""
from ray.serve.llm import LLMConfig, build_openai_app


# Fixed KV cache per engine. Without it vLLM sizes the cache from how much VRAM
# was used while it loaded, and that count includes other processes on the card.
# Ray Serve starts these engines together, so one loading next to another would
# see its share as used up and fail with "No available memory for the cache
# blocks". With a fixed size gpu_memory_utilization only checks free VRAM at boot.
CHAT_KV_BYTES = int(6 * 2**30)       # 16 GiB on H200
SIDECAR_KV_BYTES = int(0.9 * 2**30)


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
        model_source="openai/gpt-oss-120b",      # native MXFP4, ~65 GB
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=180,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # Chat gets 0.78 of a 96 GB card: 65 GB weights + 6 GiB KV
        # (≈ 340k tokens at fp8, gpt-oss-120b ≈ 18.5 KB/token) → 64k
        # context × several sessions. On H200: share 0.60, CHAT_KV_BYTES
        # 16 GiB, max_model_len 131072, max_num_seqs 32.
        max_model_len=65536,
        max_num_seqs=16,
        gpu_memory_utilization=0.78,
        kv_cache_memory_bytes=CHAT_KV_BYTES,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        dtype="auto",
        # Tool calling: harmony is native to gpt-oss — VERIFY parser flag
        # before enabling (see ../README.md "Tool calling"):
        # enable_auto_tool_choice=True,
        # tool_call_parser="openai",
    ),
    **card_share(0.78),
)

embed = LLMConfig(
    model_loading_config=dict(
        model_id="embed",
        model_source="Qwen/Qwen3-Embedding-0.6B",
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,                 # embeddings batch cheaply
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        runner="pooling",                        # vLLM 0.18 has no `task`; → /v1/embeddings
        convert="embed",
        max_model_len=8192,
        gpu_memory_utilization=0.04,             # 0.03 on H200
        kv_cache_memory_bytes=SIDECAR_KV_BYTES,
        enforce_eager=True,                      # tiny model — save the CUDA-graph VRAM
    ),
    **card_share(0.04),
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
        max_model_len=8192,
        gpu_memory_utilization=0.04,             # 0.03 on H200
        kv_cache_memory_bytes=SIDECAR_KV_BYTES,
        enforce_eager=True,
        # Qwen3-Reranker ships as a causal LM; vLLM scores it as sequence
        # classification via this documented conversion:
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    ),
    **card_share(0.04),
)


app = build_openai_app({"llm_configs": [chat, embed, rerank]})
