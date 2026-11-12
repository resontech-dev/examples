"""Qwen3 4B RAG sidecars: embeddings + reranker on one worker, no chat model.

Ray Serve LLM's OpenAI router accepts a LIST of LLMConfigs; this job ships
only the two retrieval engines, under the same model ids the bundles use:

    "embed"   -> Qwen3-Embedding-4B    POST /v1/embeddings
    "rerank"  -> Qwen3-Reranker-4B     POST /v1/score

Why 4B over the 0.6B pair in the bundles (published numbers, Qwen3
Embedding report and MTEB; ours measured only for 0.6B): SciFact nDCG@10
78.3 vs 69.7 for the embeddings, 78.4 vs 74.9 after reranking, and MTEB
retrieval 68.5 vs 61.8. Both Apache-2.0, both 32k windows, 100+ languages.

Memory. bf16 weights are ~8 GB per model, so both engines ship with
`quantization="fp8"` (weights quantized at load, ~4.3 GB each; Marlin
W8A16 on Ampere, native fp8 on Ada / Hopper / Blackwell). VERIFY: fp8
on-the-fly for pooling models on vLLM 0.18.0 is unverified. If an engine
refuses it, drop the kwarg and move to a 24 GB card (bf16, same shares).

Budget on a 16 GB card, about 13.4 GB in total (each engine also takes
~0.85 GB of CUDA context and activations):

            share   real use
    embed   0.42    6.1 GB (4.3 fp8 weights + 0.9 GB KV + context)
    rerank  0.42    6.1 GB

On a 12 GB card: shares 0.40, SIDECAR_KV_BYTES 0.3 GiB, max_model_len 2048
(chunks are short). On a 24 GB card: drop `quantization` for bf16 at the
same shares, or move to the 8B siblings (Qwen3-Embedding-8B,
Qwen3-Reranker-8B, ~8.5 GB each fp8).

The yaml's `vllm:` block MUST mirror the *_size values (all 1 here).
"""
from ray.serve.llm import LLMConfig, build_openai_app


# Fixed KV cache per engine. Without it vLLM sizes the cache from how much VRAM
# was used while it loaded, and that count includes other processes on the card.
# Ray Serve starts these engines together, so one loading next to another would
# see its share as used up and fail with "No available memory for the cache
# blocks". With a fixed size gpu_memory_utilization only checks free VRAM at boot.
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


embed = LLMConfig(
    model_loading_config=dict(
        model_id="embed",                        # same id as in every bundle
        model_source="Qwen/Qwen3-Embedding-4B",  # official, ~8 GB bf16, Apache-2.0
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,                 # embeddings batch cheaply
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        runner="pooling",                        # vLLM 0.18 has no `task`; -> /v1/embeddings
        convert="embed",
        quantization="fp8",                      # weights quantized at load, ~4.3 GB. VERIFY on 0.18.0
        max_model_len=8192,                      # chunks are <= 512 tokens; the model allows 32k
        gpu_memory_utilization=0.42,
        kv_cache_memory_bytes=SIDECAR_KV_BYTES,
        enforce_eager=True,                      # pooling model: save the CUDA-graph VRAM
    ),
    **card_share(0.42),
)

rerank = LLMConfig(
    model_loading_config=dict(
        model_id="rerank",                       # same id as in every bundle
        model_source="Qwen/Qwen3-Reranker-4B",   # official, ~8 GB bf16, Apache-2.0
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=32,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        runner="pooling",                        # vLLM 0.18 has no `task`; -> /v1/score
        convert="classify",
        chat_template=RERANK_TEMPLATE,
        quantization="fp8",                      # VERIFY on 0.18.0, see the module docstring
        max_model_len=8192,
        gpu_memory_utilization=0.42,
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
    **card_share(0.42),
)

# Optional third engine: mxbai-rerank-large-v2 (mixedbread, 1.5B, Apache-2.0,
# BEIR avg 61.4 published, the registry's candidate to replace the 0.6B
# reranker). It is a Qwen2-based causal LM scored as sequence classification,
# the way the vLLM scoring docs show for mxbai-rerank-base-v2. VERIFY before
# selling: a vLLM forum thread reports scores that differ from the mxbai
# library and a different order on large payloads; compare on BEIR
# (reson_inference_benchmarks/quality) first. Needs ~1.6 GB more: lower the
# two shares above to 0.36 each on a 16 GB card, or use a 24 GB card.
#
# rerank_mxbai = LLMConfig(
#     model_loading_config=dict(
#         model_id="rerank-mxbai",
#         model_source="mixedbread-ai/mxbai-rerank-large-v2",
#     ),
#     accelerator_type=None,
#     deployment_config=dict(
#         autoscaling_config=dict(min_replicas=1, max_replicas=1),
#         max_ongoing_requests=32,
#     ),
#     engine_kwargs=dict(
#         tensor_parallel_size=1,
#         pipeline_parallel_size=1,
#         runner="pooling",
#         convert="classify",
#         max_model_len=8192,
#         gpu_memory_utilization=0.12,
#         kv_cache_memory_bytes=int(0.4 * 2**30),
#         enforce_eager=True,
#         hf_overrides={
#             "architectures": ["Qwen2ForSequenceClassification"],
#             "classifier_from_token": ["0", "1"],
#             "method": "from_2_way_softmax",
#         },
#     ),
#     **card_share(0.12),
# )


app = build_openai_app({"llm_configs": [embed, rerank]})
# With the optional engine: build_openai_app({"llm_configs": [embed, rerank, rerank_mxbai]})
