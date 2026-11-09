"""Agents bundle — Qwen3-8B + embed + rerank (+ guard), one 16 GB GPU.

The smallest complete assistant stack on a single consumer card,
Apache-clean end to end:

    "assistant" → Qwen3-8B AWQ            POST /v1/chat/completions
    "embed"     → Qwen3-Embedding-0.6B    POST /v1/embeddings
    "rerank"    → Qwen3-Reranker-0.6B     POST /v1/score
    "guard"     → Qwen3Guard-Gen-0.6B     POST /v1/chat/completions   (WITH_GUARD)

One model id serves every text task (classification, Q&A, drafting,
answer checks): callers vary temperature per task, the engine is shared. Tool calling (hermes parser) is on. Qwen3 thinks by
default — turn it off per request with
chat_template_kwargs={"enable_thinking": False}, as predict.py does.

⚠️ Every engine takes its share of the card at boot.
Budget on a 16 GB card (fractions of total VRAM, sum 0.85; the 15 % left
is for the 3–4 CUDA contexts, ~0.4 GB each):

    3 engines (default)                    4 engines (WITH_GUARD = True)
    assistant 0.62  → 9.9 GB               assistant 0.50  → 8.0 GB
      5.9 weights + ~0.8 activations         5.9 + 0.8 → KV ≈ 1.3 GB ≈ 17k tok
      → KV ≈ 3.2 GB ≈ 43k tokens fp8         (max_model_len 8192, 4 seqs)
    embed     0.115 → 1.8 GB               guard     0.12  → 1.9 GB
    rerank    0.115 → 1.8 GB               embed / rerank  0.115 each

The KV cache sizes are fixed (CHAT_KV_GIB, SIDECAR_KV_GIB below), so they do
not grow with the card. On a 24 GB card keep the same fractions and raise
CHAT_KV_GIB to ≈ 8 (32k context × 8 seqs is comfortable, the guard fits with
room).

⚠️ The yaml's `vllm:` block MUST mirror the *_size values (all 1 here).
"""
from ray.serve.llm import LLMConfig, build_openai_app


WITH_GUARD = False   # True → adds Qwen3Guard-Gen-0.6B as model "guard" (16 GB: chat context drops to 8k)

if WITH_GUARD:
    FRACTIONS = {"assistant": 0.50, "guard": 0.12, "embed": 0.115, "rerank": 0.115}
    CHAT_MAX_LEN, CHAT_MAX_SEQS, CHAT_KV_GIB = 8192, 4, 1.3
else:
    FRACTIONS = {"assistant": 0.62, "embed": 0.115, "rerank": 0.115}
    CHAT_MAX_LEN, CHAT_MAX_SEQS, CHAT_KV_GIB = 16384, 8, 3.2

# Fixed KV cache per engine. Without it vLLM sizes the cache from how much VRAM
# was used while it loaded, and that count includes other processes on the card.
# Ray Serve starts these engines together, so one loading next to another would
# see its share as used up and fail with "No available memory for the cache
# blocks". With a fixed size gpu_memory_utilization only checks free VRAM at boot.
SIDECAR_KV_GIB = 0.4
CHAT_KV_BYTES = int(CHAT_KV_GIB * 2**30)
SIDECAR_KV_BYTES = int(SIDECAR_KV_GIB * 2**30)


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


assistant = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",                    # one id for every text task — keep stable
        model_source="Qwen/Qwen3-8B-AWQ",        # official AWQ 4-bit, ~5.9 GB on disk
    ),
    accelerator_type=None,                       # consumer cards carry no Ray label
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=CHAT_MAX_SEQS * 2,  # ~2× max_num_seqs — batcher never starves
        max_queued_requests=64,                  # fast 503 beats a silent queue
        graceful_shutdown_timeout_s=120,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=CHAT_MAX_LEN,
        max_num_seqs=CHAT_MAX_SEQS,
        gpu_memory_utilization=FRACTIONS["assistant"],
        kv_cache_memory_bytes=CHAT_KV_BYTES,
        # Qwen3-8B KV ≈ 74 KB/token at fp8. The Qwen2.5-Coder-7B-AWQ
        # benchmark on a 5070 Ti ran fp8 KV + prefix caching cleanly; if
        # this checkpoint returns garbage, switch both off first
        # (reson_docs llmconfig.md → "Garbled output from quantized models").
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,              # agents resend one long system prompt every call
        enable_chunked_prefill=True,
        dtype="auto",

        # ── Tool calling (Qwen2.5 / Qwen3 → hermes) ──
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",

        # ── Thinking ──
        # Qwen3 thinks by default: slow, and JSON-parsing callers do not
        # expect a <think> block. The reasoning parser moves it into the
        # `reasoning` field. Callers turn thinking off per request. Don't set
        # `default_chat_template_kwargs` here: Ray 2.55.1 turns it into a
        # Namespace and every chat request then fails with a TypeError
        # (reson_docs inference/vllm/known_issues.md).
        reasoning_parser="qwen3",

        # ── Step 2: fine-tuned adapters ──
        # Uncomment to serve LoRA adapters trained on this base:
        # model="assistant:drafts-v3" → <dynamic_lora_loading_path>/drafts-v3/
        # (a PEFT folder). See ../Qwen2.5-7B_multi_lora_min_10gb_vram.
        # enable_lora=True,
        # max_loras=4,
        # max_lora_rank=32,
    ),
    # lora_config=dict(
    #     dynamic_lora_loading_path="s3://YOUR-BUCKET/loras/qwen3-8b/",
    #     max_num_adapters_per_replica=8,
    # ),
    **card_share(FRACTIONS["assistant"]),
)

guard = LLMConfig(
    model_loading_config=dict(
        model_id="guard",
        # Generative classifier: reply is "Safety: Safe|Unsafe|Controversial"
        # + "Categories: …" (+ "Refusal: …" for assistant messages).
        # 4B / 8B siblings exist for a 24 GB card (see the guarded_rag_chat job).
        model_source="Qwen/Qwen3Guard-Gen-0.6B",  # ~1.2 GB bf16
    ),
    accelerator_type=None,
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,                  # verdicts are short — high turnover
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=2048,                      # moderation inputs are short
        max_num_seqs=2,
        gpu_memory_utilization=FRACTIONS.get("guard", 0.12),
        kv_cache_memory_bytes=SIDECAR_KV_BYTES,
        kv_cache_dtype="fp8",
        enforce_eager=True,                      # classifier — save CUDA-graph VRAM
        dtype="auto",
    ),
    **card_share(FRACTIONS.get("guard", 0.12)),
)

embed = LLMConfig(
    model_loading_config=dict(
        model_id="embed",
        model_source="Qwen/Qwen3-Embedding-0.6B",  # ~1.2 GB bf16
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
        # 1.8 GB share − 1.2 weights − ~0.2 activations ≈ 0.4 GB KV ≈ 3.5k
        # tokens at bf16 — vLLM needs ONE max-length sequence to fit, so the
        # window stays at 2k (RAG chunks are ≤ 512 tokens anyway).
        max_model_len=2048,
        gpu_memory_utilization=FRACTIONS["embed"],
        kv_cache_memory_bytes=SIDECAR_KV_BYTES,
        enforce_eager=True,
    ),
    **card_share(FRACTIONS["embed"]),
)

rerank = LLMConfig(
    model_loading_config=dict(
        model_id="rerank",
        model_source="Qwen/Qwen3-Reranker-0.6B",   # ~1.2 GB bf16
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
        max_model_len=2048,                      # query + one candidate chunk
        gpu_memory_utilization=FRACTIONS["rerank"],
        kv_cache_memory_bytes=SIDECAR_KV_BYTES,
        enforce_eager=True,
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    ),
    **card_share(FRACTIONS["rerank"]),
)


app = build_openai_app(
    {"llm_configs": [assistant, embed, rerank] + ([guard] if WITH_GUARD else [])}
)
