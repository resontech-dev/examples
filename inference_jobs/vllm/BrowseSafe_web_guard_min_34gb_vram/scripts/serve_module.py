"""BrowseSafe via vLLM: prompt-injection detector for fetched web pages.

    "web-guard" → perplexity-ai/browsesafe   POST /v1/chat/completions

Perplexity's open-weight classifier (December 2025, MIT) fine-tuned from
Qwen3-30B-A3B-Instruct-2507 on BrowseSafe-Bench. Protocol from the model
card: one user message whose content is the RAW HTML of the page, chat
template with the generation prompt, and the model answers a single
token, "yes" or "no". Pages above the 16,384-token window are split into
non-overlapping chunks and the verdicts OR-ed (any "yes" flags the page).
predict.py implements the chunking.

Where it sits: on the chat-server side, between "page fetched" and "page
goes into the context". Not through LiteLLM, and not on user text: for
that use Qwen3Guard / Nemotron 3.5 Content Safety / gpt-oss-safeguard.

Weights and memory. Only a bf16 checkpoint is published (~61 GB).
Shipped: `quantization="fp8"` on-the-fly weight quantization (~31 GB) on
one 48 GB card: 48×0.92 = 44.2 − 31 − ~1.5 activations ≈ 11 GB KV.
Qwen3-30B-A3B KV ≈ 98 KB/token bf16, ~49 KB fp8 → ~225k cached tokens:
8 full 16k pages in flight.

⚠️ VERIFY on first boot: dynamic fp8 on the Qwen3 MoE expert layers is
expected to work on vLLM 0.18.0 (official Qwen3-30B-A3B-FP8 checkpoints
run there, and ../Qwen3-Coder-30B-A3B…pipeline_parallel is the same
architecture in AWQ), but this exact combination has not been run. If a
vetted community FP8 / AWQ re-upload of browsesafe appears, point
`model_source` at it and drop the `quantization` kwarg (auto-detected).

Other shapes:
  * 2×24 GB (4090 / 3090), fp8: yaml vllm.tensor_parallel_size: 2,
    cluster.gpus_per_worker: 2, tensor_parallel_size=2 below, and the
    commented placement_group_config. ~15.5 GB weights per card.
  * 80 GB+ (H100 / A100-80G / RTX PRO 6000 96 GB), bf16: drop
    `quantization`, raise max_num_seqs to 16.

⚠️ The yaml's `vllm:` block MUST mirror the two *_size values below.
"""
from ray.serve.llm import LLMConfig, build_openai_app


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="web-guard",                    # client-facing name: keep stable
        # Official Perplexity checkpoint, bf16 ~61 GB, MIT. Qwen3MoeForCausalLM
        # is vLLM-native: no trust_remote_code.
        model_source="perplexity-ai/browsesafe",
    ),

    # Datacenter cards may carry Ray accelerator labels; the fleet is
    # unlabeled today, so keep None (non-None = PENDING forever on consumer cards).
    accelerator_type=None,

    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=16,                 # ~2× max_num_seqs: batcher never starves
        max_queued_requests=64,                  # fast 503 beats a silent queue
        graceful_shutdown_timeout_s=60,          # answers are one token: nothing to drain
    ),

    engine_kwargs=dict(
        tensor_parallel_size=1,                  # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                # = yaml vllm.pipeline_parallel_size
        quantization="fp8",                      # on-the-fly weight quant, see banner: VERIFY
        max_model_len=16384,                     # the model's trained window (card)
        max_num_seqs=8,                          # 16 on 80 GB bf16
        kv_cache_dtype="fp8",                    # standard attention: safe here
        enable_prefix_caching=True,              # chat-template prefix is shared by every check
        enable_chunked_prefill=True,             # 16k-token prefills don't stall other checks
        gpu_memory_utilization=0.92,
        dtype="auto",
        # No tool parser, no reasoning parser: the base is the Instruct-2507
        # (non-thinking) model and the answer is one token.
    ),

    # TP=2 shape only: uncomment with the three TP=2 edits:
    # placement_group_config=dict(
    #     bundle_per_worker={"GPU": 1, "CPU": 4},
    #     strategy="STRICT_PACK",                # TP never crosses machines
    # ),
)


app = build_openai_app({"llm_configs": [llm_config]})
