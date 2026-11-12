# ⚠️ TEMPLATE: Qwen3.5-397B-A17B (FP8), TP=8 multimodal on one 8-GPU node

**Not deployable on the current fleet.** This is the serving job from
[run_qwen_3.5-397B-A17B_plan.md](../../../../reson_docs/docs/inference/run_qwen_3.5-397B-A17B_plan.md)
§7: the first large model on the platform, 397B MoE with 17B active,
text + image in one checkpoint, 262K context, Apache-2.0. It ships now so
the day the node is onboarded (plan step 4) the submit is one command;
nothing in this folder has run, and every number is napkin math from the
plan until the benchmark (plan §8) replaces it.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources (planning numbers from the plan, unvalidated)

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 640 GB (8× H100 80 GB) → 1,128 GB (8× H200) | 32 | 1 TB | ≥ 2 TB local NVMe (550 GB for this job: image 15 + 1.3×400 GB checkpoint; the rest for a second model version and the model cache) | 8x80gb / 8xh200 |

| Node | After weights + ~10 % runtime | Verdict (plan §3) |
|---|---|---|
| 8× H200, NVLink | ~690 GB for KV and the vision encoder | target: many concurrent long sessions |
| 8× H100 SXM, NVLink | ~200 GB | fallback: works, fewer long sessions at once (Unihost AZM-12) |
| 4× RTX PRO 6000, PCIe, Int4 build | ~130 GB | proof-of-pipeline only: no NVLink, Int4 costs some quality |

Node requirements beyond the table: NVLink/NVSwitch between every GPU
pair (`nvidia-smi topo -m` shows `NV#` everywhere; `require_nvlink: true`
in the yaml once spec item F is live), driver R570+ with
`nvidia-fabricmanager`, 10–25 Gbps uplink. Consumer fleet: no.

## Before first deploy (plan steps 1 to 4)

1. **Prove the family on today's fleet** (plan §4):
   `../Qwen3.5-27B-AWQ_agents_vision_min_28gb_vram` on the pinned image
   must answer a streamed chat, an `image_url` question and a tool call.
   That job is this file with `model_source` and TP changed.
2. **Platform spec items B, D, E** (plan §5): model cache and pre-stage,
   deploy deadline, per-job request limits. The yaml carries the D and E
   fields commented out until they exist.
3. **Order the node** (plan §3, decision 1): 8× H200 from a second
   provider, or AZM-12 (8× H100) at Unihost.
4. **Onboard it** (plan §6): bootstrap with all 8 GPUs in one worker,
   `nccl-tests` across 8 GPUs, burn-in, **pre-stage the ~400 GB FP8
   checkpoint on local NVMe** and verify checksums, pre-pull the ray-llm
   image, register the node as the job's head host too. Never pull the
   weights at first boot.
5. **Verify the engine arguments** against the image:
   `docker run --rm <ray-llm-image> vllm serve --help | grep -A3 -E 'reasoning-parser|tool-call-parser|mm-encoder-tp-mode|mm-processor-cache-type'`.
6. If the node carries a Ray accelerator label, set
   `accelerator_type="H200"` (or `"H100"`); otherwise keep `None`.
7. Drop the TEMPLATE banners once the benchmark page exists under
   `reson_docs/docs/benchmarks/inference/`.

## Deploy (once the node exists)

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                       # text
python predict.py --image               # sample_data/invoice.png → JSON
python predict.py --tools               # tool-calling round trip
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

**Exit (plan §7):** a deploy goes from submit to a first answer without
downloading weights, without manual steps on the node and without a
timeout.

## Client notes

One model id `qwen3.5-397b` for text and images. Thinking is on by
default; send `chat_template_kwargs: {enable_thinking: false}` per request
(`extra_body` in openai-python) when the answer should come without the
`reasoning` field. Images go as standard OpenAI `image_url` parts; the
vision encoder turns each into tokens (a 1024×1024 image is roughly 1,000
tokens, check `preprocessor_config.json`) and those count against the
context window and the KV cache like text. Up to 8 images per prompt,
video off. The key goes as both `Authorization: Bearer` and `X-API-Key`.
Until spec item E ships, the platform path caps a request at 300 s and
32 MB (plan §5), which limits image batches and the longest reasoning
answers.

## Variants

| Variant | Where | Edits |
|---|---|---|
| **TP=8, FP8 (shipped)** | 8× H200 or 8× H100 80 GB, one worker | none |
| DP=8 + expert parallel | 8× H200 | the vLLM recipe's throughput layout: each GPU holds the attention weights and one eighth of the experts. Through Ray Serve LLM it is a data-parallel deployment; measure it in the benchmark after TP=8 works. `enable_expert_parallel=True` is the intermediate flag, shipped commented out |
| Official GPTQ-Int4 (~220 GB) | 4× H200 TP=4 or 4× RTX PRO 6000 (plan's proof-of-pipeline node) | `model_source="Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"` (VERIFY the repo id), `tensor_parallel_size=4` in both files, `gpus_per_worker: 4`, `min_vram_gb_per_worker: 320`; VERIFY the `moe_wna16` kernel path for GPTQ MoE experts on vLLM 0.18.0 |
| FP8 KV cache | any | `kv_cache_dtype="fp8"`: roughly doubles long sessions per node; needs its own quality check first (garbage output seen on AWQ models, llmconfig.md §4) |
| 1M context | any | YaRN past 262K changes quality and memory; decision 2 and a separate benchmark first |

## Benchmark (plan §8)

Same harness as `reson_inference_benchmarks/`: chat 2K in / 512 out;
document Q&A at 32K / 128K / 262K; the prefix-cache effect on a second
question; 1 and 4 images per prompt; TP=8 against DP+EP=8; needle in a
haystack at 128K and 262K; 2 hours at 70 % load. The output is the capacity
sheet (concurrent users per context size per node) that sets decision 4
and the price per GPU-hour. No sales claim names this model before that
page exists.

## Notes

- The model card and the vLLM recipe override this folder wherever they
  differ; re-read both before ordering hardware.
- One node is a single point of failure for the endpoint: fine for a demo
  and a pilot; production wants a second replica with prefix-aware routing
  (plan §9).
- Placement is by hand until spec items A to C are fully live: point the
  job at the one node that fits.
- Cost reference (plan §10): AZM-12 at Unihost is €13,890.90 per month,
  €2.38 per GPU-hour at full use; an 8× H200 quote needs its own arithmetic.
