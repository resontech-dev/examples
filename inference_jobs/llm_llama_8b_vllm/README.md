# 07 — Llama 3.1 8B (vLLM)

Reference deployment for an 8B chat model. Ships with three placement
modes pre-written in `serve_module.py` — uncomment the one that matches
your cluster:

| Mode                          | Hardware                                     | When to use                                |
|-------------------------------|----------------------------------------------|--------------------------------------------|
| **(A) Single-node TP=2** (default) | 2× consumer GPUs on one box (L4, 4090, A10G) | Cheapest realistic 8B deployment           |
| (B) Single-GPU                | 1× A100 80 GB or H100                        | Highest throughput per dollar              |
| (C) Cross-node TP=2 × PP=2    | 4× small GPUs across 2 nodes                 | Last resort if (A) doesn't fit topology    |

## Setup: Llama is a gated model

```bash
# 1. Accept the license:
#    https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

# 2. Create a read-scope token:
#    https://huggingface.co/settings/tokens

# 3. Pass it to the dispatcher so it forwards to every replica:
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

The dispatcher should set this in the env it passes to the head and
worker containers. `serve_module.py` reads it from `os.environ` and
forwards it to vLLM workers via `LLMConfig.runtime_env`.

## Use it

```bash
cp -r inference_templates/07_llm_llama_8b_vllm jobs/llama_8b
./scripts/setup.sh jobs/llama_8b

python3 -c '
import openai
c = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
print(c.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Summarize the Treaty of Versailles in 3 bullets."}],
).choices[0].message.content)
'
```

## VRAM sizing — Llama 3.1 8B

| Precision | Weights | KV-cache (64 seq × 8k ctx) | Total |
|-----------|---------|----------------------------|-------|
| fp16      | 16 GB   | ~4 GB                       | ~20 GB |
| bf16      | 16 GB   | ~4 GB                       | ~20 GB |
| int8 (quantised, requires GPTQ/AWQ weights) | 8 GB | ~4 GB | ~12 GB |

For mode (A) at TP=2 each GPU holds ~8 GB weights + ~2 GB activations +
~2 GB KV-cache ≈ 12 GB. L4 (24 GB) has ample headroom; 16 GB cards
(4080, A4000) fit at lower `max_num_seqs`.

## What to change

| where (`serve_module.py`)                                | for what                                          |
|----------------------------------------------------------|---------------------------------------------------|
| Uncomment (A), (B), or (C)                               | match your cluster topology                       |
| `accelerator_type`                                       | `"L4"`, `"A10G"`, `"A100"`, `"H100"`, or `None`   |
| `engine_kwargs.max_model_len`                            | context window (8k default; 32k+ at higher VRAM)  |
| `engine_kwargs.max_num_seqs`                             | concurrent sequences per replica                  |
| `deployment_config.autoscaling.max_replicas`             | data-parallel scale-out on top                    |

## Drop-in alternates

The same template runs other 8B-ish models with just a `model_source`
change:

| `model_source`                                       | Notes                                              |
|------------------------------------------------------|----------------------------------------------------|
| `mistralai/Mistral-7B-Instruct-v0.3`                  | gated; same shape as Llama 8B                      |
| `Qwen/Qwen2.5-7B-Instruct`                            | not gated; identical TP/PP behaviour              |
| `microsoft/Phi-3-mini-128k-instruct`                  | 3.8B but 128k context — drop TP=1, raise max_model_len |
| `mistralai/Mixtral-8x7B-Instruct-v0.1`                | MoE 47B; needs TP=4 minimum, much bigger budget    |
