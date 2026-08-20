# Integrate LLM from scratch (predictor mode)

Use the platform's standard predictor contract to serve an LLM. You write
the inference code; the platform provides HTTP + auth + lifecycle.

For OpenAI API compatibility, large models, tensor/pipeline parallelism,
continuous batching, or vLLM in general — use the OpenAI-compatible mode
instead ([openai_compatible_llm_flow.md](openai_compatible_llm_flow.md)).
This doc is for the cases where the predictor-mode tradeoffs are right.

---

## When predictor-mode LLM beats OpenAI-mode

| Want this                                                 | Use predictor mode |
|-----------------------------------------------------------|--------------------|
| Custom request/response shape (not OpenAI's)              | YES                |
| Inline RAG (fetch context, then generate) in one call     | YES                |
| Custom tokenizer / pre-prompt manipulation                | YES                |
| Non-generative models (embeddings, rerankers, classifiers)| YES                |
| Models under ~7B on a single GPU, modest QPS              | works fine         |

Use OpenAI mode instead when:

- Existing OpenAI SDK clients have to talk to you unmodified
- Throughput matters (vLLM's continuous batching is ~5–10× faster)
- The model is large enough to need tensor or pipeline parallelism
- You want PagedAttention / KV-cache sharing for free

---

## Reference example — chat LLM via transformers

`my_llm_serve.py`:

```python
import json, time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatLLM:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        dtype: str = "auto",
        default_max_new_tokens: int = 256,
        default_temperature: float = 0.7,
        default_top_p: float = 0.9,
        max_input_tokens: int = 2048,
        system_prompt: str | None = None,
    ):
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.max_input_tokens = max_input_tokens

        self.device = self._pick_device()
        torch_dtype = self._pick_dtype(dtype, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        ).to(self.device).eval()

    def warmup(self):
        # 4-token throwaway generation so the first real request doesn't
        # pay cuDNN autotune + kv-cache allocation cost.
        with torch.inference_mode():
            inputs = self.tokenizer("hi", return_tensors="pt").to(self.device)
            self.model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                pad_token_id=self.tokenizer.pad_token_id)

    @torch.inference_mode()
    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            req = json.loads(data.decode("utf-8"))
            if not isinstance(req, dict):
                req = {"prompt": data.decode("utf-8")}
        except json.JSONDecodeError:
            req = {"prompt": data.decode("utf-8")}

        prompt = req.get("prompt") or req.get("text") or ""
        if not prompt:
            raise ValueError("missing 'prompt' field")

        max_new = int(req.get("max_new_tokens", self.default_max_new_tokens))
        temperature = float(req.get("temperature", self.default_temperature))
        top_p = float(req.get("top_p", self.default_top_p))
        system = req.get("system", self.system_prompt)
        do_sample = temperature > 0

        # Apply chat template if the tokenizer has one.
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=self.max_input_tokens).to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new, do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        t_gen0 = time.perf_counter()
        out = self.model.generate(**inputs, **gen_kwargs)
        gen_ms = (time.perf_counter() - t_gen0) * 1000

        new_tokens = out[0, prompt_tokens:]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "text": completion,
            "model_id": self.model_id,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(new_tokens.numel()),
            "generate_ms": gen_ms,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model_id": self.model_id, "device": str(self.device)}

    @staticmethod
    def _pick_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _pick_dtype(spec, device):
        if spec == "auto":
            return torch.float16 if device.type in ("cuda", "mps") else torch.float32
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[spec]
```

`inference.yaml`:

```yaml
model:
  # No `path` — weights pulled from HF Hub inside __init__.
  module: "my_llm_serve"
  class_name: "ChatLLM"
  init_args:
    model_id: "Qwen/Qwen2.5-0.5B-Instruct"
    default_max_new_tokens: 256
    system_prompt: "You are a concise, helpful assistant."

cluster:
  num_workers: 1
  mode: "data_parallel"
  gpus_per_worker: 1
  cpus_per_worker: 8

  # generate() is sequential per replica — keep the queue tiny to avoid
  # head-of-line blocking. Add more replicas for throughput, not depth.
  max_ongoing_requests: 2
  device: "auto"

requirements:
  - "torch"                           # bare — bound transitively by transformers
  - "transformers==4.40.0"            # pin: generate kwarg names churn across minors
  - "accelerate==0.29.3"
  - "tokenizers>=0.15.0,<0.20"
  - "sentencepiece>=0.2.0"
  - "safetensors>=0.4.0"
  - "huggingface_hub>=0.22,<0.25"
```

---

## Memory budget

Per replica:

```
weights × bytes_per_param  +  kv-cache  +  workspace
       (model size)            (model_id-specific)    (~1.5 GB)
```

| dtype  | bytes/param |
|--------|-------------|
| fp32   | 4           |
| bf16   | 2           |
| fp16   | 2           |
| int8   | 1 (needs GPTQ/AWQ quantised weights) |

| model                                  | params | fp16 weights | total per-replica |
|----------------------------------------|--------|--------------|-------------------|
| SmolLM2-360M-Instruct                  | 360M   | 720 MB        | ~2 GB             |
| Qwen2.5-0.5B-Instruct                  | 500M   | 1.0 GB        | ~2.5 GB           |
| Qwen2.5-1.5B-Instruct                  | 1.5B   | 3 GB          | ~4.5 GB           |
| Mistral-7B-Instruct                    | 7B     | 14 GB         | ~16 GB            |
| Llama-3.1-8B-Instruct                  | 8B     | 16 GB         | ~17.5 GB          |

Multiply per-replica by `num_replicas` to size each worker host's GPU.
If you're on Apple Silicon Docker with the default 7.65 GB VM, you can
fit one ≤ 1.5 GB-weights replica comfortably. Larger models OOM at boot
or cycle through Ray's 95% OOM-kill threshold.