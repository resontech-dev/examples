"""Chat LLM template — JSON prompt -> generated text.

Request shape (JSON body posted to /predict):
    {
      "prompt": "Explain attention in one sentence.",
      "max_new_tokens": 128,      # optional, init default
      "temperature": 0.7,          # optional
      "top_p": 0.9,                # optional
      "system": "Be terse."        # optional, overrides init system_prompt
    }

A bare text body is also accepted and treated as `{"prompt": <text>}`.

Response:
    {
      "text": "...",                       # the generated completion (prompt stripped)
      "model_id": "Qwen/...",
      "prompt_tokens": int,
      "completion_tokens": int,
      "generate_ms": float,
      "throughput_tokens_per_s": float,
      "device": "cuda"
    }

Notes:
- No `load_weights()` — `from_pretrained(model_id)` is what loads the
  weights. The platform skips load_weights when inference.yaml has no
  `model.path`, which is what we want here.
- For larger models, swap the LLM template for a vLLM- or llama.cpp-
  backed one. Transformers is fine up to ~7B at modest QPS.
"""
from __future__ import annotations

import json
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class ChatLLM:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        dtype: str = "auto",
        default_max_new_tokens: int = 256,
        default_temperature: float = 0.7,
        default_top_p: float = 0.9,
        max_input_tokens: int = 2048,
        chat_template: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.use_chat_template = bool(chat_template)
        self.system_prompt = system_prompt
        self.default_max_new_tokens = int(default_max_new_tokens)
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)
        self.max_input_tokens = int(max_input_tokens)

        self.device = _pick_device()
        self.torch_dtype = _resolve_dtype(dtype, self.device)

        print(f"[ChatLLM] loading tokenizer for {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[ChatLLM] loading model {model_id} dtype={self.torch_dtype} device={self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        print(f"[ChatLLM] ready on {self.device}")

    def warmup(self) -> None:
        # Run a tiny generation so the first real request doesn't pay the
        # cuDNN autotune / kv-cache allocation tax.
        with torch.inference_mode():
            inputs = self.tokenizer("hello", return_tensors="pt").to(self.device)
            self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

    @torch.inference_mode()
    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()

        text = data.decode("utf-8", errors="replace")
        try:
            req = json.loads(text)
            if not isinstance(req, dict):
                req = {"prompt": text}
        except json.JSONDecodeError:
            req = {"prompt": text}

        prompt = req.get("prompt") or req.get("text") or ""
        if not prompt:
            return {"error": "request must include a 'prompt' field (or be raw text)"}

        max_new = int(req.get("max_new_tokens", self.default_max_new_tokens))
        temperature = float(req.get("temperature", self.default_temperature))
        top_p = float(req.get("top_p", self.default_top_p))
        system = req.get("system")
        do_sample = temperature > 0

        inputs = self._tokenize(prompt, system)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        gen_kwargs = dict(
            max_new_tokens=max_new,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        t_gen0 = time.perf_counter()
        out = self.model.generate(**inputs, **gen_kwargs)
        gen_ms = (time.perf_counter() - t_gen0) * 1000

        new_tokens = out[0, prompt_tokens:]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        completion_tokens = int(new_tokens.numel())
        return {
            "text": completion,
            "model_id": self.model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generate_ms": gen_ms,
            "throughput_tokens_per_s": (completion_tokens / (gen_ms / 1000)) if gen_ms > 0 else 0.0,
            "device": str(self.device),
            "params": {
                "max_new_tokens": max_new,
                "temperature": temperature,
                "top_p": top_p,
            },
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    def _tokenize(self, prompt: str, system: str | None) -> dict:
        if (
            self.use_chat_template
            and hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            messages: list[dict] = []
            sys_msg = system if system is not None else self.system_prompt
            if sys_msg:
                messages.append({"role": "system", "content": sys_msg})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = prompt
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.torch_dtype).replace("torch.", ""),
        }


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    if spec == "auto":
        return torch.float16 if device.type in ("cuda", "mps") else torch.float32
    if spec not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype '{spec}'; expected one of {list(_DTYPE_MAP) + ['auto']}")
    return _DTYPE_MAP[spec]
