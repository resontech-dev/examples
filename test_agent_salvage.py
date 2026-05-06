"""Salvage attempt for the degenerate federated LoRA.

The trained LoRA collapsed to "always emit `{`" due to gradient explosion.
This script tries scaling the LoRA contribution down (alpha/rank ratio) to
see if any setting blends usefully with the base model.

If even 0.05 scaling produces broken output, the LoRA is unsalvageable.
"""
import unsloth  # noqa: F401 — must be first
from unsloth import FastLanguageModel

import torch
from peft import set_peft_model_state_dict


CKPT_PATH = "/home/pyaremenko/examples/GLOBAL_MODEL (2).pt"
BASE_MODEL = "unsloth/mistral-7b-bnb-4bit"
SCALING_FACTORS = [0.0, 0.05, 0.25, 1.0, 2.0]   # 0=no LoRA, 2.0=as-trained

PROMPT = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"


def _set_lora_scaling(model, factor):
    """Override the LoRA scaling on every adapter layer."""
    n = 0
    for name, module in model.named_modules():
        if hasattr(module, "scaling") and isinstance(module.scaling, dict):
            for k in module.scaling:
                module.scaling[k] = factor
                n += 1
    return n


def main():
    print(f"Loading {BASE_MODEL}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=1024, dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
    )

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    set_peft_model_state_dict(model, sd)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    for factor in SCALING_FACTORS:
        n = _set_lora_scaling(model, factor)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Detect the "{ { {" failure
        unique_tokens = len(set(response.split()))
        verdict = "DEGENERATE" if unique_tokens < 5 else "COHERENT"
        print(f"\n--- LoRA scaling = {factor:.2f} ({n} layers patched) — {verdict} ---")
        print(f"  Output (first 200 chars): {response[:200]!r}")


if __name__ == "__main__":
    main()
