"""Smoke test for the federated LoRA adapter trained on job_agent (Mistral-7B + Glaive).

Loads:
  - Base model: unsloth/mistral-7b-bnb-4bit
  - LoRA adapter: federated state_dict from `GLOBAL_MODEL (2).pt`
Then prompts the merged model with sample function-calling queries and
prints what it generates. This verifies the FL training produced something
coherent — not just that the plumbing ran.
"""
import sys
from pathlib import Path

# Unsloth must be imported first
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import torch
from peft import set_peft_model_state_dict


CKPT_PATH = "/home/pyaremenko/examples/GLOBAL_MODEL (2).pt"
BASE_MODEL = "unsloth/mistral-7b-bnb-4bit"
LORA_RANK = 16
MAX_SEQ_LENGTH = 1024


PROMPTS = [
    # Generic instruction (tests baseline coherence)
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",

    # Function-calling style prompt (matches Glaive training format)
    """SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city name"}
        },
        "required": ["location"]
    }
}

USER: What's the weather in Tokyo right now?


ASSISTANT:""",

    """SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "calculate_tip",
    "description": "Calculate the tip amount for a given bill",
    "parameters": {
        "type": "object",
        "properties": {
            "bill_amount": {"type": "number", "description": "The total bill amount"},
            "tip_percentage": {"type": "number", "description": "Tip percentage (0-100)"}
        },
        "required": ["bill_amount", "tip_percentage"]
    }
}

USER: I had dinner that cost $85, please calculate a 20% tip.


ASSISTANT:""",
]


def main():
    print("=" * 72)
    print(f"Loading base model: {BASE_MODEL}")
    print("=" * 72)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
    )

    print(f"\nLoading federated LoRA: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    print(f"  {len(sd)} LoRA tensors, {sum(v.numel() for v in sd.values()):,} params")

    # Apply LoRA weights to peft model
    try:
        set_peft_model_state_dict(model, sd)
        print("  ✓ LoRA weights applied via set_peft_model_state_dict")
    except Exception as e:
        # Fallback: drop the `base_model.model.` prefix and try again
        print(f"  set_peft_model_state_dict failed ({type(e).__name__}); trying direct load_state_dict")
        model.load_state_dict(sd, strict=False)
        print("  ✓ LoRA weights applied via load_state_dict (strict=False)")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n{'=' * 72}")
        print(f"PROMPT {i}:")
        print(prompt[-400:] if len(prompt) > 400 else prompt)
        print("-" * 72)
        print("RESPONSE:")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(response)


if __name__ == "__main__":
    sys.exit(main())
