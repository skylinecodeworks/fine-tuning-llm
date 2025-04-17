#!/usr/bin/env python3
"""
Simple tester for the fine-tuned PEFT model.
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(
        description="Simple tester for fine-tuned PEFT model"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./finetuned",
        help="Path to the PEFT adapter directory"
    )
    parser.add_argument(
        "--base-model-dir", type=str, default=None,
        help="Path to the base model directory (overrides adapter_config)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Prompt text for generation; if not set, will prompt interactively"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50,
        help="Number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Nucleus sampling (top-p)"
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    print(f"Loading tokenizer from '{model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model_dir = args.base_model_dir
    if base_model_dir is None:
        config_path = os.path.join(model_dir, "adapter_config.json")
        with open(config_path, "r") as f:
            adapter_cfg = json.load(f)
        base_model_dir = adapter_cfg.get("base_model_name_or_path")
        if base_model_dir is None:
            raise ValueError(
                "base_model_name_or_path not found in adapter_config.json; "
                "please set --base-model-dir"
            )

    print(f"Loading base model from '{base_model_dir}'...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir, trust_remote_code=True
    )
    print(f"Loading PEFT adapter from '{model_dir}'...")
    model = PeftModel.from_pretrained(base_model, model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device).eval()

    prompt = args.prompt or input("Prompt: ")

    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated output:\n{generated}")

if __name__ == "__main__":
    main()