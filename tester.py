#!/usr/bin/env python3
"""
Interactive tester for the fine-tuned LLM model.
Provides a simple REPL with commands to adjust generation parameters.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Interactive console for fine-tuned LLM")
    parser.add_argument("--model-dir", type=str, default="./finetuned",
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum number of tokens to generate (including prompt)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling (top-p)")
    parser.add_argument("--top-k", type=int, default=60,
                        help="Top-k sampling")
    parser.add_argument("--no-sample", action="store_true",
                        help="Disable sampling (use greedy decoding)")
    parser.add_argument("--log-file", type=str,
                        help="File to log conversation history (optional)")
    args = parser.parse_args()

    # Load tokenizer and model
    print(f"Loading tokenizer and model from '{args.model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Generation settings
    gen_config = {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": not args.no_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Open log file if specified
    log_f = open(args.log_file, "a", encoding="utf-8") if args.log_file else None

    help_text = (
        "\nCommands:\n"
        "  /help               Show this help message\n"
        "  /exit               Exit tester\n"
        "  /settings           Show current generation settings\n"
        "  /set <param> <val>  Update a generation parameter\n"
    )
    print(help_text)

    while True:
        try:
            text = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting tester...")
            break
        if not text:
            continue

        # Handle commands
        if text.startswith("/"):
            parts = text.split()
            cmd = parts[0]
            if cmd == "/help":
                print(help_text)
            elif cmd == "/exit":
                print("Exiting tester...")
                break
            elif cmd == "/settings":
                print("Current generation settings:")
                for k, v in gen_config.items():
                    print(f"  {k}: {v}")
            elif cmd == "/set" and len(parts) == 3:
                key, val = parts[1], parts[2]
                if key in gen_config:
                    if isinstance(gen_config[key], bool):
                        gen_config[key] = val.lower() in ("true", "1", "yes")
                    else:
                        try:
                            gen_config[key] = type(gen_config[key])(val)
                        except ValueError:
                            print(f"Invalid value for {key}: {val}")
                            continue
                    print(f"Set {key} = {gen_config[key]}")
                else:
                    print(f"Unknown parameter: {key}")
            else:
                print("Unknown command. Type /help for available commands.")
            continue

        # Tokenize and generate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_config)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(response + "\n")

        # Log conversation
        if log_f:
            log_f.write(f"User: {text}\nBot: {response}\n\n")
            log_f.flush()

if __name__ == "__main__":
    main()