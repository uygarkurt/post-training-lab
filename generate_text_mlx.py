#!/usr/bin/env python3
"""
Simple text generation script using MLX models.
"""

import argparse
import json
import os

from mlx_lm import load, generate
from mlx_lm.tuner.utils import load_adapters


def read_adapter_config(adapter_dir):
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"--load-adapter requires adapter_config.json in {adapter_dir}"
        )
    with open(config_path) as f:
        return json.load(f)


def load_model(model_path, load_adapter):
    if load_adapter:
        adapters_file = os.path.join(model_path, "adapters.safetensors")
        if not os.path.isfile(adapters_file):
            raise FileNotFoundError(
                f"--load-adapter requires adapters.safetensors in {model_path}"
            )

        adapter_cfg = read_adapter_config(model_path)
        base_model = adapter_cfg.get("base_model")
        if not base_model:
            raise ValueError(
                f"adapter_config.json in {model_path} is missing 'base_model'"
            )

        print(f"Loading base model {base_model} ...")
        model, tokenizer = load(base_model)
        print(f"Loading adapters from {model_path} ...")
        load_adapters(model, model_path)
        return model, tokenizer

    print(f"Loading {model_path} ...")
    return load(model_path)


def main():
    parser = argparse.ArgumentParser(description="Generate text using MLX model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-0.5B-Instruct-MLX",
                        help="Path to MLX model or checkpoint directory")
    parser.add_argument(
        "--load-adapter", action="store_true",
        help="Load LoRA adapters from --model_path (requires adapters.safetensors + adapter_config.json)",
    )
    parser.add_argument("--prompt", type=str, 
                        default="Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
                        help="Prompt for text generation")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model_path, args.load_adapter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Apply chat template
    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Add <think> token to start generation
    formatted_prompt = formatted_prompt + "<think>\n"
    
    print("=== Formatted Prompt ===")
    print(formatted_prompt)
    
    # Generate and print output
    output = generate(model, tokenizer, prompt=formatted_prompt, verbose=False)
    print("=== Generated Output ===")
    print(output)


if __name__ == "__main__":
    main()