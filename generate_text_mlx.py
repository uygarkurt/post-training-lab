#!/usr/bin/env python3
"""
Simple text generation script using MLX models.
"""

import argparse
from mlx_lm import load, generate


def main():
    parser = argparse.ArgumentParser(description="Generate text using MLX model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-0.5B-Instruct-MLX",
                        help="Path to MLX model (default: Qwen/Qwen2-0.5B-Instruct-MLX)")
    parser.add_argument("--prompt", type=str, 
                        default="Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
                        help="Prompt for text generation")
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load(args.model_path)
    
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