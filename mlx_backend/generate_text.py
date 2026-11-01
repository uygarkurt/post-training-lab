#!/usr/bin/env python3
"""Simple text generation entrypoint for the MLX backend."""

import argparse
import json
import os

from mlx_lm import load, generate
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import load_adapters


def generate_completions(policy, prompt_ids, prompt_attention_mask, tokenizer, max_new_tokens):
    """Greedily generate raw completion IDs, retaining EOS for validation."""
    # batch_generate returns only text; validation also needs EOS/truncation IDs.
    # BatchGenerator pads prompts itself, so remove DataLoader padding by its mask.
    prompts = [
        ids[mask.bool()].tolist()
        for ids, mask in zip(prompt_ids, prompt_attention_mask)
    ]
    generator = BatchGenerator(
        policy,
        max_tokens=max_new_tokens,
        stop_tokens=[[token] for token in tokenizer.eos_token_ids],
        prefill_batch_size=len(prompts),
        completion_batch_size=len(prompts),
    )
    try:
        uids = generator.insert(prompts)
        completions = {uid: [] for uid in uids}
        while responses := generator.next_generated():
            for response in responses:
                completions[response.uid].append(response.token)
    finally:
        generator.close()
    return [completions[uid] for uid in uids]


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
    """Parse generation options, load the model, and print its completion."""
    parser = argparse.ArgumentParser(description="Generate text using MLX model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Path to Hugging Face model or native MLX checkpoint directory",
    )
    parser.add_argument(
        "--load-adapter",
        action="store_true",
        help="Load native MLX adapters from --model_path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Randy has 60 mango trees on his farm. He also has 5 less than "
            "half as many coconut trees as mango trees. How many trees does "
            "Randy have in all on his farm?"
        ),
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--tools-json",
        type=str,
        default=None,
        help="JSON array of available function schemas",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    try:
        tools = json.loads(args.tools_json) if args.tools_json else None
    except json.JSONDecodeError as error:
        parser.error(f"--tools-json must be valid JSON: {error}")
    if tools is not None and not isinstance(tools, list):
        parser.error("--tools-json must contain a JSON array")

    model, tokenizer = load_model(args.model_path, args.load_adapter)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("=== Formatted Prompt ===")
    print(formatted_prompt)

    output = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=args.max_new_tokens,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )

    print("=== Generated Output ===")
    print(output)


if __name__ == "__main__":
    main()
