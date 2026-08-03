import argparse

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path, load_adapter):
    """Load a Hugging Face model, optionally with a PEFT adapter."""
    if load_adapter:
        adapter_config = PeftConfig.from_pretrained(model_path)
        base_model = adapter_config.base_model_name_or_path
        if not base_model:
            raise ValueError(
                f"adapter_config.json in {model_path} is missing "
                "'base_model_name_or_path'"
            )

        print(f"Loading base model {base_model} ...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        print(f"Loading adapter from {model_path} ...")
        model = PeftModel.from_pretrained(model, model_path)
        return model, tokenizer

    print(f"Loading {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def main():
    """Parse generation options, load the model, and print its completion."""
    parser = argparse.ArgumentParser(description="Generate text using CUDA model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Path to Hugging Face model or checkpoint directory",
    )
    parser.add_argument(
        "--load-adapter",
        action="store_true",
        help="Load a PEFT adapter from --model_path",
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
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.load_adapter)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("=== Formatted Prompt ===")
    print(formatted_prompt)

    model_inputs = tokenizer(formatted_prompt, return_tensors="pt")
    model_inputs = {
        name: tensor.to("cuda")
        for name, tensor in model_inputs.items()
    }
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = model_inputs["input_ids"].shape[1]
    completion_ids = generated_ids[0, prompt_length:]
    output = tokenizer.decode(completion_ids, skip_special_tokens=True)

    print("=== Generated Output ===")
    print(output)


if __name__ == "__main__":
    main()
