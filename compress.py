import argparse
import subprocess
from pathlib import Path

from get_model_stats import get_model_params, get_model_size_gb
from evaluate import calculate_score

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot


compression_type_to_recipes = {
    "GPTQ:W4A16": [GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])],
    "GPTQ:W8A8": [GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"])],
    "SMQ+GPTQ:W4A16": [SmoothQuantModifier(smoothing_strength=0.8),GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"]),]
}

def perform_compression(path_to_model: str, compression_type: str, output_dir: str) -> None:
    if "BNB" not in compression_type:
        recipe = compression_type_to_recipes[compression_type]

        oneshot(
            model=path_to_model,
            dataset="open_platypus",
            recipe=recipe,
            output_dir=output_dir,
            max_seq_length=256,
            num_calibration_samples=256,
        )
    else:
        if "INT8" in compression_type:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                                 llm_int8_enable_fp32_cpu_offload=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

        tokenizer = AutoTokenizer.from_pretrained(path_to_model)

        model8bit = AutoModelForCausalLM.from_pretrained(
            path_to_model,
            dtype="auto",
            device_map="auto",
            quantization_config=quantization_config
        )

        model8bit.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-model",
        type=str,
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["GPTQ:W4A16", "GPTQ:W8A8", "BNB:INT8", "BNB:INT4", "SMQ+GPTQ:W4A16", "all"],
        default='all',
    )
    args = parser.parse_args()

    output_dir = Path(f"Qwen3-8B-{args.type.replace(':', '_')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir)

    model = AutoModelForCausalLM.from_pretrained(args.path_to_model, device_map="auto")

    original_model_params = get_model_params(model)
    original_model_size = get_model_size_gb(model)

    print(f"Total model parameters: {original_model_params}")
    print(f"Model size (Gb): {original_model_size}")

    if args.type == "all":
        modifications = list(compression_type_to_recipes.keys())
    else:
        modifications = [args.type]

    original_performance = None
    for modification in modifications:
        print(f"Performing model compression using {modification}...")
        perform_compression(args.path_to_model, modification, output_path)

        _, original_performance = calculate_score(args.path_to_model, output_path, original_performance=original_performance)


if __name__ == "__main__":
    main()
