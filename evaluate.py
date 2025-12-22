import os
import argparse
import json
import subprocess
from typing import Optional, Tuple

from get_model_stats import get_model_params, get_model_size_gb        

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def evaluate_mmlu(path_to_model: str) -> float:
    model_dir = '"' + path_to_model + '"'
    tp_size = torch.cuda.device_count()
    
    os.environ["MODEL_PATH"] = model_dir
    os.environ["TP_SIZE"] = str(tp_size)
    os.environ["RUN_NAME"] = path_to_model.split("/")[-1] + "_results"

    subprocess.run("./lm_eval.sh", shell=True)

    with open(os.environ["RUN_NAME"], 'r', encoding='utf-8') as file:
        results = json.load(file)
    
    return results["results"]["mmlu"]["acc,none"]

def calculate_score(source_model: str, compressed_model_dir: str, original_performance: Optional[float] = None) -> Tuple[float, float]:
    original_model = AutoModelForCausalLM.from_pretrained(source_model, device_map="auto")
    original_model_size = get_model_size_gb(original_model)

    compressed_model = AutoModelForCausalLM.from_pretrained(compressed_model_dir, device_map="auto")

    compressed_model_params = get_model_params(compressed_model)
    compressed_model_size = get_model_size_gb(compressed_model)

    print(f"Total model parameters after compression: {compressed_model_params}")
    print(f"Compressed model size (Gb): {compressed_model_size}")

    compression_ratio = original_model_size / compressed_model_size

    if original_performance is None:
        original_performance = evaluate_mmlu(source_model)
    
    compressed_performance = evaluate_mmlu(compressed_model_dir)
    
    performance_drop = (original_performance - compressed_performance) / original_performance
    total_score = compression_ratio / (1 + performance_drop)

    print(f"Compression Ratio: {compression_ratio}")
    print(f"Performance drop: {performance_drop}")
    print(f"Final Score = {total_score}")

    return total_score, original_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-original-model",
        type=str,
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--path-to-compressed-model",
        type=str,
    )
    args = parser.parse_args()

    _, _ = calculate_score(args.path_to_original_model, args.path_to_compressed_model)
