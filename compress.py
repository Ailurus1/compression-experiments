import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize a model using AutoAWQ")
    parser.add_argument(
        "--path-to-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Path to the original model (local or Hugging Face hub). Default: Qwen/Qwen3-8B"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./compressed_model",
        help="Directory to save the quantized model. Default: ./compressed_model"
    )
    parser.add_argument(
        "--quant-config",
        type=str,
        default='{"zero_point": true, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}',
        help="Quantization configuration as a JSON string. Default: 4-bit GEMM"
    )
    parser.add_argument(
        "--calib-data",
        type=str,
        default="wikitext",
        choices=["wikitext", "dolly"],
        help="Calibration dataset: 'wikitext' or 'dolly'. Default: wikitext"
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=128,
        help="Maximum number of calibration samples. Default: 128"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_config = json.loads(args.quant_config)

    try:
        model = AutoAWQForCausalLM.from_pretrained(args.path_to_model)
        tokenizer = AutoTokenizer.from_pretrained(
            args.path_to_model, trust_remote_code=True
        )

        if args.calib_data == "wikitext":
            calib_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            calib_data = [text for text in calib_data['text'] 
                          if text.strip() != '' and len(text.split()) > 20]
        else:
            calib_data = load_dataset('databricks/databricks-dolly-15k', split='train')
            calib_data = [
                f"{example['instruction']}\n{example['context']}\n{example['response']}"
                for example in calib_data
            ]
        calib_data = calib_data[:args.max_calib_samples]

        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_data,
            max_calib_samples=args.max_calib_samples
        )

        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)

        with open(output_dir / "quant_config.json", "w", encoding="utf-8") as f:
            json.dump(quant_config, f, indent=2)

        print(f"Model saved to: {output_dir.absolute()}")

    except ImportError:
        print("Error: AutoAWQ is not installed")
        sys.exit(1)
    except Exception as e:
        print(f"Error during quantization: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
