import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from lighteval.models.transformers.transformers_model import (
    TransformersModel,
    TransformersModelConfig,
)
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

def get_model_size_gb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_gb = (param_size + buffer_size) / (1024**3)
    return size_all_gb

def get_model_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Evaluate models using")
    parser.add_argument(
        "--path-to-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Path to the model (local or Hugging Face hub). Default: Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--tasks", type=str, default="mmlu", help="Task to evaluate on. Default: mmlu"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for debugging only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.path_to_model, device_map="auto")

    print(f"Total parameters: {get_model_params(model)}")
    print(f"Total model size (Gb): {get_model_size_gb(model)}")

    config = TransformersModelConfig(model_name=args.path_to_model, batch_size=1)
    model = TransformersModel.from_model(model, config)

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE, max_samples=args.limit
    )

    pipeline = Pipeline(
        model=model,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=EvaluationTracker(output_dir=output_dir),
        tasks=args.tasks,
    )

    pipeline.evaluate()
    pipeline.show_results()

    result_dict = pipeline.get_results()
    with open(Path(args.output_dit) / 'results.json', 'w', encoding="utf-8") as file:
        json.dump(result_dict, file)


if __name__ == "__main__":
    main()