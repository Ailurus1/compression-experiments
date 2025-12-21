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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.path_to_model, device_map="auto")

    config = TransformersModelConfig(model_name=args.path_to_model, batch_size=1)
    model = TransformersModel.from_model(model, config)

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE, max_samples=args.limit
    )

    pipeline = Pipeline(
        model=model,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=EvaluationTracker(output_dir=output_dir),
        tasks=args.tasks,
    )

    results = pipeline.evaluate()

    for task_name, task_results in results.items():
        if "results" in task_results:
            print(f"\n{task_name}:")
            for metric, value in task_results["results"].items():
                print(f"  {metric}: {value}")
        else:
            print(f"\n{task_name}: No results available")

    results_file = output_dir / "results.json"

    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Results are saved in: {results_file}")


if __name__ == "__main__":
    main()
