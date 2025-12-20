import argparse
import json

from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA on ELI5")
    parser.add_argument(
        "--path-to-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Path to the base model (local or Hugging Face hub). Default: Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lora_finetuned",
        help="Directory to save the fine-tuned model. Default: ./lora_finetuned",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sentence-transformers/eli5",
        help="Hugging Face dataset to use. Default: sentence-transformers/eli5",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs. Default: 3",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Training batch size. Default: 4"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate. Default: 2e-4"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length. Default: 512",
    )
    parser.add_argument(
        "--lora-r", type=int, default=16, help="LoRA rank (r). Default: 16"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA alpha. Default: 32"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.1, help="LoRA dropout. Default: 0.1"
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of target modules for LoRA. Default: q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.path_to_model,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.path_to_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = [m.strip() for m in args.target_modules.split(",")]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset(args.dataset, split="train")

    def format_example(example):
        return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=(device == "cuda"),
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        logging_dir=output_dir / "logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")

    config = {
        "base_model": args.path_to_model,
        "dataset": args.dataset,
        "training_args": {**training_args.to_dict()},
        "lora_config": peft_config.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
