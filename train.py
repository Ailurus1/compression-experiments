from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import glob
import os
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='cofofprom/QWEN3-8B-BnB-8bit')
    parser.add_argument('--mmlu_train_path', type=str, default='./data/auxiliary_train')
    parser.add_argument('--n_train_samples', type=int, default=1000)
    parser.add_argument('--train_outdir', type=str, default='./qwen3-peft-mmlu')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        load_in_8bit=True,
    )
    
    model.config.use_cache = False
    
    model = prepare_model_for_kbit_training(model)
    model.print_trainable_parameters()
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
    )
    
    model = get_peft_model(model, lora_config)
    
    train_files = glob.glob(f"{args.mmlu_train_path}/*.csv")
    print("Number of train files:", len(train_files))
    
    
    dataset = load_dataset(
        "csv",
        data_files=train_files,
        column_names=['question', 'A', 'B', 'C', 'D', 'answer'],
        header=None
    )
    
    train_dataset = dataset
    train_dataset['train'] = dataset["train"].shuffle(seed=42).select(range(args.n_train_samples))
    
    
    def tokenize_mmlu(example):
        prompt = (
            "Answer the multiple-choice question.\n\n"
            f"Question: {example['question']}\n"
            f"A. {example['A']}\n"
            f"B. {example['B']}\n"
            f"C. {example['C']}\n"
            f"D. {example['D']}\n\n"
            "Answer:"
        )
    
        answer = " " + example["answer"]
    
        full_text = prompt + answer
    
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=512,
        )
    
        labels = tokenized["input_ids"].copy()
    
        prompt_len = len(
            tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
        )
    
        labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels
    
        return tokenized
    
    tokenized_train = train_dataset['train'].map(
        tokenize_mmlu,
        remove_columns=train_dataset['train'].column_names,
        batched=False
    )
    
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    
    
    training_args = TrainingArguments(
        output_dir=args.train_outdir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator
    )
    
    trainer.train()
