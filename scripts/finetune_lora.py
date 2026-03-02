from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def tokenize_function(examples, tokenizer, max_len: int):
    # 'text' já contém o chat template pronto.
    out = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA/QLoRA para Qwen2.5-7B-Instruct (sem TRL).")
    parser.add_argument("--model", default=os.path.join(PROJECT_ROOT, "models", "qwen2.5-7b"))
    parser.add_argument("--train", default=os.path.join(PROJECT_ROOT, "src", "app", "data", "training", "sft_train.jsonl"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "models", "qwen2.5-7b-lora"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--use-4bit", action="store_true", help="Ativa QLoRA 4-bit (recomendado na 3060 12GB).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    if not os.path.isfile(args.train):
        raise RuntimeError(f"Arquivo de treino não encontrado: {args.train}")

    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.float16,
        quantization_config=quant_config,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora)

    dataset = load_dataset("json", data_files=args.train, split="train")

    tokenized = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, args.max_len),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    # salva só o adapter LoRA
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print(f"OK: LoRA salvo em: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())