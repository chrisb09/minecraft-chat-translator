#!/usr/bin/env python3
"""
train_lora.py

Fine-tunes Qwen2.5-0.5B-Instruct in FP16 with LoRA on AMD Vega 56 GPU (ROCm).

Features:
- Pure FP16 mixed precision training.
- Gradient checkpointing for minimal VRAM usage (< 3.0 GB).
- Strict loss masking (calculates loss only on assistant translation turns).
- Evaluates validation loss and saves PEFT checkpoints.
"""

import os
import sys
import json
import torch
import argparse
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.0"
os.environ["ROCM_TARGET_GFX_VERSION"] = "900"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TRAIN_DATA = "/git/finetune/data/train.jsonl"
DEFAULT_VAL_DATA = "/git/finetune/data/val.jsonl"
DEFAULT_OUTPUT_DIR = "/git/finetune/models/lora_checkpoint"


class ChatTranslationDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Data file not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        self.samples.append(json.loads(line))
                    except Exception:
                        pass

        print(f"--> Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        messages = sample["messages"]

        # Build full conversation tokens with loss masking on non-assistant tokens
        input_ids = []
        labels = []

        # Start with system prompt if present
        sys_msg = next((m for m in messages if m["role"] == "system"), None)
        if sys_msg:
            sys_text = f"<|im_start|>system\n{sys_msg['content']}<|im_end|>\n"
            sys_ids = self.tokenizer.encode(sys_text, add_special_tokens=False)
            input_ids.extend(sys_ids)
            labels.extend([-100] * len(sys_ids))

        # Iterate user/assistant turns
        for m in messages:
            if m["role"] == "system":
                continue
            elif m["role"] == "user":
                u_text = f"<|im_start|>user\n{m['content']}<|im_end|>\n"
                u_ids = self.tokenizer.encode(u_text, add_special_tokens=False)
                input_ids.extend(u_ids)
                labels.extend([-100] * len(u_ids))
            elif m["role"] == "assistant":
                # Prefix before generation
                header_text = "<|im_start|>assistant\n"
                header_ids = self.tokenizer.encode(header_text, add_special_tokens=False)
                input_ids.extend(header_ids)
                labels.extend([-100] * len(header_ids))

                # Assistant content (active loss calculation)
                body_text = f"{m['content']}<|im_end|>\n"
                body_ids = self.tokenizer.encode(body_text, add_special_tokens=False)
                input_ids.extend(body_ids)
                labels.extend(body_ids)

        # Truncate to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = 100 * trainable_params / all_param
    print(f"--> Trainable params: {trainable_params:,} / {all_param:,} ({pct:.2f}%)")


def train(args):
    print("=========================================================")
    print("   Qwen2.5-0.5B LoRA Fine-Tuning on Vega 56 (ROCm FP16)  ")
    print("=========================================================")

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA GPU not detected.")

    device_name = torch.cuda.get_device_name(0)
    print(f"--> Target GPU: {device_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"--> Loading base model in FP16 (eager attention): {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="eager",
        trust_remote_code=True
    )

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    train_dataset = ChatTranslationDataset(args.train_data, tokenizer, max_length=args.max_length)
    val_dataset = ChatTranslationDataset(args.val_data, tokenizer, max_length=args.max_length) if os.path.exists(args.val_data) else None

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        fp16=True,
        logging_steps=10,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("\n--> Commencing LoRA training loop...")
    trainer.train()

    print(f"\n--> Saving LoRA adapter checkpoint to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[SUCCESS] Training complete and adapter saved.")


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Qwen2.5-0.5B on Vega 56.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-data", type=str, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=str, default=DEFAULT_VAL_DATA)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
