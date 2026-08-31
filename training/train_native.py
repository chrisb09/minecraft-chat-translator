#!/usr/bin/env python3
"""
train_native.py

Pure PyTorch Native LoRA Training Loop for Qwen2.5-0.5B on AMD Vega 56 (ROCm FP16).
Provides real-time unbuffered progress, precise memory tracking, clean validation loss evaluation,
and rock-solid execution without Trainer overhead.
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# Target ROCm variables
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.0"
os.environ["ROCM_TARGET_GFX_VERSION"] = "900"
os.environ["MIOPEN_DEBUG_DISABLE_AMD_ASM_KERNELS"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_DATA = "/git/finetune/data/train.jsonl"
VAL_DATA = "/git/finetune/data/val.jsonl"
OUTPUT_DIR = "/git/finetune/models/lora_checkpoint"


class ChatDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        self.data.append(json.loads(line))
                    except Exception:
                        pass
        print(f"--> Loaded {len(self.data):,} conversations from {jsonl_path}", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        messages = sample["messages"]

        input_ids = []
        labels = []

        sys_msg = next((m for m in messages if m["role"] == "system"), None)
        if sys_msg:
            sys_text = f"<|im_start|>system\n{sys_msg['content']}<|im_end|>\n"
            sys_ids = self.tokenizer.encode(sys_text, add_special_tokens=False)
            input_ids.extend(sys_ids)
            labels.extend([-100] * len(sys_ids))

        for m in messages:
            if m["role"] == "system":
                continue
            elif m["role"] == "user":
                u_text = f"<|im_start|>user\n{m['content']}<|im_end|>\n"
                u_ids = self.tokenizer.encode(u_text, add_special_tokens=False)
                input_ids.extend(u_ids)
                labels.extend([-100] * len(u_ids))
            elif m["role"] == "assistant":
                h_text = "<|im_start|>assistant\n"
                h_ids = self.tokenizer.encode(h_text, add_special_tokens=False)
                input_ids.extend(h_ids)
                labels.extend([-100] * len(h_ids))

                b_text = f"{m['content']}<|im_end|>\n"
                b_ids = self.tokenizer.encode(b_text, add_special_tokens=False)
                input_ids.extend(b_ids)
                labels.extend(b_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def format_time(seconds: float) -> str:
    s = int(seconds)
    hrs = s // 3600
    mins = (s % 3600) // 60
    secs = s % 60
    if hrs > 0:
        return f"{hrs:02d}h:{mins:02d}m:{secs:02d}s"
    return f"{mins:02d}m:{secs:02d}s"


def evaluate_val_loss(model, val_loader, device, max_eval_batches: int = 100) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= max_eval_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            count += 1
    model.train()
    return total_loss / max(1, count)


def main():
    print("=================================================================", flush=True)
    print("    Pure PyTorch Native LoRA Training (Vega 56 ROCm FP16)        ", flush=True)
    print("=================================================================", flush=True)

    device = torch.device("cuda:0")
    print(f"--> Target Device: {torch.cuda.get_device_name(device)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"--> Loading base model in FP16 (eager attention)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="eager",
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = ChatDataset(TRAIN_DATA, tokenizer, max_length=1024)
    val_dataset = ChatDataset(VAL_DATA, tokenizer, max_length=1024)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Hyperparameters
    epochs = 1
    grad_accum_steps = 8
    lr = 2e-4
    eval_interval = 50  # evaluate every 50 optimizer steps
    save_interval = 100

    total_optimizer_steps = (len(train_loader) * epochs) // grad_accum_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine scheduler with warmup
    warmup_steps = int(total_optimizer_steps * 0.05)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"--> Total Optimization Steps: {total_optimizer_steps:,} (Epochs: {epochs}, GradAccum: {grad_accum_steps})", flush=True)
    print(f"--> Warmup Steps: {warmup_steps} | Initial Evaluation...", flush=True)

    init_val_loss = evaluate_val_loss(model, val_loader, device, max_eval_batches=50)
    print(f"--> Initial Zero-Shot Val Loss: {init_val_loss:.4f}\n", flush=True)
    print("----------------------------------------------------------------------------------------------------------------", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = init_val_loss

    model.train()
    optimizer.zero_grad()

    global_step = 0
    running_loss = 0.0
    step_tokens = 0
    t_start = time.time()
    t_last_log = time.time()

    for epoch in range(1, epochs + 1):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            step_tokens += input_ids.shape[1]

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            running_loss += outputs.loss.item()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Periodic Logging (every 5 steps)
                if global_step % 5 == 0 or global_step == total_optimizer_steps:
                    t_now = time.time()
                    elapsed = t_now - t_start
                    dt_step = t_now - t_last_log
                    t_last_log = t_now

                    avg_step_loss = running_loss / (5 * grad_accum_steps)
                    running_loss = 0.0

                    tok_sec = (step_tokens / max(0.1, dt_step))
                    step_tokens = 0

                    cur_lr = scheduler.get_last_lr()[0]
                    pct = (global_step / total_optimizer_steps) * 100.0
                    eta_sec = (total_optimizer_steps - global_step) * (elapsed / max(1, global_step))

                    vram_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

                    print(
                        f"Step [{global_step:>4}/{total_optimizer_steps:<4}] ({pct:>5.1f}%) | "
                        f"Loss: {avg_step_loss:>6.4f} | "
                        f"LR: {cur_lr:.2e} | "
                        f"Speed: {tok_sec:>5.0f} tok/s | "
                        f"VRAM: {vram_mb:>5.0f}MB | "
                        f"Elapsed: {format_time(elapsed)} | "
                        f"ETA: {format_time(eta_sec)}",
                        flush=True
                    )

                # Periodic Validation Evaluation (every eval_interval)
                if global_step % eval_interval == 0:
                    t_eval_0 = time.time()
                    val_loss = evaluate_val_loss(model, val_loader, device, max_eval_batches=80)
                    t_eval_elapsed = time.time() - t_eval_0

                    is_best = val_loss < best_val_loss
                    best_tag = " [NEW BEST]" if is_best else ""
                    print(f"\n    >>> [VAL EVAL @ Step {global_step}] Val Loss: {val_loss:.4f}{best_tag} (Eval Time: {t_eval_elapsed:.1f}s)\n", flush=True)

                    if is_best:
                        best_val_loss = val_loss
                        best_dir = os.path.join(OUTPUT_DIR, "best")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)

                # Periodic Checkpoint Saving
                if global_step % save_interval == 0 or global_step == total_optimizer_steps:
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

    print("\n=================================================================", flush=True)
    print("                       TRAINING COMPLETE                         ", flush=True)
    print(f"Total Time Taken:  {format_time(time.time() - t_start)}", flush=True)
    print(f"Final Val Loss:    {best_val_loss:.4f}", flush=True)
    print(f"Saving final adapter to {OUTPUT_DIR}...", flush=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[SUCCESS] Model fine-tuning complete and saved!", flush=True)
    print("=================================================================\n", flush=True)


if __name__ == "__main__":
    main()
