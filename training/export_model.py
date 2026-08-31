#!/usr/bin/env python3
"""
export_model.py

Merges trained LoRA adapter weights with base Qwen2.5-0.5B-Instruct into a
standalone FP16 Hugging Face model checkpoint for zero-overhead inference.
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER_DIR = "/git/finetune/models/lora_checkpoint/best" if os.path.exists("/git/finetune/models/lora_checkpoint/best") else "/git/finetune/models/lora_checkpoint"
DEFAULT_EXPORT_DIR = "/git/finetune/models/qwen2.5-0.5b-chat-translator-fp16"


def export(base_model_id: str, adapter_dir: str, export_dir: str):
    print("=========================================================")
    print("      Merging LoRA Adapter into Standalone Model         ")
    print("=========================================================")

    print(f"--> Base model: {base_model_id}")
    print(f"--> Adapter dir: {adapter_dir}")
    print(f"--> Output dir: {export_dir}")

    print("--> Loading base model in FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    print("--> Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("--> Merging weights...")
    merged_model = peft_model.merge_and_unload()

    print(f"--> Saving merged standalone model to {export_dir}...")
    os.makedirs(export_dir, exist_ok=True)
    merged_model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)

    print(f"[SUCCESS] Export completed successfully! Saved to {export_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export and merge LoRA adapter into standalone FP16 model.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--export-dir", type=str, default=DEFAULT_EXPORT_DIR)
    args = parser.parse_args()

    export(args.base_model, args.adapter_dir, args.export_dir)


if __name__ == "__main__":
    main()
