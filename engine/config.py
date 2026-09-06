"""
config.py

Configuration settings for the Persistent Session KV Cache Translation Engine.
"""

import os
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = "float16" if torch.cuda.is_available() else "bfloat16"

# Model paths: Prefer merged standalone model, then base model
MERGED_MODEL_DIR = "/git/finetune/models/qwen2.5-0.5b-chat-translator-fp16"
ADAPTER_MODEL_DIR = "/git/finetune/models/lora_checkpoint/best"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# KV Cache & Session Management
# Kept small: on CPU every cached token sits in system RAM and this host also
# runs the whole Minecraft network (~12 GiB of 14 GiB already in use).
MAX_SESSION_TOKENS = 2048       # Per-session context budget, overflow resets the session
MAX_ACTIVE_SESSIONS = 10        # Max rooms held in RAM (LRU-evicted beyond this)
CACHE_IDLE_TIMEOUT_SECONDS = 3600  # Evict inactive rooms after 1 hour

# Server settings
SERVER_HOST = "127.0.0.1"       # localhost only — only the local paper servers call this
SERVER_PORT = 8000
