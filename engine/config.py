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
MAX_SESSION_TOKENS = 4096       # Sliding window threshold per session
MAX_ACTIVE_SESSIONS = 100       # Max rooms held in VRAM/RAM
CACHE_IDLE_TIMEOUT_SECONDS = 3600  # Evict inactive rooms after 1 hour

# Server settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
