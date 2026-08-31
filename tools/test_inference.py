#!/usr/bin/env python3
"""
test_inference.py

Tests the merged fine-tuned Qwen2.5-0.5B multi-participant chat translation model.
"""

import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/git/finetune/models/qwen2.5-0.5b-chat-translator-fp16"

SYSTEM_PROMPT = """You are a fast real-time chat translation engine.
Room Languages: [en, de]

[DOMAIN CONTEXT & NOMENCLATURE]
Context: Minecraft Multiplayer Server (Survival & Modded Development).
Glossary:
- "LoW" -> "Legend of War"
- "Ansturm" -> "Rush / Charge ability"
- "Doppelsprung" -> "Double Jump"
- "base" -> "Basis / Stützpunkt"
- "netherite" -> "Netherit"

[PROTOCOL INSTRUCTIONS]
1. Read the incoming message formatted as `[<timestamp>] [<user>]: <message>`.
2. Output detected source language on line 1: `src: <en|de>`.
3. Output translation for the other room language on line 2: `<target_lang>: <translated_message>`.
4. Maintain speaker tags, tone, emojis, and exact domain glossary terms."""

TEST_MESSAGES = [
    "[18.12.2024][13:48:50] [tobi20]: der ansturm geht im wasser",
    "[18.12.2024][13:49:04] [FreeSoccerHDX]: naja er sollte gehn wenn du nicht schwimsmt vlt",
    "[09/02/2020 2:15 AM] [Cpl_iPatch]: i have half a stack of netherite blocks at my base",
    "[2026-08-23 14:32:05] [Lukas]: Hat jemand die Hausaufgaben für den Mathe-LK verstanden?",
    "[2026-08-23 14:32:40] [Sarah]: Yes, exercise 3 on page 42 is about derivatives."
]


def test_model():
    print("=================================================================")
    print("       Testing Fine-Tuned Chat Translation Model                 ")
    print("=================================================================")

    print(f"--> Loading model from {MODEL_PATH} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    model.eval()

    print("\n--> Running Test Inferences:\n")

    history = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"

    for msg in TEST_MESSAGES:
        history += f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(history, return_tensors="pt")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
            )
        elapsed = (time.time() - t0) * 1000

        gen_tokens = outputs[0][input_ids.shape[1]:]
        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        print(f"INPUT:  {msg}")
        print(f"OUTPUT:\n{response_text}")
        print(f"Latency: {elapsed:.1f} ms | Tokens: {len(gen_tokens)}")
        print("-" * 65)

        history += f"{response_text}<|im_end|>\n"


if __name__ == "__main__":
    test_model()
