#!/usr/bin/env python3
"""
compare_models.py

Side-by-side comparative evaluation of Multi-Participant Chat Translation across:
1. Base Untuned Qwen2.5-0.5B-Instruct (Local CPU)
2. Fine-Tuned Qwen2.5-0.5B-Chat-Translator (Local CPU)
3. Cerebras Qwen 3.8 27B (Cerebras Inference API, reasoning_effort=none, 5 RPM rate limited)
4. Gemini 3.5 Flash Lite (Google AI Studio)
5. InclusionAI Ling 3.0 Flash (OpenRouter: inclusionai/ling-3.0-flash-fin:free)
6. Liquid LFM 2.5 2.6B (OpenRouter: liquid/lfm-2.5-2.6b:free)
"""

import os
import sys
import re
import time
import json
import asyncio
import torch
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv("/git/finetune/.env")

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_MODEL_DIR = "/git/finetune/models/qwen2.5-0.5b-chat-translator-fp16"

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
- "dc" -> "Discord / Disconnected"
- "LK" -> "Leistungskurs / Advanced Course"

[PROTOCOL INSTRUCTIONS]
1. Read the incoming message formatted as `[<timestamp>] [<user>]: <message>`.
2. Output detected source language on line 1: `src: <en|de>`.
3. Output translation for the other room language on line 2: `<target_lang>: <translated_message>`.
4. Maintain speaker tags, tone, emojis, and exact domain glossary terms.
5. Output ONLY the two protocol lines, no introduction, no conversational filler."""

TEST_CASES = [
    {
        "name": "Case 1: Game Ability & Typos (DE -> EN)",
        "input": "[18.12.2024][13:48:50] [tobi20]: der ansturm geht im wasser",
        "description": "Tests if 'ansturm' maps to glossary 'Rush' and strict output formatting."
    },
    {
        "name": "Case 2: Colloquial Slang & Heavy Typos (DE -> EN)",
        "input": "[18.12.2024][13:49:04] [FreeSoccerHDX]: naja er sollte gehn wenn du nicht schwimsmt vlt",
        "description": "Tests normalization of 'schwimsmt', 'vlt', 'gehn'."
    },
    {
        "name": "Case 3: Gaming Terms & Item Stacks (EN -> DE)",
        "input": "[09/02/2020 2:15 AM] [Cpl_iPatch]: i have half a stack of netherite blocks at my base",
        "description": "Tests gaming terminology (stack, netherite, base) into German."
    },
    {
        "name": "Case 4: Server Acronym & Slang (DE -> EN)",
        "input": "[18.12.2024][15:10:41] [Moritzwied]: Damn das neue schwimmen is so aids für die LoW version",
        "description": "Tests acronym 'LoW' -> 'Legend of War' and slang tone."
    },
    {
        "name": "Case 5: Domain Nomenclature / School Context (DE -> EN)",
        "input": "[2026-08-23 14:32:05] [Lukas]: Hat jemand die Hausaufgaben für den Mathe-LK verstanden?",
        "description": "Tests 'LK' -> 'advanced course / Leistungskurs'."
    }
]


def clean_output(raw_text: str) -> str:
    text = raw_text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for i, l in enumerate(lines):
        if l.lower().startswith("src:"):
            return "\n".join(lines[i:])
    return text


def run_local_model(model, tokenizer, input_text: str, device: torch.device) -> Tuple[str, float]:
    history = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(history, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=48,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        )
    elapsed = (time.time() - t0) * 1000

    gen_tokens = outputs[0][input_ids.shape[1]:]
    res = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return res, elapsed


_last_cerebras_call = 0.0

async def call_cerebras(client, input_text: str) -> Tuple[str, float]:
    global _last_cerebras_call
    # Strict 5 RPM rate limit = minimum 12.5 seconds between requests
    now = time.time()
    wait_time = 12.5 - (now - _last_cerebras_call)
    if _last_cerebras_call > 0 and wait_time > 0:
        print(f"    [Cerebras Rate Limiter: Waiting {wait_time:.1f}s to respect 5 RPM limit...]")
        await asyncio.sleep(wait_time)

    t0 = time.time()
    _last_cerebras_call = time.time()
    try:
        response = await client.chat.completions.create(
            model="qwen-3.8-27b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ],
            temperature=0.1,
            max_tokens=200,
            extra_body={"reasoning_effort": "none"}
        )
        elapsed = (time.time() - t0) * 1000
        msg = response.choices[0].message
        content = msg.content or (msg.reasoning if hasattr(msg, "reasoning") else "")
        return clean_output(content), elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return f"[ERROR: {e}]", elapsed


async def call_gemini(client, model_name: str, input_text: str) -> Tuple[str, float]:
    t0 = time.time()
    try:
        from google.genai import types
        full_prompt = f"{SYSTEM_PROMPT}\n\n[USER INPUT]:\n{input_text}"
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=64
            )
        )
        elapsed = (time.time() - t0) * 1000
        return clean_output(response.text.strip()), elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return f"[ERROR: {e}]", elapsed


async def call_ling_flash(client, input_text: str) -> Tuple[str, float]:
    t0 = time.time()
    try:
        response = await client.chat.completions.create(
            model="inclusionai/ling-3.0-flash-fin:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ],
            temperature=0.1,
            max_tokens=200,
            extra_body={"reasoning": {"effort": "none"}}
        )
        elapsed = (time.time() - t0) * 1000
        msg = response.choices[0].message
        content = msg.content or (msg.reasoning if hasattr(msg, "reasoning") else "")
        return clean_output(content), elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return f"[ERROR: {e}]", elapsed


async def call_liquid_lfm(client, input_text: str) -> Tuple[str, float]:
    t0 = time.time()
    try:
        response = await client.chat.completions.create(
            model="liquid/lfm-2.5-2.6b:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ],
            temperature=0.1,
            max_tokens=600
        )
        elapsed = (time.time() - t0) * 1000
        msg = response.choices[0].message
        content = msg.content or (msg.reasoning if hasattr(msg, "reasoning") else "")
        return clean_output(content), elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return f"[ERROR: {e}]", elapsed


async def main():
    print("===================================================================================================")
    print("           Multi-Model Comparative Translation Benchmark                                           ")
    print("===================================================================================================")

    device = torch.device("cpu")
    print(f"--> Local Compute Device: {device}\n")

    # 1. Load Base Untuned Model
    print(f"--> [1/2] Loading Base Untuned Model: {BASE_MODEL_ID}...")
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    base_model.eval()

    # 2. Load Fine-Tuned Model
    print(f"--> [2/2] Loading Fine-Tuned Model: {FINETUNED_MODEL_DIR}...")
    ft_tok = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
    ft_model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_DIR, torch_dtype=torch.bfloat16).to(device)
    ft_model.eval()

    # 3. OpenRouter Client
    openrouter_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEYS", "").split(",")[0].strip()
    from openai import AsyncOpenAI
    or_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key) if openrouter_key else None

    # 4. Gemini Client
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS", "").split(",")[0].strip()
    from google import genai
    gem_client = genai.Client(api_key=gemini_key) if gemini_key else None

    # 5. Cerebras Client
    cerebras_key = os.environ.get("CEREBRAS_API_KEY", "").strip()
    cerebras_client = AsyncOpenAI(base_url="https://api.cerebras.ai/v1", api_key=cerebras_key) if cerebras_key else None

    print("--> All models initialized! Running test cases...\n")

    for case_idx, case in enumerate(TEST_CASES, 1):
        print("=" * 100)
        print(f"TEST {case_idx}: {case['name']}")
        print(f"INPUT: {case['input']}")
        print(f"FOCUS: {case['description']}")
        print("-" * 100)

        # 1. Base Untuned Model
        base_out, base_ms = run_local_model(base_model, base_tok, case['input'], device)

        # 2. Fine-Tuned Model
        ft_out, ft_ms = run_local_model(ft_model, ft_tok, case['input'], device)

        # 3. Cerebras Qwen 3.8 27B
        cerebras_out, cerebras_ms = ("[Cerebras key missing]", 0)
        if cerebras_client:
            cerebras_out, cerebras_ms = await call_cerebras(cerebras_client, case['input'])

        # 4. Gemini 3.5 Flash Lite
        gem_out, gem_ms = ("[Gemini key missing]", 0)
        if gem_client:
            gem_out, gem_ms = await call_gemini(gem_client, "gemini-3.5-flash-lite", case['input'])

        # 5. InclusionAI Ling 3.0 Flash
        ling_out, ling_ms = ("[OpenRouter key missing]", 0)
        if or_client:
            ling_out, ling_ms = await call_ling_flash(or_client, case['input'])

        # 6. Liquid LFM 2.5 2.6B
        liquid_out, liquid_ms = ("[OpenRouter key missing]", 0)
        if or_client:
            liquid_out, liquid_ms = await call_liquid_lfm(or_client, case['input'])

        print(f"[1. Base Untuned Qwen 0.5B] ({base_ms:.0f}ms):")
        print(f"{base_out}\n")

        print(f"[2. Fine-Tuned Qwen 0.5B] ({ft_ms:.0f}ms):")
        print(f"{ft_out}\n")

        print(f"[3. Cerebras Qwen 3.8 27B (Rate Limited 5 RPM)] ({cerebras_ms:.0f}ms):")
        print(f"{cerebras_out}\n")

        print(f"[4. Gemini 3.5 Flash Lite (Cloud)] ({gem_ms:.0f}ms):")
        print(f"{gem_out}\n")

        print(f"[5. InclusionAI Ling 3.0 Flash (OpenRouter)] ({ling_ms:.0f}ms):")
        print(f"{ling_out}\n")

        print(f"[6. Liquid LFM 2.5 2.6B (OpenRouter)] ({liquid_ms:.0f}ms):")
        print(f"{liquid_out}\n")

    print("=" * 100)
    print("Benchmark Comparison Complete.")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
