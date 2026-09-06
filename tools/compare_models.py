#!/usr/bin/env python3
"""
compare_models.py

Side-by-side comparative evaluation of Multi-Participant Chat Translation across:
1. Local Models:
   - Base Untuned Qwen2.5-0.5B-Instruct (CPU)
   - Fine-Tuned Qwen2.5-0.5B-Chat-Translator (CPU)
2. Cerebras Inference API:
   - Qwen 3.8 27B (qwen-3.8-27b, 5 RPM rate limited)
3. OpenRouter:
   - InclusionAI Ling 3.0 Flash (inclusionai/ling-3.0-flash-fin:free)
   - Liquid LFM 2.5 2.6B (liquid/lfm-2.5-2.6b:free)
4. Google Gemini Suite (Multi-key round-robin pool):
   - gemini-3.1-flash-lite
   - gemini-3.5-flash-lite
   - gemini-3.5-flash
   - gemini-3.6-flash
   - gemini-3.7-flash
   - gemini-3.8-flash
   - gemma-4-26b-a4b-it
   - gemma-4-31b-it
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
- "tp" -> "teleportieren / Teleport"
- "grief" -> "griefen / zerstören"
- "elytra" -> "Elytren / Elytra"
- "raid" -> "Raid / Überfall"
- "wipe" -> "wipen / sterben / Niederlage"

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
    },
    {
        "name": "Case 6: Gamer Slang, Teleport & Griefing (EN -> DE)",
        "input": "[12/04/2021 6:42 PM] [xX_Shadow_Xx]: bro can someone tp to me? someone griefed my chest room and stole my elytra",
        "description": "Tests abbreviations (tp), gaming loanwords (griefed), items (elytra), and casual tone into German."
    },
    {
        "name": "Case 7: Server Latency & Technical Slang (DE -> EN)",
        "input": "[18.12.2024][16:05:12] [CraftingKing]: laggt der server bei euch auch so hart oder hab ich packet loss?",
        "description": "Tests German technical gaming terms (laggt, so hart, packet loss) into English."
    },
    {
        "name": "Case 8: Raid Coordination & Combat Verbs (EN -> DE)",
        "input": "[23/05/2022 11:15 AM] [Speedy_99]: afk for 5 mins, don't start the raid without me or we wipe",
        "description": "Tests gaming acronyms (afk), multiplayer mechanics (raid), and combat failure terms (wipe) into German."
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
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        )
    elapsed = (time.time() - t0) * 1000

    gen_tokens = outputs[0][input_ids.shape[1]:]
    res = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return res, elapsed


_last_cerebras_call = 0.0

async def call_cerebras(client, input_text: str, rpm: int = 450) -> Tuple[str, float]:
    global _last_cerebras_call
    min_interval = 60.0 / rpm if rpm > 0 else 0.0
    now = time.time()
    wait_time = min_interval - (now - _last_cerebras_call)
    if _last_cerebras_call > 0 and wait_time > 0.05:
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


class GeminiPool:
    """Round-robins requests across multiple API keys to maximize quota & speed."""
    def __init__(self, keys: List[str]):
        from google import genai
        self.clients = [genai.Client(api_key=k) for k in keys if k.strip()]
        self._idx = 0

    def get_client(self):
        if not self.clients:
            return None
        c = self.clients[self._idx % len(self.clients)]
        self._idx += 1
        return c


async def call_gemini(pool: GeminiPool, model_name: str, input_text: str) -> Tuple[str, float]:
    client = pool.get_client()
    if not client:
        return "[Gemini keys missing]", 0

    t0 = time.time()
    try:
        from google.genai import types
        full_prompt = f"{SYSTEM_PROMPT}\n\n[USER INPUT]:\n{input_text}"

        cfg_kwargs = {
            "temperature": 0.1,
            "max_output_tokens": 600,
        }
        # Disable reasoning where supported to eliminate latency and reasoning token overflow
        if model_name in ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash"]:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        response = await client.aio.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(**cfg_kwargs)
        )
        elapsed = (time.time() - t0) * 1000

        text = ""
        # Multi-part extraction: Google GenAI puts reasoning in parts[0] and output in subsequent parts
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            parts = [p.text for p in response.candidates[0].content.parts if hasattr(p, "text") and p.text]
            for p in reversed(parts):
                if "src:" in p.lower():
                    text = p
                    break
            else:
                text = parts[-1] if parts else ""

        if not text:
            try:
                text = response.text or ""
            except Exception:
                pass

        return clean_output(text.strip()), elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return f"[ERROR: {e}]", elapsed


async def call_openrouter(client, model_id: str, input_text: str, max_tokens: int = 300, disable_reasoning: bool = False) -> Tuple[str, float]:
    t0 = time.time()
    try:
        kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
        if disable_reasoning:
            kwargs["extra_body"] = {"reasoning": {"effort": "none"}}

        response = await client.chat.completions.create(**kwargs)
        elapsed = (time.time() - t0) * 1000
        msg = response.choices[0].message
        content = msg.content or (msg.reasoning if hasattr(msg, "reasoning") else "")
        return clean_output(content), elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return f"[ERROR: {e}]", elapsed


async def main():
    print("===================================================================================================")
    print("           Multi-Model Comparative Translation Benchmark (Full Suite)                              ")
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

    # 4. Cerebras Client
    cerebras_key = os.environ.get("CEREBRAS_API_KEY", "").strip()
    cerebras_client = AsyncOpenAI(base_url="https://api.cerebras.ai/v1", api_key=cerebras_key) if cerebras_key else None

    # 5. Gemini Pool (3 API keys load-balanced)
    gemini_keys = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
    gem_pool = GeminiPool(gemini_keys)

    gemini_models = [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it"
    ]

    print(f"--> All models initialized! Running {len(TEST_CASES)} test cases...\n")

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
        cerebras_rpm = int(os.environ.get("CEREBRAS_RPMS", "450").split(",")[0].strip())
        if cerebras_client:
            cerebras_out, cerebras_ms = await call_cerebras(cerebras_client, case['input'], rpm=cerebras_rpm)

        # 4. OpenRouter Models
        ling_out, ling_ms = ("[OpenRouter key missing]", 0)
        liquid_out, liquid_ms = ("[OpenRouter key missing]", 0)
        if or_client:
            ling_out, ling_ms = await call_openrouter(or_client, "inclusionai/ling-3.0-flash-fin:free", case['input'], max_tokens=200, disable_reasoning=True)
            liquid_out, liquid_ms = await call_openrouter(or_client, "liquid/lfm-2.5-2.6b:free", case['input'], max_tokens=600, disable_reasoning=False)

        # 5. Gemini / Gemma Models
        gemini_results: Dict[str, Tuple[str, float]] = {}
        for g_model in gemini_models:
            gemini_results[g_model] = await call_gemini(gem_pool, g_model, case['input'])

        print(f"--- [LOCAL COMPUTE] ---")
        print(f"[Base Untuned Qwen 0.5B] ({base_ms:.0f}ms):\n{base_out}\n")
        print(f"[Fine-Tuned Qwen 0.5B] ({ft_ms:.0f}ms):\n{ft_out}\n")

        print(f"--- [CEREBRAS WSE-3 CLOUD] ---")
        print(f"[Cerebras Qwen 3.8 27B] ({cerebras_ms:.0f}ms):\n{cerebras_out}\n")

        print(f"--- [OPENROUTER FREE CLOUD] ---")
        print(f"[InclusionAI Ling 3.0 Flash] ({ling_ms:.0f}ms):\n{ling_out}\n")
        print(f"[Liquid LFM 2.5 2.6B] ({liquid_ms:.0f}ms):\n{liquid_out}\n")

        print(f"--- [GOOGLE GEMINI & GEMMA SUITE (Load-Balanced Pool)] ---")
        for g_model, (g_out, g_ms) in gemini_results.items():
            print(f"[{g_model}] ({g_ms:.0f}ms):\n{g_out}\n")

    print("=" * 100)
    print("Benchmark Comparison Complete.")
    print("=" * 100)
    await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
