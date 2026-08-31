#!/usr/bin/env python3
"""
test_api_keys.py

Verifies all configured API keys and models in .env:
- Tests each Google AI Studio key across Gemma & Gemini models.
- Tests OpenRouter key with stealth/ox-alpha.
- Masks API keys for security in output logs.
"""

import os
import sys
import time
import json
import asyncio
from typing import Tuple, List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv("/git/finetune/.env")

sys.path.insert(0, "/git/finetune")
from data.translate_corpus import parse_keys_from_env, clean_json_response


def mask_key(k: str) -> str:
    if len(k) <= 8:
        return "****"
    return f"...{k[-6:]}"


async def test_gemini_model(client, model_name: str) -> Tuple[bool, str, float]:
    from google.genai import types
    prompt = """You are a translator. Translate this message:
[0] [User]: Hallo Welt
Output strict JSON: {"0": {"clean_source_text": "Hallo Welt", "target_text": "Hello World"}}"""
    t0 = time.time()
    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        elapsed = (time.time() - t0) * 1000
        text = response.text.strip()
        data = clean_json_response(text)
        if "0" in data:
            return True, f"OK ({elapsed:.0f}ms) -> {data['0'].get('target_text', '')}", elapsed
        return True, f"OK ({elapsed:.0f}ms) -> {text[:60]}", elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return False, str(e), elapsed


async def test_openrouter_model(client, model_name: str) -> Tuple[bool, str, float]:
    prompt = """Translate this message: [0] [User]: Hallo Welt
Output strict JSON: {"0": {"clean_source_text": "Hallo Welt", "target_text": "Hello World"}}"""
    t0 = time.time()
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a translator that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        elapsed = (time.time() - t0) * 1000
        raw_text = response.choices[0].message.content.strip()
        data = clean_json_response(raw_text)
        if "0" in data:
            return True, f"OK ({elapsed:.0f}ms) -> {data['0'].get('target_text', '')}", elapsed
        return True, f"OK ({elapsed:.0f}ms) -> {raw_text[:60]}", elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return False, str(e), elapsed


async def main():
    print("=================================================================")
    print("           API Key & Model Capability Verification               ")
    print("=================================================================")

    g_keys, o_keys = parse_keys_from_env()
    print(f"--> Found {len(g_keys)} Google AI Studio key(s)")
    print(f"--> Found {len(o_keys)} OpenRouter key(s)\n")

    # Test the exact Gemma & Gemini models
    test_models = [
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite"
    ]

    # 1. Test Google AI Studio Keys & Models
    if g_keys:
        from google import genai
        print("-----------------------------------------------------------------")
        print("  GOOGLE AI STUDIO KEYS & MODELS                                 ")
        print("-----------------------------------------------------------------")
        for k_idx, key in enumerate(g_keys):
            masked = mask_key(key)
            print(f"\n[Key #{k_idx+1} ({masked})]")
            try:
                client = genai.Client(api_key=key)
            except Exception as e:
                print(f"  [ERROR] Client init failed: {e}")
                continue

            for m_name in test_models:
                ok, msg, ms = await test_gemini_model(client, m_name)
                status_tag = "[PASS]" if ok else "[FAIL]"
                print(f"  {status_tag} Model '{m_name}': {msg}")

    # 2. Test OpenRouter Keys & Models
    if o_keys:
        from openai import AsyncOpenAI
        print("\n-----------------------------------------------------------------")
        print("  OPENROUTER KEYS & MODELS                                       ")
        print("-----------------------------------------------------------------")
        for k_idx, key in enumerate(o_keys):
            masked = mask_key(key)
            print(f"\n[OpenRouter Key #{k_idx+1} ({masked})]")
            try:
                client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
            except Exception as e:
                print(f"  [ERROR] Client init failed: {e}")
                continue

            ok, msg, ms = await test_openrouter_model(client, "stealth/ox-alpha")
            status_tag = "[PASS]" if ok else "[FAIL]"
            print(f"  {status_tag} Model 'stealth/ox-alpha': {msg}")

    print("\n=================================================================")
    print("                  Verification Completed                         ")
    print("=================================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
