#!/usr/bin/env python3
"""
benchmark_latency.py

Benchmarks latency & throughput comparison on Vega 56:
- Persistent KV Cache (Zero prefill overhead)
- Cold / Stateless Evaluation (Re-evaluates full history on every turn)
"""

import sys
import time
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, "/git/finetune")
from engine import config

TEST_CONVERSATION = [
    ("tobi20", "Hallo zusammen, wie läuft der Server?"),
    ("FreeSoccerHDX", "Server läuft stabil, teste gerade die Movement-Abilities."),
    ("Moritzwied", "der Doppelsprung im Wasser hat noch einen Bug."),
    ("tobi20", "ja stimmt, der ansturm lädt sich auch beim schwimmen auf."),
    ("FreeSoccerHDX", "ich schaue mir das direkt im code an und baue einen fix."),
    ("Cpl_iPatch", "i found a fortress in the nether, anyone want to trade?"),
    ("Sweatz123", "i can trade you some diamonds at spawn."),
    ("LockiBoy", "bin auch am spawn, lass uns treffen."),
    ("tobi20", "perfekt, fix ist fast fertig getestet."),
    ("FreeSoccerHDX", "update wird in 2 minuten deployed, bitte kurz warten.")
]

SYSTEM_PROMPT = """<|im_start|>system
You are a fast real-time chat translation engine.
Room Languages: [en, de]

[DOMAIN CONTEXT & NOMENCLATURE]
Context: Minecraft Multiplayer Server (Survival & Modded Development).
Glossary:
- "LoW" -> "Legend of War"
- "Ansturm" -> "Rush / Charge ability"
- "Doppelsprung" -> "Double Jump"
- "spawn" -> "spawn area"

[PROTOCOL INSTRUCTIONS]
1. Read incoming message formatted as `[<timestamp>] [<user>]: <message>`.
2. Output detected source language on line 1: `src: <en|de>`.
3. Output translation for the other room language on line 2: `<target_lang>: <translated_message>`.<|im_end|>
"""


def benchmark():
    print("=================================================================")
    print("   Vega 56 Benchmark: Persistent KV Cache vs. Stateless Prefill  ")
    print("=================================================================")

    device = torch.device(config.DEVICE)
    print(f"--> Target GPU: {torch.cuda.get_device_name(device)}")
    print(f"--> Loading model in FP16: {config.BASE_MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=config.DEVICE,
        trust_remote_code=True
    )
    model.eval()

    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    eos_ids = [t for t in eos_ids if t is not None]

    # Warmup pass
    warmup_ids = tokenizer.encode("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(warmup_ids, max_new_tokens=10)
    torch.cuda.synchronize()

    print("\n--> Running 10 multi-turn conversation steps in both modes...\n")

    results_cached = []
    results_stateless = []

    # -------------------------------------------------------------
    # 1. PERSISTENT KV CACHE MODE
    # -------------------------------------------------------------
    print("Running Mode 1: [Persistent Session KV Cache]...")
    sys_ids = tokenizer.encode(SYSTEM_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        cache = DynamicCache()
        out_sys = model(sys_ids, past_key_values=cache, use_cache=True)
        cache = out_sys.past_key_values

    for idx, (user, text) in enumerate(TEST_CONVERSATION):
        msg_text = f"<|im_start|>user\n[14:00:{idx:02d}] [{user}]: {text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(msg_text, return_tensors="pt").to(device)

        torch.cuda.synchronize()
        t0 = time.time()
        ttft = 0.0
        generated = []

        with torch.no_grad():
            curr = input_ids
            for step in range(35):
                out = model(curr, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                if step == 0:
                    torch.cuda.synchronize()
                    ttft = time.time() - t0

                nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                tok_id = nxt.item()
                if tok_id in eos_ids:
                    break
                generated.append(tok_id)
                curr = nxt

            end_ids = tokenizer.encode("<|im_end|>\n", return_tensors="pt").to(device)
            out_end = model(end_ids, past_key_values=cache, use_cache=True)
            cache = out_end.past_key_values

        torch.cuda.synchronize()
        t_total = time.time() - t0
        cached_tokens = cache.get_seq_length()

        results_cached.append({
            "turn": idx + 1,
            "ttft_ms": ttft * 1000,
            "total_ms": t_total * 1000,
            "cached_tokens": cached_tokens,
            "gen_tokens": len(generated)
        })

    # -------------------------------------------------------------
    # 2. STATELESS / COLD PREFILL MODE (Re-evaluates full history)
    # -------------------------------------------------------------
    print("Running Mode 2: [Stateless Cold Prefill (Full History Re-evaluated)]...")
    history_text = SYSTEM_PROMPT

    for idx, (user, text) in enumerate(TEST_CONVERSATION):
        history_text += f"<|im_start|>user\n[14:00:{idx:02d}] [{user}]: {text}<|im_end|>\n<|im_start|>assistant\n"
        full_input_ids = tokenizer.encode(history_text, return_tensors="pt").to(device)
        total_prompt_tokens = full_input_ids.shape[1]

        torch.cuda.synchronize()
        t0 = time.time()
        ttft = 0.0
        generated = []

        with torch.no_grad():
            cache = DynamicCache()
            curr = full_input_ids
            for step in range(35):
                out = model(curr, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                if step == 0:
                    torch.cuda.synchronize()
                    ttft = time.time() - t0

                nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                tok_id = nxt.item()
                if tok_id in eos_ids:
                    break
                generated.append(tok_id)
                curr = nxt

        torch.cuda.synchronize()
        t_total = time.time() - t0
        gen_str = tokenizer.decode(generated, skip_special_tokens=True)
        history_text += f"{gen_str}<|im_end|>\n"

        results_stateless.append({
            "turn": idx + 1,
            "ttft_ms": ttft * 1000,
            "total_ms": t_total * 1000,
            "prompt_tokens": total_prompt_tokens,
            "gen_tokens": len(generated)
        })

    # -------------------------------------------------------------
    # SCORECARD TABLE
    # -------------------------------------------------------------
    print("\n===================================================================================================")
    print("                                      BENCHMARK SCORECARD                                          ")
    print("===================================================================================================")
    print(f"{'Turn':<6} | {'History Tokens':<15} | {'Cached TTFT':<12} | {'Stateless TTFT':<15} | {'Speedup (TTFT)':<15} | {'Total Time':<12}")
    print("---------------------------------------------------------------------------------------------------")

    for c, s in zip(results_cached, results_stateless):
        speedup = s["ttft_ms"] / max(c["ttft_ms"], 0.01)
        print(f"{c['turn']:<6} | {c['cached_tokens']:<15} | {c['ttft_ms']:>8.2f} ms | {s['ttft_ms']:>10.2f} ms   | {speedup:>12.2f}x    | {c['total_ms']:>8.2f} ms")

    print("===================================================================================================")
    avg_c_ttft = sum(r["ttft_ms"] for r in results_cached) / len(results_cached)
    avg_s_ttft = sum(r["ttft_ms"] for r in results_stateless) / len(results_stateless)
    print(f"Average Time-to-First-Token (Persistent Cache): {avg_c_ttft:.2f} ms")
    print(f"Average Time-to-First-Token (Stateless Re-eval): {avg_s_ttft:.2f} ms")
    print(f"Overall TTFT Latency Reduction: {(1 - avg_c_ttft / avg_s_ttft) * 100:.1f}% faster!")
    print("===================================================================================================\n")


if __name__ == "__main__":
    benchmark()
