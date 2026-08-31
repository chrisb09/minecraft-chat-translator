#!/usr/bin/env python3
"""
chat_cli.py

Interactive terminal simulator for Multi-Participant Chat Translation on Vega 56.
Allows typing messages from different users and inspects real-time translations & latency.
"""

import sys
import os
import time
from datetime import datetime

# Add root directory to sys.path
sys.path.insert(0, "/git/finetune")

from engine.session_kv_manager import SessionKVManager


def print_banner():
    print("=================================================================")
    print("   Vega 56 Persistent KV-Cache Chat Translation Terminal        ")
    print("=================================================================")
    print("Commands:")
    print("  /user <name>    - Switch active speaker username (default: tobi20)")
    print("  /domain <text>  - Set room domain context")
    print("  /glossary       - Add/View domain glossary terms")
    print("  /stats          - Show GPU VRAM and cached tokens")
    print("  /reset          - Clear conversation KV cache")
    print("  /exit           - Quit CLI")
    print("=================================================================\n")


import argparse

def main():
    parser = argparse.ArgumentParser(description="Interactive Chat Translation Terminal with Persistent KV Cache.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Execution device: auto (detects Vega GPU or CPU), cpu, or cuda")
    args = parser.parse_args()

    print_banner()

    session_id = "cli_room_demo"
    active_user = "tobi20"
    domain = "Minecraft Multiplayer & Mod Development"
    glossary = {
        "LoW": "Legend of War",
        "Ansturm": "Rush / Charge ability",
        "Doppelsprung": "Double Jump",
        "vanilla": "unmodded Minecraft",
        "base": "player home",
        "dc": "Discord or disconnected"
    }

    print(f"--> Initializing Session Engine (device mode: {args.device})...")
    manager = SessionKVManager(device_override=args.device)
    manager.create_session(
        session_id=session_id,
        domain=domain,
        glossary=glossary,
        room_languages=["en", "de"]
    )
    print("--> Engine ready! Start chatting below:\n")

    while True:
        try:
            prompt_str = f"[{active_user}] > "
            user_input = input(prompt_str).strip()

            if not user_input:
                continue

            if user_input.startswith("/exit") or user_input.startswith("/quit"):
                print("\nExiting chat. Goodbye!")
                break

            if user_input.startswith("/user "):
                active_user = user_input.split(" ", 1)[1].strip()
                print(f"--> Switched active speaker to: {active_user}\n")
                continue

            if user_input.startswith("/stats"):
                stats = manager.get_stats()
                print("\n--- GPU & Engine Stats ---")
                print(f"Allocated VRAM: {stats['allocated_vram_mb']} MB")
                print(f"Reserved VRAM:  {stats['reserved_vram_mb']} MB")
                print(f"Active Sessions: {stats['active_sessions']}")
                for s in stats["session_list"]:
                    print(f"  Session '{s['session_id']}': {s['cached_tokens']} cached tokens, {s['total_turns']} turns")
                print("--------------------------\n")
                continue

            if user_input.startswith("/reset"):
                manager.reset_session(session_id)
                manager.create_session(session_id=session_id, domain=domain, glossary=glossary)
                print("--> Session KV Cache wiped and reinitialized.\n")
                continue

            timestamp = datetime.now().strftime("%H:%M:%S")

            # Execute Turn via Persistent KV Cache
            result = manager.process_turn(
                session_id=session_id,
                username=active_user,
                timestamp=timestamp,
                message=user_input
            )

            # Display translation results
            print(f"    └── [Detected: {result['src_lang'].upper()}]")
            for lang, trans_text in result["translations"].items():
                print(f"    └── [{lang.upper()}]: {trans_text}")

            print(f"    [Metrics: TTFT: {result['ttft_ms']}ms | Total: {result['total_time_ms']}ms | Speed: {result['tokens_per_second']} tok/s | Cached: {result['total_cached_tokens']} tokens]\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break


if __name__ == "__main__":
    main()
