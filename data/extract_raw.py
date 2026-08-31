#!/usr/bin/env python3
"""
extract_raw.py

Extracts raw chat conversations from:
1. rawdata/protocol.txt (German server development & gameplay chat)
2. rawdata/minecraft-server-chat_clean.json (English multiplayer chat)

Filters low-quality spam/keyboard mashing and clusters messages into cohesive
multi-turn conversation threads.
"""

import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

RAW_DIR = "/git/finetune/rawdata"
OUTPUT_FILE = "/git/finetune/data/raw_extracted_threads.jsonl"

MAX_TIME_GAP_SECONDS = 15 * 60  # 15 minutes max gap between messages in same thread
MIN_THREAD_MSGS = 4
MAX_THREAD_MSGS = 14

GLOSSARY_PROTOCOL_DE = {
    "LoW": "Legend of War (Server/Mod Name)",
    "Ansturm": "Rush / Charge ability",
    "Doppelsprung": "Double Jump ability",
    "vanilla": "unmodded Minecraft",
    "dc": "Discord or disconnected",
    "velocity": "movement velocity / speed vector",
    "healthbar": "health display bar",
    "spawn": "spawn area / starting point",
    "haider": "hater / troll"
}

GLOSSARY_MINECRAFT_EN = {
    "base": "player home / settlement",
    "netherite": "netherite gear / material",
    "stack": "stack of 64 items",
    "cap": "lie / false claim",
    "grief": "destroying other players' builds",
    "tp": "teleport",
    "afk": "away from keyboard",
    "spawn": "world spawn area",
    "pvp": "player versus player combat"
}

# Regex to detect key mashing / test spam like "d", "tt", "awd", "asdf", "dfg", "xcv"
KEYMASH_PATTERN = re.compile(r"^[wasdfxcvjkltqyuiopghz1234567890\.\^\?\:\/\-_ ]{1,3}$", re.IGNORECASE)


def is_meaningful_content(content: str) -> bool:
    text = content.strip()
    if not text:
        return False
    if text.startswith("./") or text.startswith("/"):
        return False
    if text.startswith("<!>") or "joined the game" in text.lower() or "left the game" in text.lower():
        return False
    # If the text is purely 1-3 characters of keymash
    if KEYMASH_PATTERN.match(text) and text.lower() not in {"hi", "yo", "ok", "gg", "np", "ty", "wb", "bb", "ja", "no", "xd"}:
        return False
    # Check if string has at least some letter or meaningful punctuation
    if not re.search(r"[a-zA-ZäöüÄÖÜß0-9]", text):
        return False
    return True


def parse_protocol_line(line: str) -> Optional[Dict[str, Any]]:
    pattern = r"^\[(\d{2}\.\d{2}\.\d{4})\]\[(\d{2}:\d{2}:\d{2})\]\s+([^:]+):\s*(.*)$"
    match = re.match(pattern, line.strip())
    if not match:
        return None

    date_str, time_str, user, content = match.groups()
    user = user.strip()
    content = content.strip()

    if not is_meaningful_content(content):
        return None

    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M:%S")
        ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

    return {
        "timestamp": ts_str,
        "dt": dt,
        "username": user,
        "content": content
    }


def parse_json_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = item.get("content", "").strip()
    user = item.get("username", "").strip()
    date_raw = item.get("date", "").strip()

    if not user or not date_raw or not is_meaningful_content(content):
        return None

    dt = None
    for fmt in ["%m/%d/%Y %I:%M %p", "%d/%m/%Y %I:%M %p", "%Y-%m-%d %H:%M:%S"]:
        try:
            dt = datetime.strptime(date_raw, fmt)
            break
        except Exception:
            continue

    if dt is None:
        return None

    return {
        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "dt": dt,
        "username": user,
        "content": content
    }


def chunk_messages_into_threads(
    messages: List[Dict[str, Any]],
    source_prefix: str,
    source_lang: str,
    domain_name: str,
    glossary: Dict[str, str],
    max_threads: Optional[int] = None
) -> List[Dict[str, Any]]:
    if not messages:
        return []

    messages.sort(key=lambda m: m["dt"])

    threads = []
    current_thread_msgs = []
    last_dt = None

    def flush_thread():
        nonlocal current_thread_msgs
        if len(current_thread_msgs) >= MIN_THREAD_MSGS:
            cleaned_msgs = []
            for idx, msg in enumerate(current_thread_msgs[:MAX_THREAD_MSGS]):
                cleaned_msgs.append({
                    "id": idx,
                    "timestamp": msg["timestamp"],
                    "username": msg["username"],
                    "content": msg["content"]
                })
            threads.append({
                "thread_id": f"{source_prefix}_{len(threads):05d}",
                "source_file": source_prefix,
                "source_lang": source_lang,
                "domain": domain_name,
                "glossary": glossary,
                "messages": cleaned_msgs
            })
        current_thread_msgs = []

    for msg in messages:
        if max_threads and len(threads) >= max_threads:
            break

        if last_dt is None:
            current_thread_msgs.append(msg)
            last_dt = msg["dt"]
            continue

        time_gap = (msg["dt"] - last_dt).total_seconds()

        if time_gap > MAX_TIME_GAP_SECONDS or len(current_thread_msgs) >= MAX_THREAD_MSGS:
            flush_thread()
            current_thread_msgs = [msg]
        else:
            current_thread_msgs.append(msg)

        last_dt = msg["dt"]

    flush_thread()
    return threads


def extract_protocol_txt(path: str) -> List[Dict[str, Any]]:
    print(f"--> Parsing protocol.txt: {path}")
    raw_msgs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_protocol_line(line)
            if parsed:
                raw_msgs.append(parsed)

    print(f"    Extracted {len(raw_msgs)} clean messages from protocol.txt")
    threads = chunk_messages_into_threads(
        raw_msgs,
        source_prefix="protocol_de",
        source_lang="de",
        domain_name="Minecraft Server & Plugin Development",
        glossary=GLOSSARY_PROTOCOL_DE
    )
    print(f"    Constructed {len(threads)} clean German conversation threads.")
    return threads


def extract_minecraft_json(path: str, max_threads: int = 500) -> List[Dict[str, Any]]:
    print(f"--> Parsing minecraft-server-chat_clean.json: {path}")
    raw_msgs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"    Error reading JSON: {e}")
            return []

    print(f"    Loaded {len(data):,} raw JSON items. Filtering and parsing...")
    for item in data:
        parsed = parse_json_item(item)
        if parsed:
            raw_msgs.append(parsed)
            if len(raw_msgs) >= max_threads * 20:
                break

    print(f"    Extracted {len(raw_msgs)} clean messages from JSON.")
    threads = chunk_messages_into_threads(
        raw_msgs,
        source_prefix="minecraft_en",
        source_lang="en",
        domain_name="Minecraft Multiplayer Survival & PvP",
        glossary=GLOSSARY_MINECRAFT_EN,
        max_threads=max_threads
    )
    print(f"    Constructed {len(threads)} clean English conversation threads.")
    return threads


def main():
    protocol_path = os.path.join(RAW_DIR, "protocol.txt")
    json_path = os.path.join(RAW_DIR, "minecraft-server-chat_clean.json")

    all_threads = []

    if os.path.exists(protocol_path):
        de_threads = extract_protocol_txt(protocol_path)
        all_threads.extend(de_threads)

    if os.path.exists(json_path):
        en_threads = extract_minecraft_json(json_path, max_threads=400)
        all_threads.extend(en_threads)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for thread in all_threads:
            f.write(json.dumps(thread, ensure_ascii=False) + "\n")

    print(f"\n[SUCCESS] Extracted a total of {len(all_threads)} high-quality conversation threads into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
