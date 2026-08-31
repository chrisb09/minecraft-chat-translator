#!/usr/bin/env python3
"""
synthesize_training_data.py

Converts the parallel conversation corpus (data/unified_conversations.jsonl)
into rich multi-turn training data formatted in Qwen2.5 ChatML with:

1. Self-Language Normalization & Clean Bidirectional Pairing:
   - Accepts raw user typos as input, but strictly outputs clean canonical target text.
2. Dataset Weighting & Resampling:
   - Configurable weight multipliers for protocol_de vs minecraft_en.
3. Username Permutations:
   - Samples real usernames from rawdata/usernames.txt and updates in-text mentions.
4. Timestamp / Date Shifting:
   - Generates temporal offsets while maintaining relative second-deltas between turns.
5. Scenario Diversity:
   - Mono-German, Mono-English, Per-User assigned languages, Dynamic Code-Switching.

Outputs:
- data/train.jsonl (90%)
- data/val.jsonl (10%)
"""

import os
import re
import json
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

UNIFIED_CORPUS_FILE = "/git/finetune/data/unified_conversations.jsonl"
USERNAMES_FILE = "/git/finetune/rawdata/usernames.txt"
TRAIN_FILE = "/git/finetune/data/train.jsonl"
VAL_FILE = "/git/finetune/data/val.jsonl"

SYSTEM_PROMPT_TEMPLATE = """You are a fast real-time chat translation engine.
Room Languages: [en, de]

[DOMAIN CONTEXT & NOMENCLATURE]
Context: {domain}
Glossary:
{glossary_str}

[PROTOCOL INSTRUCTIONS]
1. Read the incoming message formatted as `[<timestamp>] [<user>]: <message>`.
2. Output detected source language on line 1: `src: <en|de>`.
3. Output translation for the other room language on line 2: `<target_lang>: <translated_message>`.
4. Maintain speaker tags, tone, emojis, and exact domain glossary terms."""


class UsernamePool:
    def __init__(self, filepath: str, max_pool_size: int = 50000):
        self.pool = []
        if os.path.exists(filepath):
            print(f"--> Loading username pool from {filepath}...")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    u = line.strip()
                    if u and len(u) >= 3 and len(u) <= 20 and re.match(r"^[a-zA-Z0-9_\-]+$", u):
                        self.pool.append(u)
                        if len(self.pool) >= max_pool_size:
                            break
            print(f"    Loaded {len(self.pool):,} clean usernames into memory.")
        else:
            print(f"[WARN] Usernames file not found: {filepath}. Using fallback names.")
            self.pool = ["ShadowHunter", "CyberKnight", "PixelCrafter", "VortexGamer", "LunaStar", "IronGamer99"]

    def sample_names(self, count: int) -> List[str]:
        if len(self.pool) < count:
            return [f"Player_{i}" for i in range(count)]
        return random.sample(self.pool, count)


def format_glossary(glossary: Dict[str, str]) -> str:
    if not glossary:
        return "- None"
    return "\n".join([f"- \"{k}\" -> \"{v}\"" for k, v in glossary.items()])


def build_system_message(domain: str, glossary: Dict[str, str]) -> Dict[str, str]:
    return {
        "role": "system",
        "content": SYSTEM_PROMPT_TEMPLATE.format(
            domain=domain,
            glossary_str=format_glossary(glossary)
        )
    }


def replace_usernames_in_conversation(conv: Dict[str, Any], pool: UsernamePool) -> Dict[str, Any]:
    """
    Substitutes speaker usernames and systematically rewrites in-text mentions.
    """
    original_users = list({m["username"] for m in conv["messages"]})
    new_users = pool.sample_names(len(original_users))
    user_map = {old: new for old, new in zip(original_users, new_users)}

    # Build regex to replace old usernames occurring inside messages as whole words or with @
    patterns = []
    for old_u in original_users:
        patterns.append((re.compile(rf"(?:\b|@){re.escape(old_u)}\b", re.IGNORECASE), user_map[old_u]))

    def rewrite_text(text: str) -> str:
        res = text
        for pat, rep in patterns:
            res = pat.sub(rep, res)
        return res

    new_messages = []
    for m in conv["messages"]:
        new_m = dict(m)
        new_m["username"] = user_map.get(m["username"], m["username"])
        new_m["raw_input"] = rewrite_text(m.get("raw_input", m.get("text_de", "")))
        new_m["text_de"] = rewrite_text(m["text_de"])
        new_m["text_en"] = rewrite_text(m["text_en"])
        new_messages.append(new_m)

    new_conv = dict(conv)
    new_conv["messages"] = new_messages
    return new_conv


def shift_timestamps(conv: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies a random temporal shift (days + hours + minutes) preserving relative deltas.
    """
    messages = conv["messages"]
    if not messages:
        return conv

    try:
        first_dt = datetime.strptime(messages[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return conv

    # Generate random delta shift (e.g. within past 2 years, random hour)
    shift_days = random.randint(-700, 700)
    shift_hours = random.randint(-12, 12)
    shift_minutes = random.randint(-30, 30)
    total_offset = timedelta(days=shift_days, hours=shift_hours, minutes=shift_minutes)

    new_messages = []
    for m in messages:
        new_m = dict(m)
        try:
            m_dt = datetime.strptime(m["timestamp"], "%Y-%m-%d %H:%M:%S")
            shifted_dt = m_dt + total_offset
            new_m["timestamp"] = shifted_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        new_messages.append(new_m)

    new_conv = dict(conv)
    new_conv["messages"] = new_messages
    return new_conv


def create_mono_de_sample(conv: Dict[str, Any]) -> Dict[str, Any]:
    """German user inputs -> Clean English output."""
    domain_ctx = conv.get("domain_context", {})
    messages = [build_system_message(domain_ctx.get("domain", "General Chat"), domain_ctx.get("glossary", {}))]

    for m in conv["messages"]:
        # If original was DE, use raw_input with typos; if original was EN, use clean text_de
        input_text = m.get("raw_input", m["text_de"]) if m.get("original_lang") == "de" else m["text_de"]
        target_text = m["text_en"]

        user_msg = f"[{m['timestamp']}] [{m['username']}]: {input_text}"
        asst_msg = f"src: de\nen: {target_text}"
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})

    return {"messages": messages, "scenario": "mono_de"}


def create_mono_en_sample(conv: Dict[str, Any]) -> Dict[str, Any]:
    """English user inputs -> Clean German output."""
    domain_ctx = conv.get("domain_context", {})
    messages = [build_system_message(domain_ctx.get("domain", "General Chat"), domain_ctx.get("glossary", {}))]

    for m in conv["messages"]:
        # If original was EN, use raw_input with typos; if original was DE, use clean text_en
        input_text = m.get("raw_input", m["text_en"]) if m.get("original_lang") == "en" else m["text_en"]
        target_text = m["text_de"]

        user_msg = f"[{m['timestamp']}] [{m['username']}]: {input_text}"
        asst_msg = f"src: en\nde: {target_text}"
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})

    return {"messages": messages, "scenario": "mono_en"}


def create_per_user_sample(conv: Dict[str, Any]) -> Dict[str, Any]:
    """Each speaker is assigned a fixed language (DE or EN)."""
    domain_ctx = conv.get("domain_context", {})
    messages = [build_system_message(domain_ctx.get("domain", "General Chat"), domain_ctx.get("glossary", {}))]

    users = list({m["username"] for m in conv["messages"]})
    user_lang_map = {u: ("de" if idx % 2 == 0 else "en") for idx, u in enumerate(users)}

    for m in conv["messages"]:
        lang = user_lang_map.get(m["username"], "de")
        if lang == "de":
            input_text = m.get("raw_input", m["text_de"]) if m.get("original_lang") == "de" else m["text_de"]
            user_msg = f"[{m['timestamp']}] [{m['username']}]: {input_text}"
            asst_msg = f"src: de\nen: {m['text_en']}"
        else:
            input_text = m.get("raw_input", m["text_en"]) if m.get("original_lang") == "en" else m["text_en"]
            user_msg = f"[{m['timestamp']}] [{m['username']}]: {input_text}"
            asst_msg = f"src: en\nde: {m['text_de']}"

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})

    return {"messages": messages, "scenario": "per_user"}


def create_random_switching_sample(conv: Dict[str, Any]) -> Dict[str, Any]:
    """Random language chosen per message to simulate bilingual users."""
    domain_ctx = conv.get("domain_context", {})
    messages = [build_system_message(domain_ctx.get("domain", "General Chat"), domain_ctx.get("glossary", {}))]

    for m in conv["messages"]:
        lang = "de" if random.random() < 0.5 else "en"
        if lang == "de":
            input_text = m.get("raw_input", m["text_de"]) if m.get("original_lang") == "de" else m["text_de"]
            user_msg = f"[{m['timestamp']}] [{m['username']}]: {input_text}"
            asst_msg = f"src: de\nen: {m['text_en']}"
        else:
            input_text = m.get("raw_input", m["text_en"]) if m.get("original_lang") == "en" else m["text_en"]
            user_msg = f"[{m['timestamp']}] [{m['username']}]: {input_text}"
            asst_msg = f"src: en\nde: {m['text_de']}"

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})

    return {"messages": messages, "scenario": "random_switch"}


def synthesize(
    input_file: str,
    weight_protocol: float = 8.0,
    weight_minecraft: float = 1.0,
    username_permutations: int = 2,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[int, int]:
    random.seed(seed)

    if not os.path.exists(input_file):
        print(f"[ERROR] Input parallel corpus not found: {input_file}")
        return 0, 0

    pool = UsernamePool(USERNAMES_FILE)

    conversations = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    conversations.append(json.loads(line))
                except Exception:
                    pass

    print(f"--> Loaded {len(conversations)} base parallel conversations.")

    all_samples = []

    for conv in conversations:
        # Determine dataset multiplier based on source
        source_file = conv.get("source_file", "")
        if "protocol" in source_file:
            repeat_count = int(round(weight_protocol))
        else:
            repeat_count = int(round(weight_minecraft))

        repeat_count = max(1, repeat_count)

        for rep in range(repeat_count):
            # Base or permuted version
            working_conv = conv
            if rep > 0 or random.random() < 0.5:
                working_conv = replace_usernames_in_conversation(working_conv, pool)
                working_conv = shift_timestamps(working_conv)

            # Generate scenarios
            all_samples.append(create_mono_de_sample(working_conv))
            all_samples.append(create_mono_en_sample(working_conv))
            all_samples.append(create_per_user_sample(working_conv))
            all_samples.append(create_random_switching_sample(working_conv))

    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Synthesized {len(train_samples):,} training samples -> {TRAIN_FILE}")
    print(f"[SUCCESS] Synthesized {len(val_samples):,} validation samples -> {VAL_FILE}")

    return len(train_samples), len(val_samples)


def main():
    parser = argparse.ArgumentParser(description="Synthesize weighted & permuted ChatML training samples.")
    parser.add_argument("--input", type=str, default=UNIFIED_CORPUS_FILE, help="Path to unified_conversations.jsonl")
    parser.add_argument("--weight-protocol", type=float, default=8.0, help="Oversampling weight for protocol.txt")
    parser.add_argument("--weight-minecraft", type=float, default=1.0, help="Weight for minecraft json")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Ratio for training set split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    synthesize(
        input_file=args.input,
        weight_protocol=args.weight_protocol,
        weight_minecraft=args.weight_minecraft,
        train_ratio=args.train_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
