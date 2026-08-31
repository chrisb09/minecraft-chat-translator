#!/usr/bin/env python3
"""
translate_corpus.py

Resilient parallel translation worker with:
- Rich real-time progress dashboard (Elapsed time, ETA, % bar, throughput threads/min & msgs/min).
- Capability-aware model routing (GEMINI_CAPABILITIES, OPENROUTER_CAPABILITIES).
- Multi-account key rotation for Google AI Studio & OpenRouter.
- Dynamic sliding-window tracking for RPM (Requests/min) & TPM (Tokens/min).
- Daily quota tracking (RPD) and automatic multi-tier model fallback.
- Auto-cooldown on 429 errors.
- JSON cleaning from markdown code fences.
- Auto-resume and live terminal progress.
"""

import os
import sys
import re
import json
import time
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, Counter
from dotenv import load_dotenv

load_dotenv("/git/finetune/.env")

INPUT_FILE = "/git/finetune/data/raw_extracted_threads.jsonl"
OUTPUT_FILE = "/git/finetune/data/unified_conversations.jsonl"

MODEL_ALIASES = {
    "gemma-4-31b": "gemma-4-31b-it",
    "gemma-4-26b": "gemma-4-26b-a4b-it",
    "gemma-4-26b-it": "gemma-4-26b-a4b-it",
}


def resolve_model_name(raw_name: str) -> str:
    cleaned = raw_name.strip()
    return MODEL_ALIASES.get(cleaned.lower(), cleaned)


def parse_unit_number(val_str: str, default: int = 1000) -> int:
    s = val_str.strip().lower()
    if not s:
        return default
    if s in ("unbegrenzt", "unlimited", "inf", "none", "0"):
        return 999_999_999
    if s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    try:
        return int(float(s))
    except Exception:
        return default


def parse_model_specs(provider_prefix: str) -> List[Dict[str, Any]]:
    raw_ids = os.environ.get(f"{provider_prefix}_MODEL_IDS", "") or os.environ.get(f"{provider_prefix}_MODELS", "")
    if not raw_ids.strip():
        return []

    model_ids = [resolve_model_name(m.strip()) for m in raw_ids.split(",") if m.strip()]

    raw_rpms = [r.strip() for r in os.environ.get(f"{provider_prefix}_RPMS", "").split(",") if r.strip()]
    raw_tpms = [t.strip() for t in os.environ.get(f"{provider_prefix}_TPMS", "").split(",") if t.strip()]
    raw_rpds = [d.strip() for d in os.environ.get(f"{provider_prefix}_RPDS", "").split(",") if d.strip()]
    raw_caps = [c.strip() for c in os.environ.get(f"{provider_prefix}_CAPABILITIES", "").split(",") if c.strip()]

    specs = []
    for idx, mid in enumerate(model_ids):
        default_rpm = 30 if "gemma" in mid.lower() else 15
        default_tpm = 16_000 if "gemma" in mid.lower() else 250_000
        default_rpd = 14_400 if "gemma" in mid.lower() else 500
        default_cap = 9 if "ox-alpha" in mid.lower() else (6 if "3.5" in mid.lower() else 5)

        rpm_str = raw_rpms[idx] if idx < len(raw_rpms) else (raw_rpms[-1] if raw_rpms else str(default_rpm))
        tpm_str = raw_tpms[idx] if idx < len(raw_tpms) else (raw_tpms[-1] if raw_tpms else str(default_tpm))
        rpd_str = raw_rpds[idx] if idx < len(raw_rpds) else (raw_rpds[-1] if raw_rpds else str(default_rpd))
        cap_str = raw_caps[idx] if idx < len(raw_caps) else (raw_caps[-1] if raw_caps else str(default_cap))

        specs.append({
            "model_id": mid,
            "rpm": parse_unit_number(rpm_str, default=default_rpm),
            "tpm": parse_unit_number(tpm_str, default=default_tpm),
            "rpd": parse_unit_number(rpd_str, default=default_rpd),
            "capability": int(cap_str) if cap_str.isdigit() else default_cap
        })
    return specs


def assess_thread_complexity(thread: Dict[str, Any]) -> int:
    total_text = " ".join([m["content"] for m in thread.get("messages", [])])
    has_code = bool(re.search(r"0x[0-9a-fA-F]+|\.[a-zA-Z]{1,4}\b|[a-zA-Z_]+\(\)|[a-zA-Z_]+\.[a-zA-Z_]+", total_text))
    has_heavy_slang = bool(re.search(r"\b(?:aids|haider|stuck|reference_update|velocity|koh[aä]renz)\b", total_text, re.IGNORECASE))

    if has_code and has_heavy_slang:
        return 6
    elif has_code or has_heavy_slang:
        return 5
    return 4


class ModelKeyLimiter:
    def __init__(self, provider: str, spec: Dict[str, Any], key_str: str, client: Any):
        self.provider = provider
        self.model_name = spec["model_id"]
        self.rpm = spec["rpm"]
        self.tpm = spec["tpm"]
        self.rpd = spec["rpd"]
        self.capability = spec.get("capability", 5)
        self.key_str = key_str
        self.client = client

        self.min_interval = 60.0 / max(1, self.rpm)
        self.recent_req_times = deque()
        self.recent_tok_history = deque()
        self.daily_requests = 0
        self.cooldown_until = 0.0
        self.lock = asyncio.Lock()
        self.success_count = 0
        self.error_count = 0

    def is_available(self, min_capability: int = 1) -> bool:
        if self.capability < min_capability:
            return False
        now = time.time()
        if now < self.cooldown_until:
            return False
        if self.daily_requests >= self.rpd:
            return False
        return True

    async def acquire(self, estimated_tokens: int = 500):
        async with self.lock:
            now = time.time()
            if now < self.cooldown_until:
                await asyncio.sleep(self.cooldown_until - now)

            while self.recent_req_times and (time.time() - self.recent_req_times[0]) > 60.0:
                self.recent_req_times.popleft()

            while self.recent_tok_history and (time.time() - self.recent_tok_history[0][0]) > 60.0:
                self.recent_tok_history.popleft()

            if len(self.recent_req_times) >= self.rpm:
                sleep_needed = 60.0 - (time.time() - self.recent_req_times[0]) + 0.1
                if sleep_needed > 0:
                    await asyncio.sleep(sleep_needed)

            cur_tokens = sum(t[1] for t in self.recent_tok_history)
            if cur_tokens + estimated_tokens > self.tpm:
                if self.recent_tok_history:
                    sleep_needed = 60.0 - (time.time() - self.recent_tok_history[0][0]) + 0.1
                    if sleep_needed > 0:
                        await asyncio.sleep(sleep_needed)

            if self.recent_req_times:
                elapsed = time.time() - self.recent_req_times[-1]
                if elapsed < self.min_interval:
                    await asyncio.sleep(self.min_interval - elapsed)

            cur_t = time.time()
            self.recent_req_times.append(cur_t)
            self.recent_tok_history.append((cur_t, estimated_tokens))
            self.daily_requests += 1

    def report_error(self, is_rate_limit: bool = True, is_quota_exceeded: bool = False):
        self.error_count += 1
        now = time.time()
        if is_quota_exceeded:
            self.daily_requests = self.rpd
            print(f"\n    [DAILY QUOTA EXHAUSTED] {self.provider}:{self.model_name} on key ...{self.key_str[-6:]} reached {self.rpd:,} RPD.")
        elif is_rate_limit:
            self.cooldown_until = max(self.cooldown_until, now + 35.0)
            print(f"\n    [COOLDOWN] {self.provider}:{self.model_name} on key ...{self.key_str[-6:]} rate limited. Pausing for 35s.")

    def report_success(self):
        self.success_count += 1


class MultiModelDispatcher:
    def __init__(self, gemini_keys: List[str], openrouter_keys: List[str]):
        self.limiters: List[ModelKeyLimiter] = []
        self.model_priority_order = []

        gemini_clients = {}
        for k in gemini_keys:
            try:
                from google import genai
                gemini_clients[k] = genai.Client(api_key=k)
            except Exception as e:
                print(f"[WARN] Failed to initialize Gemini client for key ...{k[-6:]}: {e}")

        openrouter_clients = {}
        for k in openrouter_keys:
            try:
                from openai import AsyncOpenAI
                openrouter_clients[k] = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=k)
            except Exception as e:
                print(f"[WARN] Failed to initialize OpenRouter client for key ...{k[-6:]}: {e}")

        gemini_specs = parse_model_specs("GEMINI")
        if not gemini_specs and gemini_keys:
            gemini_specs = [
                {"model_id": "gemma-4-26b-a4b-it", "rpm": 30, "tpm": 16000, "rpd": 14400, "capability": 4},
                {"model_id": "gemma-4-31b-it", "rpm": 30, "tpm": 16000, "rpd": 14400, "capability": 4},
                {"model_id": "gemini-3.5-flash-lite", "rpm": 15, "tpm": 250000, "rpd": 500, "capability": 6},
                {"model_id": "gemini-3.1-flash-lite", "rpm": 15, "tpm": 250000, "rpd": 500, "capability": 5},
            ]

        openrouter_specs = parse_model_specs("OPENROUTER")
        if not openrouter_specs and openrouter_keys:
            openrouter_specs = [
                {"model_id": "stealth/ox-alpha", "rpm": 20, "tpm": 100000, "rpd": 2000, "capability": 9}
            ]

        print("\n--> [Dispatcher] Initializing Model Matrix (Capacity & Capabilities):")
        for spec in gemini_specs:
            self.model_priority_order.append(("gemini", spec["model_id"], spec["capability"]))
            print(f"    - [GEMINI]     {spec['model_id']:<22} | Cap: {spec['capability']} | RPM: {spec['rpm']:>3} | TPM: {spec['tpm']:>7,} | RPD: {spec['rpd']:>7,}")
            for k, cl in gemini_clients.items():
                self.limiters.append(ModelKeyLimiter("gemini", spec, k, cl))

        for spec in openrouter_specs:
            self.model_priority_order.append(("openrouter", spec["model_id"], spec["capability"]))
            print(f"    - [OPENROUTER] {spec['model_id']:<22} | Cap: {spec['capability']} | RPM: {spec['rpm']:>3} | TPM: {spec['tpm']:>7,} | RPD: {spec['rpd']:>7,}")
            for k, cl in openrouter_clients.items():
                self.limiters.append(ModelKeyLimiter("openrouter", spec, k, cl))
        print()

    def select_limiter(self, min_capability: int = 1) -> Optional[ModelKeyLimiter]:
        for prov, mod, cap in self.model_priority_order:
            if cap >= min_capability:
                matching = [l for l in self.limiters if l.provider == prov and l.model_name == mod and l.is_available(min_capability)]
                if matching:
                    return min(matching, key=lambda l: l.daily_requests)

        for prov, mod, cap in self.model_priority_order:
            matching = [l for l in self.limiters if l.provider == prov and l.model_name == mod and l.is_available(1)]
            if matching:
                return min(matching, key=lambda l: l.daily_requests)

        active_candidates = [l for l in self.limiters if l.daily_requests < l.rpd]
        if active_candidates:
            return min(active_candidates, key=lambda l: l.cooldown_until)

        return None


def parse_keys_from_env() -> Tuple[List[str], List[str]]:
    g_keys = []
    o_keys = []

    raw_g = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    for part in re.split(r"[,;\n\s]+", raw_g):
        part = part.strip()
        if part and part not in g_keys:
            g_keys.append(part)

    for i in range(1, 20):
        val = os.environ.get(f"GEMINI_API_KEY_{i}")
        if val and val.strip() and val.strip() not in g_keys:
            g_keys.append(val.strip())

    raw_o = os.environ.get("OPENROUTER_API_KEYS", "") or os.environ.get("OPENROUTER_API_KEY", "")
    for part in re.split(r"[,;\n\s]+", raw_o):
        part = part.strip()
        if part and part not in o_keys:
            o_keys.append(part)

    for i in range(1, 20):
        val = os.environ.get(f"OPENROUTER_API_KEY_{i}")
        if val and val.strip() and val.strip() not in o_keys:
            o_keys.append(val.strip())

    return g_keys, o_keys


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    return json.loads(text)


def create_translation_prompt(thread: Dict[str, Any]) -> str:
    source_lang = thread["source_lang"]
    target_lang = "en" if source_lang == "de" else "de"
    source_lang_name = "German" if source_lang == "de" else "English"
    target_lang_name = "English" if source_lang == "de" else "German"

    domain = thread.get("domain", "General Multiplayer Gaming Chat")
    glossary = thread.get("glossary", {})
    glossary_str = "\n".join([f"- \"{k}\" -> \"{v}\"" for k, v in glossary.items()])

    messages = thread["messages"]
    msg_lines = []
    for m in messages:
        msg_lines.append(f"[{m['id']}] [{m['username']}]: {m['content']}")
    dialogue_str = "\n".join(msg_lines)

    prompt = f"""You are an expert bilingual chat translator specializing in {source_lang_name} <-> {target_lang_name} translations for online gaming and developer communities.

[DOMAIN CONTEXT]
{domain}

[GLOSSARY & NOMENCLATURE]
{glossary_str}

[TASK INSTRUCTIONS]
For each chat message in the dialogue below:
1. Produce a natural, high-quality translation into {target_lang_name} (`target_text`).
2. Produce a normalized/cleaned version in the original {source_lang_name} (`clean_source_text`), fixing accidental keystroke typos (e.g. "schwimsmt" -> "schwimmst", "sieht so ahrt aus" -> "sieht so hart aus") while preserving casual gamer tone.
3. Maintain domain terms from the glossary accurately.
4. Output MUST be valid JSON only: a JSON object mapping string message IDs to an object with `clean_source_text` and `target_text`.

[CHAT MESSAGES TO TRANSLATE]
{dialogue_str}

[OUTPUT FORMAT EXAMPLE]
{{
  "0": {{
    "clean_source_text": "clean original {source_lang_name}",
    "target_text": "translation into {target_lang_name}"
  }},
  "1": {{
    "clean_source_text": "clean original {source_lang_name}",
    "target_text": "translation into {target_lang_name}"
  }}
}}
"""
    return prompt


async def call_gemini(limiter: ModelKeyLimiter, prompt: str) -> Dict[str, Any]:
    await limiter.acquire(estimated_tokens=len(prompt) // 4 + 300)
    from google.genai import types
    response = await limiter.client.aio.models.generate_content(
        model=limiter.model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3
        )
    )
    limiter.report_success()
    return clean_json_response(response.text)


async def call_openrouter(limiter: ModelKeyLimiter, prompt: str) -> Dict[str, Any]:
    await limiter.acquire(estimated_tokens=len(prompt) // 4 + 300)
    response = await limiter.client.chat.completions.create(
        model=limiter.model_name,
        messages=[
            {"role": "system", "content": "You are a professional chat translator that outputs strict JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    limiter.report_success()
    raw_text = response.choices[0].message.content.strip()
    return clean_json_response(raw_text)


async def process_thread(
    thread: Dict[str, Any],
    dispatcher: MultiModelDispatcher,
    semaphore: asyncio.Semaphore,
    max_retries: int = 6
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    async with semaphore:
        prompt = create_translation_prompt(thread)
        source_lang = thread["source_lang"]
        base_complexity = assess_thread_complexity(thread)

        for attempt in range(max_retries):
            required_cap = min(9, base_complexity + attempt * 2)

            limiter = dispatcher.select_limiter(min_capability=required_cap)
            if not limiter:
                return None, None

            used_model = limiter.model_name
            try:
                if limiter.provider == "gemini":
                    data = await call_gemini(limiter, prompt)
                else:
                    data = await call_openrouter(limiter, prompt)

                parallel_messages = []
                for m in thread["messages"]:
                    mid_str = str(m["id"])
                    entry = data.get(mid_str) or data.get(m["id"]) or {}

                    if isinstance(entry, dict):
                        clean_src = entry.get("clean_source_text", "").strip() or m["content"]
                        target_txt = entry.get("target_text", "").strip() or m["content"]
                    elif isinstance(entry, str):
                        clean_src = m["content"]
                        target_txt = entry.strip() or m["content"]
                    else:
                        clean_src = m["content"]
                        target_txt = m["content"]

                    if source_lang == "de":
                        text_de = clean_src
                        text_en = target_txt
                    else:
                        text_en = clean_src
                        text_de = target_txt

                    parallel_messages.append({
                        "id": m["id"],
                        "timestamp": m["timestamp"],
                        "username": m["username"],
                        "original_lang": source_lang,
                        "raw_input": m["content"],
                        "text_de": text_de,
                        "text_en": text_en
                    })

                return {
                    "conversation_id": thread["thread_id"],
                    "source_file": thread["source_file"],
                    "domain_context": {
                        "domain": thread.get("domain", ""),
                        "glossary": thread.get("glossary", {})
                    },
                    "messages": parallel_messages
                }, used_model

            except Exception as e:
                err_str = str(e).lower()
                is_quota = "quota" in err_str or "daily" in err_str or "exceeded your current quota" in err_str
                is_rate = "429" in err_str or "rate" in err_str or "resource_exhausted" in err_str or is_quota

                limiter.report_error(is_rate_limit=is_rate, is_quota_exceeded=is_quota)

                wait_sec = 2.0 * (attempt + 1)
                await asyncio.sleep(wait_sec)

        return None, None


def format_duration(seconds: float) -> str:
    secs = int(max(0, seconds))
    hrs = secs // 3600
    mins = (secs % 3600) // 60
    s = secs % 60
    if hrs > 0:
        return f"{hrs:02d}h:{mins:02d}m:{s:02d}s"
    return f"{mins:02d}m:{s:02d}s"


def render_progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        pct = 100.0
        filled = width
    else:
        pct = min(100.0, (completed / total) * 100.0)
        filled = int((completed / total) * width)
    bar = "=" * filled + (">" if filled < width else "") + "-" * (width - filled - (1 if filled < width else 0))
    return f"[{bar}] {pct:>5.1f}%"


def list_gemini_models():
    g_keys, _ = parse_keys_from_env()
    if not g_keys:
        print("[ERROR] No GEMINI_API_KEY found in .env to list models.")
        return

    from google import genai
    client = genai.Client(api_key=g_keys[0])
    print(f"--> Querying Google AI Studio available models for key ...{g_keys[0][-6:]}...")
    try:
        models = client.models.list()
        print("\nAvailable Models in your Google AI Studio Account:")
        for m in models:
            print(f"  - {m.name}")
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}")


async def main_async(args):
    if args.list_gemini_models:
        list_gemini_models()
        return

    g_keys, o_keys = parse_keys_from_env()
    print("=================================================================")
    print("      Vega 56 Resilient Parallel Translation Worker Dashboard    ")
    print("=================================================================")
    print(f"--> Authenticated Accounts: {len(g_keys)} Gemini key(s), {len(o_keys)} OpenRouter key(s)")

    if not g_keys and not o_keys:
        print("\n[ERROR] No API keys found in .env or environment variables.")
        print("Please configure /git/finetune/.env with your keys.")
        sys.exit(1)

    dispatcher = MultiModelDispatcher(gemini_keys=g_keys, openrouter_keys=o_keys)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        sys.exit(1)

    existing_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_ids.add(obj["conversation_id"])
                except Exception:
                    pass
        print(f"--> Auto-Resume: Found {len(existing_ids)} previously translated conversations.")

    raw_threads = []
    total_messages_to_translate = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if t["thread_id"] not in existing_ids:
                    raw_threads.append(t)
                    total_messages_to_translate += len(t.get("messages", []))

    if args.limit and args.limit > 0:
        raw_threads = raw_threads[:args.limit]
        total_messages_to_translate = sum(len(t.get("messages", [])) for t in raw_threads)

    total_threads = len(raw_threads)
    if total_threads == 0:
        print("[SUCCESS] All conversation threads are already translated! Nothing to do.")
        return

    print(f"--> Queue: {total_threads:,} threads ({total_messages_to_translate:,} messages) | Concurrency: {args.concurrency}")
    print("---------------------------------------------------------------------------------------------------\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    completed_count = 0
    completed_messages = 0
    model_usage_counter = Counter()
    t_start = time.time()

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        tasks = [
            process_thread(t, dispatcher, semaphore)
            for t in raw_threads
        ]

        for future in asyncio.as_completed(tasks):
            result, used_model = await future
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                completed_count += 1
                msg_count = len(result.get("messages", []))
                completed_messages += msg_count
                if used_model:
                    model_usage_counter[used_model] += 1

                elapsed = max(0.1, time.time() - t_start)
                rate_threads_min = (completed_count / elapsed) * 60.0
                rate_msgs_min = (completed_messages / elapsed) * 60.0

                remaining_threads = total_threads - completed_count
                eta_seconds = (remaining_threads / (completed_count / elapsed)) if completed_count > 0 else 0.0

                bar_str = render_progress_bar(completed_count, total_threads, width=20)
                elapsed_str = format_duration(elapsed)
                eta_str = format_duration(eta_seconds)

                # Format compact model breakdown string
                top_models = ", ".join([f"{k.split('-')[0]}:{v}" for k, v in model_usage_counter.most_common(2)])

                status_line = (
                    f"\r{bar_str} | {completed_count:>3}/{total_threads:<3} thr | "
                    f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                    f"Speed: {rate_threads_min:>5.1f} thr/m ({rate_msgs_min:>5.1f} msg/m) | "
                    f"[{top_models}]"
                )
                sys.stdout.write(status_line)
                sys.stdout.flush()

    total_elapsed = time.time() - t_start
    print(f"\n\n=================================================================")
    print(f"                       TRANSLATION COMPLETE                      ")
    print(f"=================================================================")
    print(f"Total Threads Translated:  {completed_count:,} / {total_threads:,}")
    print(f"Total Messages Processed:  {completed_messages:,}")
    print(f"Total Time Taken:          {format_duration(total_elapsed)}")
    print(f"Average Throughput:        {(completed_count / max(1.0, total_elapsed)) * 60:.1f} threads/min")
    print(f"Model Utilization Breakdown:")
    for mod, count in model_usage_counter.most_common():
        pct = (count / max(1, completed_count)) * 100
        print(f"  - {mod:<26} : {count:>4} threads ({pct:>5.1f}%)")
    print(f"Output File:               {OUTPUT_FILE}")
    print("=================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Per-model rate-limited parallel translation worker with live progress.")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent async tasks")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of threads to translate (0 for all)")
    parser.add_argument("--list-gemini-models", action="store_true", help="List all available Google AI Studio models for key")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
