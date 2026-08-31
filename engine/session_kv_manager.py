"""
session_kv_manager.py

Persistent GPU Session KV Cache Manager using PyTorch DynamicCache on AMD Vega 56.
Maintains system prompt prefix and multi-turn dialogue state in GPU VRAM across requests.
"""

import time
import re
import os
import torch
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from . import config

SYSTEM_PROMPT_TEMPLATE = """You are a fast real-time chat translation engine.
Room Languages: [{languages}]

[DOMAIN CONTEXT & NOMENCLATURE]
Context: {domain}
Glossary:
{glossary_str}

[PROTOCOL INSTRUCTIONS]
1. Read the incoming message formatted as `[<timestamp>] [<user>]: <message>`.
2. Output detected source language on line 1: `src: <iso_code>`.
3. Output translation for the other room language on line 2: `<target_lang>: <translated_message>`.
4. Maintain speaker tags, tone, emojis, and exact domain glossary terms."""


def format_glossary(glossary: Dict[str, str]) -> str:
    if not glossary:
        return "- None"
    return "\n".join([f"- \"{k}\" -> \"{v}\"" for k, v in glossary.items()])


class SessionState:
    def __init__(
        self,
        session_id: str,
        domain: str,
        glossary: Dict[str, str],
        room_languages: List[str],
        system_prompt: str,
        system_token_count: int,
        past_key_values: DynamicCache
    ):
        self.session_id = session_id
        self.domain = domain
        self.glossary = glossary
        self.room_languages = room_languages
        self.system_prompt = system_prompt
        self.system_token_count = system_token_count
        self.past_key_values = past_key_values
        self.cached_token_count = system_token_count
        self.message_history: List[Dict[str, Any]] = []
        self.last_accessed = time.time()
        self.total_turns = 0


class SessionKVManager:
    def __init__(self, model_path: Optional[str] = None, device_override: Optional[str] = None):
        if device_override and device_override.lower() != "auto":
            device_str = device_override.lower()
        else:
            device_str = "cpu"
            if torch.cuda.is_available():
                dev_name = torch.cuda.get_device_name(0).lower()
                # Only use GPU if discrete Vega GPU is active
                if "vega" in dev_name:
                    device_str = "cuda:0"
                else:
                    print(f"--> [SessionKVManager] Discrete Vega GPU not linked (found: {torch.cuda.get_device_name(0)}). Using high-speed Ryzen 7 CPU.")

        self.device = torch.device(device_str)
        torch_dtype = torch.float16 if "cuda" in device_str else torch.bfloat16

        self.sessions: Dict[str, SessionState] = {}

        # Determine model path
        if model_path:
            chosen_path = model_path
        elif os.path.exists(config.MERGED_MODEL_DIR):
            chosen_path = config.MERGED_MODEL_DIR
        else:
            chosen_path = config.BASE_MODEL_ID

        print(f"--> [SessionKVManager] Loading model from: {chosen_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(chosen_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            chosen_path,
            torch_dtype=torch_dtype,
            device_map=self.device,
            attn_implementation="eager",
            trust_remote_code=True
        )
        self.model.eval()

        self.eos_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        ]
        self.eos_token_ids = [t for t in self.eos_token_ids if t is not None]

        print(f"--> [SessionKVManager] Ready. Allocated VRAM: {self.get_allocated_vram_mb():.2f} MB")

    def get_allocated_vram_mb(self) -> float:
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        return 0.0

    def get_reserved_vram_mb(self) -> float:
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        return 0.0

    def create_session(
        self,
        session_id: str,
        domain: str = "Minecraft Multiplayer & Mod Development",
        glossary: Optional[Dict[str, str]] = None,
        room_languages: Optional[List[str]] = None
    ) -> SessionState:
        if glossary is None:
            glossary = {
                "LoW": "Legend of War",
                "Ansturm": "Rush / Charge ability",
                "Doppelsprung": "Double Jump ability",
                "vanilla": "unmodded Minecraft",
                "base": "player home / base",
                "netherite": "netherite gear"
            }
        if room_languages is None:
            room_languages = ["en", "de"]

        langs_str = ", ".join(room_languages)
        glossary_str = format_glossary(glossary)
        system_text = SYSTEM_PROMPT_TEMPLATE.format(
            languages=langs_str,
            domain=domain,
            glossary_str=glossary_str
        )

        formatted_sys = f"<|im_start|>system\n{system_text}<|im_end|>\n"
        sys_ids = self.tokenizer.encode(formatted_sys, return_tensors="pt").to(self.device)

        # Prefill System Prompt into initial DynamicCache
        with torch.no_grad():
            cache = DynamicCache()
            outputs = self.model(sys_ids, past_key_values=cache, use_cache=True)
            initial_cache = outputs.past_key_values

        session = SessionState(
            session_id=session_id,
            domain=domain,
            glossary=glossary,
            room_languages=room_languages,
            system_prompt=system_text,
            system_token_count=sys_ids.shape[1],
            past_key_values=initial_cache
        )

        self.sessions[session_id] = session
        print(f"--> [SessionKVManager] Created session '{session_id}' with {session.system_token_count} prefix tokens.")
        return session

    def get_or_create_session(self, session_id: str) -> SessionState:
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_accessed = time.time()
            return session
        return self.create_session(session_id)

    def parse_translation_output(self, generated_text: str) -> Tuple[str, Dict[str, str]]:
        """
        Parses output string formatted as:
        src: de
        en: hello world
        """
        lines = [l.strip() for l in generated_text.strip().split("\n") if l.strip()]
        src_lang = "unknown"
        translations = {}

        for line in lines:
            if line.lower().startswith("src:"):
                src_lang = line.split(":", 1)[1].strip().lower()
            elif ":" in line:
                parts = line.split(":", 1)
                lang_code = parts[0].strip().lower()
                text = parts[1].strip()
                if len(lang_code) <= 5 and text:
                    translations[lang_code] = text

        return src_lang, translations

    def process_turn(
        self,
        session_id: str,
        username: str,
        timestamp: str,
        message: str,
        max_new_tokens: int = 128
    ) -> Dict[str, Any]:
        t_start = time.time()
        session = self.get_or_create_session(session_id)

        # Format incoming user message turn
        user_turn_text = f"<|im_start|>user\n[{timestamp}] [{username}]: {message}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer.encode(user_turn_text, return_tensors="pt").to(self.device)
        num_prompt_tokens = input_ids.shape[1]

        cache = session.past_key_values
        generated_token_ids = []
        ttft = 0.0

        with torch.no_grad():
            curr_input = input_ids

            for step in range(max_new_tokens):
                outputs = self.model(
                    input_ids=curr_input,
                    past_key_values=cache,
                    use_cache=True
                )
                cache = outputs.past_key_values

                if step == 0:
                    ttft = time.time() - t_start

                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                token_id = next_token.item()

                if token_id in self.eos_token_ids:
                    break

                generated_token_ids.append(token_id)
                curr_input = next_token

            # Append the trailing assistant <|im_end|>\n to keep the conversation cache clean
            end_tokens = self.tokenizer.encode("<|im_end|>\n", return_tensors="pt").to(self.device)
            outputs_end = self.model(input_ids=end_tokens, past_key_values=cache, use_cache=True)
            cache = outputs_end.past_key_values

        session.past_key_values = cache
        session.cached_token_count = cache.get_seq_length()
        session.total_turns += 1

        t_total = time.time() - t_start
        gen_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        src_lang, translations = self.parse_translation_output(gen_text)

        turn_info = {
            "turn_id": session.total_turns,
            "timestamp": timestamp,
            "username": username,
            "message": message,
            "raw_output": gen_text,
            "src_lang": src_lang,
            "translations": translations,
            "tokens_generated": len(generated_token_ids),
            "total_cached_tokens": session.cached_token_count,
            "ttft_ms": round(ttft * 1000, 2),
            "total_time_ms": round(t_total * 1000, 2),
            "tokens_per_second": round(len(generated_token_ids) / max(t_total - ttft, 0.001), 2)
        }

        session.message_history.append(turn_info)
        return turn_info

    def reset_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_sessions": len(self.sessions),
            "allocated_vram_mb": round(self.get_allocated_vram_mb(), 2),
            "reserved_vram_mb": round(self.get_reserved_vram_mb(), 2),
            "session_list": [
                {
                    "session_id": s.session_id,
                    "total_turns": s.total_turns,
                    "cached_tokens": s.cached_token_count,
                    "last_accessed_ago_sec": round(time.time() - s.last_accessed, 1)
                }
                for s in self.sessions.values()
            ]
        }
