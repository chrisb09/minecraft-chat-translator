# minecraft-chat-translator

Real-time EN↔DE chat translation fine-tuned on authentic Minecraft server logs. A Qwen2.5-0.5B-Instruct model trained with LoRA on CPU (AMD Zen 4 / AVX-512 BF16) to translate multi-participant gaming chat with strict two-line output protocol, domain glossary awareness, typo tolerance, and persistent KV cache for low-latency inference.

---

## Overview

Multi-participant gaming servers often mix languages. This project fine-tunes a small (0.5B) language model to act as a real-time translation layer between English and German speakers, running locally with sub-15ms TTFT using a persistent KV cache session engine.

**Key properties:**
- Strict two-line output protocol (`src: de` / `en: ...`) — no preamble, no filler
- Domain glossary awareness: `Ansturm` → `Rush`, `LoW` → `Legend of War`, `LK` → `Leistungskurs`, etc.
- Tolerates real chat typos (`schwimsmt`, `vlt`, `gehn`)
- Persistent KV cache per session via FastAPI engine — flat ~54ms TTFT regardless of conversation length
- Runs on CPU (1 GB RAM) or GPU (1 GB VRAM)

---

## Benchmark

Results from [`tools/compare_models.py`](tools/compare_models.py) across 8 test cases (4 DE→EN, 4 EN→DE):

| Model | Host / Provider | Protocol Adherence | Lang Detection | Glossary / Slang Mapping | Avg Latency |
|---|---|---|---|---|---|
| **Base Qwen 0.5B (untuned)** | Local CPU (BF16) | ❌ Failed (hallucinates format) | ❌ Wrong | None | ~2,500ms |
| **Fine-Tuned Qwen 0.5B (ours)** | Local CPU (BF16) |  **100% Strict** |  100% |  `Ansturm`→`Rush`, `LoW`→`Legend of War`, `wipe`→`wipen` | ~1,500ms CPU / **<15ms KV cache** |
| Cerebras Qwen 3.8 27B | Cerebras WSE-3 |  **100% Strict** |  100% |  Accurate gaming slang (`wipen`, `Chest-Room`) | **~280ms** (>1500 tok/s) |
| Gemini 3.5 Flash Lite | Google AI Studio |  100% (echoes speaker tag) |  100% |  Accurate conversational gaming German | ~680ms |
| Gemini 3.1 Flash Lite | Google AI Studio |  100% (echoes speaker tag) |  100% |  Accurate | ~1,800ms |
| Gemini 3.5 / 3.7 / 3.8 Flash | Google AI Studio |  High |  100% |  Detailed comprehension | ~1,200ms – 6,000ms (thinking overhead) |
| Gemma 4 (26B / 31B) | Google AI Studio | ⚠️ Verbose CoT before output |  100% |  Strong domain comprehension | ~9,000ms – 14,000ms |
| InclusionAI Ling 3.0 Flash | OpenRouter (free) |  **100% Strict** |  100% |  Good (`LK`→`Advanced Math Course`) | ~950ms |
| Liquid LFM 2.5 2.6B | OpenRouter (free) | ⚠️ Inconsistent (reasoning spill) |  100% | None | ~3,500ms |

Sample output (fine-tuned model):
```
Input:  [18.12.2024][15:10:41] [Moritzwied]: Damn das neue schwimmen is so aids für die LoW version
Output: src: de
        en: Damn, the new swimming is so aids for the Legend of War version
```
Input:  [18.12.2024][15:10:41] [Moritzwied]: Damn das neue schwimmen is so aids für die LoW version
Output: src: de
        en: Damn, the new swimming is so aids for the Legend of War version
```

---

## Hugging Face Models

| Variant | Link | Size |
|---|---|---|
| Merged FP16 (plug-and-play) | [duLouser/qwen2.5-0.5b-minecraft-chat-translator](https://huggingface.co/duLouser/qwen2.5-0.5b-minecraft-chat-translator) | 943 MB |
| LoRA adapter only | [duLouser/qwen2.5-0.5b-minecraft-chat-translator-lora](https://huggingface.co/duLouser/qwen2.5-0.5b-minecraft-chat-translator-lora) | 34 MB |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys (optional — only needed for cloud model comparison)

```bash
cp .env.example .env
# edit .env with your Gemini and OpenRouter keys
```

### 3. Run the interactive chat CLI

```bash
bash chat.sh
# or directly:
python tools/chat_cli.py --device cpu
```

### 4. Start the FastAPI inference server

```bash
python -m uvicorn engine.server:app --host 0.0.0.0 --port 8000
```

REST endpoint: `POST /translate` with `{"session_id": "room1", "message": "[12:00] [user]: Hallo"}`

WebSocket: `ws://localhost:8000/ws/{session_id}`

---

## Repository Structure

```
minecraft-chat-translator/
├── data/
│   ├── extract_raw.py              # Extract threads from raw server logs
│   ├── translate_corpus.py         # Multi-LLM parallel EN↔DE translation
│   ├── synthesize_training_data.py # Generate ChatML training samples
│   ├── build_glossary.py           # Build bilingual Minecraft glossary
│   └── minecraft_bilingual_glossary.json
├── engine/
│   ├── config.py                   # Auto device/dtype detection
│   ├── session_kv_manager.py       # Persistent KV cache per session
│   └── server.py                   # FastAPI REST + WebSocket server
├── training/
│   ├── train_cpu.py                # LoRA fine-tuning on CPU (BF16/AVX-512)
│   └── export_model.py             # Merge LoRA adapter into standalone model
├── tools/
│   ├── chat_cli.py                 # Interactive terminal chat simulator
│   ├── compare_models.py           # Side-by-side multi-model benchmark
│   ├── benchmark_latency.py        # KV cache vs stateless latency test
│   ├── test_inference.py           # Quick inference smoke test
│   └── test_api_keys.py            # Verify API key configuration
├── rawdata/
│   ├── minecraft_glossary.txt      # Raw domain glossary input
│   └── protocol.txt                # Translation protocol specification
├── chat.sh                         # Launcher for chat CLI
├── run_training.sh                 # ROCm env launcher (GPU training reference)
├── requirements.txt
└── PLAN.md                         # Architecture and pipeline documentation
```

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | LoRA (r=16, alpha=32, target: q/v/k/o projections) |
| Training data | 5,990 ChatML samples (4 scenarios × username permutations) |
| Validation data | 666 samples |
| Hardware | AMD Ryzen 7 H 255 (14× Zen 4), AVX-512 BF16, CPU-only |
| Throughput | ~245 tokens/sec |
| Duration | 7 hours 34 minutes (1 epoch, 1,497 steps) |
| Val loss | 1.1972 → **0.1005** (91.6% reduction) |
| Precision | BF16 native, gradient accumulation = 4 |

### Data pipeline

1. `data/extract_raw.py` — 558 threads / 7,461 messages from raw server logs → `data/raw_extracted_threads.jsonl`
2. `data/translate_corpus.py` — parallel async translation using Gemini and OpenRouter APIs → `data/unified_conversations.jsonl`
3. `data/synthesize_training_data.py` — 4 linguistic scenarios (mono-DE, mono-EN, per-user, random-switch), weighted protocol samples (8×), username permutations from 65.9M name list → `data/train.jsonl` / `data/val.jsonl`

### Reproduce training

```bash
# CPU training (recommended — GPU training on gfx900/Vega56 is unstable)
python training/train_cpu.py

# Merge LoRA adapter into standalone model
python training/export_model.py
```

---

## Inference Usage

### Merged model (standalone)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "duLouser/qwen2.5-0.5b-minecraft-chat-translator",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("duLouser/qwen2.5-0.5b-minecraft-chat-translator")

SYSTEM = """You are a fast real-time chat translation engine.
Room Languages: [en, de]
Output ONLY:
Line 1: src: <en|de>
Line 2: <target_lang>: <translation>"""

prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n[12:00] [tobi20]: der ansturm geht im wasser<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=48, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
# src: de
# en: The Rush is working in water
```

### LoRA adapter

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "duLouser/qwen2.5-0.5b-minecraft-chat-translator-lora")
tokenizer = AutoTokenizer.from_pretrained("duLouser/qwen2.5-0.5b-minecraft-chat-translator-lora")
```

---

## License

Apache 2.0 — same as the base model [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
