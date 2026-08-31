# Real-Time Multi-Participant Chat Translation Engine

A high-throughput, low-latency multi-participant chat translation system built on **Qwen2.5-0.5B-Instruct** running in native **FP16** on an **AMD Radeon RX Vega 56 (8GB HBM2, ROCm)** with persistent session **KV cache retention**.

---

## 1. System Overview & Hardware Architecture

### Hardware & Acceleration Stack
- **GPU**: AMD Radeon RX Vega 56 (`gfx900`, 8 GB HBM2, ~410 GB/s bandwidth) over OCuLink.
- **Power & Undervolt**: 140W power cap, core voltage capped at 950 mV, HBM2 at 920 MHz (managed via `/etc/systemd/system/vega-adjust.service`).
- **ROCm & Framework**: PyTorch 2.2.0 ROCm 5.6 (`HSA_OVERRIDE_GFX_VERSION=9.0.0`, `HIP_VISIBLE_DEVICES=0`).
- **Precision**: Native packed FP16 (Vega 56 dual-issue FP16 delivers ~21 TFLOPs; no INT8 emulation required).

### Base Model Selection: Qwen 2.5-0.5B-Instruct
- **Parameters**: 0.49B (FP16 model size: ~0.98 GB).
- **Context Length**: 32,768 tokens native context, 8,192 max generation.
- **Vocabulary**: 151,643 tokens (rich UTF-8 and multilingual tokenization).
- **Attention Architecture**: Grouped-Query Attention (GQA) with 2 KV heads per layer (~12.2 MB VRAM per 1,000 cached tokens).
- **VRAM Budget**: ~1.0 GB weights + ~0.5 GB runtime overhead leaves **~6.5 GB free VRAM** for holding hundreds of thousands of cached session tokens.

---

## 2. Translation Protocol & Dialogue Format

### A. System Header (Pinned in VRAM Prefix)
```text
<|im_start|>system
You are a fast real-time chat translation engine.
Room Languages: [en, de]

[DOMAIN CONTEXT & NOMENCLATURE]
Context: Minecraft Multiplayer Server (Survival & Modded Development).
Glossary:
- "LoW" -> "Legend of War"
- "Ansturm" -> "Rush / Charge ability"
- "Doppelsprung" -> "Double Jump"
- "Base" -> "Basis / Stützpunkt"
- "Netherite" -> "Netherit"
- "dc" -> "Discord / Disconnected"

[PROTOCOL INSTRUCTIONS]
1. Read the incoming message formatted as `[<timestamp>] [<user>]: <message>`.
2. Output detected source language on line 1: `src: <en|de>`.
3. Output translation for the other room language on line 2: `<target_lang>: <translated_message>`.
4. Maintain speaker tags, tone, emojis, and exact domain glossary terms.<|im_end|>
```

### B. Multi-Turn Session Stream
```text
<|im_start|>user
[18.12.2024][13:48:50] [tobi20]: der ansturm geht im wasser<|im_end|>
<|im_start|>assistant
src: de
en: the rush ability works in water<|im_end|>
<|im_start|>user
[18.12.2024][13:49:04] [FreeSoccerHDX]: naja er sollte gehn wenn du nicht schwimsmt vlt<|im_end|>
<|im_start|>assistant
src: de
en: well maybe it should work if you're not swimming<|im_end|>
<|im_start|>user
[09/02/2020 2:15 AM] [Cpl_iPatch]: i have half a stack of netherite blocks<|im_end|>
<|im_start|>assistant
src: en
de: ich habe einen halben Stack Netheritblöcke<|im_end|>
```

---

## 3. Dataset Engineering Pipeline

```
  Raw Sources:
  - protocol.txt (German Dev Chat)
  - minecraft-server-chat_clean.json (English MP Chat)
                         │
                         ▼
             [Step 1: Extract & Clean]
        data/extract_raw.py (Time-gap chunking)
                         │
                         ▼
         [Step 2: LLM Parallel Translation]
          data/translate_corpus.py
       (Google AI Studio Gemini / OpenRouter)
                         │
                         ▼
        Unified Parallel Corpus (JSONL)
        (Every message has text_de + text_en)
                         │
                         ▼
        [Step 3: Multi-Scenario Synthesis]
         data/synthesize_training_data.py
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
  [Mono-DE]         [Mono-EN]       [Multi-Lingual]
  All DE->EN        All EN->DE      User A: DE, User B: EN
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
        [Code-Switching & Hybrid Permutations]
        (Random language flips, varied glossaries)
                         │
                         ▼
          train.jsonl & val.jsonl
        (Formatted in Qwen2.5 ChatML)
```

### Unified Intermediate Format (`unified_conversations.jsonl`)
```json
{
  "conversation_id": "low_dev_20241218_01",
  "domain_context": {
    "domain": "Minecraft Mod Development",
    "description": "Discussion on player movement physics and abilities.",
    "glossary": {
      "LoW": "Legend of War",
      "Ansturm": "Rush / Charge ability",
      "Doppelsprung": "Double Jump"
    }
  },
  "messages": [
    {
      "id": 0,
      "timestamp": "2024-12-18 13:48:50",
      "username": "tobi20",
      "original_lang": "de",
      "raw_input": "der ansturm geht im wasser",
      "text_de": "der Ansturm geht im Wasser",
      "text_en": "the rush ability works in water"
    }
  ]
}
```

### Self-Language Normalization & Clean Bidirectional Translations
To prevent reverse translation degradation (e.g. noisy typos like `"schwimsmen"` $\rightarrow$ `"swimming"`, which if naively reversed would train `"swimming"` $\rightarrow$ `"schwimsmen"`):
- The dataset captures both the **raw original user input** (`raw_input`) and the **clean, canonical text in the original language** (`text_de` or `text_en`).
- When user input is raw German (`raw_input`), the target output is clean English (`text_en`).
- When the synthesized turn is English input (`text_en`), the target output is clean German (`text_de`), ensuring clean, high-quality translations in both directions.

### Dataset Augmentation & Weighting Pipeline
1. **Dataset Weighting & Resampling**:
   - `protocol.txt` (server modding) is smaller but feature-dense. Configurable dataset weights (e.g. `weight_protocol: 10.0`, `weight_minecraft: 1.0`) ensure balanced representation during training.
2. **Username Permutation with In-Text Mention Rewriting**:
   - Sample new usernames from `rawdata/usernames.txt` (65.9M genuine usernames).
   - In-text speaker mentions (e.g. `"LockiBoy, kommt das in den Vulkan?"` $\rightarrow$ `"Tom0614, kommt das in den Vulkan?"`) are systematically updated and validated.
3. **Timestamp & Date Shifting**:
   - Random date/time offsets applied across conversation sessions, preserving inter-message relative time deltas.
4. **Bilingual Minecraft Glossary Expansion**:
   - `rawdata/minecraft_glossary.txt` terms are parsed and paired into German/English nomenclature tables to enrich the domain system prompt.

### Training Scenarios & Distribution
1. **Mono-German Sessions (40%)**: All inputs are German $\rightarrow$ `src: de`, `en: ...`
2. **Mono-English Sessions (40%)**: All inputs are English $\rightarrow$ `src: en`, `de: ...`
3. **Multi-User Mixed Sessions (15%)**: Specific users speak specific languages.
4. **Per-Message Code-Switching (5%)**: Dynamic language switches per turn to prevent overfitting to usernames.
5. **Loss Masking**: System and user prompt tokens receive `labels = -100` so loss is computed strictly over assistant responses.

---

## 4. Model Fine-Tuning (LoRA on Vega 56 ROCm)

- **Script**: `training/train_lora.py`
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Precision**: Pure FP16 mixed precision.
- **LoRA Config**:
  - `r = 16`, `lora_alpha = 32`, `lora_dropout = 0.05`
  - Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Memory Optimization**: Gradient checkpointing enabled ($\approx 2.5\text{ GB}$ peak VRAM during training).
- **Export**: `training/export_model.py` merges LoRA weights back into a standalone FP16 model checkpoint.

---

## 5. Persistent KV Cache Server Engine

```
                          ┌───────────────────────────┐
                          │ Incoming Message Request  │
                          │   (session_id, user, msg) │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Check Session in VRAM?    │
                          └──────┬─────────────┬──────┘
                                 │             │
                       [YES: Cache Hit]   [NO: Cache Miss / New]
                                 │             │
                                 │             ├─► Prefill System Prompt + Glossary
                                 │             └─► Store Base DynamicCache
                                 ▼
                     ┌───────────────────────────────┐
                     │ Append ONLY New Message       │
                     │  (e.g., ~20-30 new tokens)    │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Forward Pass & Decode Output  │
                     │ (TTFT < 10ms, >150 tok/s)     │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Update KV Cache State in VRAM │
                     │ (Slide window if >4k tokens)  │
                     └───────────────────────────────┘
```

### Core Server Architecture
- **`engine/session_kv_manager.py`**:
  - Manages session lifecycle using PyTorch `DynamicCache`.
  - Zero-copy prefix pinning for domain context and glossary.
  - Granular cache slicing for sliding context windows.
- **`engine/server.py`**:
  - FastAPI + WebSockets for real-time streaming translations.
  - Endpoints:
    - `POST /session/create`: Register room with context & glossary.
    - `POST /session/translate`: Push message, get structured translations.
    - `GET /ws/chat/{session_id}`: Real-time token streaming.
    - `GET /stats`: GPU memory, active sessions, cache hit rates.

---

## 6. Testing, Verification & Benchmarking

- **`tools/chat_cli.py`**: Interactive CLI simulating a live multi-user chat room with real-time translations.
- **`tools/benchmark_latency.py`**: Benchmarks measuring:
  - **Cold Turn Latency**: Full prefill of entire chat history.
  - **Warm Turn Latency (Persistent KV Cache)**: Immediate decode latency (< 15ms target).
