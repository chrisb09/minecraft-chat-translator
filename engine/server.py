"""
server.py

FastAPI REST & WebSocket server for Real-time Multi-Participant Chat Translation
powered by Persistent GPU Session KV Cache Manager on AMD Vega 56.
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .session_kv_manager import SessionKVManager
from . import config

app = FastAPI(
    title="Vega 56 Real-Time Chat Translation Engine",
    description="Low-latency multi-participant chat translation server with persistent GPU KV cache",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Manager Instance
kv_manager: Optional[SessionKVManager] = None


@app.on_event("startup")
def startup_event():
    global kv_manager
    # TRANSLATE_DEVICE defaults to "cpu": the ROCm torch build in the vega_torch env
    # can still see the Vega 56, which the model must NOT use (GPU unstable on this host).
    device = os.environ.get("TRANSLATE_DEVICE", "cpu")
    print(f"--> [FastAPI] Initializing Session KV Manager on device '{device}'...")
    kv_manager = SessionKVManager(device_override=device)


class CreateSessionRequest(BaseModel):
    session_id: str
    domain: Optional[str] = "Minecraft Multiplayer & Mod Development"
    glossary: Optional[Dict[str, str]] = None
    room_languages: Optional[List[str]] = Field(default_factory=lambda: ["en", "de"])


class TranslateMessageRequest(BaseModel):
    session_id: str
    username: str
    message: str
    timestamp: Optional[str] = None
    max_new_tokens: Optional[int] = 128


class TranslateResponse(BaseModel):
    session_id: str
    turn_id: int
    timestamp: str
    username: str
    message: str
    src_lang: str
    translations: Dict[str, str]
    raw_output: str
    ttft_ms: float
    total_time_ms: float
    tokens_per_second: float
    total_cached_tokens: int


@app.get("/health")
def health_check():
    device = str(kv_manager.device) if kv_manager is not None else "not initialized"
    return {"status": "ok", "device": device}


@app.get("/stats")
def get_engine_stats():
    if kv_manager is None:
        raise HTTPException(status_code=503, detail="KV Manager not initialized")
    return kv_manager.get_stats()


@app.post("/session/create")
def create_session(req: CreateSessionRequest):
    if kv_manager is None:
        raise HTTPException(status_code=503, detail="KV Manager not initialized")

    session = kv_manager.create_session(
        session_id=req.session_id,
        domain=req.domain,
        glossary=req.glossary,
        room_languages=req.room_languages
    )
    return {
        "status": "created",
        "session_id": session.session_id,
        "system_tokens_cached": session.system_token_count,
        "room_languages": session.room_languages
    }


@app.post("/session/translate", response_model=TranslateResponse)
def translate_message(req: TranslateMessageRequest):
    if kv_manager is None:
        raise HTTPException(status_code=503, detail="KV Manager not initialized")

    ts = req.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        turn_info = kv_manager.process_turn(
            session_id=req.session_id,
            username=req.username,
            timestamp=ts,
            message=req.message,
            max_new_tokens=req.max_new_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"translation failed: {e}")

    # Protocol validation: callers (cinder) treat any non-200 as "backend failed"
    # and fall back to DeepL/LibreTranslate, so reject malformed generations here.
    if turn_info.get("src_lang", "unknown") in ("", "unknown") or not turn_info.get("translations"):
        raise HTTPException(status_code=502, detail="model output failed protocol validation")

    return TranslateResponse(
        session_id=req.session_id,
        **turn_info
    )


@app.post("/session/reset")
def reset_session(session_id: str):
    if kv_manager is None:
        raise HTTPException(status_code=503, detail="KV Manager not initialized")

    success = kv_manager.reset_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "reset", "session_id": session_id}


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if kv_manager is None:
        await websocket.close(code=1011, reason="KV Manager not ready")
        return

    # Ensure session exists
    kv_manager.get_or_create_session(session_id)
    await websocket.send_json({"type": "connected", "session_id": session_id})

    try:
        while True:
            data = await websocket.receive_json()
            username = data.get("username", "User")
            message = data.get("message", "")
            timestamp = data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not message:
                continue

            turn_info = kv_manager.process_turn(
                session_id=session_id,
                username=username,
                timestamp=timestamp,
                message=message
            )

            await websocket.send_json({
                "type": "translation",
                "session_id": session_id,
                **turn_info
            })

    except WebSocketDisconnect:
        print(f"--> WebSocket disconnected for session: {session_id}")


def main():
    import uvicorn
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)


if __name__ == "__main__":
    main()
