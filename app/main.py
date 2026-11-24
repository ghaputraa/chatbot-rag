import json
import mimetypes
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import DATA_DIR, STATIC_DIR, SUPPORTED_MIME_TYPES
from app.core.rag_engine import ConversationalHybridRAG
from app.core.vector_db import (
    delete_document_by_name,
    detect_mime_type,
    get_vectorstore,
    sync_documents,
)
from app.schemas.chat import ChatRequest, ChatResponse, FeedbackRequest

# Konfigurasi Log & Nomor WhatsApp Staf Manusia (Dealer Suzuki Cinere)
LOG_DIR = os.path.join(os.path.dirname(DATA_DIR), "logs")
CHAT_LOG_FILE = os.path.join(LOG_DIR, "chat_logs.json")
STAFF_WHATSAPP_NUMBER = "6281234567890"  # Ganti dengan nomor resmi Sales/CS Suzuki Cinere

mimetypes.add_type("application/pdf", ".pdf")
mimetypes.add_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")
mimetypes.add_type("text/plain", ".txt")

WEB_INDEX_PATH = Path(STATIC_DIR) / "index.html"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

vector_store = None
rag_instance: Optional[ConversationalHybridRAG] = None


# ==============================================================================
# UTILITAS LOGGING, ESKALASI & NORMALISASI
# ==============================================================================
def _load_logs() -> list[dict]:
    if os.path.exists(CHAT_LOG_FILE):
        try:
            with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_logs(logs: list[dict]):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def append_chat_log(
    turn_id: str, question: str, answer: str, citations: list, needs_escalation: bool
):
    logs = _load_logs()
    logs.append({
        "turn_id": turn_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "citations_count": len(citations),
        "sources": [c.get("source", "-") for c in citations],
        "feedback": None,
        "escalated_to_human": needs_escalation,
    })
    _save_logs(logs)


def update_chat_feedback(turn_id: str, feedback_value: str) -> Optional[dict]:
    logs = _load_logs()
    updated_entry = None
    for entry in logs:
        if entry.get("turn_id") == turn_id:
            entry["feedback"] = feedback_value
            if feedback_value == "down":
                entry["escalated_to_human"] = True
            updated_entry = entry
            break
    if updated_entry:
        _save_logs(logs)
    return updated_entry


def build_whatsapp_escalation_url(user_question: str) -> str:
    msg = (
        f"Halo Staf Suzuki Cinere, saya membutuhkan bantuan lebih lanjut terkait pertanyaan berikut:\n"
        f"\"{user_question}\""
    )
    return f"https://wa.me/{STAFF_WHATSAPP_NUMBER}?text={quote(msg)}"


def init_rag_system(force_reload: bool = False) -> ConversationalHybridRAG:
    global vector_store, rag_instance
    if vector_store is None:
        vector_store = get_vectorstore()
    if rag_instance is None or force_reload:
        sync_documents(vector_store)
        rag_instance = ConversationalHybridRAG(vector_store)
    return rag_instance


def normalize_query_typos(text: str) -> str:
    """Memperbaiki typo umum pengguna agar BM25 dan Semantic Search tetap akurat."""
    replacements = {
        r"\bjimmy\b": "JIMNY",
        r"\bmbil(nya)?\b": "mobil",
        r"\btype(nya)?\b": "tipe",
        r"\bdkumen\b": "dokumen",
        r"\bdsen\b": "dosen",
        r"\bperasi\b": "operasi",
    }
    cleaned = text
    for pattern, repl in replacements.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
    return cleaned


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_rag_system()
    yield


app = FastAPI(
    title="API Chatbot Layanan Suzuki Cinere (RAG)",
    description="Backend REST API & Web Interface untuk Chatbot Layanan Berbasis Hybrid RAG dengan Sitasi Sumber.",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==============================================================================
# ENDPOINT UTAMA & PERCAKAPAN
# ==============================================================================
@app.get("/", include_in_schema=False)
def serve_web_ui():
    if not WEB_INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="File app/static/index.html tidak ditemukan.",
        )
    return FileResponse(str(WEB_INDEX_PATH))


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Suzuki Cinere RAG API"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    raw_query = payload.query.strip()
    if not raw_query:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")

    query = normalize_query_typos(raw_query)
    rag = init_rag_system()

    if hasattr(rag, "set_document_filter"):
        rag.set_document_filter(payload.selected_doc or None)

    start_time = time.perf_counter()
    try:
        result = rag.ask(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses pertanyaan: {str(e)}")
    latency = round(time.perf_counter() - start_time, 2)

    answer_text = result.get("answer", "")
    raw_citations = result.get("citations", [])

    # Deteksi jawaban penolakan & status eskalasi ke staf manusia
    lower_ans = answer_text.lower()
    if (
        "informasi tidak tersedia" in lower_ans
        or "tidak ditemukan" in lower_ans
        or "[sumber: tidak ada]" in lower_ans
    ):
        clean_citations = []
        needs_escalation = True
        escalation_url = build_whatsapp_escalation_url(raw_query)
    else:
        clean_citations = [
            c for c in raw_citations
            if isinstance(c, dict) and str(c.get("snippet", "")).strip()
        ]
        needs_escalation = False
        escalation_url = None

    turn_id = str(uuid.uuid4())[:8]
    append_chat_log(turn_id, raw_query, answer_text, clean_citations, needs_escalation)

    return ChatResponse(
        turn_id=turn_id,
        answer=answer_text,
        citations=clean_citations,
        latency_sec=latency,
        needs_escalation=needs_escalation,
        escalation_url=escalation_url,
    )


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackRequest):
    updated = update_chat_feedback(payload.turn_id, payload.feedback)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Log percakapan dengan turn_id '{payload.turn_id}' tidak ditemukan.",
        )

    escalation_url = (
        build_whatsapp_escalation_url(updated.get("question", ""))
        if payload.feedback == "down"
        else None
    )
    return {
        "status": "ok",
        "turn_id": payload.turn_id,
        "feedback": payload.feedback,
        "needs_escalation": payload.feedback == "down",
        "escalation_url": escalation_url,
    }


@app.get("/api/logs")
def get_chat_logs():
    logs = _load_logs()
    up_count = sum(1 for item in logs if item.get("feedback") == "up")
    down_count = sum(1 for item in logs if item.get("feedback") == "down")
    esc_count = sum(1 for item in logs if item.get("escalated_to_human"))
    return {
        "summary": {
            "total": len(logs),
            "thumbs_up": up_count,
            "thumbs_down": down_count,
            "escalated": esc_count,
        },
        "logs": logs,
    }


@app.post("/api/reset")
def reset_memory():
    rag = init_rag_system()
    if hasattr(rag, "clear_history"):
        rag.clear_history()
    else:
        rag.chat_history = []
    return {"status": "ok", "message": "Memori percakapan berhasil direset."}


# ==============================================================================
# ENDPOINT PANEL ADMIN MANAJEMEN DOKUMEN (MIME-ONLY)
# ==============================================================================
@app.get("/api/documents")
def list_documents():
    if not os.path.exists(DATA_DIR):
        return {"documents": [], "total": 0}

    docs = sorted(
        [
            f
            for f in os.listdir(DATA_DIR)
            if os.path.isfile(os.path.join(DATA_DIR, f))
            and detect_mime_type(os.path.join(DATA_DIR, f)) in SUPPORTED_MIME_TYPES
        ]
    )
    return {"documents": docs, "total": len(docs)}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    safe_filename = os.path.basename(file.filename or "")
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Nama file tidak valid.")

    guessed_mime, _ = mimetypes.guess_type(safe_filename)
    incoming_mime = file.content_type or guessed_mime or "application/octet-stream"

    if incoming_mime not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"MIME type '{incoming_mime}' tidak didukung. Gunakan: {', '.join(SUPPORTED_MIME_TYPES)}.",
        )

    dest_path = os.path.join(DATA_DIR, safe_filename)
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        actual_mime = detect_mime_type(dest_path)
        if actual_mime not in SUPPORTED_MIME_TYPES:
            os.remove(dest_path)
            raise HTTPException(
                status_code=400,
                detail=f"Isi file tidak valid (terdeteksi sebagai '{actual_mime}').",
            )

        init_rag_system(force_reload=True)
        return {
            "status": "ok",
            "message": f"Dokumen '{safe_filename}' ({actual_mime}) berhasil diunggah dan diindeks.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengunggah dokumen: {str(e)}")


@app.delete("/api/documents/{filename}")
def remove_document(filename: str):
    safe_filename = os.path.basename(filename)
    target_path = os.path.join(DATA_DIR, safe_filename)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Dokumen '{safe_filename}' tidak ditemukan.")

    try:
        os.remove(target_path)
        delete_document_by_name(vector_store, safe_filename)
        init_rag_system(force_reload=True)
        return {
            "status": "ok",
            "message": f"Dokumen '{safe_filename}' berhasil dihapus dari basis pengetahuan.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghapus dokumen: {str(e)}")