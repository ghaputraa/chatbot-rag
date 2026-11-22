import os
import re
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import DATA_DIR, STATIC_DIR, SUPPORTED_EXTENSIONS
from app.core.rag_engine import ConversationalHybridRAG
from app.core.vector_db import (
    delete_document_by_name,
    get_vectorstore,
    sync_documents,
)
from app.schemas.chat import ChatRequest, ChatResponse

WEB_INDEX_PATH = Path(STATIC_DIR) / "index.html"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

vector_store = None
rag_instance: Optional[ConversationalHybridRAG] = None


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
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount folder static (berguna jika kelak Web UI memakai file CSS/JS terpisah)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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

    # Sembunyikan sitasi jika jawaban berupa penolakan atau snippet kosong
    lower_ans = answer_text.lower()
    if "informasi tidak tersedia" in lower_ans or "[sumber: tidak ada]" in lower_ans:
        clean_citations = []
    else:
        clean_citations = [
            c for c in raw_citations
            if isinstance(c, dict) and str(c.get("snippet", "")).strip()
        ]

    return ChatResponse(
        answer=answer_text,
        citations=clean_citations,
        latency_sec=latency,
    )


@app.post("/api/reset")
def reset_memory():
    rag = init_rag_system()
    if hasattr(rag, "clear_history"):
        rag.clear_history()
    else:
        rag.chat_history = []
    return {"status": "ok", "message": "Memori percakapan berhasil direset."}


@app.get("/api/documents")
def list_documents():
    if not os.path.exists(DATA_DIR):
        return {"documents": [], "total": 0}

    docs = sorted(
        [
            f
            for f in os.listdir(DATA_DIR)
            if os.path.isfile(os.path.join(DATA_DIR, f))
            and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]
    )
    return {"documents": docs, "total": len(docs)}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format file tidak didukung. Gunakan {', '.join(SUPPORTED_EXTENSIONS)}.",
        )

    dest_path = os.path.join(DATA_DIR, file.filename)
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        init_rag_system(force_reload=True)
        return {
            "status": "ok",
            "message": f"Dokumen '{file.filename}' berhasil diunggah dan diindeks.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengunggah dokumen: {str(e)}")


@app.delete("/api/documents/{filename}")
def remove_document(filename: str):
    target_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Dokumen '{filename}' tidak ditemukan.")

    try:
        os.remove(target_path)
        delete_document_by_name(vector_store, filename)
        init_rag_system(force_reload=True)
        return {
            "status": "ok",
            "message": f"Dokumen '{filename}' berhasil dihapus dari basis pengetahuan.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghapus dokumen: {str(e)}")