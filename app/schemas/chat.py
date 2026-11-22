from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Pertanyaan pengguna")
    selected_doc: Optional[str] = Field(default=None, description="Filter dokumen tertentu")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Jawaban dari mesin Hybrid RAG")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Daftar sitasi sumber")
    latency_sec: float = Field(..., description="Waktu respons dalam detik")