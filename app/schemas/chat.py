from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Pertanyaan pengguna")
    selected_doc: Optional[str] = Field(
        default=None, description="Filter nama dokumen spesifik (opsional)"
    )


class ChatResponse(BaseModel):
    turn_id: str
    answer: str
    citations: List[Dict[str, Any]] = []
    latency_sec: float
    needs_escalation: bool = False
    escalation_url: Optional[str] = None


class FeedbackRequest(BaseModel):
    turn_id: str = Field(..., min_length=1, description="ID unik percakapan")
    feedback: Literal["up", "down"] = Field(
        ..., description="Nilai umpan balik: 'up' (jempol atas) atau 'down' (jempol bawah)"
    )