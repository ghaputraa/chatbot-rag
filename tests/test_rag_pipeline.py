import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Pastikan root proyek (chatbot-rag/) terbaca saat pytest dijalankan dari mana saja
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app, normalize_query_typos
from app.core.rag_engine import ConversationalHybridRAG, clean_llm_output


# ==============================================================================
# DUMMY RETRIEVER YANG VALID SEBAGAI RUNNABLE LANGCHAIN
# ==============================================================================
class DummyVectorRetriever(BaseRetriever):
    """Retriever palsu yang sah sebagai turunan BaseRetriever untuk EnsembleRetriever."""
    docs: List[Document] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.docs


# ==============================================================================
# FIXTURE: MOCK VECTORSTORE & RAG ENGINE TERISOLASI
# ==============================================================================
@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content=(
                "| Tipe Mobil | Transmisi | Harga OTR Jakarta |\n"
                "| --- | --- | --- |\n"
                "| JIMNY 5-DOOR AT TWO TONE | AT | Rp 478.600.000 |\n"
                "| ALL NEW ERTIGA HYBRID GX AT | AT | Rp 298.400.000 |"
            ),
            metadata={"source": "pricelist_suzuki_cinere.pdf", "page": 0, "total_pages": 2},
        )
    ]


@pytest.fixture
def mock_vectorstore(sample_docs):
    """Membuat mock ChromaDB dengan retriever yang kompatibel dengan Pydantic LangChain."""
    vs = MagicMock()
    dummy_retriever = DummyVectorRetriever(docs=sample_docs)

    vs.as_retriever.return_value = dummy_retriever
    # Default jarak cosine 0.25 -> similarity_score = 0.75
    vs.similarity_search_with_score.return_value = [(sample_docs[0], 0.25)]
    return vs


@pytest.fixture
def rag_engine(mock_vectorstore, sample_docs, monkeypatch):
    """Inisialisasi ConversationalHybridRAG dengan API key dummy dan ChromaDB mock."""
    monkeypatch.setattr("app.core.rag_engine.GROQ_API_KEYS", ["gsk_test_dummy_key_12345"])
    monkeypatch.setattr(
        "app.core.rag_engine.get_all_chunks_as_documents",
        lambda vs, source_filter=None: sample_docs,
    )
    return ConversationalHybridRAG(mock_vectorstore)


# ==============================================================================
# 1. PENGUJIAN COSINE GATEKEEPER & ANTI-HALUSINASI
# ==============================================================================
class TestGatekeeperAndGuardrail:
    def test_reject_out_of_domain_low_similarity(self, rag_engine, mock_vectorstore):
        """Query di luar domain (skor cosine < MIN_RELEVANCE_SCORE) wajib ditolak tanpa panggil LLM."""
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Document(page_content="Teks acak"), 0.95)
        ]

        with patch.object(rag_engine, "_invoke_chain_with_fallback") as mock_llm:
            result = rag_engine.ask("Berapa harga mobil Toyota Avanza tipe G?")

            assert "Informasi tidak tersedia di dalam dokumen" in result["answer"]
            assert result["citations"] == []
            assert result["relevance_score"] < 0.12
            mock_llm.assert_not_called()

    def test_accept_in_domain_high_similarity(self, rag_engine, mock_vectorstore):
        """Query relevan lolos Gatekeeper dan menghasilkan jawaban beserta sitasi valid."""
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Document(page_content="JIMNY 5-DOOR AT TWO TONE Rp 478.600.000"), 0.20)
        ]

        with patch.object(
            rag_engine,
            "_invoke_chain_with_fallback",
            return_value="Harga OTR Jimny 5-Door AT Two Tone adalah Rp 478.600.000. [Sumber: pricelist_suzuki_cinere.pdf, Hal. 1]",
        ):
            result = rag_engine.ask("Berapa harga OTR Suzuki Jimny 5-Door AT Two Tone?")

            assert "Rp 478.600.000" in result["answer"]
            assert "\n\n[Sumber:" in result["answer"]
            assert len(result["citations"]) == 1
            assert result["citations"][0]["source"] == "pricelist_suzuki_cinere.pdf"
            assert result["citations"][0]["page"] == 1


# ==============================================================================
# 2. PENGUJIAN INTEGRITAS SITASI & PEMBERSIHAN OUTPUT LLM
# ==============================================================================
class TestCitationAndOutputFormatting:
    def test_clean_think_tags_from_reasoning_models(self):
        """Blok internal <think>...</think> dari model reasoning wajib dibersihkan."""
        raw_output = "<think>Mencari harga di tabel baris 1...</think>Harga OTR adalah Rp 298.400.000."
        assert clean_llm_output(raw_output) == "Harga OTR adalah Rp 298.400.000."

    def test_citation_deduplication(self, rag_engine):
        """Beberapa chunk dari dokumen & halaman yang sama tidak boleh menghasilkan sitasi duplikat."""
        dup_docs = [
            Document(
                page_content="Bagian 1 tabel harga.",
                metadata={"source": "pricelist_suzuki_cinere.pdf", "page": 0},
            ),
            Document(
                page_content="Bagian 2 tabel harga di halaman yang sama.",
                metadata={"source": "pricelist_suzuki_cinere.pdf", "page": 0},
            ),
        ]
        rag_engine.retriever = DummyVectorRetriever(docs=dup_docs)

        with patch.object(
            rag_engine,
            "_invoke_chain_with_fallback",
            return_value="Daftar harga tersedia. [Sumber: pricelist_suzuki_cinere.pdf, Hal. 1]",
        ):
            result = rag_engine.ask("Tampilkan harga mobil Suzuki")
            assert len(result["citations"]) == 1
            assert result["citations"][0]["page"] == 1


# ==============================================================================
# 3. PENGUJIAN MULTI-TURN MEMORY & NORMALISASI TYPO
# ==============================================================================
class TestConversationMemoryAndTypo:
    def test_short_followup_query_merges_previous_question(self, rag_engine):
        """Pertanyaan lanjutan singkat (<= 5 kata) digabungkan dengan konteks pertanyaan sebelumnya."""
        rag_engine.chat_history.append(
            ("Berapa harga OTR Suzuki Jimny 5-Door?", "Harga OTR Jimny 5-Door adalah Rp 478.600.000.")
        )
        merged = rag_engine._build_search_query("Kalau TDP nya berapa?")
        assert "Suzuki Jimny 5-Door" in merged
        assert "TDP" in merged

    def test_normalize_query_typos(self):
        """Fungsi normalisasi memperbaiki typo umum pelanggan sebelum masuk ke retriever."""
        assert normalize_query_typos("harga mbil jimmy type terbaru") == "harga mobil JIMNY tipe terbaru"


# ==============================================================================
# 4. PENGUJIAN ENDPOINT REST API FASTAPI (app/main.py)
# ==============================================================================
class TestFastAPIEndpoints:
    @pytest.fixture
    def client(self, rag_engine):
        with patch("app.main.init_rag_system", return_value=rag_engine):
            with TestClient(app) as test_client:
                yield test_client

    def test_health_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_chat_endpoint_rejects_empty_query(self, client):
        response = client.post("/api/chat", json={"query": "   "})
        assert response.status_code == 400

    def test_chat_endpoint_hides_citations_when_out_of_domain(self, client, rag_engine):
        """Endpoint /api/chat wajib mengosongkan list citations saat jawaban berupa penolakan."""
        with patch.object(
            rag_engine,
            "ask",
            return_value={
                "answer": "Informasi tidak tersedia di dalam dokumen. (Skor relevansi terlalu rendah: 0.04)",
                "citations": [{"source": "pricelist.pdf", "page": 1, "snippet": "teks sisa"}],
            },
        ):
            response = client.post("/api/chat", json={"query": "Siapa presiden pertama Indonesia?"})
            assert response.status_code == 200
            data = response.json()
            assert "Informasi tidak tersedia" in data["answer"]
            assert data["citations"] == []
            assert "latency_sec" in data

    def test_reset_memory_endpoint(self, client, rag_engine):
        rag_engine.chat_history = [("Q1", "A1")]
        response = client.post("/api/reset")
        assert response.status_code == 200
        assert rag_engine.chat_history == []