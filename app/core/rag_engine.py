import os
import re
import itertools
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from app.config import (
    GROQ_API_KEYS,
    LLM_MODEL,
    FALLBACK_LLM_MODELS,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    RETRIEVER_K,
    HYBRID_WEIGHTS,
    MAX_HISTORY_TURNS,
    MIN_RELEVANCE_SCORE,
)
from app.core.vector_db import get_all_chunks_as_documents


def clean_llm_output(text: str) -> str:
    """Menghapus blok pemikiran internal <think>...</think> (khusus model Qwen/DeepSeek)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def format_docs(docs, max_chars: int = 14000) -> str:
    formatted = []
    total_len = 0
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Dokumen"))
        page = doc.metadata.get("page", 0) + 1
        total_pages = doc.metadata.get("total_pages", "Tidak diketahui")
        block = f"[Dokumen: {source}, Halaman: {page} dari {total_pages}]\n{doc.page_content}"
        if total_len + len(block) > max_chars:
            break
        formatted.append(block)
        total_len += len(block)
    return "\n\n".join(formatted)


class ConversationalHybridRAG:
    """Mesin RAG Anti-Halusinasi dengan Cosine Gatekeeper, Hybrid Search, dan Memory."""

    def __init__(self, vector_store):
        self.api_keys = list(GROQ_API_KEYS)
        if not self.api_keys:
            raise ValueError("GROQ_API_KEY atau GROQ_API_KEYS tidak ditemukan di file .env!")

        self._key_cycler = itertools.cycle(self.api_keys)
        self.vector_store = vector_store
        self.active_filter = None
        self.chat_history = []

        # Urutan model: Utama -> Fallback dari config.py
        self.model_chain = list(
            dict.fromkeys([LLM_MODEL] + FALLBACK_LLM_MODELS)
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tugas Anda adalah mengekstrak jawaban HANYA dari teks dan metadata pada bagian KONTEKS DOKUMEN.
DILARANG KERAS menggunakan pengetahuan umum atau mengasumsikan hal yang tidak tertulis.

ATURAN WAJIB:
- Jika pengguna meminta penjabaran (misalnya isi pasal, ayat, varian produk, atau rincian harga/paket kredit), tuliskan rinciannya secara lengkap, persis, dan jelas sesuai teks/tabel dokumen.
- Jangan pernah menukar atau mencampur harga, DP, angsuran, maupun spesifikasi antar-tipe/varian produk.
- Selalu jawab dalam Bahasa Indonesia yang formal dan jelas sesuai bahasa pertanyaan pengguna (pertahankan istilah resmi atau nama jabatan asli seperti "Steering Committee" jika memang tertulis demikian di dokumen).
- Jika menyajikan daftar harga dalam jumlah banyak (lebih dari 10 item), gunakan format ringkas per baris: `1. NAMA TIPE: Rp XXX.XXX.XXX` agar seluruh item muat lengkap beserta sitasinya di baris paling akhir.
- Jika pengguna bertanya jumlah total item/varian/pasal/ayat (BUKAN jumlah halaman dokumen):
  1. Daftarkan dan nomori seluruh item unik terlebih dahulu dari awal hingga akhir (1, 2, 3, dst.).
  2. Setelah seluruh daftar selesai ditulis, baru tuliskan kalimat kesimpulan jumlah totalnya.
- Jika pengguna bertanya jumlah halaman dokumen, langsung sebutkan angkanya dari metadata `total_pages` tanpa membuat daftar halaman 1 sampai N.
- JANGAN tulis kalimat pembuka basa-basi (seperti "Berdasarkan dokumen..."). Langsung berikan jawaban inti.
- HANYA jika informasi benar-benar tidak ada sama sekali di KONTEKS DOKUMEN, balas PERSIS dengan kalimat ini:
  Informasi tidak tersedia di dalam dokumen.
- Letakkan sitasi pada BARIS BARU (enter) di bawah teks jawaban dengan format:
  [Sumber: nama_file, Hal. X]

CONTOH JAWABAN YANG BENAR:
Pertanyaan: Berapa lama masa garansi produk?
Jawaban: Masa garansi resmi produk adalah 2 tahun sejak tanggal pembelian.

[Sumber: panduan.pdf, Hal. 1]

RIWAYAT PERCAKAPAN:
{chat_history}

KONTEKS DOKUMEN:
{context}"""),
            ("human", "Pertanyaan: {question}\nJawaban:")
        ])

        self.retriever = self._build_hybrid_retriever()

    def _create_llm(self, model_name: str, api_key: str) -> ChatGroq:
        """Membuat instance ChatGroq dengan konfigurasi retry & reasoning otomatis."""
        llm_kwargs = {
            "model": model_name,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "groq_api_key": api_key,
            "max_retries": 0,   # Langsung pindah key/model tanpa menunggu retry lama jika kena 429
            "timeout": 12.0,
        }
        if "qwen3" in model_name.lower():
            llm_kwargs["reasoning_effort"] = "none"
        return ChatGroq(**llm_kwargs)

    def _invoke_chain_with_fallback(self, payload: dict) -> str:
        """Menjalankan chain dengan rotasi API Key & Fallback Model otomatis (termasuk jika output kosong)."""
        last_error = None

        for model_name in self.model_chain:
            for _ in range(len(self.api_keys)):
                current_key = next(self._key_cycler)
                try:
                    llm = self._create_llm(model_name, current_key)
                    chain = self.prompt | llm | StrOutputParser()
                    raw_res = chain.invoke(payload)
                    cleaned_res = clean_llm_output(raw_res)

                    # Jika model mengembalikan string kosong setelah <think> dihapus, alihkan ke model fallback
                    if not cleaned_res:
                        raise ValueError(f"Model {model_name} mengembalikan output kosong.")

                    return cleaned_res
                except Exception as e:
                    last_error = e
                    err_lower = str(e).lower()
                    if any(
                        k in err_lower
                        for k in (
                            "429",
                            "413",
                            "404",
                            "request too large",
                            "model_not_found",
                            "does not exist",
                            "rate_limit",
                            "tokens per minute",
                            "too many requests",
                            "output kosong",
                        )
                    ):
                        continue
                    raise e

        raise last_error

    def _build_hybrid_retriever(self):
        search_kwargs = {"k": RETRIEVER_K}
        if self.active_filter:
            search_kwargs["filter"] = {"source": self.active_filter}

        vector_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        filtered_docs = get_all_chunks_as_documents(self.vector_store, self.active_filter)

        if not filtered_docs:
            return vector_retriever

        bm25_retriever = BM25Retriever.from_documents(filtered_docs)
        bm25_retriever.k = RETRIEVER_K

        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=HYBRID_WEIGHTS
        )

    def refresh_retriever(self):
        self.retriever = self._build_hybrid_retriever()

    def set_document_filter(self, filename: str | None) -> bool:
        if filename is None:
            self.active_filter = None
            self.refresh_retriever()
            return True

        docs = get_all_chunks_as_documents(self.vector_store, filename)
        if not docs:
            return False

        self.active_filter = os.path.basename(docs[0].metadata.get("source", filename))
        self.refresh_retriever()
        return True

    def clear_history(self):
        self.chat_history = []

    def _format_history(self) -> str:
        if not self.chat_history:
            return "Kosong."
        lines = []
        for q, a in self.chat_history[-MAX_HISTORY_TURNS:]:
            lines.append(f"User: {q}\nAssistant: {a}")
        return "\n".join(lines)

    def _build_search_query(self, query: str) -> str:
        """Menggabungkan pertanyaan terakhir jika user mengirim pertanyaan lanjutan singkat."""
        if self.chat_history and len(query.split()) <= 5:
            last_q = self.chat_history[-1][0]
            return f"{last_q} {query}"
        return query

    def _check_relevance(self, search_query: str) -> tuple[bool, float]:
        """Menghitung skor kemiripan Cosine (0.0 - 1.0)."""
        filter_dict = {"source": self.active_filter} if self.active_filter else None
        results = self.vector_store.similarity_search_with_score(
            search_query, k=1, filter=filter_dict
        )
        if not results:
            return False, 0.0

        cosine_distance = results[0][1]
        similarity_score = max(0.0, 1.0 - cosine_distance)

        # Jika user meminta penjabaran/ringkasan utuh, izinkan lolos ke Hybrid Retriever
        command_keywords = ("jabarkan", "sebutkan", "rincikan", "jelaskan", "utuh", "semua", "seluruh", "daftar", "tabel")
        if any(cmd in search_query.lower() for cmd in command_keywords):
            return True, similarity_score

        return similarity_score >= MIN_RELEVANCE_SCORE, similarity_score

    def ask(self, query: str) -> dict:
        search_query = self._build_search_query(query)

        # 1. Gatekeeper Anti-Halusinasi
        is_relevant, top_score = self._check_relevance(search_query)
        if not is_relevant:
            fallback_msg = f"Informasi tidak tersedia di dalam dokumen. (Skor relevansi terlalu rendah: {top_score:.2f})"
            return {
                "answer": fallback_msg,
                "citations": [],
                "relevance_score": round(top_score, 4),
            }

        # 2. Ambil dokumen awal via Hybrid Retriever
        source_docs = self.retriever.invoke(search_query)
        if not source_docs:
            return {
                "answer": "Informasi tidak tersedia di dalam dokumen.",
                "citations": [],
                "relevance_score": round(top_score, 4),
            }

        primary_source = os.path.basename(source_docs[0].metadata.get("source", ""))

        # 2a. Jika user menyebut nomor halaman spesifik (misal: "halaman 2" / "hal. 2"),
        #     tarik seluruh chunk dari halaman tersebut pada dokumen terkait secara utuh
        page_match = re.search(r"\b(?:halaman|hal\.?)\s*(\d+)\b", query.lower())
        if page_match and primary_source:
            target_page_idx = int(page_match.group(1)) - 1
            all_doc_chunks = get_all_chunks_as_documents(self.vector_store, primary_source)
            page_chunks = [
                d for d in all_doc_chunks
                if d.metadata.get("page") == target_page_idx
            ]
            if page_chunks:
                source_docs = page_chunks

        # 2b. Jika user meminta daftar/penjabaran menyeluruh (tanpa filter halaman spesifik),
        #     tarik seluruh chunk dokumen utama secara berurutan agar daftar tidak terpotong
        elif primary_source:
            full_list_triggers = ("seluruh", "semua", "total", "daftar", "lengkap", "utuh", "keseluruhan")
            if any(w in search_query.lower() for w in full_list_triggers):
                all_doc_chunks = get_all_chunks_as_documents(self.vector_store, primary_source)
                if all_doc_chunks:
                    source_docs = all_doc_chunks

        # Gunakan batas 14.000 karakter agar payload tidak melampaui limit TPM Groq (error 413)
        context_text = format_docs(source_docs, max_chars=14000)
        history_text = self._format_history()

        # 3. Eksekusi LLM dengan Auto-Rotation Key, Fallback Model & Anti-Empty Output
        answer = self._invoke_chain_with_fallback({
            "context": context_text,
            "chat_history": history_text,
            "question": query
        })

        # 4. Pastikan sitasi turun ke baris baru
        answer = re.sub(r"\s*(\[Sumber:)", r"\n\n\1", answer)

        # Simpan ke memori percakapan (potong jawaban panjang ke 300 karakter agar hemat token)
        if "Informasi tidak tersedia di dalam dokumen" not in answer:
            short_answer = answer[:300] + "..." if len(answer) > 300 else answer
            self.chat_history.append((query, short_answer))

        citations = []
        seen_citations = set()
        for doc in source_docs:
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            pg = doc.metadata.get("page", 0) + 1
            key = f"{src}_{pg}"
            if key not in seen_citations:
                seen_citations.add(key)
                citations.append({
                    "source": src,
                    "page": pg,
                    "snippet": doc.page_content[:130].replace("\n", " ")
                })

        return {
            "answer": answer,
            "citations": citations,
            "relevance_score": round(top_score, 4),
        }