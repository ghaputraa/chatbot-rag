# 📊 Diagram Alur Sistem — Suzuki Cinere Hybrid RAG

Dokumen ini menyajikan visualisasi alur pemrosesan data (*ingestion*), alur tanya-jawab (*inference pipeline*), dan arsitektur komponen menggunakan sintaks **Mermaid**.

---

## 1. Diagram Arsitektur Komponen Sistem

```mermaid
flowchart TB
    subgraph UI["Presentation Layer"]
        WEB["Web UI<br/>(app/static/index.html)"]
        ST["Streamlit Dashboard<br/>(app/ui_app.py)"]
        CLI["CLI Terminal<br/>(main.py)"]
    end

    subgraph API["API Gateway Layer"]
        FAST["FastAPI Server<br/>(app/main.py)"]
        SCHEMA["Pydantic Schemas<br/>(app/schemas/chat.py)"]
    end

    subgraph CORE["Core RAG Engine (app/core/rag_engine.py)"]
        MEM["Conversation Memory<br/>& Query Builder"]
        GATE{"Cosine Gatekeeper<br/>(Score >= 0.12?)"}
        HYBRID["Ensemble Retriever<br/>(0.35 Vector + 0.65 BM25)"]
        GROQ["Groq Cloud LLM Chain<br/>(Key Rotation + Model Fallback)"]
    end

    subgraph DATA["Data & Storage Layer (app/core/vector_db.py)"]
        RAW[("Dokumen Dealer<br/>(data/raw/*.pdf,docx,txt)")]
        CHROMA[("ChromaDB Vector Store<br/>(chroma_db/)")]
        MANIFEST["indexed_manifest.json"]
    end

    WEB --> FAST
    FAST <--> SCHEMA
    FAST --> MEM
    ST --> MEM
    CLI --> MEM

    MEM --> GATE
    GATE -- "Tidak (< 0.12)" --> REJECT["Tolak: Informasi tidak tersedia"]
    GATE -- "Ya (>= 0.12)" --> HYBRID
    HYBRID --> GROQ
    GROQ --> RESP["Jawaban Terstruktur + Sitasi"]

    RAW --> CHROMA
    RAW -.-> MANIFEST
    CHROMA <--> GATE
    CHROMA <--> HYBRID
```

---

## 2. Diagram Alur Inferensi & Anti-Halusinasi (*Sequence Diagram*)

```mermaid
sequenceDiagram
    autonumber
    actor User as Pengguna (Web / Streamlit / CLI)
    participant API as FastAPI / UI Handler
    participant RAG as ConversationalHybridRAG
    participant DB as ChromaDB & BM25
    participant LLM as Groq Cloud API

    User->>API: Kirim pertanyaan (query)
    API->>API: Normalisasi typo (normalize_query_typos)
    API->>RAG: ask(query)
    RAG->>RAG: Gabungkan konteks riwayat jika follow-up singkat

    RAG->>DB: similarity_search_with_score(k=1)
    DB-->>RAG: Skor kemiripan kosinus (top_score)

    alt top_score < 0.12 (Out-of-Domain)
        RAG-->>API: "Informasi tidak tersedia di dalam dokumen."
        API-->>User: Tampilkan penolakan (tanpa sitasi)
    else top_score >= 0.12 (In-Domain)
        RAG->>DB: Hybrid Retrieval (Vector 0.35 + BM25 0.65)
        DB-->>RAG: Top-K Chunks + Metadata Halaman
        RAG->>LLM: Kirim System Prompt + Konteks + Pertanyaan
        LLM-->>RAG: Jawaban + Tag Sitasi
        RAG->>RAG: Bersihkan tag <think> & deduplikasi sitasi
        RAG->>RAG: Simpan ringkasan ke chat_history
        RAG-->>API: Dict(answer, citations, relevance_score)
        API-->>User: Tampilkan jawaban & sumber rujukan
    end
```

---

## 3. Diagram Pipeline Ingesti Dokumen (`sync_documents`)

```mermaid
flowchart LR
    A["Scan Folder data/raw/"] --> B{"Cek mtime di<br/>indexed_manifest.json"}
    B -- "Tidak Berubah" --> C["Lewati File (Skip)"]
    B -- "Baru / Diubah" --> D{"Ekstensi File?"}
    D -- ".pdf" --> E["pdfplumber:<br/>Pisahkan Teks & Tabel Markdown"]
    D -- ".docx / .txt" --> F["Docx2txtLoader / TextLoader"]
    E --> G["RecursiveCharacterTextSplitter<br/>(chunk_size=1500, overlap=250)"]
    F --> G
    G --> H["Generate MD5 Chunk ID<br/>& HuggingFace Embeddings"]
    H --> I[("Simpan ke ChromaDB &<br/>Perbarui Manifest")]
```