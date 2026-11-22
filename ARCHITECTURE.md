# 🏗️ Arsitektur Sistem — Conversational Hybrid RAG

Dokumen ini menjelaskan desain arsitektur berlapis (*layered architecture*) yang diterapkan pada sistem **Chatbot Layanan Suzuki Cinere** untuk menjamin modularitas kode, akurasi ekstraksi angka/tabel, serta kemudahan pengembangan paralel antar sub-tim.

---

## 1. Gambaran Umum Arsitektur (4 Lapisan)

Sistem dipisahkan menjadi empat lapisan terisolasi sehingga perubahan pada antarmuka pengguna tidak memengaruhi logika mesin AI maupun penyimpanan vektor:

1. **Presentation Layer (Antarmuka Pengguna):**
   * **Web UI (`app/static/index.html`):** Antarmuka ringan berbasis HTML/CSS/JS yang berkomunikasi melalui REST API JSON.
   * **Streamlit Dashboard (`app/ui_app.py`):** Antarmuka berbasis Python untuk demonstrasi interaktif, inspeksi sumber rujukan, dan manajemen dokumen visual.
   * **CLI Terminal (`main.py`):** Antarmuka baris perintah untuk inspeksi dan sinkronisasi cepat di terminal.

2. **API & Schema Gateway Layer (`app/main.py` & `app/schemas/chat.py`):**
   * Dibangun di atas **FastAPI** dengan validasi kontrak data **Pydantic** (`ChatRequest`, `ChatResponse`).
   * Bertanggung jawab melakukan pra-pemrosesan input (`normalize_query_typos`), manajemen siklus hidup aplikasi (`lifespan`), penyajian aset statis (`/static`), serta pengelolaan unggah/hapus file pada `data/raw/`.

3. **Core RAG & Reasoning Layer (`app/core/rag_engine.py`):**
   * **Conversational Query Reformulation:** Menggabungkan pertanyaan lanjutan singkat (kurang dari atau sama dengan 5 kata) dengan pertanyaan pengguna sebelumnya berdasarkan memori percakapan (`MAX_HISTORY_TURNS = 2`).
   * **Cosine Relevance Gatekeeper:** Menghitung skor kemiripan kosinus teratas terhadap ChromaDB. Jika skor berada di bawah `MIN_RELEVANCE_SCORE` (`0.12`), sistem langsung mengembalikan pesan penolakan terstandar tanpa memanggil LLM.
   * **Hybrid Ensemble Retrieval:** Menggabungkan `VectorStoreRetriever` (`bobot 0.35`) dan `BM25Retriever` (`bobot 0.65`) menggunakan *Reciprocal Rank Fusion* (RRF). Dilengkapi mode penarikan halaman/dokumen utuh otomatis apabila pengguna meminta daftar lengkap atau merujuk nomor halaman spesifik.
   * **Groq Cloud LLM Execution & Fallback:** Mengeksekusi inferensi melalui `ChatGroq` (`temperature = 0.0`) dengan rotasi *API key* dan *model fallback chain* otomatis, serta pembersihan tag `<think>...</think>` untuk model *reasoning*.

4. **Data Ingestion & Persistence Layer (`app/core/vector_db.py` & `app/config.py`):**
   * **Structured Document Extraction:** Menggunakan `pdfplumber` dengan deteksi *bounding box* tabel untuk mengonversi tabel PDF menjadi Markdown Table, serta `Docx2txtLoader` dan `TextLoader` untuk format non-PDF.
   * **Incremental Manifest Indexing:** Melacak waktu modifikasi file (`mtime`) melalui `chroma_db/indexed_manifest.json` sehingga hanya dokumen baru atau yang diperbarui yang diproses ulang saat aplikasi dijalankan.
   * **Vector & Lexical Store:** Menyimpan *dense embeddings* (`paraphrase-multilingual-MiniLM-L12-v2`, ruang metrik `cosine`) di **ChromaDB** lokal (`chroma_db/`) dan membangun indeks **BM25** di memori secara dinamis.

---

## 2. Keputusan Desain & Parameter Kunci (`app/config.py`)

| Parameter | Nilai Konfigurasi | Rasional Teknis |
| :--- | :--- | :--- |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1500` / `250` | Menjaga satu tabel pricelist atau pasal SOP utuh di dalam satu *chunk* agar baris harga tidak terpotong. |
| `HYBRID_WEIGHTS` | `[0.35, 0.65]` | Memberikan bobot lebih tinggi (`65%`) pada BM25 agar kode tipe kendaraan (misal: `XL7`, `GX`, `AT`) dan angka OTR ditemukan secara presisi. |
| `RETRIEVER_K` | `6` | Mengambil cukup banyak kandidat *chunk* lintas brosur tanpa melampaui batas token per menit (TPM) Groq. |
| `MIN_RELEVANCE_SCORE` | `0.12` | Menyaring pertanyaan *out-of-domain* sejak dini sebelum membebani kuota LLM. |
| `LLM_TEMPERATURE` | `0.0` | Memastikan keluaran deterministik dan patuh sepenuhnya pada teks konteks dokumen. |

---

## 3. Arsitektur Evaluasi & Pengujian

* **Benchmarking Pipeline (`scripts/evaluate_rag.py`):** Menguji tiga konfigurasi *retriever* (`vector_only`, `bm25_only`, `hybrid_rag`) secara terotomasi terhadap `data/eval/golden_dataset.json` dan mengekspor metrik *Keyword Recall*, *Citation Accuracy*, *Guardrail Accuracy*, serta *Latency*.
* **Automated Test Suite (`tests/test_rag_pipeline.py`):** Menggunakan `pytest` dan *mocking* (`DummyVectorRetriever`, `TestClient`) untuk memvalidasi logika *Gatekeeper*, deduplikasi sitasi, memori percakapan, dan *endpoint* FastAPI secara deterministik dalam hitungan detik.