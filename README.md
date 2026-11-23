# 🚗 Suzuki Cinere — Conversational Hybrid RAG Assistant

Asisten tanya-jawab dokumen berbasis **Conversational Hybrid Retrieval-Augmented Generation (RAG)** dengan perlindungan anti-halusinasi (*Cosine Gatekeeper*), sitasi sumber halaman otomatis, serta dukungan **tiga mode antarmuka** (CLI Terminal, Web UI FastAPI, dan Dashboard Streamlit).

Sistem ini dirancang khusus untuk melayani pencarian informasi produk, daftar harga OTR, simulasi paket kredit/TDP, spesifikasi varian kendaraan, dan SOP layanan resmi **Dealer Suzuki Cinere** secara akurat tanpa fabrikasi data.

---

## ✨ Fitur Utama

1. **Hybrid Search Retrieval (`0.35 Vector + 0.65 BM25`):** Menggabungkan pencarian semantik (`paraphrase-multilingual-MiniLM-L12-v2` via ChromaDB) dengan pencarian kata kunci eksak (`BM25Okapi` via `EnsembleRetriever`) agar kode varian kendaraan dan nominal harga Rupiah terbaca presisi.
2. **Ekstraksi Tabel PDF Presisi (`pdfplumber`):** Memisahkan teks narasi dan tabel harga PDF menjadi format Markdown Table terstruktur sehingga kolom tipe, transmisi, dan harga tidak bergeser saat dipecah (*chunking*).
3. **Cosine Gatekeeper & Anti-Halusinasi:** Menyaring pertanyaan di luar domain menggunakan ambang batas kemiripan kosinus (`MIN_RELEVANCE_SCORE = 0.12`) sebelum memanggil LLM, serta mewajibkan setiap jawaban menyertakan sitasi `[Sumber: nama_file.pdf, Hal. X]`.
4. **Resilient Groq Cloud LLM Chain:** Mendukung rotasi otomatis multi-API key (`GROQ_API_KEYS`) dan *fallback* berantai antar-model (`openai/gpt-oss-20b` → `qwen/qwen3.8-27b` → `openai/gpt-oss-120b`) untuk mengantisipasi batas *rate-limit* (HTTP 429).
5. **Conversational Memory & Query Normalization:** Menyimpan konteks percakapan multi-giliran (*multi-turn*) dan memperbaiki *typo* umum pelanggan sebelum melakukan pencarian.
6. **Multi-Interface Ready:** Mendukung pengujian lewat **CLI Terminal** (`main.py`), **REST API + Web UI** (`app/main.py` + `app/static/index.html`), dan **Streamlit App** (`app/ui_app.py`).

---

## 📁 Struktur Direktori Repositori

```text
chatbot-rag/
├── main.py                               # Entry point CLI Terminal interaktif
├── pytest.ini                            # Konfigurasi ruang lingkup pengujian otomatis
├── requirements.txt                      # Daftar dependensi Python
├── .env.example                          # Template variabel environment
├── .gitignore                            # Pengecualian file lokal (.env, venv, chroma_db)
│
├── app/                                  # Paket utama aplikasi
│   ├── __init__.py
│   ├── main.py                           # Server REST API FastAPI (/api/chat, upload, docs)
│   ├── ui_app.py                         # Antarmuka visual berbasis Streamlit
│   ├── config.py                         # Konfigurasi terpusat (path, parameter RAG, keys)
│   ├── core/                             # Mesin inti pemrosesan dokumen & AI
│   │   ├── __init__.py
│   │   ├── vector_db.py                  # Ekstraksi PDF, chunking, manifest & ChromaDB
│   │   └── rag_engine.py                 # Hybrid Search, Gatekeeper, Memory & Groq LLM
│   ├── schemas/                          # Validasi request/response Pydantic
│   │   ├── __init__.py
│   │   └── chat.py
│   └── static/                           # Antarmuka Web UI (HTML/CSS/JS)
│       └── index.html
│
├── data/                                 # Penyimpanan dokumen & dataset evaluasi
│   ├── raw/                              # Dokumen sumber (.pdf, .docx, .txt)
│   └── eval/                             # Golden Dataset & hasil benchmark (T4)
│       └── golden_dataset.json
│
├── scripts/                              # Skrip evaluasi & eksperimen retrieval
│   └── evaluate_rag.py                   # Benchmark Vector vs BM25 vs Hybrid RAG
│
├── tests/                                # Pengujian otomatis terisolasi (T5)
│   ├── __init__.py
│   └── test_rag_pipeline.py              # Unit & integration tests (pytest)
│
├── docs/                                 # Dokumentasi akademik & laporan (T1 – T5)
│   └── pdf/
│
├── README.md                             # Panduan utama proyek
├── ARCHITECTURE.md                       # Penjelasan arsitektur berlapis
└── DIAGRAMS.md                           # Diagram alur sistem (Mermaid)
```

---

## 🚀 Panduan Instalasi & Menjalankan Aplikasi

### 1. Persiapan Lingkungan Virtual & Dependensi
```powershell
# 1. Buat lingkungan virtual (virtual environment)
python -m venv venv

# 2. Aktifkan virtual environment (Windows PowerShell)
.\venv\Scripts\activate

# 3. Instal seluruh dependensi ke dalam venv aktif
python -m pip install -r requirements.txt
```

### 2. Konfigurasi API Key (`.env`)
Salin `.env.example` menjadi `.env` di *root* folder proyek dan masukkan API key Groq Anda:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Opsional: pisahkan beberapa key dengan koma untuk rotasi otomatis
GROQ_API_KEYS=gsk_key_1,gsk_key_2
```

### 3. Menambahkan Dokumen Pengetahuan
Letakkan seluruh file brosur, *pricelist* OTR/TDP, dan SOP dealer (`.pdf`, `.docx`, `.txt`) ke dalam folder:
```text
data/raw/
```

### 4. Menjalankan Antarmuka Pilihan

* **Mode 1 — Web UI & REST API FastAPI (Port `8000`):**
  ```powershell
  python -m uvicorn app.main:app --reload
  ```
  Akses antarmuka web di `http://127.0.0.1:8000` atau dokumentasi API Swagger di `http://127.0.0.1:8000/docs`.

* **Mode 2 — Dashboard Interaktif Streamlit (Port `8501`):**
  ```powershell
  python -m streamlit run app/ui_app.py
  ```
  Akses aplikasi di `http://localhost:8501`.

* **Mode 3 — CLI Terminal Interaktif:**
  ```powershell
  python main.py
  ```
  Mendukung perintah cepat `/list`, `/sync`, `/fokus <nama_file>`, `/hapus <nama_file>`, dan `/reset`.

---

## 🧪 Evaluasi & Pengujian Otomatis

* **Menjalankan Benchmark 3 Metode Retrieval (Tugas 4):**
  ```powershell
  python scripts/evaluate_rag.py
  ```
  Menghasilkan perbandingan performa **Vector Only**, **BM25 Only**, dan **Hybrid RAG** ke dalam `data/eval/benchmark_summary.csv`.

* **Menjalankan Unit & API Integration Tests (Tugas 5):**
  ```powershell
  pytest -v
  ```

---

## 👥 Struktur Tim Pengembang (Produk B)

* **Ketua Tim Besar:** Angga Putra Prasetia Nugraha
* **Sub-Tim Data & Domain:** Kayla *(Koordinator)*, Itsna, Sendi
* **Sub-Tim Model & Analitik:** Neisya *(Koordinator)*, Tia, Angga
* **Sub-Tim Pengembangan Aplikasi:** Hafidz *(Koordinator)*, Bintang, Habibah 
* **Sub-Tim Pengujian & Kelayakan:** Aisyah *(Koordinator)*, Selvi, Oyang
