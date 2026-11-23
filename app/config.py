import os
from pathlib import Path
from dotenv import load_dotenv

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Tentukan root folder proyek (chatbot-rag/) secara dinamis
BASE_DIR = Path(__file__).resolve().parent.parent

# Muat file .env dari root folder proyek
load_dotenv(BASE_DIR / ".env")

# Path Direktori (Disinkronkan dengan struktur folder baru)
DATA_DIR = str(BASE_DIR / "data" / "raw")
EVAL_DIR = str(BASE_DIR / "data" / "eval")
STATIC_DIR = str(BASE_DIR / "app" / "static")
CHROMA_DIR = str(BASE_DIR / "chroma_db")

COLLECTION_NAME = "rag_docs_collection"
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")

# Konfigurasi Model & Embedding
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-20b"
FALLBACK_LLM_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1200

# Konfigurasi Chunking & Hybrid Retrieval
CHUNK_SIZE = 1500             # 1 tabel/pasal utuh masuk dalam 1 chunk
CHUNK_OVERLAP = 250           # Batas antar-baris/ayat tetap terhubung
RETRIEVER_K = 6               # Jumlah kandidat dokumen teratas yang diambil
HYBRID_WEIGHTS = [0.35, 0.65] # 0.35 Vector + 0.65 BM25 agar nama varian & angka harga akurat
MIN_RELEVANCE_SCORE = 0.12    # Ambang batas Cosine Gatekeeper

# Konfigurasi Memori Percakapan
MAX_HISTORY_TURNS = 2

# API Keys (Mendukung banyak key dipisah koma di GROQ_API_KEYS atau 1 key di GROQ_API_KEY)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = [
    k.strip()
    for k in (os.getenv("GROQ_API_KEYS") or GROQ_API_KEY).split(",")
    if k.strip()
]

