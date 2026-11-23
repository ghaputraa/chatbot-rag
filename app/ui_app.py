import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import re
import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

from app.config import DATA_DIR, LLM_MODEL, FALLBACK_LLM_MODELS, SUPPORTED_EXTENSIONS
from app.core.vector_db import (
    get_vectorstore,
    sync_documents,
    list_indexed_documents,
    delete_document_by_name,
)
from app.core.rag_engine import ConversationalHybridRAG

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Suzuki Cinere — Hybrid RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS UNTUK TAMPILAN ELEGAN & RESPONSIF
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sembunyikan menu/footer bawaan agar bersih saat presentasi */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Kontainer utama lebih proporsional */
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 5rem;
        max-width: 960px;
    }

    /* Header Banner Utama */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.28) 0%, rgba(15, 23, 42, 0.65) 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
        color: #F8FAFC;
    }
    .hero-subtitle {
        font-size: 0.86rem;
        color: #CBD5E1;
        margin: 0;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 500;
        background: rgba(59, 130, 246, 0.16);
        border: 1px solid rgba(96, 165, 250, 0.35);
        color: #93C5FD;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(52, 211, 153, 0.35);
        color: #6EE7B7;
    }

    /* Styling Chat Bubble */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background-color: rgba(30, 41, 59, 0.38);
    }

    /* Styling Expander Sitasi */
    [data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background-color: rgba(15, 23, 42, 0.45);
        margin-top: 0.5rem;
    }

    /* Kartu Statistik di Sidebar */
    .sidebar-stat {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 0.75rem 0.95rem;
        margin-bottom: 0.85rem;
        font-size: 0.82rem;
    }

    /* Tombol lebih halus */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }

    /* Responsif layar HP */
    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.25rem;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }
        .hero-title {
            font-size: 1.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


# ==============================================================================
# INISIALISASI RESOURCE & SESSION STATE
# ==============================================================================
@st.cache_resource(show_spinner="Memuat model embedding & ChromaDB...")
def init_vectorstore():
    os.makedirs(DATA_DIR, exist_ok=True)
    vs = get_vectorstore()
    sync_documents(vs)
    return vs


vector_store = init_vectorstore()

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = ConversationalHybridRAG(vector_store)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Halo! Saya asisten virtual dokumen **Suzuki Cinere**. Silakan ajukan pertanyaan seputar brosur, spesifikasi kendaraan, daftar harga OTR, atau unggah dokumen baru melalui panel sebelah kiri.",
            "citations": [],
        }
    ]

rag_engine: ConversationalHybridRAG = st.session_state.rag_engine

# ==============================================================================
# SIDEBAR: MANAJEMEN DOKUMEN & PENGATURAN RAG
# ==============================================================================
with st.sidebar:
    st.markdown("### 📂 Manajemen Dokumen")

    indexed_docs = list_indexed_documents(vector_store)
    doc_names = sorted(list(indexed_docs.keys()))
    total_chunks_all = sum(indexed_docs.values())

    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div style="color:#94A3B8; font-size:0.75rem; margin-bottom:4px;">STATUS KNOWLEDGE BASE</div>
            <div style="font-weight:600; font-size:0.95rem; color:#F1F5F9;">
                📄 {len(doc_names)} Dokumen &nbsp;•&nbsp; 🧩 {total_chunks_all} Chunks
            </div>
            <div style="color:#94A3B8; font-size:0.74rem; margin-top:6px;">
                LLM: <code>{LLM_MODEL}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Upload Dokumen Multi-Format
    st.markdown("**1. Unggah Dokumen Baru**")
    allowed_exts = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    uploaded_files = st.file_uploader(
        "Pilih file (.pdf, .docx, .txt)",
        type=allowed_exts,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("📥 Simpan & Indeks Dokumen", use_container_width=True, type="primary"):
        with st.spinner("Menyimpan dan mengindeks dokumen ke ChromaDB..."):
            saved_count = 0
            for uf in uploaded_files:
                file_path = os.path.join(DATA_DIR, uf.name)
                with open(file_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_count += 1

            sync_res = sync_documents(vector_store)
            rag_engine.refresh_retriever()
            total_chunks = sync_res.get("total_chunks", 0)
            st.success(f"{saved_count} file berhasil diindeks ({total_chunks} chunks).")
            st.rerun()

    st.divider()

    # 2. Dropdown Filter Fokus Dokumen
    st.markdown("**2. Filter Cakupan Pencarian**")
    filter_options = ["🌐 Semua Dokumen"] + doc_names
    selected_option = st.selectbox(
        "Fokus pencarian pada:",
        options=filter_options,
        index=0,
        label_visibility="collapsed",
    )

    if selected_option == "🌐 Semua Dokumen":
        if rag_engine.active_filter is not None:
            rag_engine.set_document_filter(None)
    else:
        if rag_engine.active_filter != selected_option:
            rag_engine.set_document_filter(selected_option)

    st.divider()

    # 3. Daftar Dokumen & Hapus Dokumen
    st.markdown(f"**3. Dokumen Terindeks ({len(doc_names)})**")
    if not doc_names:
        st.info("Belum ada dokumen di database. Unggah file di atas untuk memulai.")
    else:
        for name in doc_names:
            col_name, col_del = st.columns([4.5, 1])
            with col_name:
                st.markdown(f"📄 **{name}**  \n<span style='color:#94A3B8;font-size:0.78rem;'>{indexed_docs[name]} chunks</span>", unsafe_allow_html=True)
            with col_del:
                if st.button("🗑️", key=f"del_{name}", help=f"Hapus {name}"):
                    delete_document_by_name(vector_store, name)
                    local_path = os.path.join(DATA_DIR, name)
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    if rag_engine.active_filter == name:
                        rag_engine.set_document_filter(None)
                    else:
                        rag_engine.refresh_retriever()
                    st.toast(f"Dokumen '{name}' berhasil dihapus.")
                    st.rerun()

    st.divider()

    # 4. Reset Percakapan & Memori
    if st.button("🧹 Reset Riwayat Chat", use_container_width=True):
        rag_engine.clear_history()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Riwayat percakapan dan memori konteks telah direset. Silakan ajukan pertanyaan baru!",
                "citations": [],
            }
        ]
        st.rerun()

# ==============================================================================
# AREA UTAMA: HERO HEADER & ANTARMUKA CHAT
# ==============================================================================
active_scope_text = (
    f"🎯 Fokus: {rag_engine.active_filter}"
    if rag_engine.active_filter
    else "🌐 Semua Dokumen"
)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">🤖 Asisten Virtual Suzuki Cinere</div>
        <div class="hero-subtitle">
            <span class="badge-pill badge-green">● Online</span>
            <span class="badge-pill">{active_scope_text}</span>
            <span class="badge-pill">⚡ Hybrid Search + Anti-Halusinasi</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tampilkan seluruh riwayat chat bubble
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"].replace("\n", "  \n"))
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"📚 Sumber Rujukan ({len(citations)} bagian)"):
                for idx, c in enumerate(citations, 1):
                    src_name = c.get("source", c.get("document", "Dokumen"))
                    page_num = c.get("page", "-")
                    snippet = str(c.get("snippet", "")).strip()
                    st.markdown(f"**{idx}. {src_name}** — *Halaman {page_num}*")
                    if snippet:
                        st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))

# Input pertanyaan pengguna di bagian bawah layar
if prompt := st.chat_input("Tanyakan harga OTR, spesifikasi mobil, atau isi dokumen..."):
    if not doc_names:
        st.warning("⚠️ Database dokumen masih kosong. Silakan unggah dokumen terlebih dahulu melalui sidebar kiri.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        clean_query = normalize_query_typos(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Mencari informasi & menyusun jawaban..."):
                clean_citations = []
                try:
                    result = rag_engine.ask(clean_query)
                    answer_text = result.get("answer", "")
                    raw_citations = result.get("citations", [])

                    lower_ans = answer_text.lower()
                    if "informasi tidak tersedia" not in lower_ans and "[sumber: tidak ada]" not in lower_ans:
                        clean_citations = [
                            c for c in raw_citations
                            if isinstance(c, dict) and str(c.get("snippet", "")).strip()
                        ]
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        answer_text = "⏳ Batas kecepatan API Groq tercapai. Tunggu beberapa detik lalu coba kembali."
                    else:
                        answer_text = f"❌ Terjadi kesalahan sistem: {e}"

            st.markdown(answer_text.replace("\n", "  \n"))
            if clean_citations:
                with st.expander(f"📚 Sumber Rujukan ({len(clean_citations)} bagian)"):
                    for idx, c in enumerate(clean_citations, 1):
                        src_name = c.get("source", c.get("document", "Dokumen"))
                        page_num = c.get("page", "-")
                        snippet = str(c.get("snippet", "")).strip()
                        st.markdown(f"**{idx}. {src_name}** — *Halaman {page_num}*")
                        if snippet:
                            st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "citations": clean_citations,
            })
