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

from app.config import DATA_DIR, LLM_MODEL, SUPPORTED_EXTENSIONS
from app.core.vector_db import (
    get_vectorstore,
    sync_documents,
    list_indexed_documents,
    delete_document_by_name,
)
from app.core.rag_engine import ConversationalHybridRAG

st.set_page_config(
    page_title="Suzuki Cinere — AI Concierge",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

footer {visibility: hidden;}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 10.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 920px !important;
}

/* Hero Banner dengan Aksen Suzuki (Biru & Merah) */
.hero-wrapper {
    position: relative;
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 1.1rem;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 58, 138, 0.45) 100%);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-top: 3px solid #E11D48;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
}
.hero-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #F87171;
    margin-bottom: 0.25rem;
}
.hero-title {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.55rem 0;
}
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 500;
    background: rgba(59, 130, 246, 0.16);
    border: 1px solid rgba(96, 165, 250, 0.38);
}
.chip-live {
    background: rgba(16, 185, 129, 0.16);
    border-color: rgba(52, 211, 153, 0.4);
}

/* Chat Message Card */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 0.9rem 1rem !important;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(148, 163, 184, 0.2);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}
[data-testid="stChatMessageContent"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow-wrap: break-word !important;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    white-space: normal !important;
    overflow-wrap: break-word !important;
    line-height: 1.6 !important;
}
[data-testid="stChatMessageContent"] table {
    display: block;
    width: 100%;
    overflow-x: auto;
    border-collapse: collapse;
}

/* Sidebar Card */
.kb-card {
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    margin-bottom: 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.25);
    background: rgba(30, 41, 59, 0.35);
    font-size: 0.82rem;
}

.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


def normalize_query_typos(text: str) -> str:
    replacements = {
        r"\bjimmy\b": "JIMNY",
        r"\bmbil(nya)?\b": "mobil",
        r"\btype(nya)?\b": "tipe",
        r"\bdkumen\b": "dokumen",
        r"\bdsen\b": "dosen",
        r"\bperasi\b": "operasi",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


@st.cache_resource(show_spinner="Menyiapkan AI Assistant & Knowledge Base...")
def init_vectorstore():
    os.makedirs(DATA_DIR, exist_ok=True)
    vs = get_vectorstore()
    sync_documents(vs)
    return vs


vector_store = init_vectorstore()

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = ConversationalHybridRAG(vector_store)

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Selamat datang di **Suzuki Cinere AI Concierge**. Tanyakan informasi harga OTR, spesifikasi kendaraan, promo, atau pilih pertanyaan cepat di bawah ini.",
        "citations": [],
    }]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

rag_engine: ConversationalHybridRAG = st.session_state.rag_engine

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### 📂 Knowledge Base")
    indexed_docs = list_indexed_documents(vector_store)
    doc_names = sorted(list(indexed_docs.keys()))
    total_chunks_all = sum(indexed_docs.values())

    st.markdown(
        f'<div class="kb-card">'
        f'<div style="font-size:0.72rem;opacity:0.7;margin-bottom:3px;">DATABASE TERINDEKS</div>'
        f'<div style="font-weight:700;font-size:0.95rem;">📄 {len(doc_names)} File &nbsp;•&nbsp; 🧩 {total_chunks_all} Chunks</div>'
        f'<div style="font-size:0.74rem;opacity:0.75;margin-top:4px;">Engine: <code>{LLM_MODEL}</code></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**1. Unggah Dokumen (.pdf, .docx, .txt)**")
    allowed_exts = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    uploaded_files = st.file_uploader("Upload", type=allowed_exts, accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files and st.button("📥 Simpan & Indeks Dokumen", use_container_width=True, type="primary"):
        with st.spinner("Mengindeks dokumen..."):
            for uf in uploaded_files:
                with open(os.path.join(DATA_DIR, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
            sync_res = sync_documents(vector_store)
            rag_engine.refresh_retriever()
            st.success(f"Berhasil diindeks ({sync_res.get('total_chunks', 0)} chunks).")
            st.rerun()

    st.divider()
    st.markdown("**2. Filter Fokus Dokumen**")
    selected_option = st.selectbox("Filter", ["🌐 Semua Dokumen"] + doc_names, label_visibility="collapsed")
    target_filter = None if selected_option == "🌐 Semua Dokumen" else selected_option
    if rag_engine.active_filter != target_filter:
        rag_engine.set_document_filter(target_filter)

    st.divider()
    st.markdown(f"**3. Daftar Dokumen ({len(doc_names)})**")
    if not doc_names:
        st.info("Belum ada dokumen terindeks.")
    else:
        for name in doc_names:
            c1, c2 = st.columns([4.5, 1])
            c1.markdown(f"📄 **{name}**  \n`{indexed_docs[name]} chunks`")
            if c2.button("🗑️", key=f"del_{name}", help=f"Hapus {name}"):
                delete_document_by_name(vector_store, name)
                p = os.path.join(DATA_DIR, name)
                if os.path.exists(p):
                    os.remove(p)
                if rag_engine.active_filter == name:
                    rag_engine.set_document_filter(None)
                else:
                    rag_engine.refresh_retriever()
                st.rerun()

    st.divider()
    if st.button("🧹 Reset Percakapan", use_container_width=True):
        rag_engine.clear_history()
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Riwayat percakapan telah dibersihkan. Silakan ajukan pertanyaan baru!",
            "citations": [],
        }]
        st.rerun()

# ==============================================================================
# HEADER UTAMA
# ==============================================================================
scope_label = f"🎯 {rag_engine.active_filter}" if rag_engine.active_filter else "🌐 Semua Dokumen"
st.markdown(
    f'<div class="hero-wrapper">'
    f'<div class="hero-kicker">SUZUKI CINERE • DIGITAL ASSISTANT</div>'
    f'<div class="hero-title">🚘 Asisten Virtual & Katalog Produk</div>'
    f'<div class="chip-row">'
    f'<span class="chip chip-live">● Sistem Aktif</span>'
    f'<span class="chip">{scope_label}</span>'
    f'<span class="chip">⚡ Hybrid RAG + Sitasi</span>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# ==============================================================================
# TAMPILAN RIWAYAT CHAT
# ==============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"📚 Sumber Rujukan ({len(citations)} bagian)"):
                for idx, c in enumerate(citations, 1):
                    src = c.get("source", c.get("document", "Dokumen"))
                    pg = c.get("page", "-")
                    snip = str(c.get("snippet", "")).strip()
                    st.markdown(f"**{idx}. {src}** (Hal. {pg})")
                    if snip:
                        st.caption(snip[:300] + ("..." if len(snip) > 300 else ""))

# ==============================================================================
# QUICK PROMPTS (MUNCUL SAAT CHAT BARU DIMULAI)
# ==============================================================================
if len(st.session_state.messages) == 1 and doc_names:
    st.caption("💡 Pertanyaan cepat:")
    q_cols = st.columns(3)
    quick_questions = [
        "Daftar harga OTR mobil Suzuki terbaru?",
        "Rekomendasi mobil keluarga 7 penumpang?",
        "Spesifikasi dan keunggulan Suzuki XL7 Hybrid?",
    ]
    for col, q_text in zip(q_cols, quick_questions):
        if col.button(q_text, use_container_width=True):
            st.session_state.pending_prompt = q_text
            st.rerun()

# ==============================================================================
# PROSES INPUT PERTANYAAN (DARI CHAT INPUT ATAU QUICK PROMPT)
# ==============================================================================
user_input = st.chat_input("Tanyakan harga OTR, spesifikasi mobil, atau isi dokumen...")
active_prompt = st.session_state.pending_prompt or user_input

if active_prompt:
    st.session_state.pending_prompt = None
    if not doc_names:
        st.warning("⚠️ Database dokumen masih kosong. Unggah dokumen di panel kiri.")
    else:
        st.session_state.messages.append({"role": "user", "content": active_prompt, "citations": []})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        clean_query = normalize_query_typos(active_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Menelusuri dokumen & menyusun jawaban..."):
                clean_citations = []
                try:
                    res = rag_engine.ask(clean_query)
                    answer_text = res.get("answer", "")
                    if "informasi tidak tersedia" not in answer_text.lower():
                        clean_citations = [
                            c for c in res.get("citations", [])
                            if isinstance(c, dict) and str(c.get("snippet", "")).strip()
                        ]
                except Exception as e:
                    answer_text = "⏳ Batas API Groq tercapai, silakan coba beberapa detik lagi." if "429" in str(e) else f"❌ Terjadi kesalahan: {e}"

            st.markdown(answer_text)
            if clean_citations:
                with st.expander(f"📚 Sumber Rujukan ({len(clean_citations)} bagian)"):
                    for idx, c in enumerate(clean_citations, 1):
                        src = c.get("source", c.get("document", "Dokumen"))
                        pg = c.get("page", "-")
                        snip = str(c.get("snippet", "")).strip()
                        st.markdown(f"**{idx}. {src}** (Hal. {pg})")
                        if snip:
                            st.caption(snip[:300] + ("..." if len(snip) > 300 else ""))

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "citations": clean_citations,
            })
        st.rerun()
