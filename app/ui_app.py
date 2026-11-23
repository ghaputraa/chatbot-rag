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
    page_title="Suzuki Cinere — Hybrid RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
footer {visibility: hidden;}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 9.5rem !important;
    max-width: 100%;
}
.hero-card {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(15, 23, 42, 0.55) 100%);
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.hero-title {font-size: 1.3rem; font-weight: 700; margin: 0 0 0.45rem 0;}
.badge-pill {
    display: inline-block; padding: 0.18rem 0.6rem; margin: 0.15rem 0.25rem 0 0;
    border-radius: 999px; font-size: 0.75rem; font-weight: 500;
    background: rgba(59,130,246,0.18); border: 1px solid rgba(96,165,250,0.45);
}
.badge-green {background: rgba(16,185,129,0.18); border-color: rgba(52,211,153,0.45);}
[data-testid="stChatMessage"] {
    border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.7rem;
    border: 1px solid rgba(148, 163, 184, 0.2);
}
.sidebar-stat {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 10px; padding: 0.7rem 0.85rem; margin-bottom: 0.8rem; font-size: 0.82rem;
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
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Halo! Silakan ajukan pertanyaan seputar brosur, spesifikasi kendaraan, harga OTR Suzuki Cinere, atau unggah dokumen baru di panel kiri.",
        "citations": [],
    }]

rag_engine: ConversationalHybridRAG = st.session_state.rag_engine

with st.sidebar:
    st.markdown("### 📂 Manajemen Dokumen")
    indexed_docs = list_indexed_documents(vector_store)
    doc_names = sorted(list(indexed_docs.keys()))
    total_chunks_all = sum(indexed_docs.values())

    st.markdown(
        f'<div class="sidebar-stat"><b>📄 {len(doc_names)} Dokumen • 🧩 {total_chunks_all} Chunks</b><br>'
        f'<span style="opacity:0.75;font-size:0.75rem;">Model: {LLM_MODEL}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("**1. Unggah Dokumen Baru**")
    allowed_exts = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    uploaded_files = st.file_uploader("Upload", type=allowed_exts, accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files and st.button("📥 Simpan & Indeks", use_container_width=True, type="primary"):
        with st.spinner("Mengindeks dokumen..."):
            for uf in uploaded_files:
                with open(os.path.join(DATA_DIR, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
            sync_res = sync_documents(vector_store)
            rag_engine.refresh_retriever()
            st.success(f"Tersimpan ({sync_res.get('total_chunks', 0)} chunks).")
            st.rerun()

    st.divider()
    st.markdown("**2. Filter Cakupan Pencarian**")
    selected_option = st.selectbox("Filter", ["🌐 Semua Dokumen"] + doc_names, label_visibility="collapsed")
    target_filter = None if selected_option == "🌐 Semua Dokumen" else selected_option
    if rag_engine.active_filter != target_filter:
        rag_engine.set_document_filter(target_filter)

    st.divider()
    st.markdown(f"**3. Dokumen Terindeks ({len(doc_names)})**")
    if not doc_names:
        st.info("Belum ada dokumen.")
    else:
        for name in doc_names:
            c1, c2 = st.columns([4.5, 1])
            c1.markdown(f"📄 **{name}**  \n`{indexed_docs[name]} chunks`")
            if c2.button("🗑️", key=f"del_{name}"):
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
    if st.button("🧹 Reset Riwayat Chat", use_container_width=True):
        rag_engine.clear_history()
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Riwayat percakapan direset. Silakan ajukan pertanyaan baru!",
            "citations": [],
        }]
        st.rerun()

scope_label = f"🎯 {rag_engine.active_filter}" if rag_engine.active_filter else "🌐 Semua Dokumen"
st.markdown(
    f'<div class="hero-card"><div class="hero-title">🤖 Asisten Virtual Suzuki Cinere</div>'
    f'<span class="badge-pill badge-green">● Online</span>'
    f'<span class="badge-pill">{scope_label}</span>'
    f'<span class="badge-pill">⚡ Hybrid RAG</span></div>',
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"].replace("\n", "  \n"))
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

if prompt := st.chat_input("Tanyakan harga OTR, spesifikasi mobil, atau isi dokumen..."):
    if not doc_names:
        st.warning("⚠️ Database dokumen masih kosong.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        clean_query = normalize_query_typos(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Mencari informasi..."):
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
                    answer_text = "⏳ Batas API Groq tercapai, coba lagi sebentar." if "429" in str(e) else f"❌ Error: {e}"

            st.markdown(answer_text.replace("\n", "  \n"))
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
