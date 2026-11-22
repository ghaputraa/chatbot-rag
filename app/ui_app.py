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
            "content": "Halo! Silakan ajukan pertanyaan seputar brosur, harga OTR, atau layanan Suzuki Cinere, atau unggah dokumen baru melalui panel di sebelah kiri.",
            "citations": [],
        }
    ]

rag_engine: ConversationalHybridRAG = st.session_state.rag_engine

# ==============================================================================
# SIDEBAR: MANAJEMEN DOKUMEN & PENGATURAN RAG
# ==============================================================================
with st.sidebar:
    st.title("📂 Manajemen Dokumen")
    st.caption(f"Model Utama: `{LLM_MODEL}`\nFallback: `{', '.join(FALLBACK_LLM_MODELS)}`")
    st.divider()

    # 1. Upload Dokumen Multi-Format
    st.subheader("1. Unggah Dokumen Baru")
    allowed_exts = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    uploaded_files = st.file_uploader(
        "Pilih file (.pdf, .docx, .txt)",
        type=allowed_exts,
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("📥 Simpan & Indeks Dokumen", use_container_width=True):
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
            st.success(f"{saved_count} file disimpan (Total database: {total_chunks} chunks).")
            st.rerun()

    st.divider()

    # 2. Dropdown Filter Fokus Dokumen
    st.subheader("2. Filter Cakupan Pencarian")
    indexed_docs = list_indexed_documents(vector_store)
    doc_names = sorted(list(indexed_docs.keys()))

    filter_options = ["🌐 Semua Dokumen"] + doc_names
    selected_option = st.selectbox(
        "Fokus pencarian pada:",
        options=filter_options,
        index=0,
    )

    if selected_option == "🌐 Semua Dokumen":
        if rag_engine.active_filter is not None:
            rag_engine.set_document_filter(None)
    else:
        if rag_engine.active_filter != selected_option:
            rag_engine.set_document_filter(selected_option)

    st.divider()

    # 3. Daftar Dokumen & Hapus Dokumen
    st.subheader(f"3. Dokumen Terindeks ({len(doc_names)})")
    if not doc_names:
        st.info("Belum ada dokumen di database. Unggah file di atas untuk memulai.")
    else:
        for name in doc_names:
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.markdown(f"📄 **{name}**  \n`{indexed_docs[name]} chunks`")
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
    if st.button("🧹 Reset Riwayat Chat & Memori", use_container_width=True):
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
# AREA UTAMA: ANTARMUKA CHAT BUBBLE
# ==============================================================================
st.header("🤖 Asisten Tanya-Jawab Suzuki Cinere (Hybrid RAG)")

active_scope_label = (
    f"🎯 Fokus Dokumen: **{rag_engine.active_filter}**"
    if rag_engine.active_filter
    else "🌐 Cakupan: **Semua Dokumen Terindeks**"
)
st.caption(f"{active_scope_label} | Mode: Hybrid Search (Cosine Vector + BM25) + Anti-Halusinasi")

# Tampilkan seluruh riwayat chat bubble
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("\n", "  \n"))
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"📚 Lihat Sumber Rujukan ({len(citations)} bagian)"):
                for idx, c in enumerate(citations, 1):
                    src_name = c.get("source", c.get("document", "Dokumen"))
                    page_num = c.get("page", "-")
                    snippet = str(c.get("snippet", "")).strip()
                    st.markdown(f"**{idx}. {src_name}** (Halaman {page_num})")
                    if snippet:
                        st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))

# Input pertanyaan pengguna di bagian bawah layar
if prompt := st.chat_input("Ketik pertanyaan Anda berdasarkan isi dokumen..."):
    if not doc_names:
        st.warning("⚠️ Database dokumen masih kosong. Silakan unggah dokumen terlebih dahulu melalui sidebar kiri.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        clean_query = normalize_query_typos(prompt)
        with st.chat_message("assistant"):
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
                with st.expander(f"📚 Lihat Sumber Rujukan ({len(clean_citations)} bagian)"):
                    for idx, c in enumerate(clean_citations, 1):
                        src_name = c.get("source", c.get("document", "Dokumen"))
                        page_num = c.get("page", "-")
                        snippet = str(c.get("snippet", "")).strip()
                        st.markdown(f"**{idx}. {src_name}** (Halaman {page_num})")
                        if snippet:
                            st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "citations": clean_citations,
            })