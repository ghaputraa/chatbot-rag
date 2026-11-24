import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import re
import time
import json
import uuid
import mimetypes
import warnings
from datetime import datetime
from urllib.parse import quote
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

from app.config import DATA_DIR, LLM_MODEL, SUPPORTED_MIME_TYPES
from app.core.vector_db import (
    get_vectorstore,
    sync_documents,
    list_indexed_documents_info,
    delete_document_by_name,
)
from app.core.rag_engine import ConversationalHybridRAG

# Konfigurasi Log, Keamanan Admin & Nomor WhatsApp Staf Manusia (Dealer Suzuki Cinere)
LOG_DIR = os.path.join(os.path.dirname(DATA_DIR), "logs")
CHAT_LOG_FILE = os.path.join(LOG_DIR, "chat_logs.json")
STAFF_WHATSAPP_NUMBER = os.getenv("STAFF_WHATSAPP_NUMBER", "6281234567890")  # Ganti dengan nomor resmi Sales/CS Suzuki Cinere
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminsuzuki2026")

mimetypes.add_type("application/pdf", ".pdf")
mimetypes.add_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")
mimetypes.add_type("text/plain", ".txt")

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

.kb-card {
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    margin-bottom: 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.25);
    background: rgba(30, 41, 59, 0.35);
    font-size: 0.82rem;
}

.escalation-box {
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-top: 0.65rem;
    background: rgba(225, 29, 72, 0.12);
    border: 1px solid rgba(244, 63, 94, 0.45);
    font-size: 0.85rem;
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


# ==============================================================================
# UTILITAS KEAMANAN (UU PDP), LOGGING & NORMALISASI
# ==============================================================================
def mask_pii(text: str) -> str:
    """Menyensor data pribadi (Email, NIK 16 digit, No. HP) sebelum disimpan ke log sesuai UU PDP."""
    if not text:
        return text
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_DISENSOR]", text)
    text = re.sub(r"\b\d{16}\b", "[NIK_DISENSOR]", text)
    text = re.sub(r"(\+62|62|0)8[1-9][0-9]{7,10}\b", "[NO_HP_DISENSOR]", text)
    return text


def _load_logs() -> list[dict]:
    os.makedirs(LOG_DIR, exist_ok=True)
    if os.path.exists(CHAT_LOG_FILE):
        try:
            with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_logs(logs: list[dict]):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def append_chat_log(
    turn_id: str,
    question: str,
    answer: str,
    citations: list,
    needs_escalation: bool,
    response_time: float,
):
    logs = _load_logs()
    logs.append({
        "turn_id": turn_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": mask_pii(question),
        "answer": mask_pii(answer),
        "citations_count": len(citations),
        "sources": [c.get("source", "-") for c in citations],
        "response_time_sec": response_time,
        "feedback": None,
        "escalated_to_human": needs_escalation,
    })
    _save_logs(logs)


def update_chat_feedback(turn_id: str, feedback_value: str, escalate: bool = False):
    logs = _load_logs()
    for entry in logs:
        if entry.get("turn_id") == turn_id:
            entry["feedback"] = feedback_value
            if escalate:
                entry["escalated_to_human"] = True
            break
    _save_logs(logs)


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


def build_whatsapp_escalation_url(user_question: str) -> str:
    msg = (
        f"Halo Staf Suzuki Cinere, saya membutuhkan bantuan lebih lanjut terkait pertanyaan berikut:\n"
        f"\"{user_question}\""
    )
    return f"https://wa.me/{STAFF_WHATSAPP_NUMBER}?text={quote(msg)}"


# ==============================================================================
# INISIALISASI RESOURCE & SESSION STATE
# ==============================================================================
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
        "content": "Selamat datang di **Suzuki Cinere AI Concierge**. Tanyakan informasi harga OTR, paket kredit SUFI Eazy-Flexi, paket kredit Reguler ADDB, atau pilih pertanyaan cepat di bawah ini.",
        "citations": [],
        "turn_id": None,
        "question": "",
        "feedback": None,
        "needs_escalation": False,
        "response_time": None,
    }]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

rag_engine: ConversationalHybridRAG = st.session_state.rag_engine

# ==============================================================================
# SIDEBAR: FILTER DOKUMEN & PANEL ADMIN TERPROTEKSI
# ==============================================================================
with st.sidebar:
    st.markdown("### 📂 Knowledge Base & Panel Admin")
    docs_info = list_indexed_documents_info(vector_store)
    doc_names = sorted(list(docs_info.keys()))
    total_chunks_all = sum(item["chunks"] for item in docs_info.values())

    st.markdown(
        f'<div class="kb-card">'
        f'<div style="font-size:0.72rem;opacity:0.7;margin-bottom:3px;">DATABASE TERINDEKS</div>'
        f'<div style="font-weight:700;font-size:0.95rem;">📄 {len(doc_names)} File &nbsp;•&nbsp; 🧩 {total_chunks_all} Chunks</div>'
        f'<div style="font-size:0.74rem;opacity:0.75;margin-top:4px;">Engine: <code>{LLM_MODEL}</code></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Filter Fokus Dokumen (Dapat diakses semua pengguna/pelanggan)
    st.markdown("**1. Filter Fokus Dokumen**")
    selected_option = st.selectbox("Filter", ["🌐 Semua Dokumen"] + doc_names, label_visibility="collapsed")
    target_filter = None if selected_option == "🌐 Semua Dokumen" else selected_option
    if rag_engine.active_filter != target_filter:
        rag_engine.set_document_filter(target_filter)

    st.divider()

    # Area Terkunci Password (Pembatasan Akses Dokumen & Audit Log)
    st.markdown("**🔒 2. Kontrol Akses Admin**")
    if not st.session_state.admin_authenticated:
        admin_pwd_input = st.text_input(
            "Kata Sandi Admin",
            type="password",
            placeholder="Masukkan sandi admin...",
            label_visibility="collapsed",
        )
        if st.button("🔑 Masuk Mode Admin", use_container_width=True):
            if admin_pwd_input == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Kata sandi admin salah.")
    else:
        st.success("Mode Admin Aktif")
        if st.button("🚪 Keluar Mode Admin", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()

        st.divider()
        st.markdown("**Unggah Dokumen (Validasi MIME)**")
        uploader_types = [
            ext.lstrip(".")
            for mime in SUPPORTED_MIME_TYPES
            if (ext := mimetypes.guess_extension(mime))
        ]
        uploaded_files = st.file_uploader(
            "Upload",
            type=uploader_types,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files and st.button("📥 Simpan & Indeks Dokumen", use_container_width=True, type="primary"):
            valid_files = []
            rejected_files = []

            for uf in uploaded_files:
                guessed_mime, _ = mimetypes.guess_type(uf.name)
                file_mime = uf.type or guessed_mime or "application/octet-stream"
                if file_mime in SUPPORTED_MIME_TYPES:
                    valid_files.append(uf)
                else:
                    rejected_files.append(f"{uf.name} ({file_mime})")

            if rejected_files:
                st.error(f"MIME type ditolak: {', '.join(rejected_files)}")

            if valid_files:
                with st.spinner("Mengindeks dokumen..."):
                    for uf in valid_files:
                        with open(os.path.join(DATA_DIR, uf.name), "wb") as f:
                            f.write(uf.getbuffer())
                    sync_res = sync_documents(vector_store)
                    rag_engine.refresh_retriever()
                    st.success(f"{len(valid_files)} file berhasil diindeks ({sync_res.get('total_chunks', 0)} chunks).")
                    st.rerun()

        st.divider()
        st.markdown(f"**Daftar Dokumen ({len(doc_names)})**")
        if not doc_names:
            st.info("Belum ada dokumen terindeks.")
        else:
            for name in doc_names:
                c1, c2 = st.columns([4.5, 1])
                chunk_cnt = docs_info[name]["chunks"]
                mime_val = docs_info[name]["mime_type"]
                c1.markdown(f"📄 **{name}**  \n`{chunk_cnt} chunks` • `{mime_val}`")
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
        st.markdown("**Log Percakapan & Latensi**")
        all_logs = _load_logs()
        up_count = sum(1 for l in all_logs if l.get("feedback") == "up")
        down_count = sum(1 for l in all_logs if l.get("feedback") == "down")
        esc_count = sum(1 for l in all_logs if l.get("escalated_to_human"))
        timed_logs = [l.get("response_time_sec") for l in all_logs if l.get("response_time_sec") is not None]
        avg_latency = round(sum(timed_logs) / len(timed_logs), 2) if timed_logs else 0.0

        st.caption(f"Total Log: **{len(all_logs)}** | ⏱️ Rata-rata: **{avg_latency} dtk**")
        st.caption(f"👍 {up_count} | 👎 {down_count} | 🧑 Eskalasi: {esc_count}")

        if all_logs:
            st.download_button(
                label="📥 Unduh Log Percakapan (JSON)",
                data=json.dumps(all_logs, indent=2, ensure_ascii=False),
                file_name="suzuki_cinere_chat_logs.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()
    if st.button("🧹 Reset Percakapan", use_container_width=True):
        rag_engine.clear_history()
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Riwayat percakapan telah dibersihkan. Silakan ajukan pertanyaan baru!",
            "citations": [],
            "turn_id": None,
            "question": "",
            "feedback": None,
            "needs_escalation": False,
            "response_time": None,
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
    f'<span class="chip">⚡ Hybrid RAG + UU PDP Safe</span>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# ==============================================================================
# TAMPILAN RIWAYAT CHAT, LATENSI, SITASI, FEEDBACK & ESKALASI STAF
# ==============================================================================
for idx_msg, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # Tampilkan waktu respons (detik) di bawah setiap balasan AI
            if msg.get("response_time") is not None:
                st.caption(f"⏱️ Dijawab dalam **{msg['response_time']} detik**")

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

            # Kotak Eskalasi ke Staf Manusia jika informasi tidak ditemukan atau user memberi jempol bawah
            if msg.get("needs_escalation") or msg.get("feedback") == "down":
                wa_url = build_whatsapp_escalation_url(msg.get("question", "Pertanyaan layanan Suzuki Cinere"))
                st.markdown(
                    f'<div class="escalation-box">'
                    f'<strong>🧑 Butuh Bantuan Staf Manusia?</strong><br>'
                    f'Informasi spesifik belum tersedia di dokumen atau Anda memerlukan bantuan lebih lanjut. '
                    f'Hubungi langsung tim Sales / Customer Service Suzuki Cinere:<br><br>'
                    f'<a href="{wa_url}" target="_blank" style="display:inline-block;padding:0.4rem 0.85rem;'
                    f'background:#16A34A;color:#FFFFFF;border-radius:8px;text-decoration:none;font-weight:600;">'
                    f'💬 Hubungi Staf Suzuki Cinere (WhatsApp)</a>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Tombol Umpan Balik Jempol Atas / Bawah untuk setiap respons AI
            turn_id = msg.get("turn_id")
            if turn_id:
                fb_state = msg.get("feedback")
                col_fb1, col_fb2, col_fb3 = st.columns([1.2, 1.2, 5.6])
                with col_fb1:
                    if st.button(
                        "👍 Membantu" if fb_state == "up" else "👍",
                        key=f"up_{turn_id}",
                        type="primary" if fb_state == "up" else "secondary",
                    ):
                        st.session_state.messages[idx_msg]["feedback"] = "up"
                        update_chat_feedback(turn_id, "up")
                        st.rerun()
                with col_fb2:
                    if st.button(
                        "👎 Kurang Tepat" if fb_state == "down" else "👎",
                        key=f"down_{turn_id}",
                        type="primary" if fb_state == "down" else "secondary",
                    ):
                        st.session_state.messages[idx_msg]["feedback"] = "down"
                        st.session_state.messages[idx_msg]["needs_escalation"] = True
                        update_chat_feedback(turn_id, "down", escalate=True)
                        st.rerun()

# ==============================================================================
# QUICK PROMPTS
# ==============================================================================
if len(st.session_state.messages) == 1 and doc_names:
    st.caption("💡 Pertanyaan cepat:")
    q_cols = st.columns(3)
    quick_questions = [
        "Daftar harga OTR resmi Suzuki Jabodetabek?",
        "Simulasi kredit SUFI Eazy-Flexi Jadetabek?",
        "Ketentuan paket kredit Reguler ADDB Serang-Bogor?",
    ]
    for col, q_text in zip(q_cols, quick_questions):
        if col.button(q_text, use_container_width=True):
            st.session_state.pending_prompt = q_text
            st.rerun()

# ==============================================================================
# PROSES INPUT PERTANYAAN
# ==============================================================================
user_input = st.chat_input("Tanyakan harga OTR, kredit SUFI Eazy-Flexi, atau kredit Reguler ADDB...")
active_prompt = st.session_state.pending_prompt or user_input

if active_prompt:
    st.session_state.pending_prompt = None
    if not doc_names:
        st.warning("⚠️ Database dokumen masih kosong. Unggah dokumen di panel kiri.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": active_prompt,
            "citations": [],
        })
        with st.chat_message("user"):
            st.markdown(active_prompt)

        clean_query = normalize_query_typos(active_prompt)
        turn_id = str(uuid.uuid4())[:8]

        with st.chat_message("assistant"):
            with st.spinner("Menelusuri dokumen & menyusun jawaban..."):
                clean_citations = []
                needs_escalation = False
                start_time = time.perf_counter()
                try:
                    res = rag_engine.ask(clean_query)
                    answer_text = res.get("answer", "")
                    lower_ans = answer_text.lower()

                    if (
                        "informasi tidak tersedia" in lower_ans
                        or "tidak ditemukan" in lower_ans
                        or "[sumber: tidak ada]" in lower_ans
                    ):
                        needs_escalation = True
                        clean_citations = []
                    else:
                        clean_citations = [
                            c for c in res.get("citations", [])
                            if isinstance(c, dict) and str(c.get("snippet", "")).strip()
                        ]
                except Exception as e:
                    needs_escalation = True
                    answer_text = (
                        "⏳ Batas API Groq tercapai, silakan coba beberapa detik lagi."
                        if "429" in str(e)
                        else f"❌ Terjadi kesalahan sistem: {e}"
                    )
                response_time = round(time.perf_counter() - start_time, 2)

            append_chat_log(
                turn_id,
                active_prompt,
                answer_text,
                clean_citations,
                needs_escalation,
                response_time,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "citations": clean_citations,
                "turn_id": turn_id,
                "question": active_prompt,
                "feedback": None,
                "needs_escalation": needs_escalation,
                "response_time": response_time,
            })
        st.rerun()
