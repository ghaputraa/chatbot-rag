import json
import mimetypes
import os
import re
import uuid
import warnings
from datetime import datetime
from urllib.parse import quote

warnings.filterwarnings("ignore", category=UserWarning)

from app.config import DATA_DIR, SUPPORTED_MIME_TYPES
from app.core.rag_engine import ConversationalHybridRAG
from app.core.vector_db import (
    delete_document_by_name,
    detect_mime_type,
    get_vectorstore,
    sync_documents,
)

# Konfigurasi Log & Nomor WhatsApp Staf Manusia (Dealer Suzuki Cinere)
LOG_DIR = os.path.join(os.path.dirname(DATA_DIR), "logs")
CHAT_LOG_FILE = os.path.join(LOG_DIR, "chat_logs.json")
STAFF_WHATSAPP_NUMBER = "6281234567890"  # Ganti dengan nomor resmi Sales/CS Suzuki Cinere

mimetypes.add_type("application/pdf", ".pdf")
mimetypes.add_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")
mimetypes.add_type("text/plain", ".txt")


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


def append_chat_log(turn_id: str, question: str, answer: str, citations: list, needs_escalation: bool):
    logs = _load_logs()
    logs.append({
        "turn_id": turn_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "citations_count": len(citations),
        "sources": [c.get("source", "-") for c in citations],
        "feedback": None,
        "escalated_to_human": needs_escalation,
    })
    _save_logs(logs)


def update_chat_feedback(turn_id: str, feedback_value: str) -> bool:
    logs = _load_logs()
    for entry in logs:
        if entry.get("turn_id") == turn_id:
            entry["feedback"] = feedback_value
            if feedback_value == "down":
                entry["escalated_to_human"] = True
            _save_logs(logs)
            return True
    return False


def build_whatsapp_escalation_url(user_question: str) -> str:
    msg = (
        f"Halo Staf Suzuki Cinere, saya membutuhkan bantuan lebih lanjut terkait pertanyaan berikut:\n"
        f"\"{user_question}\""
    )
    return f"https://wa.me/{STAFF_WHATSAPP_NUMBER}?text={quote(msg)}"


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


def list_valid_documents() -> list[str]:
    if not os.path.exists(DATA_DIR):
        return []
    return sorted(
        [
            f
            for f in os.listdir(DATA_DIR)
            if os.path.isfile(os.path.join(DATA_DIR, f))
            and detect_mime_type(os.path.join(DATA_DIR, f)) in SUPPORTED_MIME_TYPES
        ]
    )


def run_cli():
    print("⏳ Menginisialisasi ChromaDB & Hybrid Engine (MIME-Type Verified)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    vector_store = get_vectorstore()
    sync_res = sync_documents(vector_store)
    rag = ConversationalHybridRAG(vector_store)

    docs = list_valid_documents()
    total_chunks = sync_res.get("total_chunks", 0)
    print(f"✅ Database siap ({total_chunks} chunks dari {len(docs)} file).\n")

    print("=" * 72)
    print("🤖 CHATBOT RAG HYBRID + MEMORY SIAP! (Validasi MIME-Only)")
    print("Perintah Khusus:")
    print("- /list                  -> Lihat daftar dokumen & tipe MIME")
    print("- /sync                  -> Muat dokumen baru dari folder data/raw")
    print("- /hapus <nama_file>     -> Hapus satu dokumen dari database")
    print("- /fokus <nama_file>     -> Batasi pencarian hanya ke 1 file")
    print("- /fokus semua           -> Kembalikan pencarian ke semua file")
    print("- /like                  -> Beri jempol atas (👍) pada jawaban terakhir")
    print("- /dislike               -> Beri jempol bawah (👎) & eskalasi ke staf")
    print("- /logs                  -> Lihat ringkasan log percakapan tersimpan")
    print("- /reset                 -> Hapus ingatan/riwayat percakapan")
    print("- exit / q / keluar      -> Keluar dari program")
    print("=" * 72)

    last_turn_id = None
    last_question = ""

    while True:
        scope = rag.active_filter if getattr(rag, "active_filter", None) else "Semua Dokumen"
        try:
            user_input = input(f"\n📝 [{scope}] Pertanyaan / Perintah: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Keluar dari program.")
            break

        if not user_input:
            continue

        cmd_lower = user_input.lower()
        if cmd_lower in ("exit", "q", "keluar"):
            print("👋 Sampai jumpa!")
            break

        if cmd_lower == "/list":
            docs = list_valid_documents()
            if not docs:
                print("📂 Belum ada dokumen valid di folder data/raw.")
            else:
                print(f"📂 Daftar Dokumen Valid ({len(docs)} file):")
                for idx, d in enumerate(docs, 1):
                    mime = detect_mime_type(os.path.join(DATA_DIR, d))
                    print(f"   {idx}. {d} [{mime}]")
            continue

        if cmd_lower == "/sync":
            res = sync_documents(vector_store)
            rag.refresh_retriever()
            print(f"🔄 Sinkronisasi selesai ({res.get('total_chunks', 0)} chunks aktif).")
            continue

        if cmd_lower.startswith("/hapus "):
            target = user_input[7:].strip()
            target_path = os.path.join(DATA_DIR, os.path.basename(target))
            if os.path.exists(target_path):
                os.remove(target_path)
                delete_document_by_name(vector_store, os.path.basename(target))
                if getattr(rag, "active_filter", None) == os.path.basename(target):
                    rag.set_document_filter(None)
                else:
                    rag.refresh_retriever()
                print(f"🗑️ Dokumen '{target}' berhasil dihapus.")
            else:
                print(f"⚠️ File '{target}' tidak ditemukan.")
            continue

        if cmd_lower.startswith("/fokus "):
            target = user_input[7:].strip()
            if target.lower() == "semua":
                rag.set_document_filter(None)
                print("🌐 Fokus pencarian dikembalikan ke semua dokumen.")
            else:
                rag.set_document_filter(target)
                print(f"🎯 Fokus pencarian dibatasi ke: {target}")
            continue

        if cmd_lower == "/like":
            if last_turn_id and update_chat_feedback(last_turn_id, "up"):
                print(f"👍 Umpan balik positif tersimpan untuk ID [{last_turn_id}].")
            else:
                print("⚠️ Belum ada jawaban untuk dinilai.")
            continue

        if cmd_lower == "/dislike":
            if last_turn_id and update_chat_feedback(last_turn_id, "down"):
                wa_url = build_whatsapp_escalation_url(last_question)
                print(f"👎 Umpan balik negatif tersimpan untuk ID [{last_turn_id}].")
                print(f"🧑 Eskalasi ke Staf Suzuki Cinere (WhatsApp): {wa_url}")
            else:
                print("⚠️ Belum ada jawaban untuk dinilai.")
            continue

        if cmd_lower == "/logs":
            logs = _load_logs()
            up_cnt = sum(1 for l in logs if l.get("feedback") == "up")
            down_cnt = sum(1 for l in logs if l.get("feedback") == "down")
            esc_cnt = sum(1 for l in logs if l.get("escalated_to_human"))
            print(f"📊 Ringkasan Log: Total {len(logs)} | 👍 {up_cnt} | 👎 {down_cnt} | 🧑 Eskalasi {esc_cnt}")
            print(f"📁 Lokasi file log: {CHAT_LOG_FILE}")
            continue

        if cmd_lower == "/reset":
            rag.clear_history()
            last_turn_id = None
            last_question = ""
            print("🧹 Memori percakapan berhasil direset.")
            continue

        # Proses pertanyaan Hybrid RAG
        print("\n🔍 Mencari (Hybrid Semantic + Keyword) & menyusun jawaban...\n")
        clean_query = normalize_query_typos(user_input)
        turn_id = str(uuid.uuid4())[:8]

        try:
            res = rag.ask(clean_query)
            answer_text = res.get("answer", "")
            lower_ans = answer_text.lower()

            if (
                "informasi tidak tersedia" in lower_ans
                or "tidak ditemukan" in lower_ans
                or "[sumber: tidak ada]" in lower_ans
            ):
                citations = []
                needs_escalation = True
            else:
                citations = [
                    c for c in res.get("citations", [])
                    if isinstance(c, dict) and str(c.get("snippet", "")).strip()
                ]
                needs_escalation = False

            print(f"🤖 Jawaban:\n{answer_text}")

            if citations:
                print("\n📚 Sumber Rujukan:")
                for idx, c in enumerate(citations, 1):
                    print(f"   [{idx}] {c.get('source', '-')} (Hal. {c.get('page', '-')})")

            if needs_escalation:
                wa_url = build_whatsapp_escalation_url(user_input)
                print(f"\n🧑 Informasi belum tersedia di dokumen. Hubungi Staf Suzuki Cinere:\n   👉 {wa_url}")

            append_chat_log(turn_id, user_input, answer_text, citations, needs_escalation)
            last_turn_id = turn_id
            last_question = user_input
            print(f"\n💡 Ketik /like (👍) atau /dislike (👎) untuk memberi nilai jawaban ini [ID: {turn_id}].")

        except Exception as e:
            print(f"❌ Terjadi kesalahan: {e}")


if __name__ == "__main__":
    run_cli()