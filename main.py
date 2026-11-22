import os
from app.config import DATA_DIR, SUPPORTED_EXTENSIONS
from app.core.vector_db import (
    get_vectorstore,
    sync_documents,
    list_indexed_documents,
    delete_document_by_name,
)
from app.core.rag_engine import ConversationalHybridRAG


def main():
    print("⏳ Menginisialisasi ChromaDB & Hybrid Engine...")
    vector_store = get_vectorstore()
    sync_result = sync_documents(vector_store)

    if sync_result["total_chunks"] == 0:
        print(f"❌ [Error] Folder '{DATA_DIR}/' kosong dan database belum memiliki data.")
        print(f"   Letakkan file {SUPPORTED_EXTENSIONS} di folder '{DATA_DIR}/' lalu jalankan ulang.")
        return

    if sync_result["status"] == "synced":
        print(f"📄 Menambahkan dokumen baru: {sync_result['new_files']}")
    print(f"✅ Database siap ({sync_result['total_chunks']} chunks dari {sync_result['total_files']} file).")

    try:
        rag_engine = ConversationalHybridRAG(vector_store)
    except ValueError as e:
        print(f"❌ [Error] {e}")
        return

    print("\n" + "=" * 70)
    print("🤖 CHATBOT RAG HYBRID + MEMORY SIAP! (Mendukung .pdf, .docx, .txt)")
    print("Perintah Khusus:")
    print("  • /list                  -> Lihat daftar dokumen di database")
    print("  • /sync                  -> Muat dokumen baru dari folder data")
    print("  • /hapus <nama_file>     -> Hapus satu dokumen dari database")
    print("  • /fokus <nama_file>     -> Batasi pencarian hanya ke 1 file")
    print("  • /fokus semua           -> Kembalikan pencarian ke semua file")
    print("  • /reset                 -> Hapus ingatan/riwayat percakapan")
    print("  • exit / q / keluar      -> Keluar dari program")
    print("=" * 70 + "\n")

    while True:
        fokus_label = f"[{rag_engine.active_filter}]" if rag_engine.active_filter else "[Semua Dokumen]"
        try:
            query = input(f"\n📝 {fokus_label} Pertanyaan / Perintah: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Keluar dari aplikasi. Sampai jumpa!")
            break
        except UnicodeEncodeError:
            query = input(f"\n{fokus_label} Pertanyaan / Perintah: ").strip()

        if not query:
            continue

        if query.lower() in ["exit", "q", "keluar"]:
            print("Terima kasih! Sampai jumpa.")
            break

        if query.lower() == "/list":
            docs = list_indexed_documents(vector_store)
            if not docs:
                print("📂 Database ChromaDB kosong.")
            else:
                print("\n📚 Daftar Dokumen di ChromaDB:")
                for idx, (fname, count) in enumerate(sorted(docs.items()), 1):
                    print(f"  {idx}. {fname} ({count} chunks)")
            continue

        if query.lower() == "/sync":
            res = sync_documents(vector_store)
            if res["status"] == "synced":
                rag_engine.refresh_retriever()
                print(f"💾 Berhasil menambahkan {res['new_files']}! Total chunks: {res['total_chunks']}")
            else:
                print(f"⚡ Tidak ada dokumen baru di folder '{DATA_DIR}/'.")
            continue

        if query.lower() == "/reset":
            rag_engine.clear_history()
            print("🧹 Riwayat percakapan telah dibersihkan.")
            continue

        if query.lower().startswith("/fokus"):
            parts = query.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print("⚠️ Format salah. Gunakan: /fokus nama_file.pdf atau /fokus semua")
                continue

            target = parts[1].strip()
            if target.lower() == "semua":
                rag_engine.set_document_filter(None)
                print("🌐 Filter dihapus. Pencarian kini mencakup seluruh dokumen.")
            else:
                if rag_engine.set_document_filter(target):
                    print(f"🎯 Fokus pencarian dikunci hanya pada: {rag_engine.active_filter}")
                else:
                    print(f"⚠️ Dokumen '{target}' tidak ditemukan di ChromaDB.")
            continue

        if query.lower().startswith("/hapus"):
            parts = query.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print("⚠️ Format salah. Gunakan: /hapus nama_file.pdf")
                continue

            target = parts[1].strip()
            deleted_count = delete_document_by_name(vector_store, target)
            if deleted_count == 0:
                print(f"⚠️ Dokumen '{target}' tidak ditemukan di ChromaDB.")
            else:
                if rag_engine.active_filter and rag_engine.active_filter.lower() == target.lower():
                    rag_engine.set_document_filter(None)
                else:
                    rag_engine.refresh_retriever()
                print(f"🗑️ Berhasil menghapus {deleted_count} chunks milik '{target}'.")

                file_path = os.path.join(DATA_DIR, target)
                if os.path.exists(file_path):
                    confirm = input(f"❓ Hapus juga file fisik '{file_path}'? (y/n): ")
                    if confirm.lower() == "y":
                        os.remove(file_path)
                        print(f"✅ File fisik '{file_path}' dihapus.")
            continue

        # Proses Tanya Jawab Hybrid RAG + Memory
        print("\n🔍 Mencari (Hybrid Semantic + Keyword) & menyusun jawaban...\n")
        try:
            result = rag_engine.ask(query)
            print("🤖 Jawaban:")
            print(result["answer"])
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                print(f"\n⏳ Batas kecepatan API Groq tercapai.\n   Detail: {err_msg}\n")
            else:
                print(f"\n❌ Terjadi kesalahan: {err_msg}\n")


if __name__ == "__main__":
    main()