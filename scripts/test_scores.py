import os
import sys
import warnings

# Sembunyikan peringatan internal library pihak ketiga agar terminal bersih
warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.config import MIN_RELEVANCE_SCORE, EMBEDDING_MODEL
from app.core.vector_db import get_vectorstore, sync_documents


def test_query_score(vector_store, query: str, top_k: int = 3):
    """Mencetak skor relevansi Cosine (0.0 - 1.0) tanpa memicu UserWarning."""
    results = vector_store.similarity_search_with_score(query, k=top_k)

    print(f"\n❓ Pertanyaan: \"{query}\"")
    if not results:
        print("   ⚠️ Database ChromaDB kosong.")
        return

    # Konversi Cosine Distance ChromaDB menjadi Similarity Score (0.0 s.d. 1.0)
    top_score = max(0.0, 1.0 - results[0][1])
    status = "✅ LOLOS (Dikirim ke LLM)" if top_score >= MIN_RELEVANCE_SCORE else "🛑 DITOLAK (Anti-Halusinasi)"
    print(f"   Status Gatekeeper (Threshold = {MIN_RELEVANCE_SCORE}): {status}")

    for idx, (doc, distance) in enumerate(results, 1):
        score = max(0.0, 1.0 - distance)
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", 0) + 1
        snippet = doc.page_content[:95].replace("\n", " ")
        print(f"   {idx}. Skor: {score:.4f} | [{source}, Hal. {page}] -> \"{snippet}...\"")


def main():
    print("⏳ Memuat ChromaDB & model embedding...")
    vector_store = get_vectorstore()
    sync_documents(vector_store)

    total_chunks = vector_store._collection.count()
    if total_chunks == 0:
        print("❌ Database kosong. Letakkan dokumen di folder 'data/' terlebih dahulu.")
        return

    print(f"✅ Siap menguji pada {total_chunks} chunks (Model: {EMBEDDING_MODEL})")
    print("=" * 75)

    sample_queries = [
        "Siapa nama yang mengundurkan diri?",
        "Apa alasan pengunduran diri dalam surat tersebut?",
        "Ada berapa halaman dokumen ini?",
        "Berapa harga tiket pesawat ke Mars tahun depan?",
        "Resep cara membuat nasi goreng spesial"
    ]

    print("🧪 MENJALANKAN PENGUJIAN BATCH OTOMATIS:")
    for q in sample_queries:
        test_query_score(vector_store, q, top_k=2)

    print("\n" + "=" * 75)
    print("🔍 MODE UJI INTERAKTIF (Ketik pertanyaan Anda untuk cek skornya, 'q' untuk keluar)")
    while True:
        user_q = input("\n📝 Uji Pertanyaan: ").strip()
        if user_q.lower() in ["q", "exit", "keluar"]:
            break
        if user_q:
            test_query_score(vector_store, user_q, top_k=3)


if __name__ == "__main__":
    main()