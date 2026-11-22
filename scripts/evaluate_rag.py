import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Pastikan root proyek (chatbot-rag/) masuk ke sys.path agar import `app.*` berjalan
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_community.retrievers import BM25Retriever
from app.config import EVAL_DIR, RETRIEVER_K
from app.core.rag_engine import ConversationalHybridRAG
from app.core.vector_db import (
    get_all_chunks_as_documents,
    get_vectorstore,
    sync_documents,
)

GOLDEN_DATASET_PATH = Path(EVAL_DIR) / "golden_dataset.json"
OUTPUT_JSON_PATH = Path(EVAL_DIR) / "benchmark_results.json"
OUTPUT_CSV_PATH = Path(EVAL_DIR) / "benchmark_summary.csv"


def load_golden_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"File Golden Dataset tidak ditemukan di: {path}\n"
            "Pastikan Tim Data & Domain telah menyiapkan data/eval/golden_dataset.json."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Format golden_dataset.json harus berupa list JSON yang tidak kosong.")
    return data


def build_retriever_by_mode(rag: ConversationalHybridRAG, mode: str):
    """Mengatur retriever pada instance ConversationalHybridRAG sesuai mode eksperimen."""
    rag.active_filter = None
    if mode == "vector_only":
        return rag.vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

    if mode == "bm25_only":
        all_docs = get_all_chunks_as_documents(rag.vector_store)
        if not all_docs:
            raise RuntimeError("Database ChromaDB kosong. Tidak dapat membangun indeks BM25.")
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = RETRIEVER_K
        return bm25

    if mode == "hybrid_rag":
        return rag._build_hybrid_retriever()

    raise ValueError(f"Mode evaluasi tidak dikenal: {mode}")


def check_keyword_match(answer: str, expected_keywords: List[str]) -> float:
    """Menghitung proporsi kata kunci jawaban (harga, varian, angka) yang muncul di respons."""
    if not expected_keywords:
        return 1.0
    ans_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if str(kw).lower() in ans_lower)
    return round(matched / len(expected_keywords), 4)


def check_citation_match(citations: List[Dict[str, Any]], expected_source: str | None) -> bool:
    """Memeriksa apakah dokumen sumber rujukan utama sesuai dengan target dataset."""
    if not expected_source:
        # Untuk pertanyaan out-of-domain, sitasi harus kosong
        return len(citations) == 0
    target = expected_source.lower().strip()
    return any(target in str(c.get("source", "")).lower() for c in citations)


def evaluate_mode(
    rag: ConversationalHybridRAG,
    dataset: List[Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    print(f"\n🚀 Menjalankan evaluasi mode: [{mode.upper()}] ({len(dataset)} soal)...")
    rag.retriever = build_retriever_by_mode(rag, mode)

    detailed_logs = []
    total_recall = 0.0
    total_citation_hits = 0
    total_guardrail_hits = 0
    total_latency = 0.0

    for idx, item in enumerate(dataset, 1):
        # Reset riwayat chat tiap soal independen (kecuali jika soal menandai multi-turn)
        if not item.get("keep_history", False):
            rag.clear_history()

        question = item["question"]
        expected_source = item.get("expected_source")  # None jika out-of-domain
        expected_keywords = item.get("expected_keywords", [])
        is_out_of_domain = item.get("out_of_domain", expected_source is None)

        t0 = time.perf_counter()
        try:
            result = rag.ask(question)
            answer = result.get("answer", "")
            citations = result.get("citations", [])
            rel_score = result.get("relevance_score", 0.0)
            error_msg = None
        except Exception as e:
            answer = ""
            citations = []
            rel_score = 0.0
            error_msg = str(e)
        latency = round(time.perf_counter() - t0, 3)

        # Evaluasi Metrik
        is_refused = "informasi tidak tersedia di dalam dokumen" in answer.lower()

        if is_out_of_domain:
            # Pertanyaan di luar dokumen wajib ditolak oleh Gatekeeper / LLM
            keyword_recall = 1.0 if is_refused else 0.0
            citation_ok = len(citations) == 0
            guardrail_ok = is_refused
        else:
            keyword_recall = check_keyword_match(answer, expected_keywords)
            citation_ok = check_citation_match(citations, expected_source)
            guardrail_ok = not is_refused

        total_recall += keyword_recall
        total_citation_hits += int(citation_ok)
        total_guardrail_hits += int(guardrail_ok)
        total_latency += latency

        status_icon = "✅" if (keyword_recall >= 0.8 and citation_ok and guardrail_ok) else "⚠️"
        print(
            f"  [{idx:02d}/{len(dataset):02d}] {status_icon} Recall: {keyword_recall:.2f} | "
            f"Sitasi: {'OK' if citation_ok else 'MISS'} | {latency:.2f}s | {question[:45]}..."
        )

        detailed_logs.append(
            {
                "id": item.get("id", idx),
                "category": item.get("category", "general"),
                "question": question,
                "expected_source": expected_source,
                "expected_keywords": expected_keywords,
                "answer": answer,
                "citations": citations,
                "relevance_score": rel_score,
                "keyword_recall": keyword_recall,
                "citation_match": citation_ok,
                "guardrail_passed": guardrail_ok,
                "latency_sec": latency,
                "error": error_msg,
            }
        )

        # Jeda singkat 1.5 detik untuk mencegah rate-limit TPM Groq saat pengujian massal
        time.sleep(1.5)

    n = len(dataset)
    summary = {
        "mode": mode,
        "total_questions": n,
        "avg_keyword_recall": round((total_recall / n) * 100, 2),
        "citation_accuracy_pct": round((total_citation_hits / n) * 100, 2),
        "guardrail_accuracy_pct": round((total_guardrail_hits / n) * 100, 2),
        "avg_latency_sec": round(total_latency / n, 3),
    }
    return {"summary": summary, "details": detailed_logs}


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    dataset = load_golden_dataset(GOLDEN_DATASET_PATH)

    print("⏳ Menginisialisasi ChromaDB & menyinkronkan dokumen...")
    vector_store = get_vectorstore()
    sync_res = sync_documents(vector_store)
    if sync_res["total_chunks"] == 0:
        print("❌ Database ChromaDB kosong. Isi folder data/raw/ terlebih dahulu.")
        return

    rag = ConversationalHybridRAG(vector_store)

    modes = ["vector_only", "bm25_only", "hybrid_rag"]
    all_results = {}
    summaries = []

    for mode in modes:
        res = evaluate_mode(rag, dataset, mode)
        all_results[mode] = res
        summaries.append(res["summary"])

    # Simpan laporan lengkap JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Simpan ringkasan tabel CSV untuk lampiran Laporan Tugas 4 (T4)
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\n" + "=" * 78)
    print("📊 RINGKASAN HASIL BENCHMARK RETRIEVAL (TUGAS 4 - EXPERIMENT LOG)")
    print("=" * 78)
    print(f"{'Metode':<15} | {'Keyword Recall':<15} | {'Akurasi Sitasi':<15} | {'Guardrail':<12} | {'Latensi':<10}")
    print("-" * 78)
    for s in summaries:
        print(
            f"{s['mode']:<15} | "
            f"{s['avg_keyword_recall']:>13.2f}% | "
            f"{s['citation_accuracy_pct']:>13.2f}% | "
            f"{s['guardrail_accuracy_pct']:>10.2f}% | "
            f"{s['avg_latency_sec']:>7.2f}s"
        )
    print("=" * 78)
    print(f"💾 Log detail JSON tersimpan di : {OUTPUT_JSON_PATH}")
    print(f"💾 Tabel ringkasan CSV tersimpan : {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()