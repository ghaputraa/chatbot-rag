import os
import json
import hashlib
import pdfplumber
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import (
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    SUPPORTED_EXTENSIONS,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

_EMBEDDINGS_CACHE = None
MANIFEST_PATH = os.path.join(CHROMA_DIR, "indexed_manifest.json")


def get_embeddings():
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        try:
            _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu", "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
        except Exception:
            _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
    return _EMBEDDINGS_CACHE


def get_vectorstore():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _load_manifest() -> dict[str, float]:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_manifest(manifest: dict[str, float]):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _format_table_as_markdown(table: list[list[str | None]]) -> str:
    """Mengonversi tabel PDF menjadi format Markdown Table agar kolom & nominal tidak bergeser."""
    cleaned_rows = []
    for row in table:
        if not row or all(cell is None or not str(cell).strip() for cell in row):
            continue
        cleaned_cells = [
            " ".join(str(cell).split()) if cell is not None else ""
            for cell in row
        ]
        cleaned_rows.append(cleaned_cells)

    if not cleaned_rows:
        return ""

    max_cols = max(len(r) for r in cleaned_rows)
    padded_rows = [r + [""] * (max_cols - len(r)) for r in cleaned_rows]

    lines = [
        "| " + " | ".join(padded_rows[0]) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    for row in padded_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_pdf_with_pdfplumber(file_path: str) -> list[Document]:
    """Mengekstrak teks paragraf dan tabel PDF secara terpisah menggunakan pdfplumber."""
    docs = []
    filename = os.path.basename(file_path)

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

        for idx, page in enumerate(pdf.pages):
            page_sections = []
            tables = page.find_tables()

            if tables:
                table_bboxes = [t.bbox for t in tables]

                def not_within_bboxes(obj):
                    obj_x = (obj.get("x0", 0) + obj.get("x1", 0)) / 2
                    obj_top = (obj.get("top", 0) + obj.get("bottom", 0)) / 2
                    for (x0, top, x1, bottom) in table_bboxes:
                        if x0 <= obj_x <= x1 and top <= obj_top <= bottom:
                            return False
                    return True

                non_table_page = page.filter(not_within_bboxes)
                outside_text = non_table_page.extract_text()
                if outside_text and outside_text.strip():
                    page_sections.append(outside_text.strip())

                for t_idx, table_obj in enumerate(tables, 1):
                    raw_table = table_obj.extract()
                    md_table = _format_table_as_markdown(raw_table)
                    if md_table:
                        page_sections.append(f"[Tabel {t_idx}]\n{md_table}")
            else:
                plain_text = page.extract_text(layout=False)
                if plain_text and plain_text.strip():
                    page_sections.append(plain_text.strip())

            full_page_content = "\n\n".join(page_sections).strip()
            if full_page_content:
                docs.append(
                    Document(
                        page_content=full_page_content,
                        metadata={
                            "source": filename,
                            "page": idx,
                            "total_pages": total_pages,
                        },
                    )
                )
    return docs


def load_single_document(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    if ext == ".pdf":
        return _extract_pdf_with_pdfplumber(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        return []

    docs = loader.load()
    total_pages = len(docs)

    for idx, doc in enumerate(docs):
        doc.metadata["source"] = filename
        if "page" not in doc.metadata:
            doc.metadata["page"] = idx
        if "total_pages" not in doc.metadata:
            doc.metadata["total_pages"] = total_pages

    return docs


def sync_documents(vector_store: Chroma) -> dict:
    """Memindai folder DATA_DIR dan mengindeks file baru menggunakan manifest cepat."""
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = _load_manifest()

    if not manifest and vector_store._collection.count() > 0:
        existing = list_indexed_documents(vector_store)
        for fname in existing:
            fpath = os.path.join(DATA_DIR, fname)
            manifest[fname] = os.path.getmtime(fpath) if os.path.exists(fpath) else 1.0
        _save_manifest(manifest)

    all_new_chunks = []
    chunk_ids = []
    added_files = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n| ", "\n", ". ", " ", ""],
    )

    for filename in os.listdir(DATA_DIR):
        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        file_path = os.path.join(DATA_DIR, filename)
        current_mtime = os.path.getmtime(file_path)

        if filename in manifest and manifest[filename] == current_mtime:
            continue

        if filename in manifest:
            delete_document_by_name(vector_store, filename)

        raw_docs = load_single_document(file_path)
        if not raw_docs:
            continue

        added_files.append(filename)
        manifest[filename] = current_mtime
        chunks = text_splitter.split_documents(raw_docs)

        for i, chunk in enumerate(chunks):
            page_num = chunk.metadata.get("page", 0)
            content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{filename}_p{page_num}_c{i}_{content_hash}"
            all_new_chunks.append(chunk)
            chunk_ids.append(chunk_id)

    if all_new_chunks:
        vector_store.add_documents(documents=all_new_chunks, ids=chunk_ids)
        _save_manifest(manifest)

    total_chunks = vector_store._collection.count()
    total_files = len(manifest)
    status = "synced" if added_files else "up_to_date"

    return {
        "status": status,
        "new_files": added_files,
        "added_files": added_files,
        "new_chunks": len(all_new_chunks),
        "total_chunks": total_chunks,
        "total_files": total_files,
    }


def get_all_chunks_as_documents(vector_store: Chroma, source_filter: str | None = None) -> list[Document]:
    """Mengambil seluruh chunk dari ChromaDB untuk dibangun ulang menjadi indeks BM25."""
    where_clause = {"source": source_filter} if source_filter else None
    raw_data = vector_store.get(where=where_clause, include=["documents", "metadatas"])

    docs = []
    texts = raw_data.get("documents") or []
    metadatas = raw_data.get("metadatas") or []

    for text, meta in zip(texts, metadatas):
        if text:
            docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def list_indexed_documents(vector_store: Chroma) -> dict[str, int]:
    """Mengembalikan dictionary {nama_file: jumlah_chunk} dari seluruh dokumen di ChromaDB."""
    raw_data = vector_store.get(include=["metadatas"])
    metadatas = raw_data.get("metadatas") or []

    doc_counts: dict[str, int] = {}
    for meta in metadatas:
        if not meta:
            continue
        src = os.path.basename(meta.get("source", "Unknown"))
        doc_counts[src] = doc_counts.get(src, 0) + 1
    return doc_counts


def delete_document_by_name(vector_store: Chroma, filename: str) -> int:
    """Menghapus seluruh chunk milik dokumen tertentu dari ChromaDB."""
    target_name = os.path.basename(filename)
    raw_data = vector_store.get(where={"source": target_name})
    ids_to_delete = raw_data.get("ids") or []

    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)

    manifest = _load_manifest()
    if target_name in manifest:
        del manifest[target_name]
        _save_manifest(manifest)

    return len(ids_to_delete)


def list_documents(vector_store: Chroma) -> list[str]:
    """Alias kompatibilitas untuk main.py: mengembalikan daftar nama file terindeks."""
    return sorted(list(list_indexed_documents(vector_store).keys()))


def delete_document(vector_store: Chroma, filename: str) -> int:
    """Alias kompatibilitas untuk main.py: menghapus dokumen berdasarkan nama file."""
    return delete_document_by_name(vector_store, filename)