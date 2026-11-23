"""PDF extraction, chunking, embedding, and hybrid BM25+vector RAG storage."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import chromadb
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import load_settings

LEGACY_COLLECTION_NAME = "pdf_knowledge"
COLLECTION_NAME = "pdf_knowledge_cosine"
DEFAULT_MAX_PDF_BYTES = 25 * 1024 * 1024
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f\[\]]+")


def sanitize_filename(filename: str) -> str:
    """Normalize an uploaded filename and strip control/bracket characters."""
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string.")
    base_name = Path(filename.replace("\\", "/")).name
    cleaned = _CONTROL_CHARS_RE.sub("", base_name).strip()
    if not cleaned:
        raise ValueError("Filename cannot be empty.")
    return cleaned


def compute_document_id(filename: str) -> str:
    """Compute a case-insensitive deterministic document ID from filename."""
    safe_name = sanitize_filename(filename)
    return hashlib.sha256(safe_name.casefold().encode("utf-8")).hexdigest()


def _extract_document_text(document: pymupdf.Document) -> list[dict[str, Any]]:
    if getattr(document, "needs_pass", False):
        raise ValueError("Password-protected PDFs are not supported.")
    pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(document, start=1):
        pages.append({"page": page_number, "text": page.get_text("text")})
    return pages


def extract_pdf_bytes(
    pdf_bytes: bytes, *, max_bytes: int = DEFAULT_MAX_PDF_BYTES
) -> list[dict[str, Any]]:
    """Validate and extract per-page text from raw PDF bytes."""
    if not pdf_bytes:
        raise ValueError("Uploaded PDF file is empty.")
    if len(pdf_bytes) > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        raise ValueError(f"PDF exceeds the {max_mb} MB upload limit.")
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise ValueError("Uploaded file is not a valid PDF document.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            return _extract_document_text(document)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not read PDF file: {error}") from error


def extract_pdf_text(
    pdf_path: str | Path, *, max_bytes: int = DEFAULT_MAX_PDF_BYTES
) -> list[dict[str, Any]]:
    """Validate and extract per-page text from a PDF file path."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")
    return extract_pdf_bytes(path.read_bytes(), max_bytes=max_bytes)


def split_text_into_chunks(
    pages: list[dict[str, Any]],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict[str, Any]]:
    """Split extracted PDF pages into overlapping character chunks with page numbers."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks: list[dict[str, Any]] = []
    for page in pages:
        text = page.get("text", "")
        if not text.strip():
            continue
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk, "page": page["page"]})
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _rank_bm25(question: str, records: list[dict[str, Any]]) -> list[tuple[int, float]]:
    """Score document records against a query using Okapi BM25."""
    query_terms = set(_tokenize(question))
    if not query_terms or not records:
        return []

    tokenized_documents = [_tokenize(record["document"]) for record in records]
    document_count = len(tokenized_documents)
    average_length = (
        sum(map(len, tokenized_documents)) / document_count if document_count else 0.0
    )
    if average_length == 0:
        return []

    document_frequencies = Counter(
        term for tokens in tokenized_documents for term in set(tokens)
    )
    scores: list[tuple[int, float]] = []
    k1 = 1.5
    b = 0.75

    for index, tokens in enumerate(tokenized_documents):
        term_frequencies = Counter(tokens)
        document_length = len(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_frequencies[term]
            if not frequency:
                continue
            document_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            score += (
                inverse_document_frequency
                * frequency
                * (k1 + 1)
                / (
                    frequency
                    + k1 * (1 - b + b * document_length / average_length)
                )
            )
        if score > 0:
            scores.append((index, score))

    scores.sort(key=lambda result: result[1], reverse=True)
    return scores


def _to_embedding_list(embeddings: Any) -> list[list[float]]:
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(vector) for vector in embeddings]


class RAGStore:
    """Persistent ChromaDB + SentenceTransformer store with hybrid RRF retrieval."""

    def __init__(
        self,
        db_dir: Path | str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        *,
        embed_fn: Callable[[list[str]], Any] | None = None,
    ) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self._embed_fn = embed_fn
        self._model: Any = None

        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        space = (self.collection.configuration or {}).get("hnsw", {}).get("space")
        if space != "cosine":
            raise RuntimeError(
                f"Chroma collection '{COLLECTION_NAME}' must use cosine distance."
            )

        self._migrate_legacy_collection()

    def close(self) -> None:
        """Release underlying ChromaDB file handles cleanly."""
        if hasattr(self.client, "close"):
            self.client.close()
        else:
            system = getattr(self.client, "_system", None)
            if system is not None and hasattr(system, "stop"):
                system.stop()

    def _get_embedder(self) -> Callable[[list[str]], Any]:
        if self._embed_fn is not None:
            return self._embed_fn
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model.encode

    def _migrate_legacy_collection(self) -> None:
        """Migrate legacy L2 collection rows once and drop the legacy collection."""
        collection_names = {
            item.name if hasattr(item, "name") else item
            for item in self.client.list_collections()
        }
        if LEGACY_COLLECTION_NAME not in collection_names:
            return

        legacy_collection = self.client.get_collection(name=LEGACY_COLLECTION_NAME)
        legacy_ids = legacy_collection.get(include=[])["ids"]
        target_ids = set(self.collection.get(include=[])["ids"])
        missing_ids = [
            record_id for record_id in legacy_ids if record_id not in target_ids
        ]
        for offset in range(0, len(missing_ids), 500):
            records = legacy_collection.get(
                ids=missing_ids[offset : offset + 500],
                include=["documents", "metadatas", "embeddings"],
            )
            if records["ids"]:
                self.collection.upsert(
                    ids=records["ids"],
                    documents=records["documents"],
                    embeddings=_to_embedding_list(records["embeddings"]),
                    metadatas=records["metadatas"],
                )

        migrated_ids = set(self.collection.get(include=[])["ids"])
        if all(record_id in migrated_ids for record_id in legacy_ids):
            # Remove the legacy collection so deleted documents cannot resurrect on restart.
            self.client.delete_collection(name=LEGACY_COLLECTION_NAME)

    def _get_document_records(
        self, filename: str, *, include: list[str]
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        safe_name = sanitize_filename(filename)
        document_id = compute_document_id(safe_name)
        where_filter: dict[str, Any] = {"document_id": document_id}
        records = self.collection.get(where=where_filter, include=include)
        if not records["ids"]:
            legacy_filter: dict[str, Any] = {"filename": safe_name}
            legacy_records = self.collection.get(where=legacy_filter, include=include)
            if legacy_records["ids"]:
                return document_id, legacy_filter, legacy_records
        return document_id, where_filter, records

    def generate_embeddings(self, chunks: list[dict[str, Any]]) -> Any:
        if not chunks:
            raise ValueError("Cannot generate embeddings for an empty document.")
        embedder = self._get_embedder()
        return embedder([chunk["text"] for chunk in chunks])

    def store_chunks(
        self, chunks: list[dict[str, Any]], embeddings: Any, filename: str
    ) -> int:
        if not chunks:
            raise ValueError("Cannot store an empty document.")
        embedding_rows = _to_embedding_list(embeddings)
        if len(chunks) != len(embedding_rows):
            raise ValueError("Each document chunk must have one embedding.")

        safe_name = sanitize_filename(filename)
        document_id = compute_document_id(safe_name)
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "filename": safe_name,
                "document_id": document_id,
                "page": int(chunk["page"]),
            }
            for chunk in chunks
        ]

        _, _, previous = self._get_document_records(safe_name, include=["metadatas"])
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embedding_rows,
            metadatas=metadatas,
        )

        id_set = set(ids)
        stale_ids = [
            record_id for record_id in previous["ids"] if record_id not in id_set
        ]
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        return len(chunks)

    def get_indexed_documents(self) -> list[dict[str, Any]]:
        records = self.collection.get(include=["metadatas"])
        documents: dict[str, dict[str, Any]] = {}
        for metadata in records["metadatas"]:
            if not metadata:
                continue
            filename = metadata.get("filename")
            if not filename:
                continue
            document = documents.setdefault(filename, {"chunks": 0, "pages": set()})
            document["chunks"] += 1
            if "page" in metadata and metadata["page"] is not None:
                document["pages"].add(metadata["page"])

        return [
            {
                "filename": filename,
                "chunks": details["chunks"],
                "pages": sorted(details["pages"]),
            }
            for filename, details in sorted(
                documents.items(), key=lambda item: item[0].casefold()
            )
        ]

    def remove_document(self, filename: str) -> int:
        _, _, records = self._get_document_records(filename, include=[])
        if records["ids"]:
            self.collection.delete(ids=records["ids"])
        return len(records["ids"])

    def retrieve_relevant_chunks(
        self, question: str, filename: str, k: int = 5
    ) -> dict[str, list[list[Any]]]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("A document filename must be selected.")
        if k < 1:
            raise ValueError("The number of retrieved chunks must be positive.")

        _, where_filter, document_records = self._get_document_records(
            filename, include=["documents", "metadatas"]
        )
        record_count = len(document_records["ids"])
        if record_count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        records = [
            {
                "id": record_id,
                "document": document,
                "metadata": metadata,
            }
            for record_id, document, metadata in zip(
                document_records["ids"],
                document_records["documents"],
                document_records["metadatas"],
            )
        ]

        candidate_count = min(record_count, max(k * 3, 20))
        embedder = self._get_embedder()
        question_embedding = _to_embedding_list(embedder([question]))
        vector_results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=candidate_count,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        bm25_ranking = _rank_bm25(question, records)[:candidate_count]
        vector_ids = vector_results["ids"][0]
        rankings = [
            [records[index]["id"] for index, _ in bm25_ranking],
            vector_ids,
        ]
        combined_scores: dict[str, float] = {}
        rrf_constant = 60
        for ranking in rankings:
            for rank, record_id in enumerate(ranking, start=1):
                combined_scores[record_id] = combined_scores.get(record_id, 0.0) + (
                    1.0 / (rrf_constant + rank)
                )

        records_by_id = {record["id"]: record for record in records}
        vector_distances = dict(zip(vector_ids, vector_results["distances"][0]))
        ranked_ids = sorted(
            combined_scores,
            key=lambda record_id: combined_scores[record_id],
            reverse=True,
        )[:k]

        return {
            "documents": [
                [records_by_id[record_id]["document"] for record_id in ranked_ids]
            ],
            "metadatas": [
                [records_by_id[record_id]["metadata"] for record_id in ranked_ids]
            ],
            "distances": [
                [vector_distances.get(record_id) for record_id in ranked_ids]
            ],
        }


@lru_cache(maxsize=1)
def get_default_store() -> RAGStore:
    """Lazily initialize the process-wide RAGStore from Settings."""
    settings = load_settings()
    return RAGStore(
        db_dir=settings.chroma_db_dir,
        embedding_model_name=settings.embedding_model_name,
    )


def generate_embeddings(chunks: list[dict[str, Any]]) -> Any:
    return get_default_store().generate_embeddings(chunks)


def store_chunks(
    chunks: list[dict[str, Any]], embeddings: Any, filename: str
) -> int:
    return get_default_store().store_chunks(chunks, embeddings, filename)


def get_indexed_documents() -> list[dict[str, Any]]:
    return get_default_store().get_indexed_documents()


def remove_document(filename: str) -> int:
    return get_default_store().remove_document(filename)


def retrieve_relevant_chunks(
    question: str, filename: str, k: int = 5
) -> dict[str, list[list[Any]]]:
    return get_default_store().retrieve_relevant_chunks(question, filename, k=k)
