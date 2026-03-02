from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RagPaths:
    dataset_path: str
    index_dir: str
    embedding_model_dir: str


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise RuntimeError(f"Não foi possível criar o diretório do índice: {path}")


def _normalize(v: np.ndarray) -> np.ndarray:
    # Cosine similarity via inner product (normalize vectors)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def _chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap)

    return chunks


def build_chunks_from_dataset(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    documents = dataset.get("documents", [])
    chunks: List[Dict[str, Any]] = []

    for doc in documents:
        doc_id = doc.get("id")
        doc_type = doc.get("type")
        title = doc.get("title") or ""
        content = doc.get("content") or ""
        source = doc.get("source") or ""
        metadata = doc.get("metadata") or {}

        # Para melhorar recall, mistura título + conteúdo no texto indexado
        full_text = f"Título: {title}\nFonte: {source}\n\n{content}".strip()

        for idx, part in enumerate(_chunk_text(full_text)):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::c{idx}",
                    "doc_id": doc_id,
                    "type": doc_type,
                    "title": title,
                    "source": source,
                    "metadata": metadata,
                    "text": part,
                }
            )

    return chunks


def build_faiss_index(paths: RagPaths, batch_size: int = 64) -> Tuple[int, int]:
    dataset = _read_json(paths.dataset_path)
    chunks = build_chunks_from_dataset(dataset)

    if not chunks:
        raise RuntimeError("Nenhum chunk gerado. Verifique o dataset.")

    _ensure_dir(paths.index_dir)

    model = SentenceTransformer(paths.embedding_model_dir)

    texts = [c["text"] for c in chunks]
    embeddings_list: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings_list.append(emb)

    embeddings = np.vstack(embeddings_list).astype("float32")
    embeddings = _normalize(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (com vetores normalizados = cosine)
    index.add(embeddings)

    # Persistência (Windows-safe): salvar via bytes para suportar paths Unicode
    index_bytes = faiss.serialize_index(index)
    index_path = os.path.join(paths.index_dir, "index.faiss")
    with open(index_path, "wb") as f:
        f.write(index_bytes)

    with open(os.path.join(paths.index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "embedding_model_dir": paths.embedding_model_dir,
        "dataset_path": paths.dataset_path,
        "num_chunks": len(chunks),
        "dim": dim,
    }
    with open(os.path.join(paths.index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return len(chunks), dim


def load_faiss_index(index_dir: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index_path = os.path.join(index_dir, "index.faiss")
    with open(index_path, "rb") as f:
        b = f.read()

    arr = np.frombuffer(b, dtype=np.uint8)
    index = faiss.deserialize_index(arr)

    chunks: List[Dict[str, Any]] = []
    with open(os.path.join(index_dir, "chunks.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return index, chunks


def search(
    embedding_model_dir: str,
    index_dir: str,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    model = SentenceTransformer(embedding_model_dir)
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    q_emb = _normalize(q_emb)

    index, chunks = load_faiss_index(index_dir)
    scores, ids = index.search(q_emb, top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        c = chunks[int(idx)]
        results.append(
            {
                "score": float(score),
                "chunk": c,
            }
        )

    return results

class RagSearcher:
    def __init__(self, embedding_model_dir: str, index_dir: str):
        self.model = SentenceTransformer(embedding_model_dir)
        self.index, self.chunks = load_faiss_index(index_dir)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        q_emb = _normalize(q_emb)

        scores, ids = self.index.search(q_emb, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            c = self.chunks[int(idx)]
            results.append({"score": float(score), "chunk": c})
        return results