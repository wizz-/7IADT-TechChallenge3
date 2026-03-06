# src/app/rag/faiss_index.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from app.llm.openai_client import embed_texts, load_config


@dataclass(frozen=True)
class RagPaths:
    dataset_path: str
    index_dir: str
    embedding_model_dir: str | None = None


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise RuntimeError(f"Não foi possível criar o diretório do índice: {path}")


def _normalize(v: np.ndarray) -> np.ndarray:
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


def _embed_with_retry(texts: List[str], max_retries: int = 6) -> List[List[float]]:
    delay = 0.5
    last_err: Exception | None = None

    for _ in range(max_retries):
        try:
            return embed_texts(texts)
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2.0, 8.0)

    raise RuntimeError(f"Falha ao gerar embeddings após {max_retries} tentativas: {last_err}") from last_err


def build_faiss_index(paths: RagPaths, batch_size: int = 64) -> Tuple[int, int]:
    cfg = load_config()
    dataset = _read_json(paths.dataset_path)
    chunks = build_chunks_from_dataset(dataset)

    if not chunks:
        raise RuntimeError("Nenhum chunk gerado. Verifique o dataset.")

    _ensure_dir(paths.index_dir)

    texts = [c["text"] for c in chunks]
    embeddings_list: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = _embed_with_retry(batch)
        embeddings_list.append(np.asarray(emb, dtype="float32"))

    embeddings = np.vstack(embeddings_list).astype("float32")
    embeddings = _normalize(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_bytes = faiss.serialize_index(index)
    index_path = os.path.join(paths.index_dir, "index.faiss")
    with open(index_path, "wb") as f:
        f.write(index_bytes)

    with open(os.path.join(paths.index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "embedding_provider": "openai",
        "embedding_model": cfg.embed_model,
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


class RagSearcher:
    def __init__(self, index_dir: str):
        self.index, self.chunks = load_faiss_index(index_dir)
        self._cache: Dict[str, np.ndarray] = {}

    def _embed_query(self, query: str) -> np.ndarray:
        q = (query or "").strip()
        if not q:
            return np.zeros((1, self.index.d), dtype="float32")

        cached = self._cache.get(q)
        if cached is not None:
            return cached

        emb = _embed_with_retry([q])
        q_emb = np.asarray(emb, dtype="float32")
        q_emb = _normalize(q_emb)

        if len(self._cache) >= 128:
            self._cache.clear()
        self._cache[q] = q_emb
        return q_emb

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self._embed_query(query)
        scores, ids = self.index.search(q_emb, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            c = self.chunks[int(idx)]
            results.append({"score": float(score), "chunk": c})
        return results