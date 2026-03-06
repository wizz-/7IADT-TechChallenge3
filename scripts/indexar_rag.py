from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from app.llm.openai_client import load_config
from app.rag.faiss_index import RagPaths, build_faiss_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Indexa o dataset unificado (data.json) em um índice FAISS (OpenAI embeddings).")
    parser.add_argument(
        "--dataset",
        default=os.path.join(PROJECT_ROOT, "src", "app", "data", "processed", "data.json"),
        help="Caminho do data.json unificado",
    )
    parser.add_argument(
        "--index-dir",
        default=os.path.join(PROJECT_ROOT, "src", "app", "data", "index"),
        help="Pasta de saída do índice FAISS",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch de textos para embeddings (OpenAI).")
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        print(f"ERRO: dataset não encontrado: {args.dataset}")
        return 2

    try:
        cfg = load_config()
    except Exception as e:
        print("ERRO: configuração OpenAI inválida.")
        print("Crie _secret/.env com OPENAI_API_KEY=... e opcionalmente OPENAI_EMBED_MODEL=...")
        print(f"Detalhe: {e}")
        return 2

    os.makedirs(args.index_dir, exist_ok=True)

    paths = RagPaths(
        dataset_path=args.dataset,
        index_dir=args.index_dir,
    )

    num_chunks, dim = build_faiss_index(paths, batch_size=args.batch_size)
    print(f"OK: índice criado em {args.index_dir}")
    print(f" - chunks: {num_chunks}")
    print(f" - dim: {dim}")
    print(f" - embedding_model: {cfg.embed_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())