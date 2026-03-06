from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from app.rag.faiss_index import RagSearcher


def main():

    index_dir = os.path.join(PROJECT_ROOT, "src", "app", "data", "index")

    print("Carregando índice RAG...")
    rag = RagSearcher(index_dir)

    while True:

        question = input("\nPergunta (ou 'sair'): ").strip()

        if question.lower() in ("sair", "exit", "quit"):
            break

        results = rag.search(question, top_k=5)

        print("\nResultados encontrados:\n")

        for i, r in enumerate(results, start=1):

            chunk = r["chunk"]

            print(f"--- Resultado {i} ---")
            print(f"score: {r['score']:.4f}")
            print(f"type: {chunk.get('type')}")
            print(f"title: {chunk.get('title')}")
            print(f"source: {chunk.get('source')}")
            print()

            text = chunk.get("text", "")[:500]
            print(text)
            print("\n")


if __name__ == "__main__":
    main()