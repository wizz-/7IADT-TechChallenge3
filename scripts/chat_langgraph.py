from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from app.llm.openai_client import load_config
from app.rag.faiss_index import RagSearcher
from app.workflow.medical_graph import MedicalAssistantGraph


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat no terminal com LangGraph + LangChain + RAG.")
    parser.add_argument(
        "--index-dir",
        default=os.path.join(PROJECT_ROOT, "src", "app", "data", "index"),
        help="Pasta do índice FAISS (index.faiss + chunks.jsonl)",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Número de trechos recuperados do RAG.")
    parser.add_argument("--max-context-chars", type=int, default=3000, help="Tamanho máximo do contexto injetado.")
    args = parser.parse_args()

    if not os.path.isdir(args.index_dir):
        print(f"ERRO: índice não encontrado em: {args.index_dir}")
        return 2

    try:
        cfg = load_config()
    except Exception as e:
        print("ERRO: configuração OpenAI inválida.")
        print("Verifique o arquivo .env na raiz do projeto.")
        print(f"Detalhe: {e}")
        return 2

    print(f"Modelo configurado: {cfg.chat_model}")
    print("Carregando RAG (índice FAISS)...")
    t0 = time.time()
    rag = RagSearcher(args.index_dir)
    print(f"OK. RAG carregado em {time.time() - t0:.1f}s")

    print("Construindo workflow com LangGraph...")
    t1 = time.time()
    assistant = MedicalAssistantGraph(rag)
    print(f"OK. Workflow pronto em {time.time() - t1:.1f}s")
    print("Digite sua pergunta. Comandos: /sair, /help\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaindo...")
            return 0

        if not question:
            continue

        if question.lower() in ("/sair", "/exit", "/quit"):
            print("Saindo...")
            return 0

        if question.lower() == "/help":
            print(
                "Comandos:\n"
                "  /help  - mostra ajuda\n"
                "  /sair  - encerra\n"
                "\nDica: pergunte sobre protocolos, exames iniciais, critérios de risco e encaminhamentos.\n"
            )
            continue

        try:
            t_call = time.time()
            state = assistant.invoke(
                question=question,
                top_k=args.top_k,
                max_context_chars=args.max_context_chars,
            )
            dt = time.time() - t_call
        except Exception as e:
            print(f"\nERRO ao executar o workflow: {e}\n")
            continue

        answer = state.get("validated_answer", "").strip()
        sources = state.get("sources", [])
        warnings = state.get("warnings", [])
        needs_escalation = state.get("needs_escalation", False)

        print("\n" + answer + "\n")
        print(f"(tempo: {dt:.1f}s)")

        if sources:
            print("\nFontes usadas (top-k):")
            for src in sources:
                print(f"- {src}")
            print("")

        if warnings:
            print("Sinais de validação do workflow:")
            for warning in warnings:
                print(f"- {warning}")
            print("")

        if needs_escalation:
            print("Workflow: resposta marcada com atenção de segurança.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())