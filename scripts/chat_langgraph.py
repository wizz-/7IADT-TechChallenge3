from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from app.llm.openai_client import load_config
from app.observability.workflow_logger import WorkflowLogger
from app.rag.faiss_index import RagSearcher
from app.workflow.medical_graph import MedicalAssistantGraph
from app.workflow.state import ChatMessage


def print_help() -> None:
    print(
        "Comandos:\n"
        "  /help    - mostra ajuda\n"
        "  /sair    - encerra\n"
        "  /novo    - limpa o histórico da conversa\n"
        "  /debug   - alterna exibição de detalhes técnicos\n"
        "\nDicas:\n"
        "  - Você pode citar o paciente na própria mensagem, por exemplo: 'me fale do paciente P001'\n"
        "  - Depois disso, pode continuar naturalmente: 'quais exames estão pendentes?'\n"
        "  - Para trocar de paciente, basta mencionar outro ID, como P002.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat no terminal com LangGraph + histórico + RAG.")
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

    logger = WorkflowLogger()
    session_id = str(uuid.uuid4())

    print(f"Modelo configurado: {cfg.chat_model}")
    print("Carregando RAG (índice FAISS)...")
    t0 = time.time()
    rag = RagSearcher(args.index_dir)
    print(f"OK. RAG carregado em {time.time() - t0:.1f}s")

    print("Construindo workflow com LangGraph...")
    t1 = time.time()
    assistant = MedicalAssistantGraph(rag)
    print(f"OK. Workflow pronto em {time.time() - t1:.1f}s")

    print("\nChat iniciado. Digite /help para ajuda.\n")

    messages: List[ChatMessage] = []
    current_patient_id = ""
    debug_mode = False
    turn_index = 0

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaindo...")
            return 0

        if not question:
            continue

        command = question.lower()

        if command in ("/sair", "/exit", "/quit"):
            print("Saindo...")
            return 0

        if command == "/help":
            print_help()
            continue

        if command == "/novo":
            messages = []
            current_patient_id = ""
            session_id = str(uuid.uuid4())
            turn_index = 0
            print("Conversa reiniciada.\n")
            continue

        if command == "/debug":
            debug_mode = not debug_mode
            status = "ativado" if debug_mode else "desativado"
            print(f"Modo debug {status}.\n")
            continue

        try:
            t_call = time.time()
            state = assistant.invoke(
                question=question,
                messages=messages,
                current_patient_id=current_patient_id,
                top_k=args.top_k,
                max_context_chars=args.max_context_chars,
            )
            dt = time.time() - t_call
        except Exception as e:
            print(f"\nERRO ao executar o workflow: {e}\n")
            logger.log_interaction(
                {
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "question": question,
                    "error": str(e),
                }
            )
            continue

        answer = state.get("validated_answer", "").strip()
        sources = state.get("sources", [])
        warnings = state.get("warnings", [])
        guardrail_flags = state.get("guardrail_flags", [])
        needs_escalation = state.get("needs_escalation", False)
        current_patient_id = state.get("current_patient_id", current_patient_id)

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

        print("")
        print(answer)
        print(f"\n(tempo: {dt:.1f}s)\n")

        if debug_mode:
            if current_patient_id:
                print(f"Paciente em foco: {current_patient_id}")

            if sources:
                print("\nFontes usadas (top-k):")
                for src in sources:
                    print(f"- {src}")

            if warnings:
                print("\nSinais de validação do workflow:")
                for warning in warnings:
                    print(f"- {warning}")

            if guardrail_flags:
                print("\nGuardrails acionados:")
                for flag in guardrail_flags:
                    print(f"- {flag}")

            if needs_escalation:
                print("\nWorkflow: resposta marcada com atenção de segurança.")

            print("")

        logger.log_interaction(
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "question": question,
                "current_patient_id": current_patient_id,
                "patient_data": state.get("patient_data", {}),
                "medical_record": state.get("medical_record", {}),
                "retrieved_docs_count": len(state.get("retrieved_docs", [])),
                "sources": sources,
                "raw_answer": state.get("answer", ""),
                "validated_answer": answer,
                "warnings": warnings,
                "guardrail_flags": guardrail_flags,
                "needs_escalation": needs_escalation,
                "duration_seconds": round(dt, 3),
                "chat_model": cfg.chat_model,
            }
        )

        turn_index += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())