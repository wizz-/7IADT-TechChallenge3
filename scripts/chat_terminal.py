from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from app.llm.openai_client import chat_complete, load_config
from app.rag.faiss_index import RagSearcher


def build_context_block(results: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    parts: List[str] = []
    used = 0

    for i, r in enumerate(results, start=1):
        c = r["chunk"]
        score = r["score"]

        header = (
            f"[Fonte {i}] score={score:.4f} | "
            f"type={c.get('type')} | "
            f"doc_id={c.get('doc_id')} | "
            f"source={c.get('source')}\n"
            f"title={c.get('title')}\n"
        )

        text = c.get("text", "")
        block = f"{header}{text}\n"

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n".join(parts).strip()


def build_messages(question: str, context_block: str) -> List[Dict[str, str]]:
    system = (
        "Você é um assistente médico institucional de um hospital fictício. "
        "Responda em português do Brasil. Seja objetivo, claro e seguro.\n"
        "- NÃO prescreva doses.\n"
        "- Não faça diagnóstico definitivo.\n"
        "- Se houver sinais de gravidade, oriente procurar atendimento.\n"
        "- Use o CONTEXTO fornecido quando ele for relevante.\n"
        "- Cite as fontes como [Fonte N] ao final das frases relevantes.\n"
        "- Se o contexto não trouxer base suficiente, diga isso claramente e responda com cautela.\n"
        "- Não invente fatos.\n"
        "- Não use introduções como 'Com base em evidência científica'. Responda direto."
    )

    user = (
        "CONTEXTO (trechos recuperados):\n"
        f"{context_block}\n\n"
        "PERGUNTA:\n"
        f"{question}\n\n"
        "INSTRUÇÕES DE RESPOSTA:\n"
        "1) Responda em 4 a 8 linhas.\n"
        "2) Se usar o contexto, cite as fontes no fim das frases relevantes (ex.: ... [Fonte 2]).\n"
        "3) Termine com a linha exata: 'Aviso: esta orientação não substitui avaliação médica.'\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def sanitize_answer(answer: str) -> str:
    text = (answer or "").strip()

    prefixes_to_remove = [
        "Com base em evidência científica:",
        "Com base em evidência científica",
        "Com base no contexto fornecido:",
        "Com base no contexto fornecido",
    ]

    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    if "Aviso: esta orientação não substitui avaliação médica." not in text:
        if text:
            text += "\n\nAviso: esta orientação não substitui avaliação médica."
        else:
            text = "Aviso: esta orientação não substitui avaliação médica."

    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat no terminal com RAG + OpenAI.")
    parser.add_argument(
        "--index-dir",
        default=os.path.join(PROJECT_ROOT, "src", "app", "data", "index"),
        help="Pasta do índice FAISS (index.faiss + chunks.jsonl)",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Número de trechos recuperados do RAG.")
    parser.add_argument("--max-output-tokens", type=int, default=220, help="Máximo de tokens de saída do chat.")
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

    print(f"Modelo de chat configurado: {cfg.chat_model}")
    print("Carregando RAG (índice FAISS)...")
    t_rag = time.time()
    rag = RagSearcher(args.index_dir)
    print(f"OK. RAG carregado em {time.time() - t_rag:.1f}s")
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

        results = rag.search(question, top_k=args.top_k)
        context_block = build_context_block(results)
        messages = build_messages(question, context_block)

        t1 = time.time()
        try:
            answer = chat_complete(
                messages=messages,
                max_output_tokens=args.max_output_tokens,
                temperature=0.0,
            )
        except Exception as e:
            print(f"\nERRO ao consultar a OpenAI: {e}\n")
            continue

        dt = time.time() - t1
        answer = sanitize_answer(answer)

        print("\n" + answer + "\n")
        print(f"(tempo: {dt:.1f}s)")

        if results:
            print("\nFontes usadas (top-k):")
            for i, r in enumerate(results, start=1):
                c = r["chunk"]
                print(
                    f"- [Fonte {i}] score={r['score']:.4f} | "
                    f"type={c.get('type')} | "
                    f"doc_id={c.get('doc_id')} | "
                    f"source={c.get('source')} | "
                    f"title={c.get('title')}"
                )
            print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())