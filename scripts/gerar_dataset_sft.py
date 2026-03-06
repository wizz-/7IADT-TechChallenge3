# scripts/gerar_dataset_sft_openai.py
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

SYSTEM_PROMPT = (
    "Você é um assistente médico institucional de um hospital fictício. "
    "Responda em português do Brasil, de forma objetiva, clara e segura. "
    "Não prescreva doses de medicamentos. "
    "Não faça diagnóstico definitivo. "
    "Quando houver sinais de gravidade ou incerteza relevante, recomende avaliação médica presencial. "
    "Sempre mantenha tom profissional e direto."
)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_example(user: str, assistant: str, system_prompt: str = SYSTEM_PROMPT) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def rows_from_faq(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    content = doc.get("content", "")
    question = ""
    answer = ""

    for line in content.splitlines():
        lower = line.lower()
        if lower.startswith("pergunta:"):
            question = line.split(":", 1)[1].strip()
        elif lower.startswith("resposta:"):
            answer = line.split(":", 1)[1].strip()

    if not question or not answer:
        return []

    return [{"user": question, "assistant": answer}]


def rows_from_protocol(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    title = doc.get("title", "").replace("Protocolo -", "").strip()
    content = doc.get("content", "")

    compact = "\n".join([ln.strip() for ln in content.splitlines() if ln.strip()])
    if not compact:
        return []

    if len(compact) > 2200:
        compact = compact[:2200].rstrip() + "..."

    questions = [
        f"Resuma o protocolo institucional de {title} em passos objetivos.",
        f"Quais são os exames iniciais recomendados no protocolo de {title}?",
        f"Quais critérios de alto risco aparecem no protocolo de {title}?",
        f"Quando devo encaminhar no protocolo de {title}?",
    ]

    return [{"user": q, "assistant": compact} for q in questions]


def rows_from_scientific(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    title = doc.get("title", "").strip()
    content = doc.get("content", "").strip()

    if not title or not content:
        return []

    compact = "\n".join([ln.strip() for ln in content.splitlines() if ln.strip()])
    if len(compact) > 1800:
        compact = compact[:1800].rstrip() + "..."

    question_variants = [
        f"Resuma a evidência científica sobre: {title}",
        f"O que a literatura apresentada informa sobre: {title}?",
    ]

    return [{"user": q, "assistant": compact} for q in question_variants]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera dataset SFT em JSONL no formato da OpenAI a partir do data.json unificado."
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(PROJECT_ROOT, "src", "app", "data", "processed", "data.json"),
        help="Caminho do data.json unificado",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(PROJECT_ROOT, "src", "app", "data", "training", "sft_train_openai.jsonl"),
        help="Arquivo de saída JSONL para fine-tuning OpenAI",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-scientific",
        type=int,
        default=200,
        help="Máximo de documentos científicos aproveitados no dataset de treino",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    data = read_json(args.dataset)
    docs = data.get("documents", [])

    qa_pairs: List[Dict[str, str]] = []
    scientific_count = 0

    for doc in docs:
        doc_type = doc.get("type")

        if doc_type == "faq":
            qa_pairs.extend(rows_from_faq(doc))

        elif doc_type == "protocol":
            qa_pairs.extend(rows_from_protocol(doc))

        elif doc_type == "scientific" and scientific_count < args.max_scientific:
            qa_pairs.extend(rows_from_scientific(doc))
            scientific_count += 1

    random.shuffle(qa_pairs)

    rows = [build_example(r["user"], r["assistant"]) for r in qa_pairs]

    write_jsonl(args.out, rows)

    print(f"OK: dataset SFT OpenAI gerado em: {args.out}")
    print(f" - exemplos: {len(rows)}")
    print(f" - max_scientific: {args.max_scientific}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())