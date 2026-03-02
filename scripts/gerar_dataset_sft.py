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

from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "Você é um assistente médico institucional de um hospital fictício. "
    "Responda em português do Brasil, de forma objetiva e segura. "
    "Não prescreva doses. Não faça diagnóstico definitivo. "
    "Quando necessário, recomende avaliação médica presencial."
)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def make_chat_text(tokenizer, user: str, assistant: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def rows_from_faq(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = doc.get("content", "")
    # content no seu formato tem "Pergunta:" e "Resposta:"
    q = ""
    a = ""
    for line in content.splitlines():
        if line.lower().startswith("pergunta:"):
            q = line.split(":", 1)[1].strip()
        elif line.lower().startswith("resposta:"):
            a = line.split(":", 1)[1].strip()
    if not q or not a:
        return []
    return [{"user": q, "assistant": a}]

def rows_from_protocol(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    title = doc.get("title", "").replace("Protocolo -", "").strip()
    content = doc.get("content", "")

    # Gera perguntas padrão a partir do protocolo.
    # Simples, mas funciona muito bem pra LoRA: ele aprende o "jeito institucional" de responder.
    questions = [
        f"Resuma o protocolo institucional de {title} em passos objetivos.",
        f"Quais são os exames iniciais recomendados no protocolo de {title}?",
        f"Quais critérios de alto risco aparecem no protocolo de {title}?",
        f"Quando devo encaminhar no protocolo de {title}?",
    ]

    # Como resposta-alvo, usamos o próprio texto do protocolo (compactado).
    # Para não ficar gigante, pegamos trechos e removemos excesso.
    compact = "\n".join([ln.strip() for ln in content.splitlines() if ln.strip()])
    if len(compact) > 2200:
        compact = compact[:2200] + "..."

    rows = []
    for q in questions:
        rows.append({"user": q, "assistant": compact})
    return rows

def rows_from_scientific(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Opcional: usa pouco para não “cientificar” demais o assistente.
    title = doc.get("title", "").strip()
    content = doc.get("content", "")
    if not title or not content:
        return []
    q = f"Com base em evidência científica, responda: {title}"
    # reduz
    compact = "\n".join([ln.strip() for ln in content.splitlines() if ln.strip()])
    if len(compact) > 1800:
        compact = compact[:1800] + "..."
    return [{"user": q, "assistant": compact}]

def main() -> int:
    parser = argparse.ArgumentParser(description="Gera dataset SFT (jsonl) a partir do data.json unificado.")
    parser.add_argument(
        "--dataset",
        default=os.path.join(PROJECT_ROOT, "src", "app", "data", "processed", "data.json"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(PROJECT_ROOT, "src", "app", "data", "training", "sft_train.jsonl"),
    )
    parser.add_argument(
        "--model",
        default=os.path.join(PROJECT_ROOT, "models", "qwen2.5-7b"),
        help="Tokenizer do modelo (Transformers) para aplicar chat template",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_scientific", type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed)

    data = read_json(args.dataset)
    docs = data.get("documents", [])

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    qa_pairs: List[Dict[str, str]] = []
    scientific_count = 0

    for doc in docs:
        t = doc.get("type")
        if t == "faq":
            qa_pairs.extend(rows_from_faq(doc))
        elif t == "protocol":
            qa_pairs.extend(rows_from_protocol(doc))
        elif t == "scientific" and scientific_count < args.max_scientific:
            qa_pairs.extend(rows_from_scientific(doc))
            scientific_count += 1

    # embaralha
    random.shuffle(qa_pairs)

    # converte para texto com chat template
    rows = [{"text": make_chat_text(tokenizer, r["user"], r["assistant"])} for r in qa_pairs]

    write_jsonl(args.out, rows)
    print(f"OK: dataset SFT gerado: {args.out}")
    print(f" - exemplos: {len(rows)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())