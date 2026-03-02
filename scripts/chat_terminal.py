from __future__ import annotations

import argparse
import os
import sys
import time
from peft import PeftModel
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from app.rag.faiss_index import RagSearcher


def build_context_block(results: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    parts: List[str] = []
    used = 0

    for i, r in enumerate(results, start=1):
        c = r["chunk"]
        score = r["score"]
        header = (
            f"[Fonte {i}] score={score:.4f} | type={c.get('type')} | doc_id={c.get('doc_id')} | source={c.get('source')}\n"
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
        "Responda em português do Brasil. Seja objetivo e seguro.\n"
        "- NÃO prescreva doses.\n"
        "- Não faça diagnóstico definitivo.\n"
        "- Se houver risco, oriente procurar atendimento.\n"
        "- Use o CONTEXTO fornecido quando for relevante e cite as fontes como [Fonte N].\n"
        "- Se o contexto não tiver a resposta, diga que não encontrou no material e responda de forma cautelosa.\n"
        "- Não use introduções como \"Com base em evidência científica\". Responda direto."
    )

    user = (
        "CONTEXTO (trechos recuperados):\n"
        f"{context_block}\n\n"
        "PERGUNTA:\n"
        f"{question}\n\n"
        "INSTRUÇÃO DE RESPOSTA:\n"
        "1) Responda em 4 a 8 linhas.\n"
        "2) Se usar o contexto, cite as fontes no fim de frases relevantes (ex.: ... [Fonte 2]).\n"
        "3) Termine com uma linha: 'Aviso: esta orientação não substitui avaliação médica.'\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_llm(model_dir: str, lora_dir: str | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. Instale torch com CUDA.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    if lora_dir and os.path.isdir(lora_dir):
        model = PeftModel.from_pretrained(model, lora_dir)
        print(f"LoRA aplicado: {lora_dir}")
    else:
        print("LoRA não aplicado (pasta não encontrada).")

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, messages, max_new_tokens: int = 140) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
    )

    with torch.no_grad():
        out = model.generate(**inputs, **generate_kwargs)

    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat no terminal com RAG + Qwen (Transformers).")
    parser.add_argument(
        "--llm",
        default=os.path.join(PROJECT_ROOT, "models", "qwen2.5-7b"),
        help="Pasta local do modelo Qwen (Transformers)",
    )
    parser.add_argument(
        "--emb",
        default=os.path.join(PROJECT_ROOT, "models", "bge-small-en-v1.5"),
        help="Pasta local do modelo de embeddings (sentence-transformers)",
    )
    parser.add_argument(
        "--index-dir",
        default=os.path.join(PROJECT_ROOT, "src", "app", "data", "index"),
        help="Pasta do índice FAISS (index.faiss + chunks.jsonl)",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Número de trechos recuperados do RAG.")
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument(
        "--lora",
        default=os.path.join(PROJECT_ROOT, "models", "qwen2.5-7b-lora"),
        help="Pasta do adapter LoRA. Se não existir, roda sem LoRA.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.llm):
        print(f"ERRO: modelo LLM não encontrado em: {args.llm}")
        return 2
    if not os.path.isdir(args.emb):
        print(f"ERRO: modelo de embeddings não encontrado em: {args.emb}")
        return 2
    if not os.path.isdir(args.index_dir):
        print(f"ERRO: índice não encontrado em: {args.index_dir}")
        return 2

    print("Carregando LLM (4-bit)...")
    t0 = time.time()
    tokenizer, model = load_llm(args.llm, args.lora)
    print(f"OK. LLM carregado em {time.time() - t0:.1f}s")
    print("Digite sua pergunta. Comandos: /sair, /help\n")

    print("Carregando RAG (embeddings + índice)...")
    t_rag = time.time()
    rag = RagSearcher(args.emb, args.index_dir)
    print(f"OK. RAG carregado em {time.time() - t_rag:.1f}s")

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

        if question.lower() in ("/help",):
            print(
                "Comandos:\n"
                "  /help  - mostra ajuda\n"
                "  /sair  - encerra\n"
                "\nDica: pergunte coisas clínicas (protocolos, fluxo, exames iniciais) e eu citarei fontes.\n"
            )
            continue

        # 1) RAG search
        results = rag.search(question, top_k=args.top_k)

        context_block = build_context_block(results)

        # 2) LLM answer
        messages = build_messages(question, context_block)

        t1 = time.time()
        answer = generate_answer(tokenizer, model, messages, max_new_tokens=args.max_new_tokens)
        dt = time.time() - t1

        answer = answer.replace("Com base em evidência científica:", "").strip()
        answer = answer.replace("Com base em evidência científica", "").strip()

        print("\n" + answer + "\n")
        print(f"(tempo: {dt:.1f}s)")

        # 3) Print sources
        if results:
            print("\nFontes usadas (top-k):")
            for i, r in enumerate(results, start=1):
                c = r["chunk"]
                print(
                    f"- [Fonte {i}] score={r['score']:.4f} | type={c.get('type')} | doc_id={c.get('doc_id')} | source={c.get('source')} | title={c.get('title')}"
                )
            print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())