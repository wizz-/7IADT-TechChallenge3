from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_community.llms import HuggingFacePipeline

from app.rag.faiss_index import RagSearcher


@dataclass
class LangchainRagConfig:
    project_root: str
    llm_dir: str
    emb_dir: str
    index_dir: str
    top_k: int = 6
    max_new_tokens: int = 140


def _load_local_llm(model_dir: str, max_new_tokens: int) -> HuggingFacePipeline:
    """
    Cria um LLM LangChain a partir de um modelo local HuggingFace.
    Usa CPU ou GPU automaticamente, dependendo de CUDA.
    """
    device = 0 if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    return HuggingFacePipeline(pipeline=gen_pipeline)


def _build_context_block(results: List[Dict[str, Any]], max_chars: int = 3000) -> str:
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


def build_langchain_rag_chain(config: Optional[LangchainRagConfig] = None):
    """
    Constrói um fluxo RAG com LangChain reutilizando o RagSearcher existente.

    Entrada esperada:
        {
            "question": str,
            "patient": { ... dados estruturados do paciente ... }
        }
    """
    if config is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config = LangchainRagConfig(
            project_root=project_root,
            llm_dir=os.path.join(project_root, "models", "qwen2.5-7b"),
            emb_dir=os.path.join(project_root, "models", "bge-small-en-v1.5"),
            index_dir=os.path.join(project_root, "src", "app", "data", "index"),
        )

    llm = _load_local_llm(config.llm_dir, max_new_tokens=config.max_new_tokens)
    rag_searcher = RagSearcher(config.emb_dir, config.index_dir)

    def _retrieve(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question: str = inputs["question"]
        results = rag_searcher.search(question, top_k=config.top_k)
        context_block = _build_context_block(results)
        return {
            "question": question,
            "patient": inputs.get("patient") or {},
            "context_block": context_block,
        }

    retrieve_runnable = RunnableLambda(_retrieve)

    system_prompt = (
        "Você é um assistente médico institucional de um hospital fictício. "
        "Responda em português do Brasil. Seja objetivo e seguro.\n"
        "- NÃO prescreva doses.\n"
        "- Não faça diagnóstico definitivo.\n"
        "- Se houver risco, oriente procurar atendimento.\n"
        "- Use o CONTEXTO e os DADOS DO PACIENTE quando forem relevantes e cite as fontes como [Fonte N].\n"
        "- Se o contexto não tiver a resposta, diga que não encontrou no material e responda de forma cautelosa.\n"
        "- Não use introduções como \"Com base em evidência científica\". Responda direto."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                (
                    "DADOS DO PACIENTE (estruturados):\n"
                    "{patient}\n\n"
                    "CONTEXTO (trechos recuperados):\n"
                    "{context_block}\n\n"
                    "PERGUNTA:\n"
                    "{question}\n\n"
                    "INSTRUÇÃO DE RESPOSTA:\n"
                    "1) Responda em 4 a 8 linhas.\n"
                    "2) Se usar o contexto, cite as fontes no fim de frases relevantes (ex.: ... [Fonte 2]).\n"
                    "3) Termine com uma linha: 'Aviso: esta orientação não substitui avaliação médica.'\n"
                ),
            ),
        ]
    )

    chain = (
        RunnableMap(
            question=lambda x: x["question"],
            patient=lambda x: x.get("patient") or {},
        )
        | retrieve_runnable
        | prompt
        | llm
    )

    return chain

