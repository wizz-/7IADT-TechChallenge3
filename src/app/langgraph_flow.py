from __future__ import annotations

import os
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from app.langchain_assistant import build_langchain_rag_chain, LangchainRagConfig
from app.rag.faiss_index import RagSearcher


class ClinicalState(TypedDict, total=False):
    """
    Estado do fluxo clínico no LangGraph.

    Campos principais:
      - question: pergunta clínica em linguagem natural
      - patient: dados estruturados do paciente (dict)
      - context_block: texto com trechos recuperados do RAG
      - sources: lista de fontes usadas pelo RAG
      - answer: resposta final do assistente
    """

    question: str
    patient: Dict[str, Any]
    context_block: str
    sources: List[Dict[str, Any]]
    answer: str


def _get_default_config() -> LangchainRagConfig:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return LangchainRagConfig(
        project_root=project_root,
        llm_dir=os.path.join(project_root, "models", "qwen2.5-7b"),
        emb_dir=os.path.join(project_root, "models", "bge-small-en-v1.5"),
        index_dir=os.path.join(project_root, "src", "app", "data", "index"),
    )


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


def node_rag(state: ClinicalState) -> ClinicalState:
    """
    Nó responsável por:
      - rodar busca semântica no índice FAISS
      - montar o bloco de contexto textual
      - registrar fontes utilizadas
    """
    config = _get_default_config()
    rag = RagSearcher(config.emb_dir, config.index_dir)

    question = state["question"]
    results = rag.search(question, top_k=config.top_k)
    context_block = _build_context_block(results)

    state["context_block"] = context_block
    state["sources"] = results
    return state


def node_llm(state: ClinicalState) -> ClinicalState:
    """
    Nó responsável por chamar o fluxo LangChain (LLM + prompt)
    usando o contexto montado pelo nó de RAG.
    """
    config = _get_default_config()
    chain = build_langchain_rag_chain(config)

    question = state["question"]
    patient = state.get("patient", {})
    context_block = state.get("context_block", "")

    answer = chain.invoke(
        {
            "question": question,
            "patient": patient,
            "context_block": context_block,
        }
    )

    # HuggingFacePipeline retorna string
    state["answer"] = str(answer)
    return state


def build_clinical_graph():
    """
    Cria o grafo de fluxo clínico no LangGraph.

    Fluxo:
        question + patient
           ↓
        RAG (node_rag)
           ↓
        LLM (node_llm)
           ↓
        END
    """
    graph = StateGraph(ClinicalState)

    graph.add_node("rag", node_rag)
    graph.add_node("llm", node_llm)

    graph.set_entry_point("rag")
    graph.add_edge("rag", "llm")
    graph.add_edge("llm", END)

    return graph.compile()

