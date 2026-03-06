from __future__ import annotations

from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.llm.openai_client import load_config
from app.rag.faiss_index import RagSearcher
from app.workflow.state import MedicalWorkflowState


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

    final_warning = "Aviso: esta orientação não substitui avaliação médica."
    if final_warning not in text:
        if text:
            text += f"\n\n{final_warning}"
        else:
            text = final_warning

    return text


def detect_escalation_need(question: str, answer: str, context_block: str) -> tuple[bool, List[str]]:
    warnings: List[str] = []

    q = (question or "").lower()
    a = (answer or "").lower()
    c = (context_block or "").lower()

    risk_terms = [
        "dor no peito",
        "dor torácica",
        "falta de ar",
        "desmaio",
        "convuls",
        "sepse",
        "choque",
        "hemorrag",
        "rebaixamento",
        "saturação",
        "troponina",
        "instabilidade hemodinâmica",
    ]

    if any(term in q for term in risk_terms):
        warnings.append("Pergunta com possível sinal de gravidade.")

    if "não encontrou no material" in a or "base suficiente" in a:
        warnings.append("Resposta com baixa cobertura de contexto.")

    if not c.strip():
        warnings.append("Nenhum contexto recuperado pelo RAG.")

    return (len(warnings) > 0, warnings)


class MedicalAssistantGraph:
    def __init__(self, rag: RagSearcher):
        cfg = load_config()
        self.rag = rag
        self.llm = ChatOpenAI(
            model=cfg.chat_model,
            temperature=0,
        )
        self.graph = self._build_graph()

    def _retrieve_context(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        question = state["question"]
        top_k = state.get("top_k", 6)
        max_context_chars = state.get("max_context_chars", 3000)

        results = self.rag.search(question, top_k=top_k)
        context_block = build_context_block(results, max_chars=max_context_chars)

        sources: List[str] = []
        for i, r in enumerate(results, start=1):
            c = r["chunk"]
            sources.append(
                f"[Fonte {i}] score={r['score']:.4f} | "
                f"type={c.get('type')} | doc_id={c.get('doc_id')} | "
                f"source={c.get('source')} | title={c.get('title')}"
            )

        return {
            "retrieved_docs": results,
            "context_block": context_block,
            "sources": sources,
        }

    def _generate_answer(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        question = state["question"]
        context_block = state.get("context_block", "")

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

        response = self.llm.invoke(
            [
                ("system", system),
                ("human", user),
            ]
        )

        answer = response.content if isinstance(response.content, str) else str(response.content)
        return {"answer": answer}

    def _validate_answer(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        answer = state.get("answer", "")
        question = state.get("question", "")
        context_block = state.get("context_block", "")

        validated = sanitize_answer(answer)
        needs_escalation, warnings = detect_escalation_need(question, validated, context_block)

        return {
            "validated_answer": validated,
            "needs_escalation": needs_escalation,
            "warnings": warnings,
        }

    def _safe_finalize(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        answer = state.get("validated_answer", "").strip()
        warnings = state.get("warnings", [])

        if warnings:
            answer += "\n\nObservação de segurança: em caso de sinais de gravidade ou piora clínica, procurar avaliação médica presencial imediatamente."

        return {"validated_answer": answer}

    def _normal_finalize(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        return {"validated_answer": state.get("validated_answer", "")}

    def _route_after_validation(self, state: MedicalWorkflowState) -> str:
        if state.get("needs_escalation", False):
            return "safe_finalize"
        return "normal_finalize"

    def _build_graph(self):
        graph = StateGraph(MedicalWorkflowState)

        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("validate_answer", self._validate_answer)
        graph.add_node("safe_finalize", self._safe_finalize)
        graph.add_node("normal_finalize", self._normal_finalize)

        graph.add_edge(START, "retrieve_context")
        graph.add_edge("retrieve_context", "generate_answer")
        graph.add_edge("generate_answer", "validate_answer")

        graph.add_conditional_edges(
            "validate_answer",
            self._route_after_validation,
            {
                "safe_finalize": "safe_finalize",
                "normal_finalize": "normal_finalize",
            },
        )

        graph.add_edge("safe_finalize", END)
        graph.add_edge("normal_finalize", END)

        return graph.compile()

    def invoke(
        self,
        question: str,
        top_k: int = 6,
        max_context_chars: int = 3000,
    ) -> MedicalWorkflowState:
        initial_state: MedicalWorkflowState = {
            "question": question,
            "top_k": top_k,
            "max_context_chars": max_context_chars,
        }
        return self.graph.invoke(initial_state)