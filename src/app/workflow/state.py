from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class MedicalWorkflowState(TypedDict, total=False):
    question: str
    rewritten_question: str
    messages: List[ChatMessage]

    current_patient_id: str
    patient_data: Dict[str, Any]
    medical_record: Dict[str, Any]

    top_k: int
    max_context_chars: int

    retrieved_docs: List[Dict[str, Any]]
    context_block: str
    sources: List[str]

    answer: str
    validated_answer: str

    needs_escalation: bool
    warnings: List[str]
    guardrail_flags: List[str]