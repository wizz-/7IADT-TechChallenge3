from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class MedicalWorkflowState(TypedDict, total=False):
    question: str
    top_k: int
    max_context_chars: int

    retrieved_docs: List[Dict[str, Any]]
    context_block: str
    sources: List[str]

    answer: str
    validated_answer: str

    needs_escalation: bool
    warnings: List[str]