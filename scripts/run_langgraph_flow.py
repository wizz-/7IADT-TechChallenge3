from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from app.langgraph_flow import build_clinical_graph, ClinicalState  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa o fluxo clínico baseado em LangGraph + LangChain."
    )
    parser.add_argument(
        "--question",
        required=False,
        help="Pergunta clínica em linguagem natural. Se omitida, será perguntado via input.",
    )
    parser.add_argument(
        "--patient-age",
        type=int,
        default=None,
        help="Idade do paciente (opcional).",
    )
    parser.add_argument(
        "--patient-sex",
        type=str,
        default=None,
        help="Sexo do paciente (ex.: M/F/Outro, opcional).",
    )
    parser.add_argument(
        "--patient-notes",
        type=str,
        default=None,
        help="Notas livres sobre o caso (opcional).",
    )
    args = parser.parse_args()

    question = args.question or input("Pergunta clínica: ").strip()
    if not question:
        print("Pergunta vazia. Encerrando.")
        return 0

    patient: Dict[str, Any] = {}
    if args.patient_age is not None:
        patient["age"] = args.patient_age
    if args.patient_sex:
        patient["sex"] = args.patient_sex
    if args.patient_notes:
        patient["notes"] = args.patient_notes

    app = build_clinical_graph()

    state: ClinicalState = {
        "question": question,
        "patient": patient,
    }

    for step in app.stream(state):
        for node_name, node_state in step.items():
            print(f"\n--- NODE: {node_name} ---")
            if "answer" in node_state:
                print("\nRESPOSTA:")
                print(node_state["answer"])
            if "sources" in node_state:
                print("\nFONTES (top-k):")
                for i, r in enumerate(node_state["sources"], start=1):
                    c = r["chunk"]
                    print(
                        f"- [Fonte {i}] score={r['score']:.4f} | type={c.get('type')} "
                        f"| doc_id={c.get('doc_id')} | source={c.get('source')} | title={c.get('title')}"
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

