from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _stable_id(prefix: str, seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def _slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "_", text, flags=re.UNICODE)
    text = text.strip("_")
    return (text[:max_len] or "item").strip("_")


def _format_protocol_content(p: Dict[str, Any]) -> str:
    protocol_name = (p.get("protocol") or "Protocolo").strip()

    lines: List[str] = []
    definition = p.get("definition")
    if definition:
        lines.append(f"Definição: {definition}")

    def add_list(label: str, key: str) -> None:
        items = p.get(key) or []
        if items:
            lines.append(f"{label}:")
            for it in items:
                it = str(it).strip()
                if it:
                    lines.append(f"- {it}")

    add_list("Critérios de suspeita", "suspicion_criteria")
    add_list("Exames iniciais", "initial_exams")
    add_list("Conduta inicial", "initial_management")
    add_list("Critérios de alto risco", "high_risk_criteria")
    add_list("Encaminhamento", "referral")

    notes = p.get("notes")
    if notes:
        lines.append(f"Observações: {notes}")

    header = f"{protocol_name}\n" + ("=" * len(protocol_name))
    body = "\n".join(lines).strip()
    return f"{header}\n{body}\n"


def _format_faq_content(faq: Dict[str, Any]) -> str:
    q = (faq.get("question") or "").strip()
    a = (faq.get("answer") or "").strip()
    src = (faq.get("source") or "").strip()
    proto = (faq.get("protocol") or "").strip()

    parts: List[str] = []
    if q:
        parts.append(f"Pergunta: {q}")
    if a:
        parts.append(f"Resposta: {a}")
    if proto:
        parts.append(f"Protocolo relacionado: {proto}")
    if src:
        parts.append(f"Fonte: {src}")

    return "\n".join(parts).strip() + "\n"


def _format_pubmedqa_content(pq: Dict[str, Any], pmid: str) -> str:
    question = (pq.get("QUESTION") or "").strip()
    contexts = pq.get("CONTEXTS") or []
    long_answer = (pq.get("LONG_ANSWER") or "").strip()
    final_decision = (pq.get("final_decision") or "").strip()

    header_title = f"PubMedQA PMID {pmid}"
    header = f"{header_title}\n" + ("=" * len(header_title))

    parts: List[str] = [header]

    if question:
        parts.append(f"Pergunta: {question}")

    if contexts:
        parts.append("Contexto:")
        for i, c in enumerate(contexts, start=1):
            c = str(c).strip()
            if c:
                parts.append(f"[{i}] {c}")

    if long_answer:
        parts.append(f"Resposta longa: {long_answer}")

    if final_decision:
        parts.append(f"Decisão final: {final_decision}")

    return "\n".join(parts).strip() + "\n"


def _map_protocols(protocols: List[Dict[str, Any]], version: str = "1.0") -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for p in protocols:
        name = (p.get("protocol") or "protocolo").strip()
        doc_id = _stable_id("protocol", name)

        docs.append(
            {
                "id": doc_id,
                "type": "protocol",
                "title": f"Protocolo - {name}",
                "content": _format_protocol_content(p),
                "source": "Hospital Fictício",
                "metadata": {
                    "protocol": name,
                    "version": version,
                },
            }
        )
    return docs


def _map_faqs(faqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for f in faqs:
        q = (f.get("question") or "").strip()
        proto = (f.get("protocol") or "").strip()
        seed = f"{proto}|{q}" if q else json.dumps(f, ensure_ascii=False, sort_keys=True)
        doc_id = _stable_id("faq", seed)

        docs.append(
            {
                "id": doc_id,
                "type": "faq",
                "title": q or f"FAQ - {proto or 'geral'}",
                "content": _format_faq_content(f),
                "source": (f.get("source") or "Hospital Fictício"),
                "metadata": {
                    "protocol": proto or None,
                },
            }
        )
    return docs


def _map_pubmedqa(pubmedqa: Dict[str, Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    count = 0

    for pmid, obj in pubmedqa.items():
        if limit is not None and count >= limit:
            break

        question = (obj.get("QUESTION") or "").strip()
        meshes = obj.get("MESHES") or []
        year = obj.get("YEAR")

        docs.append(
            {
                "id": f"pubmedqa_{pmid}",
                "type": "scientific",
                "title": question or f"PubMedQA PMID {pmid}",
                "content": _format_pubmedqa_content(obj, pmid),
                "source": "PubMedQA",
                "metadata": {
                    "pmid": pmid,
                    "year": year,
                    "meshes": meshes,
                    "final_decision": obj.get("final_decision"),
                    "labels": obj.get("LABELS") or [],
                },
            }
        )
        count += 1

    return docs


@dataclass(frozen=True)
class DatasetPaths:
    faq_path: str
    protocolos_path: str
    pubmedqa_path: str
    out_path: str


def build_unified_dataset(
    paths: DatasetPaths,
    pubmed_limit: Optional[int] = None,
    protocol_version: str = "1.0",
) -> Dict[str, Any]:
    faqs_raw = _read_json(paths.faq_path)
    protocolos_raw = _read_json(paths.protocolos_path)
    pubmed_raw = _read_json(paths.pubmedqa_path)

    if not isinstance(faqs_raw, list):
        raise ValueError(f"FAQ deve ser uma LISTA de objetos. Arquivo: {paths.faq_path}")
    if not isinstance(protocolos_raw, list):
        raise ValueError(f"Protocolos deve ser uma LISTA de objetos. Arquivo: {paths.protocolos_path}")
    if not isinstance(pubmed_raw, dict):
        raise ValueError(f"PubMedQA (ori_pqal.json) deve ser um OBJETO (dict) por PMID. Arquivo: {paths.pubmedqa_path}")

    documents: List[Dict[str, Any]] = []
    documents.extend(_map_protocols(protocolos_raw, version=protocol_version))
    documents.extend(_map_faqs(faqs_raw))
    documents.extend(_map_pubmedqa(pubmed_raw, limit=pubmed_limit))

    pubmed_count = pubmed_limit if pubmed_limit is not None else len(pubmed_raw)

    dataset = {
        "schema_version": "1.0",
        "generated_at": _utc_now_iso(),
        "counts": {
            "protocols": len(protocolos_raw),
            "faqs": len(faqs_raw),
            "pubmedqa": pubmed_count,
            "total_docs": len(documents),
        },
        "documents": documents,
    }
    return dataset


def save_unified_dataset(dataset: Dict[str, Any], out_path: str) -> None:
    _write_json(out_path, dataset)