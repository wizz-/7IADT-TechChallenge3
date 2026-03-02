from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from app.data.dataset_builder import DatasetPaths, build_unified_dataset, save_unified_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera o dataset unificado (data.json) a partir dos datasets RAW do projeto."
    )
    parser.add_argument(
        "--pubmed-limit",
        type=int,
        default=None,
        help="Limita a quantidade de registros do PubMedQA (útil para testes rápidos).",
    )
    parser.add_argument(
        "--protocol-version",
        default="1.0",
        help="Versão dos protocolos hospitalares (default: 1.0).",
    )
    args = parser.parse_args()

    base = os.path.join(PROJECT_ROOT, "src", "app", "data")

    paths = DatasetPaths(
        faq_path=os.path.join(base, "raw", "hospital_ficticio", "faq.json"),
        protocolos_path=os.path.join(base, "raw", "hospital_ficticio", "protocolos.json"),
        pubmedqa_path=os.path.join(base, "raw", "pubmedqa", "ori_pqal.json"),
        out_path=os.path.join(base, "processed", "data.json"),
    )

    missing = [p for p in [paths.faq_path, paths.protocolos_path, paths.pubmedqa_path] if not os.path.isfile(p)]
    if missing:
        print("ERRO: arquivos RAW não encontrados:")
        for m in missing:
            print(f" - {m}")
        return 2

    dataset = build_unified_dataset(
        paths=paths,
        pubmed_limit=args.pubmed_limit,
        protocol_version=args.protocol_version,
    )

    save_unified_dataset(dataset, paths.out_path)

    counts = dataset["counts"]
    print(f"OK: dataset unificado gerado em: {paths.out_path}")
    print(f" - Protocolos: {counts['protocols']}")
    print(f" - FAQs: {counts['faqs']}")
    print(f" - PubMedQA: {counts['pubmedqa']}")
    print(f" - Total docs: {counts['total_docs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())