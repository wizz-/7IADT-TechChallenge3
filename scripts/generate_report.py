import json
from pathlib import Path
from statistics import mean

LOG_FILE = Path("outputs/logs/workflow_log.jsonl")

def load_logs():
    with open(LOG_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_report():
    logs = load_logs()

    total = len(logs)
    avg_time = mean(x["duration_seconds"] for x in logs)

    with_patient = sum(1 for x in logs if x["current_patient_id"])
    without_patient = total - with_patient

    guardrails = sum(len(x.get("guardrail_flags", [])) for x in logs)
    escalations = sum(1 for x in logs if x.get("needs_escalation"))

    avg_sources = mean(len(x.get("sources", [])) for x in logs)

    report = f"""
# Relatório de Execução do Assistente Médico

## Visão Geral

Total de interações: {total}
Tempo médio de resposta: {avg_time:.2f}s

## Uso de Contexto de Paciente

Perguntas com paciente: {with_patient}
Perguntas gerais: {without_patient}

## Recuperação de Contexto

Média de fontes utilizadas: {avg_sources:.2f}

## Segurança

Guardrails acionados: {guardrails}
Escalações clínicas: {escalations}

"""

    Path("outputs/reports/report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    generate_report()