import json
from pathlib import Path


class PatientRepository:

    def __init__(self):

        base = Path("src/app/data/raw/hospital_ficticio")

        with open(base / "pacientes.json", encoding="utf-8") as f:
            self.pacientes = json.load(f)

        with open(base / "prontuarios.json", encoding="utf-8") as f:
            self.prontuarios = json.load(f)

    def get_patient(self, patient_id: str):

        for p in self.pacientes:
            if p["patient_id"] == patient_id:
                return p

        return None

    def get_prontuario(self, patient_id: str):

        for p in self.prontuarios:
            if p["patient_id"] == patient_id:
                return p

        return None