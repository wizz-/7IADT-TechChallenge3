from src.app.data.patient_repository import PatientRepository

repo = PatientRepository()

patient = repo.get_patient("P001")
prontuario = repo.get_prontuario("P001")

print(patient)
print(prontuario)