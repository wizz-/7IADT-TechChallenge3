from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

dataset_path = "src/app/data/training/sft_train_openai.jsonl"

print("Enviando dataset...")

with open(dataset_path, "rb") as f:
    file = client.files.create(
        file=f,
        purpose="fine-tune"
    )

print("Arquivo enviado:")
print(file.id)

print("\nCriando job de fine-tuning...")

job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo"
)

print("\nFine-tuning iniciado:")
print(job.id)

print("\nAcompanhe em:")
print("https://platform.openai.com/finetune")