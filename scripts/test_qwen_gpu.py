import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "models/qwen2.5-7b"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível. Confirme a instalação do torch com CUDA.")

    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA OK: {device_name}")

    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    load_s = time.time() - t0
    print(f"Modelo carregado em {load_s:.1f}s")

    prompt = (
        "Você é um assistente médico institucional. Responda em português, de forma curta.\n\n"
        "Pergunta: Paciente com dor torácica e sudorese. Quais exames iniciais devo solicitar?\n"
        "Resposta:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_s = time.time() - t1

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("\n--- SAÍDA ---")
    print(text)

    # VRAM (aproximação)
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print("\n--- GPU ---")
    print(f"VRAM alocada: {allocated:.2f} GB")
    print(f"VRAM reservada: {reserved:.2f} GB")
    print(f"Tempo de geração: {gen_s:.1f}s")

if __name__ == "__main__":
    main()