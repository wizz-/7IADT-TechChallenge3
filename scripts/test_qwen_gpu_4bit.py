import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_DIR = "models/qwen2.5-7b"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    print(f"CUDA OK: {torch.cuda.get_device_name(0)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"Modelo (4-bit) carregado em {time.time() - t0:.1f}s")

    messages = [
        {
            "role": "system",
            "content": (
                "Você é um assistente médico institucional. "
                "Responda em português, de forma curta, objetiva e segura. "
                "Não prescreva doses. Sempre recomende validação por um médico."
            ),
        },
        {"role": "user", "content": "Paciente com dor torácica e sudorese. Quais exames iniciais devo solicitar?"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )
    gen_s = time.time() - t1

    # ✅ Corta o prompt e decodifica só os tokens gerados
    generated = out[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print("\n--- RESPOSTA ---")
    print(answer)

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print("\n--- GPU ---")
    print(f"VRAM alocada: {allocated:.2f} GB")
    print(f"VRAM reservada: {reserved:.2f} GB")
    print(f"Tempo de geração: {gen_s:.1f}s")

if __name__ == "__main__":
    main()