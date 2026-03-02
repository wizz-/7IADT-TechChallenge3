# 🏥 Assistente Médico com RAG + Fine‑Tuning (Qwen 2.5)

Repositório do **Grupo 63 — Tech Challenge 3 (Pós‑Graduação IA para Devs)**.

Este projeto implementa um **assistente clínico institucional com RAG + LLM local**, capaz de responder perguntas médicas fundamentadas em protocolos, FAQ clínico e literatura científica.

---

# 🎯 Objetivo

Construir um sistema de QA médico que:

- interpreta perguntas clínicas em linguagem natural
- recupera evidências relevantes (RAG)
- gera respostas fundamentadas com LLM local
- cita fontes e mantém contexto clínico

---

# 🧠 Arquitetura de IA

```
Pergunta do usuário
        ↓
Embeddings (BGE)
        ↓
Busca vetorial (FAISS)
        ↓
Contexto recuperado (RAG)
        ↓
Qwen 2.5 + LoRA
        ↓
Resposta médica fundamentada
```

---

# 📁 Estrutura do Repositório

```
7IADT-TechChallenge3/
│
├── challenge/              # Materiais do Tech Challenge
│
├── scripts/                # Scripts executáveis do pipeline
│   ├── build_dataset.py
│   ├── indexar_rag.py
│   ├── chat_terminal.py
│   └── finetune_lora.py
│
├── src/app/
│   ├── rag/
│   │   └── faiss_index.py
│   │
│   └── data/
│       ├── source/         # Datasets originais
│       ├── unified/        # Dataset consolidado
│       └── index/          # Índice vetorial FAISS
│
├── requirements.txt
└── readme.md
```

---

# 📊 Datasets

O assistente utiliza três fontes principais:

## PubMedQA
Base científica com perguntas e respostas médicas.

## FAQ Hospitalar (fictício)
Perguntas clínicas baseadas em protocolos.

## Protocolos Clínicos
Fluxos hospitalares estruturados.

Todos são unificados em:

```
src/app/data/unified/data.json
```

---

# ⚙️ Requisitos

## Hardware recomendado

- GPU NVIDIA ≥ 8 GB VRAM
- 16–32 GB RAM
- SSD

Testado em:

- RTX 3060 12 GB
- 32 GB RAM

---

# 🧪 Instalação

## 1. Clonar repositório
```bash
git clone https://github.com/wizz-/7IADT-TechChallenge3
cd 7IADT-TechChallenge3
```

## 2. Ambiente virtual
```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Dependências
```bash
pip install -r requirements.txt
```

---

# 🤖 Download dos Modelos

## LLM — Qwen 2.5‑7B
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

Salvar em:
```
models/qwen2.5-7b
```

## Embeddings — BGE
https://huggingface.co/BAAI/bge-small-en-v1.5

Salvar em:
```
models/bge-small-en-v1.5
```

---

# 🔧 Pipeline de Execução

## 1️⃣ Construir dataset unificado
```bash
python scripts/build_dataset.py
```

Saída:
```
src/app/data/unified/data.json
```

## 2️⃣ Gerar índice vetorial (RAG)
```bash
python scripts/indexar_rag.py
```

Saída:
```
src/app/data/index/index.faiss
src/app/data/index/chunks.jsonl
```

## 3️⃣ Executar chat clínico
```bash
python scripts/chat_terminal.py
```

Exemplo:

```
> Quais exames iniciais para dor torácica?

ECG imediato, troponina sérica e radiografia de tórax. [Fonte 2]
Aviso: esta orientação não substitui avaliação médica.
```

---

# 🧬 Fine‑Tuning LoRA

```bash
python scripts/finetune_lora.py --use-4bit --epochs 1 --max-len 1024
```

Saída:

```
models/qwen2.5-7b-lora/
```

Se a pasta existir, o chat usa automaticamente o LoRA.

---

# ⚡ Modos de Execução

## Base (sem LoRA)
Remover:
```
models/qwen2.5-7b-lora
```

## Fine‑tuned
Pasta presente → carregamento automático

---

# 🔎 Funcionamento do RAG

1. Usuário faz pergunta clínica  
2. Sistema gera embedding  
3. Busca semântica FAISS  
4. Recupera trechos relevantes  
5. Injeta no prompt do Qwen  
6. LLM gera resposta com fontes  

---

# 🛑 Limitações

- Dataset parcialmente fictício  
- Protocolos simulados  
- Não substitui avaliação médica  
- Uso educacional  

---

# 📚 Tecnologias

- Python  
- PyTorch  
- Transformers  
- PEFT (LoRA)  
- FAISS  
- Sentence Transformers  
- Qwen 2.5  
- BGE Embeddings  

---

# 🎯 Status do Projeto

- ✅ Dataset unificado  
- ✅ Pipeline RAG  
- ✅ Chat terminal  
- ✅ Fine‑tuning LoRA  
- ⬜ Interface gráfica  
- ⬜ API REST  
- ⬜ Avaliação automática  

---

# 🚀 Próximos Passos

- Interface web clínica  
- Métricas de resposta médica  
- Avaliação RAG vs LoRA  
- Workflow LangGraph  
- Deploy local hospitalar  

---

# 👨‍💻 Grupo 63

Tech Challenge 3 — Pós‑Graduação IA para Devs

(Preencher nomes)

---

# 📜 Licença

Projeto educacional — FIAP Pós‑Tech IA para Devs
