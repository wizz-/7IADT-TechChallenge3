# 🏥 Assistente Médico com RAG + Fine-Tuning (Qwen 2.5)

Projeto desenvolvido para o **Tech Challenge – Pós-Graduação IA para Devs**.

O objetivo é construir um **chat clínico institucional** capaz de responder perguntas médicas com base em:

- Protocolos hospitalares
- FAQ clínico
- Literatura científica (PubMedQA)

Utiliza:

- ✅ RAG (Retrieval Augmented Generation)
- ✅ Fine-tuning LoRA
- ✅ LLM local (Qwen2.5-7B)
- ✅ Embeddings BGE
- ✅ FAISS
- ✅ Transformers + PyTorch

---

# 📦 Arquitetura do Projeto

TechChallenge/
│
├── models/ # modelos locais (não versionados)
│ ├── qwen2.5-7b/
│ ├── qwen2.5-7b-lora/
│ └── bge-small-en-v1.5/
│
├── scripts/
│ ├── chat_terminal.py # chat com RAG
│ ├── indexar_rag.py # cria índice FAISS
│ ├── finetune_lora.py # fine-tuning LoRA
│ └── build_dataset.py # unifica datasets
│
├── src/app/
│ ├── rag/
│ │ └── faiss_index.py
│ └── data/
│ ├── source/ # datasets originais
│ ├── unified/ # dataset unificado
│ └── index/ # índice FAISS
│
├── requirements.txt
└── README.md

# 🧠 Pipeline de IA
Datasets → Dataset Unificado → Embeddings → FAISS
↓
Recuperação (RAG)
↓
Qwen2.5 + LoRA (LLM)
↓
Resposta médica

---

# 📊 Datasets Utilizados

### 1️⃣ PubMedQA
Base científica com perguntas e respostas médicas.

### 2️⃣ FAQ Hospitalar (fictício)
Perguntas clínicas comuns baseadas em protocolos.

### 3️⃣ Protocolos Clínicos
Fluxos hospitalares estruturados.

Todos são unificados em:
src/app/data/unified/data.json

---

# ⚙️ Requisitos

## Hardware recomendado
- GPU NVIDIA ≥ 8GB VRAM
- 16–32GB RAM
- SSD

Testado em:
- RTX 3060 12GB
- 32GB RAM

---

# 🧪 Instalação

## 1️⃣ Clonar projeto
```bash
git clone <repo>
cd TechChallenge

2️⃣ Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate

3️⃣ Instalar dependências
pip install -r requirements.txt

🤖 Download dos Modelos
Baixar de:
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

Salvar em:
models/qwen2.5-7b

Embeddings BGE
https://huggingface.co/BAAI/bge-small-en-v1.5

Salvar em:
models/bge-small-en-v1.5

📚 Preparar Dataset
src/app/data/source/

Arquivos esperados:
pubmedqa.json
faq_hospital.json
protocolos.json

🔧 Gerar Dataset Unificado
python scripts/build_dataset.py

Saída:
src/app/data/unified/data.json

🔎 Criar Índice RAG (FAISS)
python scripts/indexar_rag.py

Saída:
src/app/data/index/
   index.faiss
   chunks.jsonl

💬 Rodar Chat Médico
python scripts/chat_terminal.py

Exedmplo:
> Quais exames iniciais para dor torácica?
Eletrocardiograma (ECG) imediato, troponina sérica e radiografia de tórax. [Fonte 2]
Aviso: esta orientação não substitui avaliação médica.

---

🧬 Fine-Tuning LoRA
python scripts/finetune_lora.py --use-4bit --epochs 1 --max-len 1024

Saída:
models/qwen2.5-7b-lora/
O chat automaticamente usa o LoRA se a pasta existir.

⚡ Modos de Execução
Sem LoRA
apagar pasta:
models/qwen2.5-7b-lora

Com LoRA
pasta presente → carregamento automático

---

📖 Funcionamento do RAG

Pergunta do usuário
Busca semântica FAISS
Recupera trechos relevantes
Injeta no prompt
Qwen gera resposta citando fontes

---

📖 Funcionamento do RAG

Pergunta do usuário
Busca semântica FAISS
Recupera trechos relevantes
Injeta no prompt
Qwen gera resposta citando fontes

---

🛑 Limitações
Não substitui avaliação médica
Dataset fictício
Protocolos simulados
Uso educacional

---

👨‍💻 Integrantes
(Preencher pelo grupo)

---

📚 Tecnologias

Python
PyTorch
Transformers
PEFT / LoRA
FAISS
Sentence Transformers
Qwen 2.5
BGE Embeddings

--

🎯 Status do Projeto

✅ Dataset unificado
✅ RAG funcionando
✅ Chat terminal
✅ Fine-tuning LoRA
⬜ Interface gráfica
⬜ LangChain/LangGraph
⬜ Avaliação automática


---

🚀 Próximos Passos

Interface web
Avaliação RAG vs LoRA
Métricas clínicas
LangGraph workflow
Deploy

---

📜 Licença

Projeto educacional – Pós IA para Devs
