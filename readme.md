
# 🏥 Assistente Médico com RAG + Fine‑Tuning (OpenAI)

Repositório do **Grupo 63 — Tech Challenge 3 (Pós‑Tech IA para Devs)**.

Este projeto implementa um **assistente clínico institucional** capaz de responder perguntas médicas com base em:

- protocolos hospitalares
- FAQ clínico
- literatura científica

A solução utiliza:

- **RAG (Retrieval Augmented Generation)** para recuperar evidências relevantes
- **Fine‑tuning** para ajustar o comportamento do modelo
- **OpenAI API** para embeddings, treinamento e geração de respostas

---

# 🎯 Objetivo do Projeto

Construir um sistema de **Perguntas e Respostas Médicas** que:

- interprete perguntas clínicas em linguagem natural
- recupere evidências relevantes de uma base de conhecimento
- gere respostas fundamentadas
- cite fontes utilizadas
- mantenha segurança clínica e linguagem institucional

---

# 🧠 Arquitetura de IA

Arquitetura final implementada no projeto:

Pergunta do usuário  
↓  
Embeddings da pergunta (OpenAI)  
↓  
Busca vetorial FAISS (RAG)  
↓  
Recuperação de contexto relevante  
↓  
Modelo Fine‑Tuned da OpenAI  
↓  
Resposta médica fundamentada com fontes

---

# 📁 Estrutura do Repositório

```
7IADT-TechChallenge3/
│
├── challenge/
│   └── Tech Challenge IADT - Fase 3.pdf
│
├── scripts/
│   ├── gerar_dataset_unificado.py
│   ├── gerar_dataset_sft.py
│   ├── indexar_rag.py
│   ├── testar_rag.py
│   ├── criar_finetuning.py
│   └── chat_terminal.py
│
├── src/
│   └── app/
│       ├── data/
│       │   ├── raw/
│       │   │   ├── hospital_ficticio/
│       │   │   │   ├── faq.json
│       │   │   │   └── protocolos.json
│       │   │   └── pubmedqa/
│       │   │       └── ori_pqal.json
│       │   │
│       │   ├── processed/
│       │   │   └── data.json
│       │   │
│       │   ├── index/
│       │   │   ├── index.faiss
│       │   │   ├── chunks.jsonl
│       │   │   └── meta.json
│       │   │
│       │   └── training/
│       │       └── sft_train_openai.jsonl
│       │
│       ├── llm/
│       │   └── openai_client.py
│       │
│       └── rag/
│           └── faiss_index.py
│
├── .env
├── env.sample
├── requirements.txt
└── README.md
```

---

# 📚 Fontes de Dados

O assistente utiliza três fontes principais:

## Protocolos Clínicos

Fluxos hospitalares estruturados contendo:

- definição
- critérios de suspeita
- exames iniciais
- conduta inicial
- critérios de alto risco
- encaminhamento

## FAQ Hospitalar

Perguntas e respostas baseadas diretamente nos protocolos clínicos.

## PubMedQA

Base científica com perguntas e respostas médicas utilizadas para enriquecer o comportamento do modelo.

---

# ⚙️ Tecnologias Utilizadas

- Python
- OpenAI API
- FAISS
- NumPy
- python-dotenv
- Sentence Transformers
- LangChain
- LangGraph

---

# 💻 Requisitos

Este projeto foi adaptado para rodar em **qualquer computador de escritório**, sem necessidade de GPU.

Requisitos:

- Python 3.10+
- acesso à internet
- chave de API da OpenAI

---

# 🔑 Configuração do Ambiente

## 1. Clonar o repositório

```
git clone https://github.com/wizz-/7IADT-TechChallenge3
cd 7IADT-TechChallenge3
```

## 2. Criar ambiente virtual

Windows PowerShell:

```
python -m venv .venv
.\.venv\Scripts\activate
```

## 3. Instalar dependências

```
pip install -r requirements.txt
```

---

# 🔐 Configurar a chave da OpenAI

Copie o arquivo `.env.sample`:

```
copy .env.sample .env
```

Edite o `.env` e coloque sua chave:

```
OPENAI_API_KEY=sua_chave_aqui
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

---

# 🔧 Pipeline de Execução

A execução do projeto segue **4 etapas principais**.

---

# 1️⃣ Construção do Dataset Unificado

Execute:

```
python scripts/gerar_dataset_unificado.py
```

Saída:

```
src/app/data/processed/data.json
```

---

# 2️⃣ Construção do Índice Vetorial (RAG)

Execute:

```
python scripts/indexar_rag.py
```

Saída:

```
src/app/data/index/index.faiss
src/app/data/index/chunks.jsonl
```

---

# 3️⃣ Geração do Dataset de Fine‑Tuning

Execute:

```
python scripts/gerar_dataset_sft.py
```

Saída:

```
src/app/data/training/sft_train_openai.jsonl
```

---

# 4️⃣ Treinamento do Modelo

Execute:

```
python scripts/criar_finetuning.py
```

Após terminar, será gerado um modelo:

```
ft:gpt-3.5-turbo:seu-projeto
```

Atualize o `.env` com esse modelo.

---

# 5️⃣ Executar o Chat Médico

Execute:

```
python scripts/chat_terminal.py
```

Exemplo:

```
> Quais exames iniciais para dor torácica?

ECG imediato, troponina sérica e radiografia de tórax. [Fonte 1]

Aviso: esta orientação não substitui avaliação médica.
```

---

# ⚠️ Limitações

- Dataset parcialmente fictício
- Uso educacional
- Não substitui avaliação médica

---

# 👨‍💻 Grupo 63

Tech Challenge 3 — Pós‑Tech IA para Devs
