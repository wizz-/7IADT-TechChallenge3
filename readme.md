
# 🏥 Assistente Médico com RAG + Fine-Tuning  
## Tech Challenge – Fase 3 – IA para Devs (FIAP)

Projeto acadêmico desenvolvido para o **Tech Challenge da Fase 3** da pós-graduação **IA para Devs (FIAP)**.

Este trabalho implementa um **assistente médico inteligente** com foco em apoio informacional, utilizando:

- **LLM com fine-tuning**
- **RAG (Retrieval Augmented Generation)**
- **LangChain / LangGraph**
- **consulta a dados estruturados**
- **guardrails de segurança**
- **logging e explicabilidade**

> **Aviso importante:** este projeto é **educacional** e **não substitui avaliação médica real**.  
> As respostas geradas devem ser interpretadas como apoio ao estudo e à demonstração técnica da solução.

---

# 1. Objetivo do projeto

O objetivo deste trabalho é construir um **assistente clínico hospitalar simulado** capaz de:

- responder perguntas médicas com base em **protocolos e FAQs**
- consultar **pacientes e prontuários fictícios**
- usar **RAG** para recuperar contexto relevante antes de responder
- utilizar uma **LLM fine-tuned**
- aplicar **regras de segurança** para evitar respostas perigosas
- registrar **logs estruturados** para rastreabilidade e análise posterior

A proposta foi desenhada para atender aos principais pontos esperados no Tech Challenge, com ênfase em:

- **processo de fine-tuning**
- **assistente médico com LangChain / LangGraph**
- **segurança, validação e observabilidade**
- **relatório técnico e demonstração prática**

---

# 2. Escopo da solução

A solução simula um cenário hospitalar em que o usuário pode:

- consultar protocolos clínicos
- fazer perguntas sobre pacientes fictícios
- avaliar situações de risco clínico
- verificar fontes recuperadas pelo RAG
- inspecionar o comportamento do workflow via modo debug

A arquitetura combina:

- **base documental** para perguntas médicas
- **base estruturada** para pacientes e prontuários
- **workflow com LangGraph**
- **modelo fine-tuned na OpenAI**
- **mecanismos de segurança**
- **logging em JSONL**

---

# 3. Arquitetura da solução

Fluxo simplificado do sistema:

```text
Pergunta do usuário
        ↓
Identificação do paciente citado
        ↓
Consulta da base estruturada (pacientes / prontuários)
        ↓
Busca de contexto em protocolos e documentos médicos (RAG)
        ↓
LLM fine-tuned gera resposta
        ↓
Aplicação de validações e guardrails
        ↓
Resposta final com explicabilidade e logs
```

O sistema utiliza **LangGraph** para orquestrar o fluxo e controlar as etapas de processamento.

---

# 4. Tecnologias utilizadas

- **Python**
- **OpenAI API**
- **LangChain**
- **LangGraph**
- **FAISS**
- **Embeddings OpenAI**
- **JSON / JSONL**
- **logging estruturado**

---

# 5. Estrutura do projeto

```text
scripts/
    chat_langgraph.py
    criar_finetuning.py
    generate_report.py
    gerar_dataset_sft.py
    gerar_dataset_unificado.py
    indexar_rag.py

src/
    app/
        data/
            raw/
                hospital_ficticio/
                pubmedqa/
            processed/
            rag_index/
            training/
        llm/
            openai_client.py
        observability/
            workflow_logger.py
        rag/
            faiss_index.py
        workflow/
            medical_graph.py

challenge/
    Tech Challenge IADT - Fase 3.pdf

README.md
requirements.txt
.env
```

### Resumo dos principais arquivos

- **`scripts/chat_langgraph.py`**: chat principal da aplicação
- **`scripts/gerar_dataset_unificado.py`**: consolida os dados brutos em um dataset unificado
- **`scripts/gerar_dataset_sft.py`**: gera o dataset em formato adequado para fine-tuning
- **`scripts/criar_finetuning.py`**: cria o job de fine-tuning na OpenAI
- **`scripts/indexar_rag.py`**: gera embeddings e índice FAISS para o RAG
- **`scripts/generate_report.py`**: gera relatório simples a partir dos logs
- **`src/app/workflow/medical_graph.py`**: workflow do assistente com LangGraph
- **`src/app/rag/faiss_index.py`**: indexação e recuperação de contexto
- **`src/app/observability/workflow_logger.py`**: gravação de logs estruturados

---

# 6. Base de dados utilizada

O projeto utiliza **dados sintéticos e públicos**, organizados em duas frentes.

## 6.1 Hospital fictício

A base hospitalar simulada contém:

- **FAQs médicas**
- **protocolos clínicos**
- **pacientes fictícios**
- **prontuários fictícios**

Esses dados representam um ambiente hospitalar controlado para fins acadêmicos.

## 6.2 Dataset científico

Também foi utilizado o dataset **PubMedQA**, contendo perguntas médicas baseadas em artigos científicos.

## 6.3 Tratamento dos dados

Os dados utilizados no projeto são:

- **fictícios**, no caso da base hospitalar
- **públicos**, no caso do PubMedQA
- organizados e transformados em um **dataset unificado**
- posteriormente convertidos para **JSONL** na etapa de fine-tuning

---

# 7. Pacientes fictícios da base

O sistema possui uma base de pacientes simulados que pode ser consultada no chat.

## Pacientes disponíveis

- **P001**
- **P002**
- **P003**

## Convenção de uso no chat

Para garantir identificação correta do paciente, recomenda-se usar sempre o **código do paciente**.

Exemplo:

```text
me fale do paciente P003
```

Essa convenção foi adotada para tornar a recuperação do prontuário mais confiável durante a demonstração.

## Exemplos de perguntas

```text
me fale do paciente P003
considerando o paciente P003, qual a conduta inicial se ele chegar com falta de ar?
agora o paciente P002. Ela chegou com dor torácica. Qual a conduta inicial?
```

---

# 8. Fine-tuning do modelo

Foi realizado **fine-tuning de uma LLM na OpenAI**, atendendo ao requisito do trabalho de customização do modelo.

## Modelo base

```text
gpt-3.5-turbo-0125
```

## Modelo gerado

```text
ft:gpt-3.5-turbo-0125:personal::DGFuwQ27
```

## Dataset utilizado no fine-tuning

Arquivo:

```text
sft_train_openai.jsonl
```

Conteúdo principal do dataset de treino:

- FAQs médicas
- protocolos clínicos
- conhecimento médico estruturado

## Objetivo do fine-tuning

O fine-tuning foi realizado para que o modelo:

- respondesse em estilo mais aderente ao domínio médico
- utilizasse linguagem mais consistente com protocolos
- reduzisse respostas excessivamente genéricas
- servisse como base da camada geradora do assistente

---

# 9. Observação importante sobre custo do fine-tuning

O fine-tuning foi executado na **plataforma OpenAI**.

Durante o desenvolvimento deste trabalho, a geração do modelo fine-tuned teve custo aproximado de:

```text
US$ 5
```

Esse valor pode variar conforme:

- tamanho do dataset
- quantidade de tokens processados
- política de preços vigente na OpenAI

## Recomendação acadêmica

Como este é um **trabalho de faculdade**, recomenda-se cuidado ao repetir essa etapa.

Se o script de fine-tuning for executado novamente, **novos custos podem ser gerados** na conta utilizada.

Por isso, a recomendação é:

- executar o fine-tuning apenas quando necessário
- reutilizar o modelo já gerado no projeto
- deixar claro para o grupo que essa etapa pode envolver custo financeiro real

---

# 10. RAG (Retrieval Augmented Generation)

O sistema utiliza **RAG com FAISS** para recuperar documentos relevantes antes de gerar a resposta.

## Pipeline resumido

1. os documentos médicos são consolidados
2. os textos são transformados em embeddings
3. os embeddings são indexados em FAISS
4. o sistema recupera os trechos mais relevantes para a pergunta
5. a LLM responde usando o contexto recuperado

## Benefícios do RAG no projeto

- melhora a aderência das respostas aos protocolos
- reduz respostas puramente genéricas
- permite mostrar **fontes utilizadas**
- reforça a **explicabilidade** da solução

---

# 11. Workflow com LangGraph

A aplicação utiliza **LangGraph** para organizar o fluxo do assistente.

De forma simplificada, o workflow executa etapas como:

- identificar o paciente citado
- consultar a base estruturada
- recuperar documentos relevantes via RAG
- enviar contexto consolidado para a LLM
- validar a resposta
- sinalizar casos sensíveis com guardrails
- registrar o resultado em log

Esse fluxo torna a solução mais modular, mais legível e mais apropriada para evolução futura.

---

# 12. Segurança, validação e guardrails

O sistema possui mecanismos para reduzir respostas inadequadas em contexto sensível.

## Exemplos de comportamentos tratados

- tentativa de pedir **prescrição direta**
- pedido de **dose de medicamento**
- tentativa de obter **diagnóstico definitivo**
- perguntas com **potencial gravidade clínica**

## Comportamento esperado

Nesses casos, o sistema pode:

- evitar resposta prescritiva direta
- responder com cautela
- sinalizar necessidade de avaliação médica presencial
- marcar a interação com sinais de segurança no workflow

Esses guardrails foram incluídos para atender ao caráter crítico do domínio de saúde.

---

# 13. Observabilidade, logs e rastreabilidade

Cada interação do chat gera um log estruturado.

## Informações registradas

- pergunta do usuário
- tempo de resposta
- fontes utilizadas
- guardrails acionados
- sinais de validação do workflow

## Formato

Os logs são gravados em:

```text
JSONL
```

## Objetivo dos logs

Os logs permitem:

- analisar o comportamento do assistente
- rastrear decisões
- gerar relatórios de uso
- apoiar a avaliação do sistema

---

# 14. Debug e explicabilidade

Durante a execução do chat, é possível ativar um modo de inspeção para visualizar o comportamento interno da resposta.

## Comando

```text
/debug
```

## O que o modo debug exibe

- documentos recuperados pelo RAG
- score de similaridade
- tipo da fonte (`protocol`, `faq`, `scientific`)
- sinais de validação do workflow
- guardrails acionados

## Exemplo de valor acadêmico

Esse recurso ajuda a demonstrar:

- de onde veio a resposta
- como o sistema recuperou o contexto
- quais mecanismos de segurança foram aplicados

Isso é importante para a parte de **explicabilidade** pedida no trabalho.

---

# 15. Como verificar as fontes das respostas

Quando o modo debug está ativado, o sistema exibe a lista de fontes recuperadas pelo RAG.

Exemplo de saída esperada:

```text
Fontes usadas (top-k):
- [Fonte 1] score=...
- [Fonte 2] score=...
- [Fonte 3] score=...
```

Com isso, é possível mostrar na apresentação que a resposta não foi produzida “do nada”, mas sim apoiada em:

- protocolo clínico
- FAQ hospitalar
- conteúdo científico

---

# 16. Como executar o projeto desde o começo

Esta seção foi escrita pensando no contexto de **trabalho acadêmico**, ou seja, para que o professor ou os integrantes do grupo consigam entender a sequência completa.

## 16.1 Pré-requisitos

- Python instalado
- acesso à OpenAI API
- arquivo `.env` configurado
- dependências do projeto instaladas

## 16.2 Criar ambiente virtual

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 16.3 Instalar dependências

```bash
pip install -r requirements.txt
```

## 16.4 Configurar variáveis de ambiente

Criar um arquivo `.env` na raiz do projeto com, no mínimo:

```env
OPENAI_API_KEY=SEU_TOKEN
OPENAI_CHAT_MODEL=ft:gpt-3.5-turbo-0125:personal::DGFuwQ27
```

Dependendo da implementação local, também podem existir outras variáveis relacionadas a embeddings e paths internos.

## 16.5 Gerar dataset unificado

```bash
python scripts/gerar_dataset_unificado.py
```

## 16.6 Gerar dataset de treino para fine-tuning

```bash
python scripts/gerar_dataset_sft.py
```

## 16.7 Criar índice RAG

```bash
python scripts/indexar_rag.py
```

## 16.8 Fine-tuning

```bash
python scripts/criar_finetuning.py
```

> **Atenção:** essa etapa pode gerar custo na OpenAI.  
> Após o envio do dataset, é necessário **aguardar o processamento e a criação do modelo fine-tuned** na plataforma da OpenAI.  
> Depois disso, o nome do modelo gerado deve ser configurado no `.env`.

## 16.9 Executar o chat principal

```bash
python scripts/chat_langgraph.py
```

## 16.10 Gerar relatório a partir dos logs

```bash
python scripts/generate_report.py
```

---

# 17. Relatório técnico detalhado do projeto

Esta seção consolida, em formato textual, os principais pontos técnicos exigidos pelo trabalho.

## 17.1 Processo de fine-tuning

O projeto passou por uma etapa de preparação de dados, na qual documentos médicos foram consolidados e transformados em um arquivo **JSONL** apropriado para treino supervisionado.

Em seguida, foi realizado o fine-tuning de um modelo da OpenAI, gerando um modelo customizado para o domínio do projeto:

```text
ft:gpt-3.5-turbo-0125:personal::DGFuwQ27
```

Essa etapa foi importante para atender ao requisito de uso de uma **LLM ajustada ao contexto médico**.

## 17.2 Descrição do assistente médico criado

O assistente foi desenvolvido para atuar como um **apoio conversacional clínico simulado**, combinando:

- resposta com LLM
- recuperação de contexto via RAG
- consulta a dados estruturados de pacientes
- validações de segurança
- rastreamento por logs

O foco não é substituir decisão médica, e sim demonstrar uma solução técnica coerente para o domínio de saúde.

## 17.3 Fluxo do sistema

Fluxo resumido:

```text
Pergunta
  ↓
Identificação do paciente
  ↓
Consulta a pacientes/prontuários
  ↓
Recuperação de contexto no RAG
  ↓
Geração de resposta pela LLM fine-tuned
  ↓
Validação e guardrails
  ↓
Resposta final + logs + fontes
```

## 17.4 Avaliação do modelo e análise dos resultados

A avaliação do sistema foi feita por meio de perguntas representativas sobre:

- dor torácica
- sepse
- dispneia
- uso de medicamento
- situações envolvendo pacientes fictícios

Os testes mostraram que o sistema consegue:

- recuperar protocolos relevantes
- responder com contexto documental
- consultar pacientes fictícios
- exibir fontes recuperadas
- ativar alertas em perguntas sensíveis

Além disso, os logs permitem observar:

- tempo de resposta
- fontes utilizadas
- guardrails acionados

---

# 18. Perguntas sugeridas para avaliação e apresentação

Abaixo estão perguntas recomendadas para explorar bem o projeto durante testes e apresentação.

## 18.1 Perguntas sobre protocolos

```text
quais os exames iniciais para dor torácica?
quais sinais indicam sepse?
quando solicitar troponina?
```

## 18.2 Perguntas sobre pacientes

```text
me fale do paciente P003
me fale do paciente P002
me fale do paciente P001
```

## 18.3 Perguntas sobre situação clínica

```text
considerando o paciente P003, qual a conduta inicial se ele chegar com falta de ar?
agora o paciente P002. Ela chegou com dor torácica. Qual a conduta inicial?
e se o paciente P003 vier com saturação baixa?
```

## 18.4 Perguntas para demonstrar segurança

```text
qual dose de furosemida devo prescrever?
qual remédio devo dar para esse paciente agora?
posso fechar diagnóstico só com esses sinais?
```

---

# 19. Roteiro sugerido para o vídeo

Abaixo está um roteiro simples e seguro para a gravação da apresentação.

## Etapa 1 – Introdução

Explicar rapidamente:

- objetivo do projeto
- uso de LLM em saúde
- combinação de fine-tuning, RAG e LangGraph

## Etapa 2 – Mostrar arquitetura

Apresentar de forma breve:

- dataset fictício e científico
- fine-tuning realizado
- workflow com LangGraph
- RAG com FAISS
- logs e explicabilidade

## Etapa 3 – Mostrar o chat

Executar:

```bash
python scripts/chat_langgraph.py
```

## Etapa 4 – Perguntas da demo

### Pergunta 1 – protocolo

```text
quais os exames iniciais para dor torácica?
```

### Pergunta 2 – paciente

```text
me fale do paciente P003
```

### Pergunta 3 – conduta clínica

```text
considerando o paciente P003, qual a conduta inicial se ele chegar com falta de ar?
```

### Pergunta 4 – segurança

```text
qual dose de furosemida devo prescrever?
```

### Pergunta 5 – explicabilidade

```text
/debug
```

Depois repetir uma pergunta para mostrar:

- fontes utilizadas
- score de similaridade
- guardrails acionados

## Etapa 5 – Encerramento

Concluir destacando:

- fine-tuning realizado
- uso de RAG
- consulta a pacientes fictícios
- guardrails de segurança
- logs e explicabilidade

---

# 20. Limitações e observações

- o projeto utiliza **dados fictícios e públicos**
- a identificação de pacientes foi padronizada por **código**
- o fine-tuning depende de **serviço pago**
- respostas em saúde exigem sempre cautela
- o sistema foi desenvolvido para fins **acadêmicos e demonstrativos**

---

# 21. Integrantes do grupo

**Grupo 63**

- Felipe Demétrius Martins da Silva
- Kauê Bove Martins
- Laerte Kimura
- Thiago Albuquerque Rosa
- Vinicius Otavio Soares da Silva
