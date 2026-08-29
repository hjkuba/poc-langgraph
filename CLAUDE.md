# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

POC mínima demonstrando LangChain + LangGraph juntos: tool calling, roteamento
condicional, RAG e persistência via checkpointer. Sem framework web, sem
testes — é um app de terminal (`app.py`) que conversa com um LLM local via
Ollama.

## Setup e execução

Pré-requisitos: [Ollama](https://ollama.com) rodando localmente, com os dois
modelos baixados:

```bash
ollama pull qwen2.5:1.5b        # chat/tool calling
ollama pull nomic-embed-text    # embeddings (RAG)
```

Setup do ambiente Python (3.9+):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                # instala src/agent em modo editável
```

Rodar:

```bash
python app.py
```

Não há suíte de testes nem lint configurados neste projeto.

## Arquitetura

O grafo (LangGraph) é o orquestrador; LangChain entra dentro dos nós como a
abstração de chamada ao LLM/embeddings. Fluxo:

```
START -> retrieval -> chatbot -> (tools_condition) -> tools -> chatbot -> ... -> END
```

- **`src/agent/state.py`** — `State` do grafo: `messages` (reducer
  `add_messages`, acumula histórico) e `context` (sem reducer, sobrescrito a
  cada turno pelo nó `retrieval`).
- **`src/agent/retrieval.py`** — RAG. Na importação do módulo, indexa
  `data/*.md` (chunks -> embeddings -> `InMemoryVectorStore`, recalculado a
  cada início de processo). `retrieve()` busca por similaridade e filtra por
  `SIMILARITY_THRESHOLD` — sem esse corte, perguntas sem relação com `data/`
  ainda traziam os chunks "mais próximos" e confundiam o LLM.
- **`src/agent/tools.py`** — ferramentas (`somar`, `multiplicar`) via
  decorator `@tool`; a docstring de cada função vira a descrição que o LLM
  usa para decidir quando chamá-la.
- **`src/agent/configuration.py`** — monta `llm` (`ChatOllama` +
  `bind_tools`), `embeddings` (`OllamaEmbeddings`) e `checkpointer`
  (`SqliteSaver` sobre `.sqlite/checkpoints.sqlite`, git-ignorado).
- **`src/agent/graph.py`** — monta os nós (`retrieval`, `chatbot`, `tools`) e
  arestas, injeta o `context` do RAG como `SystemMessage` extra antes do LLM
  quando existe, e faz `compile(checkpointer=checkpointer)`.
- **`app.py`** — só o loop de conversa no terminal; usa um `thread_id` fixo
  (`"cli"`, app é single-sessão) para o checkpointer recuperar o histórico
  entre execuções.

### Coisas não óbvias ao mexer nisso

- **Modelo do chat é `qwen2.5:1.5b`, não trocar por `llama3.2` sem motivo**:
  já testado (1b e 3b) e ambos chamavam ferramentas até em saudações,
  ignorando o resultado na resposta final, mesmo com system prompt reforçado.
- **Não reforçar demais o `SYSTEM_PROMPT`** (em `agent/graph.py`): tentar
  deixar mais explícito que o modelo não deve calcular de cabeça piorou o
  comportamento em vez de corrigir.
- O loop `tools -> chatbot` **não** volta a passar por `retrieval` — o
  contexto do RAG vale para o turno inteiro, buscado uma única vez.
- `InMemoryVectorStore` (RAG) não persiste em disco — reindexação completa
  toda vez que o processo sobe. Diferente disso, o checkpointer sqlite
  (histórico de conversa) sobrevive ao restart.
