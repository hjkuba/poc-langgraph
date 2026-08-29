# POC: LangChain + LangGraph

Estrutura mínima para entender como LangChain e LangGraph se encaixam, com
tool calling, roteamento condicional e RAG.

- **LangGraph** define o grafo: um `State` (o que passa de nó em nó), três
  nós (`retrieval`, `chatbot` e `tools`) e as arestas `START -> retrieval ->
  chatbot -> (decisão) -> tools -> chatbot -> ... -> END`.
- **LangChain** entra dentro dos nós, como a abstração padronizada de
  chamada ao modelo (`ChatOllama.invoke(...)`, com `bind_tools()` para tool
  calling) e de embeddings (`OllamaEmbeddings`) para o RAG.

## Estrutura

```
data/
└── manual-fluxarion.md   # documento fictício usado como base do RAG
src/agent/
├── state.py           # State do grafo (messages + context)
├── tools.py            # ferramentas disponíveis para o LLM
├── configuration.py    # LLM, modelo de embeddings
├── retrieval.py         # RAG: indexação (data/*.md) e busca por similaridade
└── graph.py             # nós, arestas (decisão + loop) e compile()
app.py                   # entrypoint: loop de conversa no terminal
```

## Pré-requisitos

1. [Ollama](https://ollama.com) instalado e rodando.
2. O modelo de chat:
   ```bash
   ollama pull qwen2.5:1.5b
   ```
   (testamos `llama3.2`, 1b e 3b: mesmo com system prompt reforçado, os dois
   chamavam ferramentas quase sempre — até em saudações — e ignoravam o
   resultado na resposta final. `qwen2.5` tem fine-tuning de function
   calling mais criterioso e não repete esse padrão)
3. O modelo de embeddings, usado pelo RAG:
   ```bash
   ollama pull nomic-embed-text
   ```
4. Python 3.9+.

## Setup (uma vez)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                # instala o pacote src/agent em modo editável
```

## Rodando

```bash
source .venv/bin/activate       # se ainda não estiver ativo
python app.py
```

Digite mensagens no terminal; digite `sair` para encerrar. O histórico da
conversa é mantido pelo `State` do LangGraph (via `add_messages`) enquanto o
loop roda. Se a pergunta envolver soma ou multiplicação, o LLM pode decidir
chamar uma ferramenta (`src/agent/tools.py`) antes de responder. Perguntas
relacionadas ao conteúdo de `data/manual-fluxarion.md` (ex: *"qual o limite
de requisições por minuto da API Fluxarion?"*) são respondidas via RAG —
tente perguntar algo que só está nesse documento pra ver o retrieval em
ação (o LLM não tem como adivinhar essas respostas sozinho).

## RAG: como funciona

1. Na importação de `src/agent/retrieval.py`, os arquivos `.md` de `data/`
   são quebrados em chunks, viram embeddings (`nomic-embed-text`) e ficam
   guardados em um `InMemoryVectorStore` (sem banco externo — recalculado a
   cada início do processo).
2. A cada turno, o nó `retrieval` gera o embedding da pergunta do usuário e
   busca os chunks mais similares por similaridade de cosseno.
3. Só chunks acima de `SIMILARITY_THRESHOLD` (calibrado testando perguntas
   relevantes vs. irrelevantes ao documento) são passados adiante — sem
   esse corte, perguntas sem relação com `data/` ainda traziam os "chunks
   mais próximos" (mesmo que pouco relevantes) e confundiam o LLM, que
   passava a recusar responder perguntas gerais.
4. O nó `chatbot` injeta esses chunks como uma `SystemMessage` extra antes
   de chamar o LLM — só quando existe contexto relevante para o turno.

## Próximos passos (fora do escopo desta POC)

- Adicionar um checkpointer do LangGraph para persistir o histórico entre
  execuções (hoje ele só vive na memória do processo).
- MCP: conectar a um servidor MCP externo como fonte de tools.
- Trocar `ChatOllama` por `ChatOpenAI`/`ChatAnthropic` se você tiver uma API
  key de um provedor pago — o resto do grafo não muda.
