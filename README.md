# POC: LangChain + LangGraph

Estrutura mínima para entender como LangChain e LangGraph se encaixam, com
tool calling e roteamento condicional.

- **LangGraph** define o grafo: um `State` (o que passa de nó em nó), dois
  nós (`chatbot` e `tools`) e as arestas `START -> chatbot -> (decisão) ->
  tools -> chatbot -> ... -> END`.
- **LangChain** entra dentro do nó `chatbot`, como a abstração padronizada
  de chamada ao modelo (`ChatOllama.invoke(...)`), com `bind_tools()` para
  permitir tool calling.

## Estrutura

```
src/agent/
├── state.py   # State do grafo
├── tools.py   # ferramentas disponíveis para o LLM
└── graph.py   # LLM, nós, arestas (decisão + loop) e compile()
app.py         # entrypoint: loop de conversa no terminal
```

## Pré-requisitos

1. [Ollama](https://ollama.com) instalado e rodando.
2. Um modelo baixado:
   ```bash
   ollama pull qwen2.5:1.5b
   ```
   (testamos `llama3.2`, 1b e 3b: mesmo com system prompt reforçado, os dois
   chamavam ferramentas quase sempre — até em saudações — e ignoravam o
   resultado na resposta final. `qwen2.5` tem fine-tuning de function
   calling mais criterioso e não repete esse padrão)
3. Python 3.9+.

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
chamar uma ferramenta (`src/agent/tools.py`) antes de responder.

## Próximos passos (fora do escopo desta POC)

- Adicionar um checkpointer do LangGraph para persistir o histórico entre
  execuções (hoje ele só vive na memória do processo).
- RAG: nó de retrieval (embeddings + vector store) antes do `chatbot`.
- MCP: conectar a um servidor MCP externo como fonte de tools.
- Trocar `ChatOllama` por `ChatOpenAI`/`ChatAnthropic` se você tiver uma API
  key de um provedor pago — o resto do grafo não muda.
