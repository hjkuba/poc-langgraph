# POC: LangChain + LangGraph

Estrutura mínima para entender como LangChain e LangGraph se encaixam.

- **LangGraph** define o grafo: um `State` (o que passa de nó em nó), um nó
  (`chatbot`) e as arestas `START -> chatbot -> END`.
- **LangChain** entra dentro do nó, como a abstração padronizada de chamada
  ao modelo (`ChatOllama.invoke(...)`).

## Pré-requisitos

1. [Ollama](https://ollama.com) instalado e rodando.
2. Um modelo baixado:
   ```bash
   ollama pull llama3.2
   ```
3. Python 3.10+.

## Rodando

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Digite mensagens no terminal; digite `sair` para encerrar. O histórico da
conversa é mantido pelo `State` do LangGraph (via `add_messages`) enquanto o
loop roda.

## Próximos passos (fora do escopo desta POC)

- Trocar o nó único por um roteamento condicional (`add_conditional_edges`)
  para decidir entre responder direto ou chamar uma ferramenta.
- Adicionar um checkpointer do LangGraph para persistir o histórico entre
  execuções (hoje ele só vive na memória do processo).
- Trocar `ChatOllama` por `ChatOpenAI`/`ChatAnthropic` se você tiver uma API
  key de um provedor pago — o resto do grafo não muda.
