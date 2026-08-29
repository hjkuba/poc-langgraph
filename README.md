# POC: LangChain + LangGraph

Estrutura mínima para entender como LangChain e LangGraph se encaixam, com
tool calling, roteamento condicional e RAG.

- **LangGraph** define o grafo: um `State` (o que passa de nó em nó), três
  nós (`retrieval`, `chatbot` e `tools`) e as arestas `START -> retrieval ->
  chatbot -> (decisão) -> tools -> chatbot -> ... -> END`.
- **LangChain** entra dentro dos nós, como a abstração padronizada de
  chamada ao modelo (`ChatOllama.invoke(...)`, com `bind_tools()` para tool
  calling) e de embeddings (`OllamaEmbeddings`) para o RAG.
- Um **checkpointer** (`SqliteSaver`) persiste o `State` em disco, então o
  histórico da conversa sobrevive ao reinício do processo.

## Estrutura

```
data/
└── manual-fluxarion.md   # documento fictício usado como base do RAG
src/agent/
├── state.py           # State do grafo (messages + context)
├── tools.py            # ferramentas disponíveis para o LLM
├── configuration.py    # LLM, modelo de embeddings, checkpointer (sqlite)
├── retrieval.py         # RAG: indexação (data/*.md) e busca por similaridade
└── graph.py             # nós, arestas (decisão + loop) e compile()
app.py                   # entrypoint: loop de conversa no terminal
.sqlite/                 # gerado em runtime: histórico persistido (git-ignorado)
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
conversa é mantido pelo `State` do LangGraph (via `add_messages`) e
persistido em `.sqlite/checkpoints.sqlite` — feche o processo e rode
`python app.py` de novo que a conversa continua de onde parou (veja
"Persistência" abaixo). Se a pergunta envolver soma ou multiplicação, o LLM pode decidir
chamar uma ferramenta (`src/agent/tools.py`) antes de responder. Perguntas
relacionadas ao conteúdo de `data/manual-fluxarion.md` (ex: *"qual o limite
de requisições por minuto da API Fluxarion?"*) são respondidas via RAG —
tente perguntar algo que só está nesse documento pra ver o retrieval em
ação (o LLM não tem como adivinhar essas respostas sozinho).

## Tool calling e roteamento condicional: como funciona

1. `src/agent/tools.py` define as ferramentas (`somar`, `multiplicar`) com
   o decorator `@tool` do LangChain — ele lê a assinatura e a docstring de
   cada função e gera a descrição que o LLM recebe para decidir quando (e
   com quais argumentos) chamá-la.
2. `agent/configuration.py` registra as ferramentas no LLM via
   `llm.bind_tools(tools)`: a partir daí, cada resposta do modelo pode vir
   como texto normal ou como uma `AIMessage` com `tool_calls` (a ferramenta
   e os argumentos que o LLM decidiu usar, sem executá-la — quem executa é
   o grafo).
3. Depois que o nó `chatbot` roda, a aresta condicional
   `tools_condition` (pronta do LangGraph) olha só a última mensagem: se
   ela tiver `tool_calls`, roteia para o nó `tools`; senão, roteia para
   `END`.
4. O nó `tools` é um `ToolNode` (também pronto do LangGraph): executa a(s)
   função(ões) pedida(s) e devolve o resultado como mensagem(ns) do tipo
   `tool`, adicionadas ao histórico.
5. A aresta fixa `tools -> chatbot` manda o resultado de volta para o
   `chatbot` gerar a resposta final com base nele — podendo, inclusive,
   decidir chamar outra ferramenta antes de responder, o que faz esse
   trecho do grafo funcionar como um loop.
6. O `SYSTEM_PROMPT` em `agent/graph.py` existe porque, sem essa instrução,
   os modelos locais testados chamavam ferramenta até em saudações — o
   texto pede explicitamente para só usar `somar`/`multiplicar` diante de
   uma conta matemática explícita.

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

## Persistência: como funciona

1. `agent/configuration.py` abre uma conexão sqlite em
   `.sqlite/checkpoints.sqlite` (pasta criada automaticamente, fora do
   controle de versão) e monta um `SqliteSaver` — o checkpointer do
   LangGraph.
2. `agent/graph.py` compila o grafo com `checkpointer=checkpointer`. A
   partir daí, toda vez que o grafo passa por um nó, o `State` inteiro
   (mensagens + contexto) é salvo no sqlite, associado a um `thread_id`.
3. `app.py` usa um `thread_id` fixo (`"cli"`, já que este app é uma sessão
   só) e passa em `graph.invoke({"messages": [...]}, config=...)` apenas a
   mensagem nova — o checkpointer carrega o histórico salvo (desta execução
   ou de uma anterior) e o reducer `add_messages` (`agent/state.py`) anexa a
   mensagem nova a ele.
4. Suportar várias conversas em paralelo é só questão de usar um
   `thread_id` diferente por sessão (hoje fixo, porque o app é
   single-sessão).

## Próximos passos (fora do escopo desta POC)

- MCP: conectar a um servidor MCP externo como fonte de tools.
- Human-in-the-loop: usar `interrupt()`/`Command` do LangGraph para pausar
  o grafo antes de agir (ex.: confirmar com o usuário antes de chamar uma
  ferramenta) e só continuar com uma resposta humana explícita.
