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
ollama pull qwen3.5:2b          # chat/tool calling
ollama pull nomic-embed-text    # embeddings (RAG)
```

Setup do ambiente Python (3.10+ — exigido pelo SDK `mcp`, ver "Coisas não
óbvias"):

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
START -> retrieval -> chatbot -> (route_tools)
  -> sem tool_calls  -> END
  -> tool_call local -> approval_local -> (aprovar: tools_local | recusar: tool_refused) -> chatbot -> ...
  -> tool_call mcp   -> approval_mcp   -> (aprovar: tools_mcp   | recusar: tool_refused) -> chatbot -> ...
```

- **`src/agent/state.py`** — `State` do grafo: `messages` (reducer
  `add_messages`, acumula histórico), `context` (sem reducer, sobrescrito a
  cada turno pelo nó `retrieval`) e `approval_decision` (sem reducer,
  escrito por `approval_local`/`approval_mcp` e lido por
  `route_after_approval`).
- **`src/agent/retrieval.py`** — RAG. Na importação do módulo, indexa
  `data/*.md` (chunks -> embeddings -> `InMemoryVectorStore`, recalculado a
  cada início de processo). `retrieve()` busca por similaridade e filtra por
  `SIMILARITY_THRESHOLD` — sem esse corte, perguntas sem relação com `data/`
  ainda traziam os chunks "mais próximos" e confundiam o LLM.
- **`src/agent/tools.py`** — ferramentas locais (`somar`, `multiplicar`,
  `consultar_saldo`) via decorator `@tool`; a docstring de cada função vira
  a descrição que o LLM usa para decidir quando chamá-la.
  `consultar_saldo` lê de um dict fixo (`_SALDOS`) que só existe ali — ao
  contrário de `somar`/`multiplicar`, o LLM não tem como responder sem
  chamar a tool.
- **`src/agent/mcp_server.py`** — servidor MCP mínimo (stdio, SDK `mcp`
  `FastMCP`), processo separado spawnado pelo client em `mcp_tools.py`.
  Expõe `obter_previsao_tempo` com dados mockados — domínio escolhido de
  propósito bem distinto das tools locais, pra o LLM não confundir as duas
  fontes.
- **`src/agent/mcp_tools.py`** — client MCP (`langchain-mcp-adapters`,
  `MultiServerMCPClient`); carrega as tools remotas uma vez na importação
  do módulo. Reembrulha cada tool (só assíncrona, por natureza do
  protocolo MCP) numa `StructuredTool` síncrona, porque o `ToolNode` do
  LangGraph invoca tools de forma síncrona — ver "Coisas não óbvias".
- **`src/agent/configuration.py`** — monta `llm` (`ChatOllama` +
  `bind_tools` com as tools locais e as do MCP juntas — o LLM não
  diferencia a fonte ao decidir o que chamar), `embeddings`
  (`OllamaEmbeddings`) e `checkpointer` (`SqliteSaver` sobre
  `.sqlite/checkpoints.sqlite`, git-ignorado).
- **`src/agent/graph.py`** — monta os nós (`retrieval`, `chatbot`,
  `approval_local`, `approval_mcp`, `tools_local`, `tools_mcp`,
  `tool_refused`) e arestas, injeta o `context` do RAG como `SystemMessage`
  extra antes do LLM quando existe, e faz `compile(checkpointer=checkpointer)`.
  `route_tools` decide entre `approval_local`/`approval_mcp`/`END` pelo
  nome da tool chamada (duas fontes de tool separadas em nós diferentes de
  propósito, por motivo didático — um único `ToolNode` com todas as tools
  funcionaria igual). `approval_local`/`approval_mcp` (mesma função,
  registrada duas vezes) pausam via `interrupt()` antes de EXECUTAR a tool
  pedida pelo LLM (não antes da resposta final — ver "Coisas não óbvias").
  `route_after_approval` lê `state["approval_decision"]` e decide entre
  rodar a tool ou desviar pra `tool_refused`, que injeta uma `ToolMessage`
  de recusa com o `tool_call_id` original. `get_pending_approval()` e
  `resume_approval()` (também aqui) encapsulam a leitura do
  `snapshot`/`Command` do LangGraph para quem chama o grafo (`app.py`) não
  precisar conhecer esses detalhes internos.
- **`app.py`** — só o loop de conversa no terminal; usa um `thread_id` fixo
  (`"cli"`, app é single-sessão) para o checkpointer recuperar o histórico
  entre execuções.

### Coisas não óbvias ao mexer nisso

- **Modelo do chat é `qwen3.5:2b`, não trocar sem medir de novo**:
  `llama3.2` (1b e 3b) já foi testado e ambos chamavam ferramentas até em
  saudações, ignorando o resultado na resposta final, mesmo com system
  prompt reforçado. Já usamos `qwen2.5:1.5b`, mas ao medir a decisão de
  tool calling (20 rodadas, thread nova por tentativa, math/saldo/clima/
  saudação) ele nunca chamou uma tool quando deveria (0/5 em cada caso que
  precisava de tool). `qwen3.5:0.8b` foi pior ainda (0/12, incluindo
  chamar `consultar_saldo` numa saudação e inventar respostas de clima/
  saldo sem chamar nada). `qwen3.5:2b` fechou 20/20 nos mesmos testes —
  se trocar de modelo de novo, repita essa bateria antes de assumir que
  está melhor.
- **Não reforçar demais o `SYSTEM_PROMPT`** (em `agent/graph.py`): tentar
  deixar mais explícito que o modelo não deve calcular de cabeça piorou o
  comportamento em vez de corrigir.
- O loop `tools -> chatbot` **não** volta a passar por `retrieval` — o
  contexto do RAG vale para o turno inteiro, buscado uma única vez.
- **A decisão de chamar ferramenta é inconsistente mesmo em
  `temperature=0`**: testamos a mesma pergunta ("quanto é 4 vezes 7?") em
  threads novas e ela só disparou `tool_calls` em parte das tentativas —
  isso vale tanto pra `somar`/`multiplicar` quanto pra `consultar_saldo`
  (testado com e sem parâmetro) e até pro exemplo canônico "que horas
  são?" com uma tool sem parâmetro nenhum. Não é algo que o formato da
  tool resolve; é ruído do próprio `qwen2.5:1.5b` — em vez de admitir que
  não sabe ou chamar a tool, ele às vezes inventa uma resposta plausível
  (ex.: uma hora falsa).
- **O gate de aprovação fica na execução da tool, não na resposta final —
  segunda versão dessa decisão**: a primeira versão (documentada aqui
  antes) fazia o oposto, pelo motivo justamente contrário — gatear a
  resposta final garantia que o fluxo de aprovação disparasse em todo
  turno, sem depender do LLM decidir chamar uma tool (a decisão de chamar
  era pouco confiável demais pra gatear nela, ver bullet acima). Voltamos
  a gatear a tool_call por pedido explícito, aceitando conscientemente que
  esse gate só dispara nos turnos em que o LLM realmente decide chamar
  algo. Com `qwen2.5:1.5b` isso era raro; com `qwen3.5:2b` a decisão ficou
  muito mais confiável (ver bullet do modelo), o que atenua bastante esse
  trade-off — mas não o elimina, é sempre heurística do LLM.
- **Tools MCP só têm implementação assíncrona, mas `ToolNode` as invoca de
  forma síncrona**: sem a ponte em `mcp_tools.py` (cada tool reembrulhada
  numa `StructuredTool` com `func` síncrono que chama `asyncio.run()` por
  invocação), a execução quebra com "StructuredTool does not support sync
  invocation". Chegamos a tentar rodar o grafo inteiro via `ainvoke()` pra
  evitar essa ponte, mas isso exige trocar o checkpointer por
  `AsyncSqliteSaver` — que fica atado ao event loop em que foi criado
  (`asyncio.get_running_loop()` no `__init__`) e não pode ser montado num
  `asyncio.run()` descartável de import, como o resto do projeto faz.
  Corrigir isso direito forçaria mover `compile(checkpointer=...)` pra
  dentro do loop de eventos do `app.py`, quebrando o padrão de `graph.py`
  expor um `graph` pronto — a ponte síncrona evita essa reestruturação.
- **MCP em Python exige 3.10+**: confirmado checando todas as versões já
  publicadas do SDK `mcp` (do qual `langchain-mcp-adapters` depende) —
  nenhuma delas roda em 3.9. O `.venv` deste projeto foi recriado com
  Python 3.12 por causa disso (era 3.9); o pré-requisito no README também
  foi atualizado.
- `InMemoryVectorStore` (RAG) não persiste em disco — reindexação completa
  toda vez que o processo sobe. Diferente disso, o checkpointer sqlite
  (histórico de conversa) sobrevive ao restart.
