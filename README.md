# POC: LangChain + LangGraph

Estrutura mínima para entender como LangChain e LangGraph se encaixam, com
tool calling, roteamento condicional e RAG.

Não conhece Python? Veja [GUIA_PYTHON.md](GUIA_PYTHON.md) — explica a
sintaxe usada neste projeto com paralelo em TypeScript.

- **LangGraph** define o grafo: um `State` (o que passa de nó em nó) e os
  nós `retrieval`, `chatbot`, `approval_local`/`approval_mcp`,
  `tools_local`/`tools_mcp` e `tool_refused` — arestas condicionais
  decidem a rota a cada turno. A aprovação humana só entra quando o LLM
  decide chamar uma ferramenta (local ou via MCP), não em toda resposta.
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
├── state.py           # State do grafo (messages + context + approval_decision)
├── tools.py            # ferramentas locais disponíveis para o LLM
├── mcp_server.py         # servidor MCP mínimo (stdio), usado por mcp_tools.py
├── mcp_tools.py            # client MCP + ponte síncrona pras tools remotas
├── configuration.py         # LLM, modelo de embeddings, checkpointer (sqlite)
├── retrieval.py               # RAG: indexação (data/*.md) e busca por similaridade
└── graph.py                    # nós, arestas (decisão + loop) e compile()
app.py                   # entrypoint: loop de conversa no terminal
.sqlite/                 # gerado em runtime: histórico persistido (git-ignorado)
```

## Pré-requisitos

1. [Ollama](https://ollama.com) instalado e rodando.
2. O modelo de chat:
   ```bash
   ollama pull qwen3.5:2b
   ```
   (testamos `llama3.2`, 1b e 3b: mesmo com system prompt reforçado, os dois
   chamavam ferramentas quase sempre — até em saudações — e ignoravam o
   resultado na resposta final. Já usamos `qwen2.5:1.5b`, mas medimos a
   decisão de tool calling em 20 rodadas e ele nunca chamou uma tool quando
   deveria (0/5 em math, saldo e clima); `qwen3.5:0.8b` foi ainda pior
   (0/12, incluindo chamar `consultar_saldo` numa saudação); `qwen3.5:2b`
   fechou 20/20 nos mesmos testes, então virou o padrão)
3. O modelo de embeddings, usado pelo RAG:
   ```bash
   ollama pull nomic-embed-text
   ```
4. Python 3.10+ (exigido pelo SDK `mcp`, usado na integração MCP —
   nenhuma versão dele roda em 3.9).

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
"Persistência" abaixo). Se a pergunta envolver soma, multiplicação, saldo
ou previsão do tempo, o LLM pode decidir chamar uma ferramenta — local
(`src/agent/tools.py`) ou via MCP (`src/agent/mcp_tools.py`, veja "MCP"
abaixo) — e, antes de executá-la, o terminal pede uma aprovação explícita
(`s`/`n` — veja "Human-in-the-loop" abaixo); turnos sem chamada de
ferramenta não passam por aprovação nenhuma. Perguntas relacionadas ao
conteúdo de `data/manual-fluxarion.md` (ex:
*"qual o limite de requisições por minuto da API Fluxarion?"*) são
respondidas via RAG — tente perguntar algo que só está nesse documento pra
ver o retrieval em ação (o LLM não tem como adivinhar essas respostas
sozinho).

## Tool calling e roteamento condicional: como funciona

1. `src/agent/tools.py` (ferramentas locais) e `src/agent/mcp_tools.py`
   (ferramentas via MCP — veja a seção "MCP" abaixo) definem as
   ferramentas disponíveis para o LLM, cada uma com nome, descrição e
   schema de argumentos que o LLM usa para decidir quando (e com quais
   argumentos) chamá-la.
2. `agent/configuration.py` registra as duas listas juntas no LLM via
   `llm.bind_tools([*tools, *mcp_tools])`: a partir daí, cada resposta do
   modelo pode vir como texto normal ou como uma `AIMessage` com
   `tool_calls` (a ferramenta e os argumentos que o LLM decidiu usar, sem
   executá-la — quem executa é o grafo). O LLM não diferencia a origem das
   ferramentas ao decidir o que chamar.
3. Depois que o nó `chatbot` roda, a aresta condicional `route_tools`
   (`agent/graph.py`) olha só a última mensagem: sem `tool_calls`, a
   resposta final já está pronta e vai direto pro `END`, sem aprovação
   nenhuma. Com `tool_calls`, olha o nome da tool chamada para escolher
   entre `approval_local` e `approval_mcp` (veja "Human-in-the-loop"
   abaixo) — duas fontes de ferramenta em nós separados de propósito, por
   motivo didático (um único `ToolNode` com todas as tools funcionaria
   igual).
4. Aprovada a execução, `tools_local`/`tools_mcp` (dois `ToolNode`, um por
   fonte) executam a função pedida e devolvem o resultado como mensagem do
   tipo `tool`, adicionada ao histórico.
5. A aresta fixa `tools_local`/`tools_mcp -> chatbot` manda o resultado de
   volta para o `chatbot` gerar a resposta final com base nele — podendo,
   inclusive, decidir chamar outra ferramenta antes de responder, o que
   faz esse trecho do grafo funcionar como um loop.
6. O `SYSTEM_PROMPT` em `agent/graph.py` existe porque, sem essa instrução,
   os modelos locais testados chamavam ferramenta até em saudações — o
   texto pede explicitamente para só usar `somar`/`multiplicar` diante de
   uma conta matemática explícita, `consultar_saldo` diante de uma
   pergunta de saldo, e `obter_previsao_tempo` diante de uma pergunta de
   previsão do tempo.

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

## MCP: como funciona

1. `src/agent/mcp_server.py` é um servidor MCP mínimo: um processo Python
   separado que fala o protocolo MCP via stdio (SDK oficial `mcp`,
   `FastMCP`). Expõe uma única tool, `obter_previsao_tempo(cidade)`, com
   dados mockados (sem chamada de API externa) — domínio escolhido de
   propósito bem diferente das tools locais (`somar`/`multiplicar`/
   `consultar_saldo`), para o LLM não confundir as duas fontes.
2. `src/agent/mcp_tools.py` é o client: usa `MultiServerMCPClient`
   (`langchain-mcp-adapters`) para spawnar `mcp_server.py` como subprocesso
   e listar suas tools uma vez, na importação do módulo.
3. As tools que o MCP devolve só têm implementação assíncrona (o protocolo
   é async-nativo), mas o `ToolNode` do LangGraph as executa de forma
   síncrona. Por isso `mcp_tools.py` reembrulha cada uma numa
   `StructuredTool` com `func` síncrono, que abre uma sessão nova com o
   servidor a cada chamada via `asyncio.run()` — funciona sem conflitar com
   nenhum loop de eventos porque o `ToolNode` já roda tools síncronas numa
   thread separada, não na principal.
4. A separação entre fontes só acontece na execução, no grafo
   (`agent/graph.py`): dois `ToolNode`, `tools_local` e `tools_mcp`,
   escolhidos por `route_tools` a partir do nome da tool chamada (veja
   "Tool calling e roteamento condicional" acima).
5. Requer Python 3.10+: o SDK `mcp` (do qual `langchain-mcp-adapters`
   depende) não publica nenhuma versão compatível com 3.9.

## Human-in-the-loop: como funciona

1. `approval_local` e `approval_mcp` (`agent/graph.py`, a mesma função
   `request_tool_approval` registrada duas vezes) ficam entre `chatbot` e a
   execução da tool — a aresta condicional `route_tools` manda para um
   desses dois nós sempre que a última mensagem do `chatbot` tem
   `tool_calls`. Sem `tool_calls`, o turno vai direto para o `END`, sem
   aprovação nenhuma.
2. Dentro do nó, `interrupt({"tool": ..., "args": ...})` (do LangGraph)
   pausa o grafo naquele ponto exato — antes de EXECUTAR a tool — e salva
   um checkpoint via o mesmo `SqliteSaver` já usado pela persistência; a
   execução só continua quando alguém retomar explicitamente.
3. `app.py` detecta a pausa com o helper `get_pending_approval(config)`
   (`agent/graph.py`) e mostra a tool + argumentos pendentes no terminal,
   pedindo `s`/`n`. A resposta volta via `resume_approval(decisao,
   config)`, que chama `graph.invoke(Command(resume=decisao), ...)` — o
   valor de `resume` vira o retorno da chamada `interrupt()` dentro do nó,
   guardado em `state["approval_decision"]`.
4. `route_after_approval` lê essa decisão: `"aprovar"` segue para
   `tools_local`/`tools_mcp`, que executa a tool normalmente; qualquer
   outro valor segue para `tool_refused`, que injeta uma `ToolMessage`
   ("Execução recusada pelo usuário — ferramenta não chamada.") com o
   mesmo `tool_call_id` do pedido — sem essa mensagem, o histórico ficaria
   com uma tool_call sem resposta correspondente.
5. Nos dois casos (tool executada ou recusada), o fluxo volta para o
   `chatbot`, que gera a resposta final considerando o resultado (ou a
   recusa) — essa resposta final não passa por nenhuma aprovação.
6. `get_pending_approval`/`resume_approval` existem para isolar em
   `agent/graph.py` os detalhes internos do LangGraph
   (`snapshot.tasks[0].interrupts[0]`) — `app.py` só lida com um contrato
   simples: `pending` é `None` ou um dict com `tool`/`args`.
7. Esta é a segunda versão desse gate: a primeira gateava a resposta final
   (disparava em todo turno, sem depender do LLM decidir chamar uma tool).
   Esta gateia a execução da tool — só dispara quando o LLM decide chamar
   algo, trade-off aceito de propósito por motivo didático (veja "Coisas
   não óbvias" no `CLAUDE.md`).

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
