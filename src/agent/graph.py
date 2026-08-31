"""Montagem do grafo: nós, arestas (incluindo decisão e loop) e compile.

Fluxo:
START -> retrieval -> chatbot -> (route_tools)
  -> sem tool_calls -> END (resposta final direto, sem aprovação)
  -> tool_call local -> approval_local -> (route_after_approval)
       -> aprovar        -> tools_local -> chatbot -> ...
       -> recusar         -> tool_refused -> chatbot -> ...
       -> repetiu demais  -> tool_loop_exceeded -> END
  -> tool_call mcp   -> approval_mcp -> (route_after_approval)
       -> aprovar        -> tools_mcp -> chatbot -> ...
       -> recusar         -> tool_refused -> chatbot -> ...
       -> repetiu demais  -> tool_loop_exceeded -> END

- Retrieval (RAG): busca, uma vez por turno, os trechos de data/*.md mais
  relevantes para a pergunta do usuário e guarda em state["context"] —
  também reseta o contador de repetição de tool_call (ver abaixo), já que
  ele vale só para o turno atual.
- Decisão de fonte: route_tools olha a última resposta do chatbot. Sem
  tool_calls, a resposta final já está pronta -> END, sem gate nenhum. Com
  tool_calls, olha o nome da tool chamada para escolher entre
  "approval_local" (tools.py) e "approval_mcp" (mcp_tools.py) — duas fontes
  de ferramenta separadas de propósito para fins didáticos, embora
  LangGraph não precise dessa distinção (um único ToolNode com todas as
  tools também funcionaria).
- Human-in-the-loop: approval_local/approval_mcp pausam o grafo via
  interrupt() antes de EXECUTAR a(s) tool_call(s) pedida(s) pelo LLM (não
  antes da resposta final — essa é a implementação original do projeto,
  revertida de propósito; ver "Coisas não óbvias" no CLAUDE.md sobre por
  que ela foi trocada). Consequência aceita conscientemente: como o LLM
  local é pouco confiável decidindo quando emitir tool_calls, esse gate só
  dispara nos turnos em que ele realmente decide chamar uma ferramenta —
  não em todo turno. Aprovar/recusar vale para a AIMessage inteira (todas
  as tool_calls dela), não só a primeira — sem isso, uma segunda tool_call
  bundlada na mesma mensagem seria executada sem nunca ter sido mostrada.
- Recusa: route_after_approval olha state["approval_decision"] (escrito
  por approval_local/approval_mcp) e desvia para "tool_refused", que
  injeta uma ToolMessage por tool_call avisando a recusa (mesmo
  tool_call_id de cada pedido) para o histórico continuar válido, e a
  resposta final some sem a tool.
- Loop excedido: request_tool_approval compara a tool_call atual com a
  anterior (mesmo nome + args); se repetir mais que MAX_TOOL_CALL_REPEATS
  vezes seguidas, pula o interrupt() e desvia direto para
  "tool_loop_exceeded", que devolve uma resposta fixa e encerra o turno
  (sem voltar pro chatbot — evita repetir o mesmo loop de novo). Existe
  porque o LLM local, com histórico de conversa mais longo, pode ficar
  chamando a mesma tool repetidamente em vez de responder com o resultado
  que já tem.
- Loop principal: tools_local/tools_mcp/tool_refused -> chatbot processa o
  resultado (ou a recusa) e gera a resposta final (ou pede outra
  ferramenta). Esse loop não passa de novo por "retrieval" — o mesmo
  contexto vale para o turno inteiro.
"""

from typing import Optional

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

# Quantas vezes seguidas a mesma tool_call (nome + args idênticos) pode se
# repetir num turno antes de desistirmos de pedir aprovação de novo — ver
# request_tool_approval. Valor pequeno de propósito: é sinal de loop, não
# de uso legítimo (uma sequência legítima chamaria tools *diferentes*).
MAX_TOOL_CALL_REPEATS = 3

from .configuration import checkpointer, llm
from .mcp_tools import mcp_tools
from .retrieval import retrieve
from .state import State
from .tools import tools

# Nomes das tools locais, pra route_tools distinguir as duas fontes olhando
# só o nome da tool chamada pelo LLM (mesma informação que já vem em
# tool_calls, sem precisar marcar a origem em nenhum outro lugar).
_LOCAL_TOOL_NAMES = {t.name for t in tools}

# Sem essa instrução, modelos locais pequenos tendem a chamar ferramentas
# até em perguntas triviais (ex: inventar uma soma para responder "oi").
# Nota: mesmo com essa instrução, o LLM decide de forma heurística quando
# usar a ferramenta — em conversas com histórico longo, ele às vezes
# resolve a conta "de cabeça" em vez de chamá-la. Reforçar ainda mais o
# texto do prompt ("nunca calcule mentalmente") piorou esse comportamento
# em vez de corrigi-lo, então mantemos a versão mais simples.
SYSTEM_PROMPT = SystemMessage(
    "Você é um assistente útil que responde em português.\n"
    "Use as ferramentas 'somar' e 'multiplicar' SOMENTE quando o usuário "
    "pedir uma conta matemática explícita, usando os números que ele "
    "mesmo informou na pergunta.\n"
    "Use a ferramenta 'consultar_saldo' quando o usuário perguntar pelo "
    "saldo de uma conta ('corrente' ou 'poupança').\n"
    "Use a ferramenta 'obter_previsao_tempo' quando o usuário perguntar "
    "pela previsão do tempo de uma cidade.\n"
    "Para saudações, conversas gerais ou qualquer pergunta que não seja "
    "uma conta, uma consulta de saldo ou uma previsão do tempo, responda "
    "diretamente em texto, sem chamar nenhuma ferramenta."
)


def retrieval_node(state: State) -> dict:
    query = state["messages"][-1].content
    return {
        "context": retrieve(query),
        "last_tool_call_signature": None,
        "tool_call_repeat_count": 0,
    }


def _context_message(context: list[str]) -> SystemMessage:
    context_text = "\n\n---\n\n".join(context)
    return SystemMessage(
        "Trechos recuperados de documentos internos (podem ou não ser "
        "relevantes para a pergunta — use-os só se forem; caso contrário, "
        f"ignore-os e responda normalmente):\n\n{context_text}"
    )


def chatbot(state: State) -> dict:
    messages = [SYSTEM_PROMPT]
    if context := state.get("context"):
        messages.append(_context_message(context))
    messages.extend(state["messages"])

    response = llm.invoke(messages)
    return {"messages": [response]}


def route_tools(state: State) -> str:
    """Decide para onde ir depois do chatbot: sem tool_calls, resposta
    final pronta -> END, sem gate. Com tool_calls, olha o nome da tool
    chamada para escolher entre "approval_local" (tools.py) e
    "approval_mcp" (mcp_tools.py) — assume que todas as tool_calls de uma
    mesma mensagem vêm da mesma fonte (só olha a primeira).
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return END
    if tool_calls[0]["name"] in _LOCAL_TOOL_NAMES:
        return "approval_local"
    return "approval_mcp"


def _tool_calls_signature(calls: list[dict]) -> str:
    return str([(call["name"], call["args"]) for call in calls])


def request_tool_approval(state: State) -> dict:
    """Pausa o grafo (approval_local e approval_mcp compartilham esta
    mesma função — a lógica de pedir aprovação não depende da fonte da
    tool, só o destino em caso de aprovação difere, e isso é decidido nas
    arestas condicionais abaixo, não aqui).

    Pausa via interrupt() antes de EXECUTAR a(s) tool_call(s) pedida(s)
    pelo LLM — todas as tool_calls da AIMessage, não só a primeira, senão
    uma segunda tool_call bundlada na mesma mensagem seria executada pelo
    ToolNode sem nunca ter sido mostrada pra aprovação. O checkpoint é
    salvo pelo SqliteSaver; quem chama o grafo (app.py) retoma com
    resume_approval(). O valor passado a interrupt() é o que app.py lê
    para mostrar o(s) pedido(s) pendente(s); o valor devolvido por
    interrupt() é a decisão humana enviada no resume, guardada em
    approval_decision para route_after_approval ler a seguir.

    Antes de pausar, compara a tool_call atual com a última vista: se for
    idêntica (mesmo nome + args) mais que MAX_TOOL_CALL_REPEATS vezes
    seguidas, é sinal de loop (o LLM repetindo a mesma chamada em vez de
    responder com o resultado que já tem) — nesse caso nem chega a pausar,
    já marca approval_decision="loop_excedido" pra route_after_approval
    desviar pro fallback.
    """
    calls = state["messages"][-1].tool_calls
    signature = _tool_calls_signature(calls)
    if signature == state.get("last_tool_call_signature"):
        repeat_count = state.get("tool_call_repeat_count", 0) + 1
    else:
        repeat_count = 1
    update = {"last_tool_call_signature": signature, "tool_call_repeat_count": repeat_count}

    if repeat_count > MAX_TOOL_CALL_REPEATS:
        return {**update, "approval_decision": "loop_excedido"}

    decision = interrupt(
        {"tools": [{"tool": call["name"], "args": call["args"]} for call in calls]}
    )
    return {**update, "approval_decision": decision}


def route_after_approval(state: State) -> str:
    decision = state.get("approval_decision")
    if decision == "aprovar":
        return "run"
    if decision == "loop_excedido":
        return "give_up"
    return "refused"


def tool_refused(state: State) -> dict:
    """Registra a recusa como ToolMessage por tool_call (mesmo
    tool_call_id de cada pedido) em vez de simplesmente pular — sem isso,
    o histórico ficaria com uma AIMessage pedindo tool(s) sem resposta
    correspondente, o que alguns modelos tratam como entrada inválida. O
    chatbot processa essas ToolMessages no próximo turno do loop e
    responde levando a recusa em conta.
    """
    calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content="Execução recusada pelo usuário — ferramenta não chamada.",
                tool_call_id=call["id"],
            )
            for call in calls
        ]
    }


def tool_loop_exceeded(state: State) -> dict:
    """Encerra o turno com uma resposta fixa quando o LLM insiste em
    repetir a mesma tool_call (ver request_tool_approval/
    MAX_TOOL_CALL_REPEATS) — não volta pro chatbot de propósito, diferente
    de tool_refused: se voltasse, nada impediria o LLM de só pedir a mesma
    tool de novo e cair no mesmo loop.
    """
    return {
        "messages": [
            AIMessage(
                "Não consegui concluir a resposta — fiquei chamando a mesma "
                "ferramenta repetidamente sem chegar a uma resposta final. "
                "Tente reformular a pergunta."
            )
        ]
    }


graph_builder = StateGraph(State)
graph_builder.add_node("retrieval", retrieval_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("approval_local", request_tool_approval)
graph_builder.add_node("approval_mcp", request_tool_approval)
graph_builder.add_node("tool_refused", tool_refused)
graph_builder.add_node("tool_loop_exceeded", tool_loop_exceeded)
# ToolNode é um nó pronto do LangGraph: olha a última mensagem, executa
# a(s) tool_call(s) pedida(s) pelo LLM e devolve o resultado como mensagens
# do tipo "tool". Dois nós — um por fonte de tool — só pra deixar essa
# fronteira visível no grafo; LangGraph roda ambos do mesmo jeito.
graph_builder.add_node("tools_local", ToolNode(tools))
graph_builder.add_node("tools_mcp", ToolNode(mcp_tools))

graph_builder.add_edge(START, "retrieval")
graph_builder.add_edge("retrieval", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"approval_local": "approval_local", "approval_mcp": "approval_mcp", END: END},
)
graph_builder.add_conditional_edges(
    "approval_local",
    route_after_approval,
    {"run": "tools_local", "refused": "tool_refused", "give_up": "tool_loop_exceeded"},
)
graph_builder.add_conditional_edges(
    "approval_mcp",
    route_after_approval,
    {"run": "tools_mcp", "refused": "tool_refused", "give_up": "tool_loop_exceeded"},
)
graph_builder.add_edge("tools_local", "chatbot")
graph_builder.add_edge("tools_mcp", "chatbot")
graph_builder.add_edge("tool_refused", "chatbot")
graph_builder.add_edge("tool_loop_exceeded", END)

# compile() transforma a definição em um grafo executável. Passar
# checkpointer aqui é o que faz o State (mensagens + contexto) ser
# persistido a cada passo do grafo, associado a um thread_id — sem isso,
# graph.invoke() continua funcionando, mas só guarda o State em memória.
graph = graph_builder.compile(checkpointer=checkpointer)


def get_pending_approval(config: dict) -> Optional[dict]:
    """Payload passado a interrupt() se o grafo estiver pausado em
    approval_local/approval_mcp (aguardando aprovação para executar uma
    tool), ou None se o turno já terminou normalmente.

    Encapsula a leitura de graph.get_state(config).tasks[0].interrupts[0] —
    esse caminho é detalhe de implementação do LangGraph (indexado por
    posição em tuplas), enquanto o restante do State (state["messages"]) é
    o contrato estável do grafo. Isolar aqui evita que quem chama o grafo
    (app.py) precise conhecer essa estrutura interna.
    """
    snapshot = graph.get_state(config)
    if not snapshot.next:
        return None
    return snapshot.tasks[0].interrupts[0].value


def resume_approval(decision: str, config: dict) -> dict:
    """Retoma o grafo pausado em approval_local/approval_mcp com a decisão
    humana sobre executar ou não a tool pedida.

    decision é o que interrupt() devolve dentro do nó — hoje só
    "aprovar" é tratado como aprovação (ver route_after_approval); qualquer
    outro valor é tratado como recusa.
    """
    return graph.invoke(Command(resume=decision), config=config)
