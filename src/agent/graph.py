"""Montagem do grafo: nós, arestas (incluindo decisão e loop) e compile.

Fluxo:
START -> retrieval -> chatbot -> (tools_condition) -> tools -> chatbot -> ... -> human_approval -> END
- Retrieval (RAG): busca, uma vez por turno, os trechos de data/*.md mais
  relevantes para a pergunta do usuário e guarda em state["context"].
- Decisão: tools_condition olha a última resposta do chatbot. Se ela contém
  tool_calls, roteia para "tools" (loop de ferramenta); caso contrário
  (resposta final pronta), roteia para "human_approval".
- Human-in-the-loop: human_approval pausa o grafo via interrupt() antes de
  qualquer resposta final ser mostrada ao usuário, e só libera com uma
  decisão humana explícita (ver app.py). Esse gate fica na resposta final,
  não na decisão de chamar uma ferramenta, porque o LLM local usado aqui
  (qwen2.5:1.5b) mostrou ser pouco confiável decidindo quando emitir
  tool_calls (ver "Coisas não óbvias" no CLAUDE.md) — gatear a resposta
  final dispara sempre, em todo turno, sem depender dessa heurística.
- Loop: tools -> chatbot processa o resultado da ferramenta e gera a
  resposta final (ou pede outra ferramenta). Esse loop não passa de novo
  por "retrieval" — o mesmo contexto vale para o turno inteiro.
"""

from typing import Optional

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from .configuration import checkpointer, llm
from .retrieval import retrieve
from .state import State
from .tools import tools

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
    "Para saudações, conversas gerais ou qualquer pergunta que não seja "
    "uma conta ou uma consulta de saldo, responda diretamente em texto, "
    "sem chamar nenhuma ferramenta."
)


def retrieval_node(state: State) -> dict:
    query = state["messages"][-1].content
    return {"context": retrieve(query)}


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


def human_approval(state: State) -> dict:
    last_message = state["messages"][-1]

    # Pausa o grafo aqui (checkpoint salvo via SqliteSaver) até alguém
    # retomar com graph.invoke(Command(resume=...), config=config). O
    # valor passado a interrupt() é o que app.py lê para mostrar a
    # resposta pendente; o valor devolvido por interrupt() é a decisão
    # humana enviada no resume. Dispara em todo turno — não depende de
    # tool_calls, só de o chatbot ter produzido uma resposta final.
    decision = interrupt({"answer": last_message.content})

    if decision == "aprovar":
        # Não muda o state: a resposta original já está lá para app.py
        # imprimir.
        return {}

    return {"messages": [AIMessage("Resposta recusada pelo usuário — não enviada.")]}


graph_builder = StateGraph(State)
graph_builder.add_node("retrieval", retrieval_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("human_approval", human_approval)
# ToolNode é um nó pronto do LangGraph: olha a última mensagem, executa
# a(s) tool_call(s) pedida(s) pelo LLM e devolve o resultado como mensagens
# do tipo "tool".
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "retrieval")
graph_builder.add_edge("retrieval", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot", tools_condition, {"tools": "tools", END: "human_approval"}
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human_approval", END)

# compile() transforma a definição em um grafo executável. Passar
# checkpointer aqui é o que faz o State (mensagens + contexto) ser
# persistido a cada passo do grafo, associado a um thread_id — sem isso,
# graph.invoke() continua funcionando, mas só guarda o State em memória.
graph = graph_builder.compile(checkpointer=checkpointer)


def get_pending_approval(config: dict) -> Optional[dict]:
    """Payload passado a interrupt() se o grafo estiver pausado em
    human_approval, ou None se o turno já terminou normalmente.

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
    """Retoma o grafo pausado em human_approval com a decisão humana.

    decision é o que interrupt() devolve dentro do nó — hoje só
    "aprovar" é tratado como aprovação (ver human_approval); qualquer
    outro valor é tratado como recusa.
    """
    return graph.invoke(Command(resume=decision), config=config)
