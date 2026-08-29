"""Montagem do grafo: nós, arestas (incluindo decisão e loop) e compile.

Fluxo: START -> chatbot -> (decisão) -> tools -> chatbot -> ... -> END
- Decisão: tools_condition olha a última resposta do chatbot. Se ela contém
  tool_calls, roteia para "tools"; caso contrário, roteia para END.
- Loop: depois de executar a ferramenta, volta para o chatbot processar o
  resultado e gerar a resposta final (ou pedir outra ferramenta).
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .configuration import llm
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
    "Para saudações, conversas gerais ou qualquer pergunta que não seja "
    "uma conta, responda diretamente em texto, sem chamar nenhuma "
    "ferramenta."
)


def chatbot(state: State) -> dict:
    response = llm.invoke([SYSTEM_PROMPT, *state["messages"]])
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
# ToolNode é um nó pronto do LangGraph: olha a última mensagem, executa
# a(s) tool_call(s) pedida(s) pelo LLM e devolve o resultado como mensagens
# do tipo "tool".
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

# compile() transforma a definição em um grafo executável.
graph = graph_builder.compile()
