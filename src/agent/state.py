"""Estado do grafo: o que passa de nó em nó."""

from typing import Annotated, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


# "messages" acumula o histórico da conversa. O reducer add_messages faz
# com que cada retorno do nó seja *anexado* à lista, em vez de substituí-la.
class State(TypedDict):
    messages: Annotated[list, add_messages]
    # Trechos recuperados do RAG para a pergunta mais recente. Sem reducer
    # (sem Annotated): cada retorno do nó "retrieval" substitui a lista
    # inteira, em vez de acumular entre turnos.
    context: list[str]
    # Decisão humana ("aprovar"/"recusar") sobre a tool_call pendente,
    # lida por route_after_approval logo após approval_local/approval_mcp
    # (agent/graph.py) para decidir se a tool é executada ou não. Sem
    # reducer: sobrescrita a cada aprovação, não acumula entre turnos.
    approval_decision: str
    # Assinatura (nome + args) da última tool_call vista por
    # request_tool_approval e quantas vezes seguidas ela se repetiu neste
    # turno — usado pra interromper um loop em que o chatbot insiste em
    # chamar a mesma tool com os mesmos args em vez de responder (ver
    # request_tool_approval em agent/graph.py). Sem reducer: resetados a
    # cada turno pelo nó "retrieval".
    last_tool_call_signature: Optional[str]
    tool_call_repeat_count: int
