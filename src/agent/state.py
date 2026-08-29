"""Estado do grafo: o que passa de nó em nó."""

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


# "messages" acumula o histórico da conversa. O reducer add_messages faz
# com que cada retorno do nó seja *anexado* à lista, em vez de substituí-la.
class State(TypedDict):
    messages: Annotated[list, add_messages]
