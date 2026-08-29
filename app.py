"""
POC mínima: LangChain + LangGraph.

O que este exemplo mostra:
- LangChain entra como a camada que representa o LLM (ChatOllama), com uma
  interface padronizada de "invoke" sobre mensagens.
- LangGraph entra como o orquestrador: define um "State" (o que passa de nó
  em nó), um grafo com um único nó (o chatbot) e as arestas START -> chatbot
  -> END.

Pré-requisitos:
1. Ollama instalado e rodando (https://ollama.com)
2. Um modelo baixado, ex:  ollama pull llama3.2
3. Dependências instaladas: pip install -r requirements.txt

Uso:
    python app.py
"""

from typing import Annotated

from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Nome do modelo local que o Ollama deve servir.
# Troque para o modelo que você baixou (ex: "llama3.2", "qwen2.5:7b", "mistral").
MODEL_NAME = "llama3.2:1b"


# 1. Estado do grafo -----------------------------------------------------
# "messages" acumula o histórico da conversa. O reducer add_messages faz
# com que cada retorno do nó seja *anexado* à lista, em vez de substituí-la.
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Nó do grafo ----------------------------------------------------------
# Um nó é só uma função: recebe o estado atual e devolve um "patch" do
# estado. Aqui, o nó chama o LLM (via LangChain) com o histórico completo
# e devolve a resposta como nova mensagem.
llm = ChatOllama(model=MODEL_NAME, temperature=0)


def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 3. Construção do grafo ---------------------------------------------------
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# compile() transforma a definição em um grafo executável.
graph = graph_builder.compile()


def main() -> None:
    print("POC LangChain + LangGraph. Digite 'sair' para encerrar.\n")

    # O estado local mantém o histórico entre turnos do loop.
    state: State = {"messages": []}

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in {"sair", "exit", "quit"}:
            print("Até mais!")
            break
        if not user_input:
            continue

        state["messages"].append(("user", user_input))

        # Cada invoke roda o grafo do START ao END uma vez.
        state = graph.invoke(state)

        print(f"Bot: {state['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
