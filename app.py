"""
POC mínima: LangChain + LangGraph.

O que este exemplo mostra:
- LangChain entra como a camada que representa o LLM (ChatOllama), com uma
  interface padronizada de "invoke" sobre mensagens.
- LangGraph entra como o orquestrador: define um "State" (agent/state.py) e
  um grafo com decisão e loop entre chatbot e tools (agent/graph.py).
- Tool calling: o LLM pode decidir chamar uma ferramenta (agent/tools.py)
  em vez de responder direto.

Estrutura (padrão src layout):
- src/agent/state.py -> definição do State do grafo
- src/agent/tools.py -> ferramentas disponíveis para o LLM
- src/agent/graph.py -> LLM, nós, arestas (decisão + loop) e compile()
- app.py             -> este arquivo: só o loop de conversa no terminal

Pré-requisitos:
1. Ollama instalado e rodando (https://ollama.com)
2. Um modelo baixado, ex:  ollama pull qwen2.5:1.5b
3. Pacote instalado em modo editável: pip install -e .
4. Dependências instaladas: pip install -r requirements.txt

Uso:
    python app.py
"""

from agent.graph import graph
from agent.state import State


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
        messages_before = len(state["messages"])

        # Cada invoke roda o grafo do START ao END uma vez (podendo passar
        # por "tools" e voltar ao chatbot mais de uma vez internamente).
        state = graph.invoke(state)

        # Mensagens novas geradas nesse turno: se alguma AIMessage tiver
        # tool_calls, o LLM decidiu usar uma ferramenta antes da resposta final.
        for message in state["messages"][messages_before:]:
            for call in getattr(message, "tool_calls", None) or []:
                print(f"[tooling: {call['name']}({call['args']})]")

        print(f"Bot: {state['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
