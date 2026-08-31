"""
POC mínima: LangChain + LangGraph.

O que este exemplo mostra:
- LangChain entra como a camada que representa o LLM (ChatOllama), com uma
  interface padronizada de "invoke" sobre mensagens.
- LangGraph entra como o orquestrador: define um "State" (agent/state.py) e
  um grafo com decisão e loop entre chatbot e tools (agent/graph.py).
- Tool calling: o LLM pode decidir chamar uma ferramenta local
  (agent/tools.py) ou via um servidor MCP (agent/mcp_tools.py,
  agent/mcp_server.py) em vez de responder direto — o grafo executa as
  duas fontes em nós separados (tools_local/tools_mcp).
- Human-in-the-loop (agent/graph.py): antes de EXECUTAR uma tool_call (local
  ou MCP) pedida pelo LLM, o grafo pausa via interrupt() e pede aprovação
  humana no terminal (s/n) — só continua com uma decisão explícita. Turnos
  sem tool_call não passam por aprovação nenhuma.
- RAG (agent/retrieval.py): a cada turno, busca no vector store (indexado a
  partir de data/*.md) os trechos mais relevantes para a pergunta e injeta
  esse contexto na chamada ao LLM.
- Persistência (agent/configuration.py): um checkpointer sqlite grava o
  State (mensagens + contexto) a cada passo do grafo, associado a um
  thread_id — o histórico sobrevive ao fim do processo.

Estrutura (padrão src layout):
- src/agent/state.py         -> definição do State do grafo
- src/agent/tools.py         -> ferramentas locais disponíveis para o LLM
- src/agent/mcp_server.py    -> servidor MCP mínimo (stdio)
- src/agent/mcp_tools.py     -> client MCP + ponte síncrona pras tools remotas
- src/agent/retrieval.py     -> indexação e busca do RAG (vector store)
- src/agent/configuration.py -> LLM, embeddings e checkpointer (persistência)
- src/agent/graph.py         -> nós, arestas (decisão + loop) e compile()
- app.py                     -> este arquivo: só o loop de conversa no terminal

Pré-requisitos:
1. Ollama instalado e rodando (https://ollama.com)
2. Um modelo baixado, ex:  ollama pull qwen3.5:2b
3. Pacote instalado em modo editável: pip install -e .
4. Dependências instaladas: pip install -r requirements.txt

Uso:
    python app.py
"""

from agent.graph import get_pending_approval, graph, resume_approval

# LangGraph identifica cada conversa persistida (checkpointer, ver
# agent/configuration.py) por um thread_id. Este app é single-sessão — um
# terminal, uma conversa por vez — então um valor fixo já garante que
# `python app.py` sempre recupera o histórico salvo na execução anterior.
# Para várias conversas em paralelo, o thread_id passaria a vir de fora
# (ex.: argumento de linha de comando).
THREAD_ID = "cli"


def main() -> None:
    print("POC LangChain + LangGraph. Digite 'sair' para encerrar.\n")

    config = {"configurable": {"thread_id": THREAD_ID}}

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in {"sair", "exit", "quit"}:
            print("Até mais!")
            break
        if not user_input:
            continue

        # Quantas mensagens já existiam antes deste turno (histórico salvo
        # pelo checkpointer, desta sessão ou de uma execução anterior do
        # processo) — usado abaixo para isolar só o que este turno gerou.
        messages_before = len(graph.get_state(config).values.get("messages", []))

        # Não montamos mais o histórico à mão: passamos só a mensagem nova.
        # Antes de rodar o grafo, o checkpointer carrega o State salvo para
        # esse thread_id e o reducer add_messages (agent/state.py) anexa a
        # mensagem nova a esse histórico.
        state = graph.invoke({"messages": [("user", user_input)]}, config=config)

        # Se "approval_local"/"approval_mcp" chamou interrupt(), o grafo
        # parou nesse ponto em vez de seguir direto pra tool — só acontece
        # nos turnos em que o LLM decidiu chamar uma ferramenta. `while`,
        # não `if`: o chatbot pode encadear mais de uma tool_call no mesmo
        # turno (ex.: chama de novo após o resultado da primeira) — cada
        # uma pausa de novo em approval_local/approval_mcp, então é preciso
        # aprovar de novo a cada pausa até o grafo realmente terminar o
        # turno. Só checar uma vez deixava a última tool_call pendente sem
        # nunca ser resolvida (resposta final vazia e log de tooling
        # repetido, contando a mesma tool_call ainda não executada).
        pending = get_pending_approval(config)
        while pending:
            # pending["tools"]: uma AIMessage pode pedir mais de uma tool
            # de uma vez — aprovar/recusar vale pra todas juntas, então
            # mostramos todas antes de perguntar (senão a(s) que não
            # aparecesse(m) aqui seria(m) executada(s) sem nunca ter sido
            # mostrada(s) pro usuário).
            for call in pending["tools"]:
                print(f"[aprovação necessária para chamar '{call['tool']}']")
                print(f"  args: {call['args']}")
            resposta = input("Aprovar? (s/n): ").strip().lower()
            decisao = "aprovar" if resposta in {"s", "sim"} else "recusar"

            state = resume_approval(decisao, config)
            pending = get_pending_approval(config)

        # Mensagens novas geradas nesse turno: se alguma AIMessage tiver
        # tool_calls, o LLM decidiu usar uma ferramenta antes da resposta
        # final (pode acontecer mais de uma vez no mesmo turno, daí olhar
        # todas as mensagens novas, não só a penúltima).
        for message in state["messages"][messages_before:]:
            for call in getattr(message, "tool_calls", None) or []:
                print(f"[tooling: {call['name']}({call['args']})]")

        print(f"Bot: {state['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
