"""Ferramentas vindas de um servidor MCP — contrapartida de tools.py para a
fonte remota.

Diferente de tools.py (funções Python locais via @tool), aqui as tools são
carregadas de um processo externo (mcp_server.py) falando o protocolo MCP.
MultiServerMCPClient.get_tools() é assíncrono; como configuration.py monta
tudo de forma síncrona na importação do módulo, usamos asyncio.run() aqui
pra buscar os metadados (nome/descrição/schema) das tools uma única vez.

As tools que o client devolve só têm implementação assíncrona (protocolo
MCP é async-nativo) — ToolNode (agent/graph.py), porém, executa cada tool
de forma síncrona (`tool.invoke()`), o que quebraria com
"StructuredTool does not support sync invocation". Por isso cada tool é
reembrulhada aqui numa StructuredTool com `func` síncrono, que abre uma
sessão nova com o servidor MCP a cada chamada via asyncio.run() — mesmo
padrão que já usamos para carregar os metadados, só que por chamada em vez
de uma vez só. ToolNode roda tools síncronas numa thread separada (não a
thread principal), então esse asyncio.run() não conflita com nenhum loop
de eventos já em execução.
"""

import asyncio
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

_MCP_SERVER_PATH = Path(__file__).resolve().parent / "mcp_server.py"

_client = MultiServerMCPClient(
    {
        "previsao_tempo": {
            "transport": "stdio",
            "command": "python",
            "args": [str(_MCP_SERVER_PATH)],
        }
    }
)


def _as_sync_tool(async_tool: BaseTool) -> StructuredTool:
    def call(**kwargs):
        return asyncio.run(async_tool.ainvoke(kwargs))

    return StructuredTool(
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
        func=call,
    )


mcp_tools = [_as_sync_tool(t) for t in asyncio.run(_client.get_tools())]
