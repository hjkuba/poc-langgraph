"""Servidor MCP mínimo, falado via stdio, usado por mcp_tools.py.

Processo separado (spawnado pelo client em mcp_tools.py) — diferente das
ferramentas locais em tools.py, que rodam como função Python direta no
mesmo processo do app. Domínio (clima) escolhido de propósito bem distinto
de somar/multiplicar/consultar_saldo, pra o LLM não confundir as duas
fontes de tool.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("previsao-tempo")

# Previsão fixa por cidade — mockada de propósito (sem chamada de API
# externa), mantendo a POC 100% local/offline como o resto do projeto.
_PREVISOES = {
    "fortaleza": "32°C, ensolarado",
    "são paulo": "19°C, nublado com chance de chuva",
    "porto alegre": "14°C, céu limpo",
}


@mcp.tool()
def obter_previsao_tempo(cidade: str) -> str:
    """Consulta a previsão do tempo atual de uma cidade."""
    cidade_normalizada = cidade.strip().lower()
    if cidade_normalizada not in _PREVISOES:
        return (
            f"Cidade '{cidade}' não encontrada. Cidades disponíveis: "
            f"{', '.join(_PREVISOES)}."
        )
    return _PREVISOES[cidade_normalizada]


if __name__ == "__main__":
    mcp.run(transport="stdio")
