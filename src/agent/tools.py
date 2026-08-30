"""Ferramentas que o LLM pode chamar via tool calling."""

from langchain_core.tools import tool


@tool
def somar(a: float, b: float) -> float:
    """Soma dois números."""
    return a + b


@tool
def multiplicar(a: float, b: float) -> float:
    """Multiplica dois números."""
    return a * b


# Saldo fictício por conta, existente só aqui — diferente de somar/
# multiplicar (que o LLM às vezes responde de cabeça sem chamar a
# ferramenta, ver CLAUDE.md), esse valor não está em nenhum lugar que o
# LLM possa "adivinhar", então a única forma de responder é chamando a
# tool. Bom pra testar de forma confiável o fluxo de aprovação humana
# (agent/graph.py, human_approval).
_SALDOS = {
    "corrente": 1250.75,
    "poupança": 4830.10,
}


@tool
def consultar_saldo(conta: str) -> float:
    """Consulta o saldo de uma conta bancária ('corrente' ou 'poupança')."""
    conta_normalizada = conta.strip().lower()
    if conta_normalizada not in _SALDOS:
        raise ValueError(
            f"Conta '{conta}' não encontrada. Contas disponíveis: "
            f"{', '.join(_SALDOS)}."
        )
    return _SALDOS[conta_normalizada]


tools = [somar, multiplicar, consultar_saldo]
