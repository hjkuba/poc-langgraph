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


tools = [somar, multiplicar]
