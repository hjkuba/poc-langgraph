"""Configuração do LLM e do modelo de embeddings usados pelo agente."""

from langchain_ollama import ChatOllama, OllamaEmbeddings

from .tools import tools

# Nome do modelo local que o Ollama deve servir.
# Testamos llama3.2 (1b e 3b): mesmo com system prompt reforçado, ambos
# chamavam ferramentas em quase qualquer pergunta, até saudações, e
# ignoravam o resultado da ferramenta na resposta final. qwen2.5 tem
# fine-tuning de function calling mais criterioso e não repete esse padrão.
MODEL_NAME = "qwen2.5:1.5b"

# Modelo separado, especializado só em gerar embeddings (vetores) — usado
# tanto para indexar os documentos em data/ quanto para vetorizar a
# pergunta do usuário na busca por similaridade (RAG).
EMBEDDING_MODEL_NAME = "nomic-embed-text"

# bind_tools() ensina o LLM a emitir "tool_calls" em vez de texto quando
# achar que uma ferramenta resolve melhor o pedido do usuário.
llm = ChatOllama(model=MODEL_NAME, temperature=0).bind_tools(tools)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
