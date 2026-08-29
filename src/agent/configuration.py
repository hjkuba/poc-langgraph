"""Configuração do LLM, do modelo de embeddings e do checkpointer (persistência)."""

import sqlite3
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver

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

# src/agent/configuration.py -> src/agent -> src -> raiz do projeto -> .sqlite/
# Pasta própria (por fora do controle de versão, como .venv/) porque o
# sqlite em modo WAL grava mais de um arquivo por banco (o .sqlite principal
# e os auxiliares -wal/-shm) — mais simples ignorar a pasta inteira no git
# do que cada sufixo.
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / ".sqlite"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Arquivo sqlite onde o checkpointer do LangGraph grava o State (mensagens +
# contexto) a cada passo do grafo. Diferente do InMemoryVectorStore do RAG,
# aqui o objetivo é justamente sobreviver ao fim do processo: numa próxima
# execução, o mesmo thread_id (ver graph.py) recupera o histórico salvo.
CHECKPOINT_DB = CHECKPOINT_DIR / "checkpoints.sqlite"

# check_same_thread=False: a conexão sqlite3 por padrão só pode ser usada na
# thread que a criou; o próprio SqliteSaver é quem sincroniza o acesso, então
# essa checagem do sqlite3 só atrapalharia (é a exceção documentada pelo
# LangGraph para esse uso).
_checkpoint_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)

# bind_tools() ensina o LLM a emitir "tool_calls" em vez de texto quando
# achar que uma ferramenta resolve melhor o pedido do usuário.
llm = ChatOllama(model=MODEL_NAME, temperature=0).bind_tools(tools)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

checkpointer = SqliteSaver(_checkpoint_conn)
