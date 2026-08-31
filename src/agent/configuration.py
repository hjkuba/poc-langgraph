"""Configuração do LLM, do modelo de embeddings e do checkpointer (persistência)."""

import sqlite3
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver

from .mcp_tools import mcp_tools
from .tools import tools

# Nome do modelo local que o Ollama deve servir.
# Testamos llama3.2 (1b e 3b): mesmo com system prompt reforçado, ambos
# chamavam ferramentas em quase qualquer pergunta, até saudações, e
# ignoravam o resultado da ferramenta na resposta final.
#
# Já usamos qwen2.5:1.5b aqui, mas medimos (20 rodadas, thread nova por
# tentativa, mesmas 4 tools locais+MCP bindadas) e ele nunca chamou
# nenhuma tool quando deveria (math/saldo/clima: 0/5 cada) — só acertava
# em não chamar à toa numa saudação (5/5). Comparamos com qwen3.5:0.8b
# (pior ainda: 0/12 no total, incluindo chamar consultar_saldo numa
# saudação e inventar respostas de clima/saldo sem chamar a tool) e
# qwen3.5:2b, que fechou 20/20 (e outras 12/12 numa rodada anterior) nos
# mesmos testes — decisão de tool calling muito mais confiável nesse
# tamanho. Não chegamos a testar gemma3:1b porque um qwen já resolveu.
MODEL_NAME = "qwen3.5:2b"

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
# achar que uma ferramenta resolve melhor o pedido do usuário. O LLM precisa
# conhecer as tools locais e as vindas do MCP juntas para decidir qual
# chamar — a separação entre as duas fontes acontece só na execução, em
# agent/graph.py (nós "tools_local" e "tools_mcp").
llm = ChatOllama(model=MODEL_NAME, temperature=0).bind_tools([*tools, *mcp_tools])

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

checkpointer = SqliteSaver(_checkpoint_conn)
