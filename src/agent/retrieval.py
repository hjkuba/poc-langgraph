"""Retrieval do RAG: indexa documentos de data/ e busca trechos relevantes.

Fluxo de indexação (roda uma vez, na importação deste módulo):
    arquivos .md em data/ -> quebrados em chunks -> embedding de cada chunk
    -> guardados no vector_store (em memória)

Fluxo de busca (a cada chamada de retrieve()):
    pergunta do usuário -> embedding da pergunta -> compara com os vetores
    do vector_store -> devolve os k chunks mais similares
"""

from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownTextSplitter

from .configuration import embeddings

# src/agent/retrieval.py -> src/agent -> src -> raiz do projeto -> data/
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# similarity_search "puro" sempre devolve os k trechos mais próximos, mesmo
# quando nada é de fato relevante (ex: pergunta sobre o Nepal ainda traz
# trechos do Fluxarion, só que com similaridade baixa) — e isso confundia o
# chatbot, fazendo-o achar que só podia responder com base no contexto.
# Medindo a similaridade de perguntas relevantes (~0.65-0.77) vs.
# irrelevantes (~0.53-0.62) para os documentos em data/, esse corte separa
# bem os dois grupos.
SIMILARITY_THRESHOLD = 0.63


def _load_chunks() -> list[str]:
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks: list[str] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        chunks.extend(splitter.split_text(path.read_text()))
    return chunks


# InMemoryVectorStore: implementação simples de vector store (sem banco
# externo) — adequada para o tamanho desta POC. Os embeddings não
# persistem em disco; são recalculados toda vez que o processo inicia.
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_texts(_load_chunks())


def retrieve(query: str, k: int = 3) -> list[str]:
    """Retorna até k trechos relevantes para query (vazio se nada passar
    do threshold de similaridade)."""
    results = vector_store.similarity_search_with_score(query, k=k)
    return [doc.page_content for doc, score in results if score >= SIMILARITY_THRESHOLD]
