"""
Retriever для RAG — поиск релевантных чанков из базы знаний.
"""

import logging

import chromadb
from chromadb.utils import embedding_functions

from config.settings import settings
from src.rag.indexer import index_knowledge_base

logger = logging.getLogger(__name__)

_collection_cache: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    """Получает или создаёт коллекцию ChromaDB."""
    global _collection_cache
    if _collection_cache is None:
        _collection_cache = index_knowledge_base()
    return _collection_cache


def retrieve_knowledge(query: str, top_k: int | None = None) -> str:
    """
    Ищет релевантные чанки в базе знаний по запросу.

    Args:
        query: Текстовый запрос для поиска.
        top_k: Количество результатов (по умолчанию из настроек).

    Returns:
        Объединённый текст найденных чанков с указанием источников.
    """
    top_k = top_k or settings.rag_top_k
    collection = get_collection()

    if collection.count() == 0:
        logger.warning("База знаний пуста. Возвращаем пустой результат.")
        return "База знаний пуста. Используй свои знания для решения задачи."

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    parts = ["=== Релевантные знания из базы ===\n"]
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
        source = meta.get("source", "неизвестно")
        parts.append(f"--- Источник {i}: {source} ---\n{chunk}\n")

    return "\n".join(parts)
