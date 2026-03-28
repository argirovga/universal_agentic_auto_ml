"""
Индексатор документов для RAG — разбивает документы на чанки и сохраняет в ChromaDB.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from config.settings import settings

logger = logging.getLogger(__name__)


def index_knowledge_base(force_reindex: bool = False) -> chromadb.Collection:
    """
    Индексирует все .md файлы из knowledge_base/documents/ в ChromaDB.

    Args:
        force_reindex: Принудительная переиндексация.

    Returns:
        Коллекция ChromaDB с проиндексированными документами.
    """
    docs_dir = settings.knowledge_base_dir / "documents"
    chroma_dir = settings.chroma_db_dir

    # Инициализация ChromaDB
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Эмбеддинг-функция (sentence-transformers, локальная)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model,
    )

    # Проверяем, нужна ли переиндексация
    collection_name = "ml_knowledge_base"
    existing_collections = [c.name for c in client.list_collections()]

    if collection_name in existing_collections and not force_reindex:
        collection = client.get_collection(name=collection_name, embedding_function=ef)
        if collection.count() > 0:
            logger.info("База знаний уже проиндексирована (%d чанков). Пропускаем.", collection.count())
            return collection
        # Коллекция пуста — удаляем и создаём заново
        client.delete_collection(name=collection_name)

    if collection_name in existing_collections and force_reindex:
        client.delete_collection(name=collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"description": "База знаний ML best practices для агентов"},
    )

    # Читаем и разбиваем документы на чанки
    all_chunks = []
    all_metadatas = []
    all_ids = []

    md_files = list(docs_dir.glob("*.md"))
    if not md_files:
        logger.warning("Нет .md файлов в %s", docs_dir)
        return collection

    chunk_id = 0
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = _split_into_chunks(text, settings.rag_chunk_size, settings.rag_chunk_overlap)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": md_file.name, "file": str(md_file)})
            all_ids.append(f"chunk_{chunk_id}")
            chunk_id += 1

    # Добавляем в коллекцию
    if all_chunks:
        collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids,
        )
        logger.info(
            "Проиндексировано %d чанков из %d файлов.",
            len(all_chunks), len(md_files),
        )

    return collection


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Разбивает текст на чанки по разделам (##) или по размеру.

    Args:
        text: Текст документа.
        chunk_size: Максимальный размер чанка в символах.
        overlap: Перекрытие между чанками.

    Returns:
        Список текстовых чанков.
    """
    # Сначала пробуем разбить по разделам markdown
    sections = []
    current_section = []
    for line in text.split("\n"):
        if line.startswith("## ") and current_section:
            sections.append("\n".join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    if current_section:
        sections.append("\n".join(current_section))

    # Если секции слишком длинные, разбиваем дальше
    chunks = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section.strip())
        else:
            # Разбиваем длинную секцию
            words = section.split()
            current_chunk = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Оставляем overlap
                    overlap_words = current_chunk[-overlap // 10:] if overlap > 0 else []
                    current_chunk = overlap_words + [word]
                    current_len = sum(len(w) + 1 for w in current_chunk)
                else:
                    current_chunk.append(word)
                    current_len += len(word) + 1
            if current_chunk:
                chunks.append(" ".join(current_chunk))

    return [c for c in chunks if c.strip()]
