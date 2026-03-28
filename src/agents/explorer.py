"""
Explorer — агент для разведочного анализа данных (EDA).
Использует LLM для анализа результатов профилирования и формулировки рекомендаций.
"""

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.benchmark.evaluator import benchmark
from src.llm.provider import get_llm
from src.rag.retriever import retrieve_knowledge
from src.tools.data_tools import get_correlations, get_data_profile, load_data

logger = logging.getLogger(__name__)

# Системный промпт для Explorer
EXPLORER_SYSTEM_PROMPT = """Ты — Explorer, агент для разведочного анализа данных (EDA).

Твоя задача:
1. Проанализировать предоставленные данные профилирования.
2. Выявить ключевые паттерны, проблемы и возможности.
3. Сформулировать конкретные рекомендации для Feature Engineering.

Формат ответа:
## Ключевые находки
- [список находок из EDA]

## Проблемы в данных
- [пропуски, выбросы, аномалии]

## Рекомендации по признакам
- [конкретные предложения по созданию/трансформации признаков]

## Рекомендация по модели
- [какую модель попробовать первой и почему]

Будь конкретным и практичным. Не пиши общих слов."""


def explorer_node(state: dict) -> dict:
    """
    Выполняет EDA: профилирует данные, анализирует корреляции,
    использует LLM для формулировки рекомендаций.

    Args:
        state: Текущее состояние графа.

    Returns:
        Состояние с заполненным eda_report.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("EXPLORER: Начинаю разведочный анализ данных")
    logger.info("=" * 60)

    # Шаг 1: Сбор информации через инструменты
    logger.info("Шаг 1: Загрузка и профилирование данных...")
    data_info = load_data.invoke({"file_type": "train"})
    profile = get_data_profile.invoke({"file_type": "train"})
    correlations = get_correlations.invoke({})

    # Шаг 2: Получаем знания из RAG
    logger.info("Шаг 2: Запрос к базе знаний...")
    rag_context = retrieve_knowledge(
        "feature engineering для данных Airbnb, обработка пропусков, регрессия табличных данных"
    )

    # Шаг 3: Анализ через LLM
    logger.info("Шаг 3: Анализ через LLM...")
    llm = get_llm()

    analysis_prompt = f"""Проанализируй результаты EDA и дай рекомендации.

=== Описание задачи ===
{state.get('task_description', 'Регрессия: предсказание target')}

=== Информация о данных ===
{data_info}

=== Профиль данных ===
{profile}

=== Корреляции ===
{correlations}

=== Знания из базы ===
{rag_context}

На основе этого анализа сформулируй подробный EDA-отчёт с конкретными рекомендациями."""

    messages = [
        SystemMessage(content=EXPLORER_SYSTEM_PROMPT),
        HumanMessage(content=analysis_prompt),
    ]

    response = llm.invoke(messages)
    eda_report = response.content

    logger.info("EDA-отчёт сформирован (%d символов).", len(eda_report))

    benchmark.track_agent("explorer", start_time, time.time())

    return {
        **state,
        "eda_report": eda_report,
    }
