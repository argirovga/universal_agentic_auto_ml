"""
Critic — агент для оценки результатов и принятия решения об улучшении.
Использует LLM для анализа метрик и формулировки рекомендаций.
"""

import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from src.benchmark.evaluator import benchmark
from src.llm.provider import get_llm
from src.memory.experiment_store import get_best_experiment, get_history_summary

logger = logging.getLogger(__name__)

# Системный промпт для Critic
CRITIC_SYSTEM_PROMPT = """Ты — Critic, агент для оценки качества ML-моделей.

Твоя задача:
1. Оценить текущие результаты обучения.
2. Сравнить с предыдущими экспериментами.
3. Решить: достаточно ли хорош результат или нужна ещё итерация.
4. Если нужна итерация — дать КОНКРЕТНЫЕ рекомендации по улучшению.

Ты ДОЛЖЕН ответить в следующем формате:

РЕШЕНИЕ: SUBMIT или IMPROVE

ОЦЕНКА:
[оценка текущих результатов]

РЕКОМЕНДАЦИИ:
[если IMPROVE — конкретные шаги: сменить модель, изменить параметры, и т.д.]

Критерии для SUBMIT:
- R² > 0.5 — приемлемый результат
- Последние 2 итерации не дали улучшения > 1% по RMSE
- Достигнут лимит итераций

Критерии для IMPROVE:
- R² < 0.5 — нужно улучшить
- Есть явные возможности для улучшения
- Не все типы моделей опробованы"""


def critic_node(state: dict) -> dict:
    """
    Оценивает результаты обучения и решает, продолжать или завершать.

    Args:
        state: Текущее состояние графа.

    Returns:
        Состояние с решением (is_satisfactory) и обратной связью.
    """
    start_time = time.time()
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", settings.max_iterations)

    logger.info("=" * 60)
    logger.info("CRITIC: Оценка итерации %d/%d", iteration, max_iterations)
    logger.info("=" * 60)

    # Собираем контекст
    model_results = state.get("model_results", "Нет результатов")
    current_metrics = state.get("current_metrics", {})
    experiment_history = get_history_summary()
    best = get_best_experiment()

    # Проверка жёсткого лимита итераций
    if iteration >= max_iterations:
        logger.info("Достигнут лимит итераций (%d). Переходим к submission.", max_iterations)
        benchmark.track_agent("critic", start_time, time.time(), {"decision": "submit_max_iter"})
        return {
            **state,
            "is_satisfactory": True,
            "critic_feedback": f"Достигнут лимит итераций ({max_iterations}). Используем лучший результат.",
        }

    # Формируем запрос к LLM
    llm_prompt = f"""Итерация: {iteration} из {max_iterations}

=== Текущие результаты ===
{model_results}

=== Текущие метрики ===
RMSE: {current_metrics.get('rmse', 'N/A')}
MAE: {current_metrics.get('mae', 'N/A')}
R²: {current_metrics.get('r2', 'N/A')}

=== История экспериментов ===
{experiment_history}

=== Лучший результат ===
{f"Модель: {best['model_type']}, RMSE: {best['metrics']['rmse']:.4f}, R²: {best['metrics']['r2']:.4f}" if best else "Нет предыдущих результатов"}

Оцени результаты и прими решение: SUBMIT (отправить) или IMPROVE (улучшить)."""

    llm = get_llm()
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=llm_prompt),
    ]

    response = llm.invoke(messages)
    critic_response = response.content

    logger.info("Ответ Critic:\n%s", critic_response)

    # Парсим решение
    is_satisfactory = _parse_decision(critic_response, current_metrics)

    decision_str = "SUBMIT" if is_satisfactory else "IMPROVE"
    logger.info("Решение Critic: %s", decision_str)

    benchmark.track_agent("critic", start_time, time.time(), {"decision": decision_str})

    return {
        **state,
        "is_satisfactory": is_satisfactory,
        "critic_feedback": critic_response,
    }


def _parse_decision(critic_response: str, metrics: dict) -> bool:
    """
    Извлекает решение из ответа Critic.

    Args:
        critic_response: Текст ответа LLM.
        metrics: Текущие метрики.

    Returns:
        True если результат удовлетворительный (SUBMIT).
    """
    response_upper = critic_response.upper()

    # Ищем явное решение
    if "РЕШЕНИЕ: SUBMIT" in response_upper or "DECISION: SUBMIT" in response_upper:
        return True
    if "РЕШЕНИЕ: IMPROVE" in response_upper or "DECISION: IMPROVE" in response_upper:
        return False

    # Фоллбэк: проверяем метрики
    r2 = metrics.get("r2", 0)
    if r2 >= settings.r2_threshold:
        logger.info("Фоллбэк: R²=%.4f >= %.2f, считаем удовлетворительным.", r2, settings.r2_threshold)
        return True

    return False
