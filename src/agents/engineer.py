"""
Engineer — агент для обучения моделей и feature engineering.
Использует LLM для выбора стратегии и инструменты для обучения.
"""

import json
import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.benchmark.evaluator import benchmark
from src.guardrails import (
    detect_prompt_injection,
    rate_limiter,
    validate_llm_json_output,
)
from src.llm.provider import get_llm
from src.memory.experiment_store import get_history_summary, save_experiment
from src.rag.retriever import retrieve_knowledge
from src.tools.ml_tools import train_model

logger = logging.getLogger(__name__)

ENGINEER_SYSTEM_PROMPT = """Ты — Engineer, агент для обучения ML-моделей.

Твоя задача:
1. На основе EDA-отчёта и обратной связи от Critic выбрать модель и стратегию.
2. Определить гиперпараметры для обучения.
3. Сформулировать решение в строгом JSON-формате.

Доступные модели: ridge, random_forest, gradient_boosting, lightgbm, xgboost

Ты ДОЛЖЕН ответить ТОЛЬКО в следующем JSON-формате (без markdown, без ```):
{
    "model_type": "название_модели",
    "hyperparams": {"param1": value1, "param2": value2},
    "reasoning": "объяснение выбора"
}

Примеры гиперпараметров:
- lightgbm: {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 7, "num_leaves": 50}
- xgboost: {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6}
- random_forest: {"n_estimators": 300, "max_depth": 15}
- ridge: {"alpha": 1.0}
- gradient_boosting: {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5}

Если это первая итерация — начни с lightgbm с дефолтными параметрами.
Если есть обратная связь от Critic — учти её при выборе."""


_LLM_RESPONSE_DEFAULTS = {
    "model_type": "lightgbm",
    "hyperparams": {"n_estimators": 500, "learning_rate": 0.05},
    "reasoning": "Фоллбэк: LLM не вернул валидный JSON",
}


def _parse_llm_response(response_text: str) -> dict:
    """
    Извлекает JSON из ответа LLM с валидацией через guardrails.

    Args:
        response_text: Текст ответа LLM.

    Returns:
        Распарсенный словарь с model_type и hyperparams.
    """
    return validate_llm_json_output(
        response_text,
        required_keys=["model_type", "hyperparams"],
        defaults=_LLM_RESPONSE_DEFAULTS,
    )


def engineer_node(state: dict) -> dict:
    """
    Обучает модель на основе рекомендаций Explorer и Critic.

    Args:
        state: Текущее состояние графа.

    Returns:
        Состояние с результатами обучения.
    """
    start_time = time.time()
    iteration = state.get("iteration", 0) + 1
    logger.info("=" * 60)
    logger.info("ENGINEER: Итерация %d — обучение модели", iteration)
    logger.info("=" * 60)

    rag_context = retrieve_knowledge("тюнинг гиперпараметров регрессия lightgbm xgboost")

    experiment_history = get_history_summary()

    _, eda_report_safe = detect_prompt_injection(state.get('eda_report', 'Нет EDA-отчёта'))
    _, critic_feedback_safe = detect_prompt_injection(
        state.get('critic_feedback', 'Нет обратной связи (первая итерация)')
    )
    _, experiment_history_safe = detect_prompt_injection(experiment_history)

    llm_prompt = f"""Итерация: {iteration} из {state.get('max_iterations', 5)}

=== EDA-отчёт ===
{eda_report_safe}

=== Обратная связь от Critic ===
{critic_feedback_safe}

=== История экспериментов ===
{experiment_history_safe}

=== Знания из базы ===
{rag_context}

Выбери модель и гиперпараметры для этой итерации. Ответь ТОЛЬКО в JSON-формате."""

    llm = get_llm()
    messages = [
        SystemMessage(content=ENGINEER_SYSTEM_PROMPT),
        HumanMessage(content=llm_prompt),
    ]

    rate_limiter.wait_if_needed()
    response = llm.invoke(messages)
    decision = _parse_llm_response(response.content)

    model_type = decision.get("model_type", "lightgbm")
    hyperparams = decision.get("hyperparams", {})
    reasoning = decision.get("reasoning", "")

    logger.info("Решение LLM: модель=%s, параметры=%s", model_type, hyperparams)
    logger.info("Обоснование: %s", reasoning)

    logger.info("Обучение модели %s...", model_type)
    train_result = train_model.invoke({
        "model_type": model_type,
        "hyperparams": json.dumps(hyperparams),
    })

    logger.info("Результат обучения:\n%s", train_result)

    metrics = _extract_metrics(train_result)

    save_experiment(
        iteration=iteration,
        model_type=model_type,
        hyperparams=hyperparams,
        metrics=metrics,
        features_info="auto-prepared features",
        notes=reasoning,
    )

    benchmark.track_agent("engineer", start_time, time.time(), {"model": model_type})

    return {
        **state,
        "iteration": iteration,
        "model_results": train_result,
        "current_model_type": model_type,
        "current_hyperparams": json.dumps(hyperparams),
        "current_metrics": metrics,
    }


def _extract_metrics(train_result: str) -> dict:
    """
    Извлекает метрики из текстового результата обучения.

    Args:
        train_result: Текст результата от train_model.

    Returns:
        Словарь с метриками {rmse, mae, r2}.
    """
    metrics = {}
    for line in train_result.split("\n"):
        line = line.strip()
        if "RMSE:" in line:
            try:
                metrics["rmse"] = float(line.split("RMSE:")[-1].strip())
            except ValueError:
                pass
        elif "MAE:" in line:
            try:
                metrics["mae"] = float(line.split("MAE:")[-1].strip())
            except ValueError:
                pass
        elif "R²:" in line or "R2:" in line:
            try:
                val = line.split(":")[-1].strip()
                metrics["r2"] = float(val)
            except ValueError:
                pass
    return metrics
