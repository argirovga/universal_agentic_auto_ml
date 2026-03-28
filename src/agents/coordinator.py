"""
Координатор — инициализирует состояние пайплайна и описывает задачу.
"""

import logging
import time

from config.settings import settings
from src.benchmark.evaluator import benchmark

logger = logging.getLogger(__name__)


def coordinator_node(state: dict) -> dict:
    """
    Точка входа пайплайна — инициализация состояния и описание задачи.

    Args:
        state: Текущее состояние графа.

    Returns:
        Обновлённое состояние с описанием задачи.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("КООРДИНАТОР: Инициализация пайплайна")
    logger.info("=" * 60)

    # Инициализируем состояние
    updated_state = {
        **state,
        "data_path": str(settings.raw_data_dir),
        "train_file": str(settings.train_file),
        "test_file": str(settings.test_file),
        "target_column": settings.target_column,
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", settings.max_iterations),
        "is_satisfactory": False,
        "experiment_history": state.get("experiment_history", []),
        "eda_report": "",
        "model_results": "",
        "critic_feedback": state.get("critic_feedback", ""),
        "submission_path": "",
        "task_description": (
            "Задача регрессии: предсказать количество дней доступности (target) "
            "для объявлений Airbnb в Нью-Йорке. "
            "Датасет содержит информацию о локации, типе жилья, цене, отзывах и хосте. "
            "Целевая переменная — непрерывная, диапазон 0-365. "
            "Метрика: RMSE. Формат submission: index, prediction."
        ),
    }

    logger.info("Задача: %s", updated_state["task_description"])
    logger.info("Данные: %s", updated_state["data_path"])
    logger.info("Максимум итераций: %d", updated_state["max_iterations"])

    benchmark.track_agent("coordinator", start_time, time.time())
    return updated_state
