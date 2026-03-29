"""
LangGraph StateGraph — определение графа мультиагентного пайплайна.

Поток:
  START → coordinator → explorer → engineer → critic → [submit | engineer]
                                                              ↓
                                                             END
"""

import json
import logging
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.coordinator import coordinator_node
from src.agents.critic import critic_node
from src.agents.engineer import engineer_node
from src.agents.explorer import explorer_node
from src.benchmark.evaluator import benchmark
from src.memory.experiment_store import get_best_experiment
from src.tools.ml_tools import predict_and_submit

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    """Состояние пайплайна, передаваемое между агентами."""
    task_description: str
    data_path: str
    train_file: str
    test_file: str
    target_column: str

    eda_report: str
    model_results: str
    critic_feedback: str

    current_model_type: str
    current_hyperparams: str
    current_metrics: dict

    iteration: int
    max_iterations: int
    is_satisfactory: bool

    submission_path: str
    experiment_history: list


def route_after_critic(state: PipelineState) -> str:
    """
    Маршрутизация после Critic: отправить submission или улучшить модель.

    Args:
        state: Текущее состояние.

    Returns:
        "submit" или "engineer".
    """
    if state.get("is_satisfactory", False):
        logger.info("Маршрут: → SUBMIT (результат удовлетворительный)")
        return "submit"
    else:
        logger.info("Маршрут: → ENGINEER (нужно улучшение, итерация %d)", state.get("iteration", 0))
        return "engineer"


def submit_node(state: PipelineState) -> PipelineState:
    """
    Финальный узел — создаёт submission файл с лучшей моделью.

    Args:
        state: Текущее состояние.

    Returns:
        Состояние с путём к submission файлу.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("SUBMIT: Создание финального submission")
    logger.info("=" * 60)

    best = get_best_experiment()
    if best:
        model_type = best["model_type"]
        hyperparams = json.dumps(best["hyperparams"])
        logger.info("Используем лучшую модель: %s (RMSE=%.4f)", model_type, best["metrics"]["rmse"])
    else:
        model_type = state.get("current_model_type", "lightgbm")
        hyperparams = state.get("current_hyperparams", "{}")
        logger.info("Используем текущую модель: %s", model_type)

    result = predict_and_submit.invoke({
        "model_type": model_type,
        "hyperparams": hyperparams,
    })

    logger.info("Submission результат:\n%s", result)

    benchmark.track_agent("submit", start_time, time.time())

    return {
        **state,
        "submission_path": "outputs/submission.csv",
        "submission_result": result,
    }


def build_graph() -> StateGraph:
    """
    Строит и компилирует LangGraph пайплайн.

    Returns:
        Скомпилированный граф.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("coordinator", coordinator_node)
    graph.add_node("explorer", explorer_node)
    graph.add_node("engineer", engineer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("submit", submit_node)

    graph.add_edge(START, "coordinator")
    graph.add_edge("coordinator", "explorer")
    graph.add_edge("explorer", "engineer")
    graph.add_edge("engineer", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"submit": "submit", "engineer": "engineer"},
    )

    graph.add_edge("submit", END)

    logger.info("Граф пайплайна построен: coordinator → explorer → engineer ⇄ critic → submit")

    return graph.compile()
