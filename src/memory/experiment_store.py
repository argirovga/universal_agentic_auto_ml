"""
Хранилище экспериментов — персистентная память для отслеживания итераций.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

EXPERIMENTS_FILE = settings.output_dir / "experiments.json"


def save_experiment(
    iteration: int,
    model_type: str,
    hyperparams: dict,
    metrics: dict,
    features_info: str = "",
    notes: str = "",
) -> dict:
    """
    Сохраняет результат эксперимента.

    Args:
        iteration: Номер итерации.
        model_type: Тип модели.
        hyperparams: Гиперпараметры модели.
        metrics: Метрики (rmse, mae, r2).
        features_info: Описание использованных признаков.
        notes: Заметки агента.

    Returns:
        Сохранённый эксперимент.
    """
    experiment = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "hyperparams": hyperparams,
        "metrics": metrics,
        "features_info": features_info,
        "notes": notes,
    }

    history = load_history()
    history.append(experiment)

    EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "Эксперимент #%d сохранён: %s, RMSE=%.4f, R²=%.4f",
        iteration, model_type,
        metrics.get("rmse", 0), metrics.get("r2", 0),
    )
    return experiment


def load_history() -> list[dict]:
    """
    Загружает историю всех экспериментов.

    Returns:
        Список экспериментов.
    """
    if not EXPERIMENTS_FILE.exists():
        return []
    try:
        with open(EXPERIMENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.warning("Не удалось загрузить историю экспериментов.")
        return []


def get_best_experiment() -> dict | None:
    """
    Возвращает лучший эксперимент по RMSE.

    Returns:
        Эксперимент с наименьшим RMSE или None.
    """
    history = load_history()
    if not history:
        return None

    valid = [e for e in history if "metrics" in e and "rmse" in e["metrics"]]
    if not valid:
        return None

    return min(valid, key=lambda e: e["metrics"]["rmse"])


def get_history_summary() -> str:
    """
    Формирует текстовое резюме истории экспериментов.

    Returns:
        Текстовое описание всех экспериментов.
    """
    history = load_history()
    if not history:
        return "История экспериментов пуста. Это первая итерация."

    parts = [f"=== История экспериментов ({len(history)} записей) ===\n"]
    for exp in history:
        metrics = exp.get("metrics", {})
        parts.append(
            f"Итерация {exp['iteration']}: {exp['model_type']} — "
            f"RMSE={metrics.get('rmse', 'N/A'):.4f}, "
            f"MAE={metrics.get('mae', 'N/A'):.4f}, "
            f"R²={metrics.get('r2', 'N/A'):.4f}"
        )
        if exp.get("notes"):
            parts.append(f"  Заметки: {exp['notes']}")

    best = get_best_experiment()
    if best:
        parts.append(
            f"\nЛучший результат: итерация {best['iteration']}, "
            f"{best['model_type']}, RMSE={best['metrics']['rmse']:.4f}"
        )

    return "\n".join(parts)
