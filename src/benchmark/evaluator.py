"""
Бенчмаркинг — отслеживание производительности агентов и ML-метрик.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


class AgentBenchmark:
    """Трекер производительности агентов и общего пайплайна."""

    def __init__(self):
        self.agent_metrics: dict[str, list[dict]] = {}
        self.pipeline_start: float | None = None
        self.pipeline_end: float | None = None

    def start_pipeline(self):
        """Запускает таймер всего пайплайна."""
        self.pipeline_start = time.time()
        logger.info("Пайплайн запущен.")

    def end_pipeline(self):
        """Останавливает таймер всего пайплайна."""
        self.pipeline_end = time.time()
        elapsed = self.pipeline_end - (self.pipeline_start or self.pipeline_end)
        logger.info("Пайплайн завершён за %.2f сек.", elapsed)

    def track_agent(self, agent_name: str, start_time: float, end_time: float, metadata: dict | None = None):
        """
        Записывает метрики выполнения агента.

        Args:
            agent_name: Имя агента.
            start_time: Время начала.
            end_time: Время завершения.
            metadata: Дополнительные метаданные.
        """
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = []

        record = {
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": end_time - start_time,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }
        self.agent_metrics[agent_name].append(record)
        logger.info("Агент '%s' выполнен за %.2f сек.", agent_name, record["duration_sec"])

    def generate_report(self) -> str:
        """
        Генерирует markdown-отчёт о производительности.

        Returns:
            Текст отчёта.
        """
        parts = [
            "# Benchmark Report",
            f"\nGenerated: {datetime.now().isoformat()}",
            "",
        ]

        # Общее время пайплайна
        if self.pipeline_start and self.pipeline_end:
            total = self.pipeline_end - self.pipeline_start
            parts.append(f"## Pipeline Total Time: {total:.2f}s\n")

        # Метрики по агентам
        parts.append("## Agent Performance\n")
        parts.append("| Agent | Calls | Total Time (s) | Avg Time (s) | Max Time (s) |")
        parts.append("|-------|-------|----------------|--------------|--------------|")

        for agent_name, records in self.agent_metrics.items():
            total_time = sum(r["duration_sec"] for r in records)
            avg_time = total_time / len(records) if records else 0
            max_time = max(r["duration_sec"] for r in records) if records else 0
            parts.append(
                f"| {agent_name} | {len(records)} | {total_time:.2f} | {avg_time:.2f} | {max_time:.2f} |"
            )

        # ML метрики из experiment store
        from src.memory.experiment_store import load_history
        history = load_history()

        if history:
            parts.append("\n## Model Performance History\n")
            parts.append("| Iteration | Model | RMSE | MAE | R² |")
            parts.append("|-----------|-------|------|-----|----|")
            for exp in history:
                m = exp.get("metrics", {})
                parts.append(
                    f"| {exp['iteration']} | {exp['model_type']} | "
                    f"{m.get('rmse', 'N/A'):.4f} | {m.get('mae', 'N/A'):.4f} | "
                    f"{m.get('r2', 'N/A'):.4f} |"
                )

        report = "\n".join(parts)

        # Сохраняем отчёт
        report_path = settings.output_dir / "benchmark_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        logger.info("Benchmark отчёт сохранён: %s", report_path)

        return report


# Глобальный экземпляр бенчмарка
benchmark = AgentBenchmark()
