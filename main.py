"""
Точка входа мультиагентной системы для автоматического решения задачи регрессии.

Использование:
    python main.py                          # Запуск с дефолтными настройками (Ollama)
    python main.py --provider huggingface   # Запуск с HuggingFace
    python main.py --max-iterations 3       # Ограничить итерации
"""

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.benchmark.evaluator import benchmark
from src.graph import build_graph


def setup_logging():
    """Настройка логирования в консоль и файл."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.output_dir / "run.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Multi-agent ML system for regression task",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "huggingface"],
        default=None,
        help="LLM provider (default: from .env or 'ollama')",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum improvement iterations (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model name",
    )
    return parser.parse_args()


def main():
    """Основная функция запуска пайплайна."""
    args = parse_args()

    if args.provider:
        settings.llm_provider = args.provider
    if args.max_iterations:
        settings.max_iterations = args.max_iterations
    if args.model:
        if settings.llm_provider == "ollama":
            settings.ollama_model = args.model
        else:
            settings.hf_model = args.model

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ЗАПУСК МУЛЬТИАГЕНТНОЙ СИСТЕМЫ")
    logger.info("=" * 60)
    logger.info("LLM провайдер: %s", settings.llm_provider)
    logger.info("Модель: %s", settings.ollama_model if settings.llm_provider == "ollama" else settings.hf_model)
    logger.info("Макс. итераций: %d", settings.max_iterations)
    logger.info("Данные: %s", settings.raw_data_dir)

    benchmark.start_pipeline()

    try:
        graph = build_graph()

        initial_state = {
            "iteration": 0,
            "max_iterations": settings.max_iterations,
            "is_satisfactory": False,
            "experiment_history": [],
        }

        logger.info("Запуск LangGraph пайплайна...")
        final_state = graph.invoke(initial_state)

        benchmark.end_pipeline()

        logger.info("=" * 60)
        logger.info("ПАЙПЛАЙН ЗАВЕРШЁН")
        logger.info("=" * 60)
        logger.info("Итераций выполнено: %d", final_state.get("iteration", 0))
        logger.info("Submission: %s", final_state.get("submission_path", "не создан"))

        if final_state.get("submission_result"):
            logger.info("\n%s", final_state["submission_result"])

        report = benchmark.generate_report()
        logger.info("\n%s", report)

    except KeyboardInterrupt:
        logger.warning("Прервано пользователем.")
        benchmark.end_pipeline()
        benchmark.generate_report()
        sys.exit(1)

    except Exception as e:
        logger.error("Критическая ошибка: %s", str(e), exc_info=True)
        benchmark.end_pipeline()
        benchmark.generate_report()
        sys.exit(1)


if __name__ == "__main__":
    main()
