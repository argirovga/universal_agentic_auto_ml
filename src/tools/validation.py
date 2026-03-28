"""
Guardrails и валидация — защита от некорректных входов и выходов.
"""

import functools
import logging
import signal
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Ошибка валидации данных."""
    pass


class TimeoutError(Exception):
    """Превышено время выполнения."""
    pass


def validate_dataframe(df: pd.DataFrame, expected_columns: list[str] | None = None) -> bool:
    """
    Проверяет корректность DataFrame.

    Args:
        df: DataFrame для проверки.
        expected_columns: Ожидаемые колонки (опционально).

    Returns:
        True если валидация прошла успешно.

    Raises:
        ValidationError: Если данные некорректны.
    """
    if df is None or df.empty:
        raise ValidationError("DataFrame пуст или None.")

    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValidationError(f"Отсутствуют колонки: {missing}")

    # Проверка на полностью пустые колонки
    fully_empty = [col for col in df.columns if df[col].isna().all()]
    if fully_empty:
        logger.warning("Полностью пустые колонки: %s", fully_empty)

    logger.info("Валидация DataFrame: %d строк, %d колонок — OK", len(df), len(df.columns))
    return True


def validate_predictions(
    predictions: np.ndarray,
    expected_length: int,
    min_value: float = 0.0,
    max_value: float = 365.0,
) -> np.ndarray:
    """
    Валидирует и корректирует предсказания.

    Args:
        predictions: Массив предсказаний.
        expected_length: Ожидаемое количество предсказаний.
        min_value: Минимальное допустимое значение.
        max_value: Максимальное допустимое значение.

    Returns:
        Скорректированный массив предсказаний.
    """
    if len(predictions) != expected_length:
        raise ValidationError(
            f"Неверное количество предсказаний: {len(predictions)}, ожидалось {expected_length}"
        )

    # Проверка на NaN
    nan_count = np.isnan(predictions).sum()
    if nan_count > 0:
        logger.warning("Обнаружено %d NaN в предсказаниях — заменяем медианой.", nan_count)
        median_val = np.nanmedian(predictions)
        predictions = np.where(np.isnan(predictions), median_val, predictions)

    # Клиппинг в допустимый диапазон
    clipped = np.clip(predictions, min_value, max_value)
    n_clipped = (predictions != clipped).sum()
    if n_clipped > 0:
        logger.warning("Обрезано %d предсказаний до диапазона [%.1f, %.1f].", n_clipped, min_value, max_value)

    return clipped


def with_timeout(timeout_sec: int):
    """
    Декоратор для ограничения времени выполнения функции.

    Args:
        timeout_sec: Максимальное время в секундах.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            def handler(signum, frame):
                raise TimeoutError(
                    f"Функция {func.__name__} превысила таймаут {timeout_sec} сек."
                )

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


def log_tool_call(func: Callable) -> Callable:
    """
    Декоратор для логирования вызовов инструментов агентов.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger.info("🔧 Вызов инструмента: %s", func.__name__)
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info("✅ %s завершён за %.2f сек.", func.__name__, elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error("❌ %s ошибка за %.2f сек: %s", func.__name__, elapsed, str(e))
            raise
    return wrapper
