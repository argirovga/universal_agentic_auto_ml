"""
Инструменты для работы с данными — загрузка, профилирование, анализ.
Используются агентами через LangGraph tool calling.
"""

import logging

import pandas as pd
from langchain_core.tools import tool

from config.settings import settings
from src.guardrails import sanitize_file_path, validate_file_size
from src.tools.validation import log_tool_call, validate_dataframe

logger = logging.getLogger(__name__)


@tool
def load_data(file_type: str = "train") -> str:
    """
    Загружает датасет и возвращает базовую информацию.

    Args:
        file_type: Тип файла — "train" или "test".

    Returns:
        Строка с описанием формы, колонок и первых строк.
    """
    path = settings.train_file if file_type == "train" else settings.test_file
    path = sanitize_file_path(path)
    if not validate_file_size(path, settings.max_csv_size_mb):
        return f"Ошибка: файл {path.name} превышает допустимый размер ({settings.max_csv_size_mb} МБ)."
    df = pd.read_csv(path)
    validate_dataframe(df)

    info_parts = [
        f"Загружен {file_type} датасет: {path.name}",
        f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок",
        f"Колонки: {list(df.columns)}",
        f"Типы данных:\n{df.dtypes.to_string()}",
        f"\nПервые 5 строк:\n{df.head().to_string()}",
    ]
    return "\n".join(info_parts)


@tool
def get_data_profile(file_type: str = "train") -> str:
    """
    Профилирует датасет — пропуски, статистика, уникальные значения категорий.

    Args:
        file_type: Тип файла — "train" или "test".

    Returns:
        Подробный профиль данных в текстовом виде.
    """
    path = settings.train_file if file_type == "train" else settings.test_file
    path = sanitize_file_path(path)
    if not validate_file_size(path, settings.max_csv_size_mb):
        return f"Ошибка: файл {path.name} превышает допустимый размер ({settings.max_csv_size_mb} МБ)."
    df = pd.read_csv(path)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_info = pd.DataFrame({"Пропуски": missing, "% пропусков": missing_pct})
    missing_info = missing_info[missing_info["Пропуски"] > 0]

    numeric_stats = df.describe().round(3).to_string()

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_info_parts = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        top_values = df[col].value_counts().head(5).to_dict()
        cat_info_parts.append(f"  {col}: {n_unique} уникальных. Топ-5: {top_values}")

    target_info = ""
    if file_type == "train" and settings.target_column in df.columns:
        target = df[settings.target_column]
        target_info = (
            f"\nЦелевая переменная '{settings.target_column}':\n"
            f"  Среднее: {target.mean():.2f}, Медиана: {target.median():.2f}\n"
            f"  Мин: {target.min()}, Макс: {target.max()}\n"
            f"  Стд. откл.: {target.std():.2f}\n"
            f"  Доля нулей: {(target == 0).mean():.2%}"
        )

    parts = [
        f"=== Профиль данных: {file_type} ===",
        f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок",
        f"\nПропуски:\n{missing_info.to_string() if len(missing_info) > 0 else '  Нет пропусков'}",
        f"\nСтатистика числовых признаков:\n{numeric_stats}",
        f"\nКатегориальные признаки:\n" + "\n".join(cat_info_parts) if cat_info_parts else "",
        target_info,
    ]
    return "\n".join(parts)


@tool
def get_correlations() -> str:
    """
    Вычисляет корреляции числовых признаков с целевой переменной.

    Returns:
        Таблица корреляций, отсортированная по абсолютному значению.
    """
    train_path = sanitize_file_path(settings.train_file)
    df = pd.read_csv(train_path)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if settings.target_column not in numeric_cols:
        return "Целевая переменная не найдена среди числовых колонок."

    correlations = df[numeric_cols].corr()[settings.target_column].drop(settings.target_column)
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

    parts = [
        "=== Корреляции с целевой переменной ===",
        correlations.round(4).to_string(),
        f"\nНаиболее коррелирующий признак: {correlations.abs().idxmax()} "
        f"(r={correlations[correlations.abs().idxmax()]:.4f})",
    ]
    return "\n".join(parts)


@tool
def get_value_distributions(column: str) -> str:
    """
    Возвращает распределение значений для указанной колонки.

    Args:
        column: Название колонки.

    Returns:
        Информация о распределении значений.
    """
    train_path = sanitize_file_path(settings.train_file)
    df = pd.read_csv(train_path)

    if column not in df.columns:
        return f"Колонка '{column}' не найдена. Доступные: {list(df.columns)}"

    col_data = df[column]
    parts = [f"=== Распределение: {column} ==="]

    if col_data.dtype in ["float64", "int64"]:
        parts.extend([
            f"Тип: числовой",
            f"Среднее: {col_data.mean():.3f}",
            f"Медиана: {col_data.median():.3f}",
            f"Стд. откл.: {col_data.std():.3f}",
            f"Мин: {col_data.min()}, Макс: {col_data.max()}",
            f"Квантили: 25%={col_data.quantile(0.25):.2f}, "
            f"75%={col_data.quantile(0.75):.2f}",
            f"Пропуски: {col_data.isna().sum()}",
        ])
    else:
        value_counts = col_data.value_counts()
        parts.extend([
            f"Тип: категориальный",
            f"Уникальных значений: {col_data.nunique()}",
            f"Пропуски: {col_data.isna().sum()}",
            f"Топ-10 значений:\n{value_counts.head(10).to_string()}",
        ])

    return "\n".join(parts)
