"""
Модуль безопасности — санитизация входов, защита от инъекций,
rate limiting LLM-запросов, валидация выходов LLM.
"""

import collections
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

ALLOWED_DATA_ROOTS = [
    settings.raw_data_dir,
    settings.clean_data_dir,
    settings.output_dir,
    settings.knowledge_base_dir,
]


def sanitize_file_path(path: str | Path, allowed_roots: list[Path] | None = None) -> Path:
    """
    Проверяет путь к файлу на path traversal атаки.

    Резолвит путь и проверяет, что он находится внутри одной из
    разрешённых директорий. Предотвращает чтение произвольных файлов.

    Args:
        path: Проверяемый путь (строка или Path).
        allowed_roots: Разрешённые корневые директории.

    Returns:
        Безопасный резолвленный Path.

    Raises:
        ValueError: Если путь выходит за пределы разрешённых директорий.
    """
    allowed = allowed_roots or ALLOWED_DATA_ROOTS
    resolved = Path(path).resolve()

    for root in allowed:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue

    logger.warning(
        "БЕЗОПАСНОСТЬ: Заблокирован доступ к пути за пределами разрешённых директорий: %s",
        resolved,
    )
    raise ValueError(
        f"Путь {resolved} находится за пределами разрешённых директорий. "
        f"Допустимые: {[str(r) for r in allowed]}"
    )


def validate_file_size(path: Path, max_size_mb: float = 500.0) -> bool:
    """
    Проверяет размер файла перед загрузкой.

    Args:
        path: Путь к файлу.
        max_size_mb: Максимально допустимый размер в мегабайтах.

    Returns:
        True если файл в допустимых пределах.
    """
    if not path.exists():
        logger.warning("Файл не найден: %s", path)
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        logger.warning(
            "БЕЗОПАСНОСТЬ: Файл %s слишком большой (%.1f МБ > %.1f МБ).",
            path, size_mb, max_size_mb,
        )
        return False

    return True


_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"(?i)you\s+are\s+now\s+"),
    re.compile(r"(?i)^system\s*:", re.MULTILINE),
    re.compile(r"(?i)###\s*instruction"),
    re.compile(r"(?i)do\s+not\s+follow\s+the\s+above"),
    re.compile(r"(?i)forget\s+(everything|all|your\s+instructions)"),
    re.compile(r"(?i)disregard\s+(all\s+)?(previous|prior|above)"),
    re.compile(r"(?i)new\s+instructions?\s*:"),
    re.compile(r"(?i)<\s*/?\s*script"),
    re.compile(r"(?i)act\s+as\s+(a\s+)?different"),
    re.compile(r"(?i)override\s+(your\s+)?(system|instructions|rules)"),
    re.compile(r"(?i)jailbreak"),
    re.compile(r"(?i)prompt\s*injection"),
]


def detect_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Проверяет текст на наличие паттернов prompt injection.

    Не блокирует пайплайн — заменяет опасные фрагменты на [FILTERED]
    и возвращает очищенный текст.

    Args:
        text: Входной текст для проверки.

    Returns:
        Кортеж (is_suspicious, sanitized_text).
    """
    if not text:
        return False, text

    is_suspicious = False
    sanitized = text

    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(sanitized)
        if match:
            is_suspicious = True
            fragment = match.group()
            logger.warning(
                "БЕЗОПАСНОСТЬ: Обнаружен паттерн prompt injection: '%s'",
                fragment[:100],
            )
            sanitized = pattern.sub("[FILTERED]", sanitized)

    if is_suspicious:
        logger.warning(
            "БЕЗОПАСНОСТЬ: Текст содержит подозрительные паттерны. "
            "Опасные фрагменты заменены на [FILTERED]."
        )

    return is_suspicious, sanitized


ALLOWED_HYPERPARAMS: dict[str, dict[str, tuple[type, float, float]]] = {
    "lightgbm": {
        "n_estimators": (int, 1, 10000),
        "learning_rate": (float, 0.001, 1.0),
        "max_depth": (int, -1, 50),
        "num_leaves": (int, 2, 1000),
        "min_child_samples": (int, 1, 1000),
        "subsample": (float, 0.1, 1.0),
        "colsample_bytree": (float, 0.1, 1.0),
        "reg_alpha": (float, 0.0, 100.0),
        "reg_lambda": (float, 0.0, 100.0),
    },
    "xgboost": {
        "n_estimators": (int, 1, 10000),
        "learning_rate": (float, 0.001, 1.0),
        "max_depth": (int, 1, 50),
        "min_child_weight": (int, 1, 1000),
        "subsample": (float, 0.1, 1.0),
        "colsample_bytree": (float, 0.1, 1.0),
        "reg_alpha": (float, 0.0, 100.0),
        "reg_lambda": (float, 0.0, 100.0),
        "gamma": (float, 0.0, 100.0),
    },
    "random_forest": {
        "n_estimators": (int, 1, 5000),
        "max_depth": (int, 1, 100),
        "min_samples_split": (int, 2, 1000),
        "min_samples_leaf": (int, 1, 500),
        "max_features": (float, 0.1, 1.0),
    },
    "gradient_boosting": {
        "n_estimators": (int, 1, 10000),
        "learning_rate": (float, 0.001, 1.0),
        "max_depth": (int, 1, 50),
        "min_samples_split": (int, 2, 1000),
        "min_samples_leaf": (int, 1, 500),
        "subsample": (float, 0.1, 1.0),
    },
    "ridge": {
        "alpha": (float, 0.0001, 10000.0),
    },
}


def sanitize_hyperparams(params: dict, model_type: str) -> dict:
    """
    Валидирует и очищает гиперпараметры модели.

    Проверяет типы значений и допустимые диапазоны.
    Неизвестные или невалидные параметры удаляются с предупреждением.

    Args:
        params: Словарь гиперпараметров от LLM.
        model_type: Тип модели.

    Returns:
        Очищенный словарь гиперпараметров.
    """
    if model_type not in ALLOWED_HYPERPARAMS:
        logger.warning("Неизвестный тип модели для валидации гиперпараметров: %s", model_type)
        return params

    allowed = ALLOWED_HYPERPARAMS[model_type]
    sanitized = {}

    for key, value in params.items():
        if key not in allowed:
            logger.warning(
                "БЕЗОПАСНОСТЬ: Неизвестный гиперпараметр '%s' для модели %s — пропущен.",
                key, model_type,
            )
            continue

        expected_type, min_val, max_val = allowed[key]
        try:
            cast_value = expected_type(value)
            if cast_value < min_val or cast_value > max_val:
                logger.warning(
                    "БЕЗОПАСНОСТЬ: Гиперпараметр %s=%s вне диапазона [%s, %s] — обрезан.",
                    key, cast_value, min_val, max_val,
                )
                cast_value = max(min_val, min(max_val, cast_value))
            sanitized[key] = cast_value
        except (ValueError, TypeError):
            logger.warning(
                "БЕЗОПАСНОСТЬ: Невалидное значение гиперпараметра %s=%s — пропущен.",
                key, value,
            )

    return sanitized


class RateLimiter:
    """
    Ограничитель частоты запросов к LLM API.

    Использует скользящее окно для контроля количества вызовов
    в секунду и в минуту. Потокобезопасный.
    """

    def __init__(self, max_per_minute: int = 30, max_per_second: int = 2):
        self._max_per_minute = max_per_minute
        self._max_per_second = max_per_second
        self._timestamps: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """
        Проверяет лимиты и при необходимости ждёт.
        Вызывается перед каждым запросом к LLM.
        """
        with self._lock:
            now = time.monotonic()

            while self._timestamps and now - self._timestamps[0] > 60.0:
                self._timestamps.popleft()

            if len(self._timestamps) >= self._max_per_minute:
                wait_time = 60.0 - (now - self._timestamps[0])
                if wait_time > 0:
                    logger.warning(
                        "RATE LIMIT: Превышен лимит %d запросов/мин. Ожидание %.1f сек.",
                        self._max_per_minute, wait_time,
                    )
                    time.sleep(wait_time)
                    now = time.monotonic()

            recent = [t for t in self._timestamps if now - t < 1.0]
            if len(recent) >= self._max_per_second:
                wait_time = 1.0 - (now - recent[0])
                if wait_time > 0:
                    logger.info("RATE LIMIT: Ожидание %.2f сек (лимит %d/сек).", wait_time, self._max_per_second)
                    time.sleep(wait_time)
                    now = time.monotonic()

            self._timestamps.append(now)


rate_limiter = RateLimiter()


def validate_llm_json_output(
    response_text: str,
    required_keys: list[str],
    defaults: dict,
) -> dict:
    """
    Извлекает и валидирует JSON из ответа LLM.

    Удаляет markdown-обёртки, ищет JSON-объект, проверяет наличие
    обязательных ключей. При ошибке возвращает значения по умолчанию.

    Args:
        response_text: Текст ответа LLM.
        required_keys: Список обязательных ключей.
        defaults: Значения по умолчанию при ошибке парсинга.

    Returns:
        Распарсенный и валидированный словарь.
    """
    cleaned = re.sub(r"```json\s*", "", response_text)
    cleaned = re.sub(r"```\s*", "", cleaned)

    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            missing_keys = [k for k in required_keys if k not in parsed]
            if missing_keys:
                logger.warning(
                    "БЕЗОПАСНОСТЬ: В ответе LLM отсутствуют ключи: %s. Используем дефолты.",
                    missing_keys,
                )
                for key in missing_keys:
                    parsed[key] = defaults.get(key)
            return parsed
        except json.JSONDecodeError:
            logger.warning("БЕЗОПАСНОСТЬ: Невалидный JSON в ответе LLM.")

    logger.warning("БЕЗОПАСНОСТЬ: Не удалось извлечь JSON из ответа LLM. Используем дефолты.")
    return dict(defaults)


def validate_llm_text_output(response_text: str, max_length: int = 50000) -> str:
    """
    Валидирует текстовый ответ LLM — обрезает и очищает.

    Args:
        response_text: Текст ответа LLM.
        max_length: Максимальная допустимая длина.

    Returns:
        Очищенный текст.
    """
    if not response_text:
        logger.warning("БЕЗОПАСНОСТЬ: Пустой ответ от LLM.")
        return ""

    if len(response_text) > max_length:
        logger.warning(
            "БЕЗОПАСНОСТЬ: Ответ LLM обрезан с %d до %d символов.",
            len(response_text), max_length,
        )
        response_text = response_text[:max_length] + "\n[...ответ обрезан...]"

    response_text = re.sub(r"<\s*script[^>]*>.*?</\s*script\s*>", "[FILTERED]", response_text, flags=re.DOTALL | re.IGNORECASE)

    return response_text
