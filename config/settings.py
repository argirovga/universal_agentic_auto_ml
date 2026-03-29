"""
Конфигурация проекта — пути, параметры LLM, настройки ML пайплайна.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Основные настройки проекта."""

    # --- Пути ---
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw_data"
    clean_data_dir: Path = PROJECT_ROOT / "data" / "clean_data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    knowledge_base_dir: Path = PROJECT_ROOT / "knowledge_base"
    chroma_db_dir: Path = PROJECT_ROOT / "knowledge_base" / "chroma_db"

    # --- Файлы данных ---
    train_file: Path = PROJECT_ROOT / "data" / "raw_data" / "train.csv"
    test_file: Path = PROJECT_ROOT / "data" / "raw_data" / "test.csv"
    sample_submission_file: Path = PROJECT_ROOT / "data" / "raw_data" / "sample_submition.csv"

    # --- LLM ---
    llm_provider: str = Field(default="ollama", description="ollama или huggingface")
    ollama_model: str = Field(default="qwen3-coder:30b", description="Модель Ollama")
    ollama_base_url: str = Field(default="http://localhost:11434", description="URL сервера Ollama")
    hf_model: str = Field(
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        description="Модель HuggingFace Inference API",
    )
    hf_token: str = Field(default="", description="Токен HuggingFace API")
    llm_temperature: float = Field(default=0.1, description="Температура генерации LLM")
    llm_max_tokens: int = Field(default=4096, description="Максимум токенов в ответе")

    # --- ML пайплайн ---
    max_iterations: int = Field(default=5, description="Максимум итераций улучшения модели")
    validation_split: float = Field(default=0.2, description="Доля валидационной выборки")
    random_seed: int = Field(default=42, description="Seed для воспроизводимости")
    target_column: str = Field(default="target", description="Название целевой колонки")

    # --- Пороги качества ---
    r2_threshold: float = Field(default=0.5, description="Минимальный R² для остановки")
    training_timeout_sec: int = Field(default=300, description="Таймаут обучения модели (сек)")

    # --- Guardrails ---
    max_csv_size_mb: float = Field(default=500.0, description="Макс. размер CSV файла (МБ)")
    rate_limit_per_minute: int = Field(default=30, description="Лимит LLM-запросов в минуту")
    rate_limit_per_second: int = Field(default=2, description="Лимит LLM-запросов в секунду")
    enable_injection_detection: bool = Field(default=True, description="Детекция prompt injection")

    # --- RAG ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Модель эмбеддингов для RAG",
    )
    rag_top_k: int = Field(default=3, description="Количество релевантных чанков из RAG")
    rag_chunk_size: int = Field(default=500, description="Размер чанка для индексации")
    rag_chunk_overlap: int = Field(default=50, description="Перекрытие чанков")

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}


settings = Settings()
