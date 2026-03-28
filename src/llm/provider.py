"""
Фабрика для создания LLM-провайдера — единый интерфейс для Ollama и HuggingFace.
"""

import logging

from langchain_core.language_models import BaseChatModel

from config.settings import settings

logger = logging.getLogger(__name__)


def get_llm(provider: str | None = None) -> BaseChatModel:
    """
    Создаёт и возвращает экземпляр LLM по указанному провайдеру.

    Args:
        provider: "ollama" или "huggingface". Если None — берётся из настроек.

    Returns:
        BaseChatModel — готовый к использованию LLM.
    """
    provider = provider or settings.llm_provider

    if provider == "ollama":
        from src.llm.ollama_backend import create_ollama_llm
        logger.info("Инициализация Ollama LLM: %s", settings.ollama_model)
        return create_ollama_llm()

    elif provider == "huggingface":
        from src.llm.hf_backend import create_hf_llm
        logger.info("Инициализация HuggingFace LLM: %s", settings.hf_model)
        return create_hf_llm()

    else:
        raise ValueError(f"Неизвестный LLM провайдер: {provider}. Используйте 'ollama' или 'huggingface'.")
