"""
Бэкенд для локальной модели через Ollama.
"""

from langchain_ollama import ChatOllama

from config.settings import settings


def create_ollama_llm() -> ChatOllama:
    """
    Создаёт экземпляр ChatOllama с параметрами из конфигурации.
    """
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
    )
