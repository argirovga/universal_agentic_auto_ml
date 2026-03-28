"""
Бэкенд для HuggingFace Inference API.
"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from config.settings import settings


def create_hf_llm() -> ChatHuggingFace:
    """
    Создаёт экземпляр ChatHuggingFace через HuggingFace Inference API.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id=settings.hf_model,
        huggingfacehub_api_token=settings.hf_token,
        temperature=settings.llm_temperature,
        max_new_tokens=settings.llm_max_tokens,
        task="text-generation",
    )
    return ChatHuggingFace(llm=endpoint)
