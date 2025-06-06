"""
Конфигурация приложения для Q/A исследовательского бота
"""
import streamlit as st
from typing import Optional


class Settings:
    """Настройки приложения"""
    
    # API Keys - будут загружены из Streamlit Secrets
    @property
    def openai_api_key(self) -> str:
        return st.secrets["OPENAI_API_KEY"]
    
    @property
    def llama_cloud_api_key(self) -> str:
        return st.secrets["LLAMA_CLOUD_API_KEY"]
    
    # LlamaCloud Index Configuration
    LLAMACLOUD_INDEX_NAME = "Arxiv 2024-2025. Key: Volatility / Parsing Preset: Balanced, Chunk size: 256, Chunk Overlap: 50"
    LLAMACLOUD_PROJECT_NAME = "Default"
    LLAMACLOUD_ORGANIZATION_ID = "858afa1e-d3dc-4a96-8783-d4f3798b0643"
    
    # OpenAI Configuration
    OPENAI_MODEL = "gpt-4o-2024-08-06"
    OPENAI_TEMPERATURE = 0.1
    OPENAI_MAX_TOKENS = 2000
    
    # Search Configuration
    DEFAULT_SIMILARITY_TOP_K = 5
    MAX_SIMILARITY_TOP_K = 10
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 500
    
    # Streamlit Configuration
    PAGE_TITLE = "Research Q/A Bot"
    PAGE_ICON = "🔬"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # Application Settings
    APP_TITLE = "Исследовательский Q/A Бот"
    APP_DESCRIPTION = "Анализ и поиск информации в научных статьях с помощью ИИ"
    MAX_RESULTS_DISPLAY = 3


# Глобальный экземпляр настроек
settings = Settings()


# Константы для валидации
SUPPORTED_ANALYSIS_TYPES = [
    "basic",
    "detailed", 
    "comparative"
]

# Сообщения для пользователя
MESSAGES = {
    "QUERY_TOO_SHORT": f"Запрос должен содержать минимум {settings.DEFAULT_SIMILARITY_TOP_K} символов",
    "QUERY_TOO_LONG": f"Запрос не должен превышать {settings.MAX_QUERY_LENGTH} символов",
    "NO_RESULTS": "По вашему запросу результаты не найдены. Попробуйте изменить формулировку.",
    "API_ERROR": "Произошла ошибка при обращении к API. Попробуйте позже.",
    "PROCESSING": "Обрабатываю ваш запрос...",
    "SUCCESS": "Анализ завершен успешно!"
}
