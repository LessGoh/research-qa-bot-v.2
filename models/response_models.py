"""
Pydantic модели для структурирования ответов исследовательского бота
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class Citation(BaseModel):
    """Модель для информации об источнике"""
    title: str = Field(description="Название статьи/источника")
    authors: List[str] = Field(default=[], description="Список авторов")
    year: Optional[int] = Field(default=None, description="Год публикации")
    source: str = Field(description="Источник публикации (журнал, конференция)")
    doi: Optional[str] = Field(default=None, description="DOI идентификатор")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Оценка релевантности")
    excerpt: str = Field(description="Ключевой отрывок из источника")


class ExtractedFact(BaseModel):
    """Модель для извлеченного факта с контекстом"""
    statement: str = Field(description="Основное утверждение/факт")
    confidence: Literal['high', 'medium', 'low'] = Field(description="Уровень уверенности")
    supporting_evidence: str = Field(description="Подтверждающие доказательства")
    context: str = Field(description="Контекст, в котором был найден факт")
    category: str = Field(description="Категория факта (методология, результат, определение)")
    contradictions: List[str] = Field(default=[], description="Противоречащие утверждения")


class QueryMetadata(BaseModel):
    """Метаданные запроса"""
    original_query: str = Field(description="Исходный запрос пользователя")
    processed_query: str = Field(description="Обработанный запрос для поиска")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время выполнения запроса")
    search_parameters: dict = Field(default={}, description="Параметры поиска")
    processing_time: Optional[float] = Field(default=None, description="Время обработки в секундах")


class ResearchResponse(BaseModel):
    """Основная модель ответа исследовательского бота"""
    
    # Метаданные запроса
    metadata: QueryMetadata = Field(description="Метаданные запроса")
    
    # Основной контент ответа
    summary: str = Field(description="Краткое резюме найденной информации")
    key_findings: List[ExtractedFact] = Field(description="Ключевые находки из анализа")
    
    # Источники и цитаты
    citations: List[Citation] = Field(description="Список релевантных источников")
    total_sources_found: int = Field(description="Общее количество найденных источников")
    
    # Аналитические выводы
    research_gaps: List[str] = Field(default=[], description="Выявленные пробелы в исследованиях")
    methodology_notes: str = Field(default="", description="Заметки о методологии исследований")
    
    # Рекомендации
    related_topics: List[str] = Field(default=[], description="Связанные темы для дальнейшего изучения")
    suggested_queries: List[str] = Field(default=[], description="Предлагаемые уточняющие запросы")
    
    # Метрики качества
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Общая уверенность в результатах")
    completeness_score: float = Field(ge=0.0, le=1.0, description="Оценка полноты ответа")


class ErrorResponse(BaseModel):
    """Модель для ошибок"""
    error_type: str = Field(description="Тип ошибки")
    error_message: str = Field(description="Сообщение об ошибке")
    details: Optional[dict] = Field(default=None, description="Дополнительные детали ошибки")
    timestamp: datetime = Field(default_factory=datetime.now)
    suggestion: Optional[str] = Field(default=None, description="Предложение по исправлению")
