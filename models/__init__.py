"""
Модели данных для Q/A исследовательского бота
"""

from .response_models import (
    Citation,
    ExtractedFact,
    ResearchResponse,
    QueryMetadata
)

__all__ = [
    "Citation",
    "ExtractedFact", 
    "ResearchResponse",
    "QueryMetadata"
]
