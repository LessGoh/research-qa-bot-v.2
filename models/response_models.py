"""
Pydantic models for structuring research bot responses
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class Citation(BaseModel):
    """Model for source information"""
    title: str = Field(description="Title of the article/source")
    authors: List[str] = Field(default=[], description="List of authors")
    year: Optional[int] = Field(default=None, description="Publication year")
    source: str = Field(description="Publication source (journal, conference)")
    doi: Optional[str] = Field(default=None, description="DOI identifier")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    excerpt: str = Field(description="Key excerpt from the source")


class ExtractedFact(BaseModel):
    """Model for extracted fact with context"""
    statement: str = Field(description="Main statement/fact")
    confidence: Literal['high', 'medium', 'low'] = Field(description="Confidence level")
    supporting_evidence: str = Field(description="Supporting evidence")
    context: str = Field(description="Context where the fact was found")
    category: str = Field(description="Fact category (methodology, result, definition)")
    contradictions: List[str] = Field(default=[], description="Contradictory statements")


class QueryMetadata(BaseModel):
    """Query metadata"""
    original_query: str = Field(description="Original user query")
    processed_query: str = Field(description="Processed query for search")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query execution time")
    search_parameters: dict = Field(default={}, description="Search parameters")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")


class ResearchResponse(BaseModel):
    """Main research bot response model"""
    
    # Query metadata
    metadata: QueryMetadata = Field(description="Query metadata")
    
    # Main response content
    summary: str = Field(description="Brief summary of found information")
    key_findings: List[ExtractedFact] = Field(description="Key findings from analysis")
    
    # Sources and citations
    citations: List[Citation] = Field(description="List of relevant sources")
    total_sources_found: int = Field(description="Total number of sources found")
    
    # Analytical conclusions
    research_gaps: List[str] = Field(default=[], description="Identified research gaps")
    methodology_notes: str = Field(default="", description="Research methodology notes")
    
    # Recommendations
    related_topics: List[str] = Field(default=[], description="Related topics for further study")
    suggested_queries: List[str] = Field(default=[], description="Suggested follow-up queries")
    
    # Quality metrics
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in results")
    completeness_score: float = Field(ge=0.0, le=1.0, description="Completeness score of the answer")


class ErrorResponse(BaseModel):
    """Model for errors"""
    error_type: str = Field(description="Error type")
    error_message: str = Field(description="Error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)
    suggestion: Optional[str] = Field(default=None, description="Suggestion for fixing the error")
