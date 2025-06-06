"""
OpenAI service for content analysis and structured response generation
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
import openai
from config import settings
from models.response_models import ResearchResponse, QueryMetadata, ErrorResponse
import logging
import time


class OpenAIService:
    """Service for working with OpenAI API and structured outputs"""
    
    def __init__(self):
        self.client: Optional[openai.OpenAI] = None
        self._initialized = False
        
    def initialize_client(self) -> bool:
        """
        Initialize OpenAI client
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if self._initialized and self.client is not None:
                return True
                
            self.client = openai.OpenAI(
                api_key=settings.openai_api_key
            )
            
            # Test the connection
            self.client.models.list()
            
            self._initialized = True
            logging.info("OpenAI client initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {str(e)}")
            st.error(f"Failed to connect to OpenAI: {str(e)}")
            return False
    
    def analyze_content(
        self, 
        query: str, 
        content: str, 
        search_metadata: Dict[str, Any]
    ) -> Optional[ResearchResponse]:
        """
        Analyze content using OpenAI and return structured response
        
        Args:
            query: Original user query
            content: Retrieved content to analyze
            search_metadata: Metadata from search results
            
        Returns:
            ResearchResponse object or None if failed
        """
        if not self._ensure_initialized():
            return None
            
        try:
            start_time = time.time()
            
            # Create system prompt for research analysis
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with query and content
            user_prompt = self._create_user_prompt(query, content)
            
            # Call OpenAI with structured output
            completion = self.client.beta.chat.completions.parse(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=ResearchResponse,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS
            )
            
            if completion.choices[0].message.parsed:
                response = completion.choices[0].message.parsed
                
                # Update metadata with processing time
                processing_time = time.time() - start_time
                response.metadata.processing_time = processing_time
                response.metadata.search_parameters = search_metadata
                
                logging.info(f"Successfully analyzed content for query: {query[:50]}...")
                return response
            else:
                logging.error("OpenAI returned no parsed response")
                return None
                
        except Exception as e:
            logging.error(f"Error analyzing content: {str(e)}")
            st.error(f"Analysis error: {str(e)}")
            return None
    
    def create_research_response(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> Optional[ResearchResponse]:
        """
        Create a structured research response from search results
        
        Args:
            query: Original query
            search_results: Results from LlamaIndex search
            
        Returns:
            ResearchResponse or None if failed
        """
        if not search_results:
            return self._create_empty_response(query)
            
        # Combine all content
        combined_content = self._combine_search_results(search_results)
        
        # Prepare metadata
        search_metadata = {
            'total_sources': len(search_results),
            'top_k': len(search_results),
            'avg_relevance_score': sum(r.get('score', 0) for r in search_results) / len(search_results)
        }
        
        # Analyze with OpenAI
        response = self.analyze_content(query, combined_content, search_metadata)
        
        if response:
            # Add source information to citations
            response.citations = self._extract_citations(search_results)
            response.total_sources_found = len(search_results)
            
        return response
    
    def handle_api_errors(self, func):
        """
        Decorator for handling API errors
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function with error handling
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except openai.RateLimitError as e:
                logging.error(f"OpenAI rate limit exceeded: {str(e)}")
                st.error("Rate limit exceeded. Please try again in a moment.")
                return None
            except openai.AuthenticationError as e:
                logging.error(f"OpenAI authentication error: {str(e)}")
                st.error("Authentication failed. Please check API key.")
                return None
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                st.error(f"An unexpected error occurred: {str(e)}")
                return None
                
        return wrapper
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for research analysis"""
        return """You are an expert research analyst specialized in scientific literature analysis. 

Your task is to analyze provided scientific content and extract structured information including:
- Key findings and facts with confidence levels
- Research methodology insights
- Identified gaps in current research
- Related topics for further exploration
- Overall summary and conclusions

Guidelines:
- Be precise and evidence-based in your analysis
- Assign appropriate confidence levels (high/medium/low) based on evidence strength
- Extract direct quotes when possible for supporting evidence
- Identify contradictions or conflicting information
- Suggest areas where more research is needed
- Maintain scientific objectivity and accuracy

Format your response according to the provided schema with proper categorization of findings."""
    
    def _create_user_prompt(self, query: str, content: str) -> str:
        """Create user prompt with query and content"""
        return f"""Please analyze the following scientific content in response to this research query:

QUERY: {query}

CONTENT TO ANALYZE:
{content}

Provide a comprehensive analysis following the structured format, including:
1. A clear summary of the main findings
2. Key facts extracted with appropriate confidence levels
3. Any research gaps or limitations identified
4. Suggestions for related topics or follow-up queries
5. Overall assessment of the information quality and completeness"""
    
    def _combine_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Combine search results into a single content string"""
        combined_parts = []
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            title = metadata.get('title', f'Source {i}')
            content = result.get('content', '')
            
            part = f"""
=== SOURCE {i}: {title} ===
Relevance Score: {result.get('score', 0.0):.3f}
Content: {content}
"""
            combined_parts.append(part)
            
        return "\n".join(combined_parts)
    
    def _extract_citations(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citation information from search results"""
        citations = []
        for result in search_results:
            metadata = result.get('metadata', {})
            
            citation = {
                'title': metadata.get('title', 'Unknown Title'),
                'authors': metadata.get('authors', '').split(',') if metadata.get('authors') else [],
                'year': metadata.get('year'),
                'source': metadata.get('source', 'Unknown Source'),
                'doi': metadata.get('doi'),
                'relevance_score': result.get('score', 0.0),
                'excerpt': result.get('content', '')[:300] + '...' if result.get('content') else ''
            }
            citations.append(citation)
            
        return citations
    
    def _create_empty_response(self, query: str) -> ResearchResponse:
        """Create empty response when no results found"""
        metadata = QueryMetadata(
            original_query=query,
            processed_query=query
        )
        
        return ResearchResponse(
            metadata=metadata,
            summary="No relevant information found for this query.",
            key_findings=[],
            citations=[],
            total_sources_found=0,
            research_gaps=["Insufficient information available on this topic"],
            methodology_notes="",
            related_topics=[],
            suggested_queries=[],
            overall_confidence=0.0,
            completeness_score=0.0
        )
    
    def _ensure_initialized(self) -> bool:
        """
        Ensure the service is initialized
        
        Returns:
            bool: True if initialized successfully
        """
        if not self._initialized:
            return self.initialize_client()
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check service health
        
        Returns:
            Dictionary with health status
        """
        status = {
            'service': 'OpenAIService',
            'initialized': self._initialized,
            'client_available': self.client is not None
        }
        
        if self._initialized:
            try:
                # Try a simple API call
                models = self.client.models.list()
                status['api_functional'] = True
                status['available_models'] = len(models.data) if hasattr(models, 'data') else 0
            except Exception as e:
                status['api_functional'] = False
                status['error'] = str(e)
        else:
            status['api_functional'] = False
            
        return status
