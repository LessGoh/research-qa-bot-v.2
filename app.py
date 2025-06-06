"""
Main Streamlit application for Research Q/A Bot
"""

import streamlit as st
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import json

# Local imports
from config import settings
from services.llama_service import LlamaService
from services.openai_service import OpenAIService
from models.response_models import ResearchResponse
from utils.helpers import (
    validate_query, 
    format_timestamp, 
    truncate_text,
    format_processing_time,
    format_authors_list,
    estimate_reading_time
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_page_config():
    """Initialize Streamlit page configuration"""
    st.set_page_config(
        page_title=settings.PAGE_TITLE,
        page_icon=settings.PAGE_ICON,
        layout=settings.LAYOUT,
        initial_sidebar_state=settings.INITIAL_SIDEBAR_STATE
    )


def init_session_state():
    """Initialize session state variables"""
    if 'llama_service' not in st.session_state:
        st.session_state.llama_service = LlamaService()
    
    if 'openai_service' not in st.session_state:
        st.session_state.openai_service = OpenAIService()
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None


def display_header():
    """Display application header"""
    st.title(settings.APP_TITLE)
    st.markdown(f"*{settings.APP_DESCRIPTION}*")
    st.divider()


def display_search_interface():
    """Display the main search interface"""
    st.subheader("🔍 Research Query")
    
    # Query input
    query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., What are the latest advances in machine learning for medical diagnosis?",
        height=100,
        help="Ask specific questions about research topics, methodologies, or findings"
    )
    
    # Search parameters
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.empty()  # Spacer
    
    with col2:
        similarity_top_k = st.selectbox(
            "Number of sources:",
            options=[3, 5, 7, 10],
            index=1,  # Default to 5
            help="How many relevant sources to analyze"
        )
    
    # Search button
    search_button = st.button(
        "🔍 Search & Analyze",
        type="primary",
        use_container_width=True
    )
    
    return query, similarity_top_k, search_button


def perform_search(query: str, similarity_top_k: int) -> Optional[ResearchResponse]:
    """
    Perform search and analysis
    
    Args:
        query: User query
        similarity_top_k: Number of sources to retrieve
        
    Returns:
        ResearchResponse or None if failed
    """
    # Validate query
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        st.error(f"Invalid query: {error_msg}")
        return None
    
    # Initialize services
    with st.status("Initializing services...", expanded=True) as status:
        st.write("Connecting to LlamaCloud...")
        if not st.session_state.llama_service.initialize_index():
            st.error("Failed to initialize LlamaCloud connection")
            return None
        
        st.write("Connecting to OpenAI...")
        if not st.session_state.openai_service.initialize_client():
            st.error("Failed to initialize OpenAI connection")
            return None
        
        st.write("Services initialized successfully!")
        status.update(label="Services ready", state="complete")
    
    # Perform search
    with st.status("Searching documents...", expanded=True) as status:
        st.write(f"Searching for: {truncate_text(query, 50)}")
        
        search_results = st.session_state.llama_service.search_documents(
            query=query, 
            top_k=similarity_top_k
        )
        
        if not search_results:
            st.error("No relevant documents found")
            return None
        
        st.write(f"Found {len(search_results)} relevant sources")
        status.update(label="Documents retrieved", state="complete")
    
    # Analyze content
    with st.status("Analyzing content with AI...", expanded=True) as status:
        st.write("Processing content with OpenAI...")
        
        response = st.session_state.openai_service.create_research_response(
            query=query,
            search_results=search_results
        )
        
        if not response:
            st.error("Failed to analyze content")
            return None
        
        st.write("Analysis completed!")
        status.update(label="Analysis complete", state="complete")
    
    # Add to search history
    st.session_state.search_history.append({
        'query': query,
        'timestamp': datetime.now(),
        'sources_count': len(search_results)
    })
    
    return response


def display_results(response: ResearchResponse):
    """
    Display search results and analysis
    
    Args:
        response: ResearchResponse object
    """
    if not response:
        return
    
    # Results header
    st.success("✅ Analysis completed successfully!")
    
    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Sources Found", 
            response.total_sources_found
        )
    
    with col2:
        processing_time = response.metadata.processing_time or 0
        st.metric(
            "Processing Time", 
            format_processing_time(processing_time)
        )
    
    with col3:
        st.metric(
            "Confidence", 
            f"{response.overall_confidence:.1%}"
        )
    
    with col4:
        st.metric(
            "Completeness", 
            f"{response.completeness_score:.1%}"
        )
    
    st.divider()
    
    # Main summary
    st.subheader("📋 Research Summary")
    st.write(response.summary)
    
    # Key findings
    if response.key_findings:
        st.subheader("🔍 Key Findings")
        
        for i, finding in enumerate(response.key_findings, 1):
            with st.expander(f"Finding {i}: {truncate_text(finding.statement, 60)}", expanded=i==1):
                
                # Confidence badge
                confidence_color = {
                    'high': '🟢',
                    'medium': '🟡', 
                    'low': '🔴'
                }
                
                st.markdown(f"**Confidence:** {confidence_color.get(finding.confidence, '⚫')} {finding.confidence.title()}")
                st.markdown(f"**Category:** {finding.category}")
                
                # Statement
                st.markdown("**Statement:**")
                st.write(finding.statement)
                
                # Supporting evidence
                if finding.supporting_evidence:
                    st.markdown("**Supporting Evidence:**")
                    st.write(finding.supporting_evidence)
                
                # Context
                if finding.context:
                    st.markdown("**Context:**")
                    st.write(finding.context)
                
                # Contradictions
                if finding.contradictions:
                    st.markdown("**Contradictions:**")
                    for contradiction in finding.contradictions:
                        st.write(f"• {contradiction}")
    
    # Sources
    if response.citations:
        st.subheader("📚 Sources")
        
        for i, citation in enumerate(response.citations[:settings.MAX_RESULTS_DISPLAY], 1):
            with st.expander(f"Source {i}: {citation.title}", expanded=False):
                
                # Authors
                if citation.authors:
                    st.markdown(f"**Authors:** {format_authors_list(citation.authors)}")
                
                # Publication info
                info_parts = []
                if citation.year:
                    info_parts.append(str(citation.year))
                if citation.source:
                    info_parts.append(citation.source)
                
                if info_parts:
                    st.markdown(f"**Publication:** {' | '.join(info_parts)}")
                
                # DOI
                if citation.doi:
                    st.markdown(f"**DOI:** {citation.doi}")
                
                # Relevance score
                st.markdown(f"**Relevance Score:** {citation.relevance_score:.3f}")
                
                # Excerpt
                if citation.excerpt:
                    st.markdown("**Excerpt:**")
                    st.write(citation.excerpt)
                    reading_time = estimate_reading_time(citation.excerpt)
                    st.caption(f"Reading time: {reading_time}")
    
    # Research gaps
    if response.research_gaps:
        st.subheader("🔬 Research Gaps Identified")
        for gap in response.research_gaps:
            st.write(f"• {gap}")
    
    # Related topics
    if response.related_topics:
        st.subheader("🔗 Related Topics")
        for topic in response.related_topics:
            st.write(f"• {topic}")
    
    # Suggested queries
    if response.suggested_queries:
        st.subheader("💡 Suggested Follow-up Queries")
        for suggestion in response.suggested_queries:
            if st.button(f"🔍 {suggestion}", key=f"suggest_{hash(suggestion)}"):
                st.session_state.suggested_query = suggestion
                st.rerun()


def display_sidebar():
    """Display sidebar with additional information and controls"""
    with st.sidebar:
        st.header("ℹ️ Information")
        
        # Health check
        if st.button("🔧 Check System Status"):
            with st.spinner("Checking system health..."):
                llama_health = st.session_state.llama_service.health_check()
                openai_health = st.session_state.openai_service.health_check()
                
                st.write("**LlamaCloud Status:**")
                st.write(f"• Initialized: {'✅' if llama_health['initialized'] else '❌'}")
                st.write(f"• Index Available: {'✅' if llama_health['index_available'] else '❌'}")
                
                st.write("**OpenAI Status:**")
                st.write(f"• Initialized: {'✅' if openai_health['initialized'] else '❌'}")
                st.write(f"• API Functional: {'✅' if openai_health.get('api_functional', False) else '❌'}")
        
        st.divider()
        
        # Search history
        st.subheader("📜 Recent Searches")
        
        if st.session_state.search_history:
            for i, search in enumerate(reversed(st.session_state.search_history[-5:]), 1):
                with st.expander(f"{i}. {truncate_text(search['query'], 30)}", expanded=False):
                    st.write(f"**Query:** {search['query']}")
                    st.write(f"**Time:** {format_timestamp(search['timestamp'], 'short')}")
                    st.write(f"**Sources:** {search['sources_count']}")
                    
                    if st.button(f"🔄 Repeat", key=f"repeat_{i}"):
                        st.session_state.repeat_query = search['query']
                        st.rerun()
        else:
            st.write("No recent searches")
        
        st.divider()
        
        # Export options
        st.subheader("📥 Export Results")
        
        if st.session_state.current_response:
            if st.button("📄 Download as JSON"):
                json_data = st.session_state.current_response.model_dump()
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(json_data, indent=2, default=str),
                    file_name=f"research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.write("No results to export")


def main():
    """Main application function"""
    
    # Initialize
    init_page_config()
    init_session_state()
    
    # Display interface
    display_header()
    display_sidebar()
    
    # Handle suggested query
    if hasattr(st.session_state, 'suggested_query'):
        query = st.session_state.suggested_query
        del st.session_state.suggested_query
    elif hasattr(st.session_state, 'repeat_query'):
        query = st.session_state.repeat_query
        del st.session_state.repeat_query
    else:
        query = ""
    
    # Main search interface
    query_input, similarity_top_k, search_button = display_search_interface()
    
    # Use suggested/repeated query if available
    if query:
        query_input = query
        search_button = True
    
    # Perform search if button clicked
    if search_button and query_input:
        response = perform_search(query_input, similarity_top_k)
        if response:
            st.session_state.current_response = response
    
    # Display results
    if st.session_state.current_response:
        st.divider()
        display_results(st.session_state.current_response)


if __name__ == "__main__":
    main()
