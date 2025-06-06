"""
LlamaIndex service for document retrieval and search
"""

import streamlit as st
from typing import List, Optional, Dict, Any
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from config import settings
import logging


class LlamaService:
    """Service for working with LlamaCloudIndex"""
    
    def __init__(self):
        self.index: Optional[LlamaCloudIndex] = None
        self._initialized = False
        
    def initialize_index(self) -> bool:
        """
        Initialize LlamaCloudIndex connection
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if self._initialized and self.index is not None:
                return True
                
            self.index = LlamaCloudIndex(
                name=settings.LLAMACLOUD_INDEX_NAME,
                project_name=settings.LLAMACLOUD_PROJECT_NAME,
                organization_id=settings.LLAMACLOUD_ORGANIZATION_ID,
                api_key=settings.llama_cloud_api_key,
            )
            
            self._initialized = True
            logging.info("LlamaCloudIndex initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize LlamaCloudIndex: {str(e)}")
            st.error(f"Failed to connect to LlamaCloud: {str(e)}")
            return False
    
    def search_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search documents using the query
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        if not self._ensure_initialized():
            st.error("❌ LlamaIndex service not initialized")
            return []
            
        try:
            if top_k is None:
                top_k = settings.DEFAULT_SIMILARITY_TOP_K
                
            # Ensure top_k is within limits
            top_k = min(max(top_k, 1), settings.MAX_SIMILARITY_TOP_K)
            
            st.write(f"🔍 Searching with top_k={top_k}")
            
            # Get retriever and perform search
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            st.write("✅ Retriever created successfully")
            
            nodes = retriever.retrieve(query)
            st.write(f"📄 Retrieved {len(nodes)} nodes from index")
            
            if not nodes:
                st.warning("⚠️ No nodes returned from search")
                return []
            
            # Format results
            results = []
            for i, node in enumerate(nodes):
                # Debug node structure
                st.write(f"🔸 Processing node {i+1}")
                st.write(f"   - Node type: {type(node)}")
                st.write(f"   - Has text: {hasattr(node, 'text')}")
                st.write(f"   - Has metadata: {hasattr(node, 'metadata')}")
                st.write(f"   - Has score: {hasattr(node, 'score')}")
                
                node_text = getattr(node, 'text', '') or getattr(node, 'node', {}).get('text', '')
                node_metadata = getattr(node, 'metadata', {}) or getattr(node, 'node', {}).get('metadata', {})
                node_score = getattr(node, 'score', 0.0)
                
                if not node_text:
                    st.warning(f"   - ⚠️ Node {i+1} has no text content")
                    continue
                
                result = {
                    "rank": i + 1,
                    "content": node_text,
                    "score": node_score,
                    "metadata": node_metadata,
                    "node_id": getattr(node, 'node_id', '') or f"node_{i}",
                }
                results.append(result)
                
                # Show preview of content
                preview = node_text[:100] + "..." if len(node_text) > 100 else node_text
                st.write(f"   - Content preview: {preview}")
                
            logging.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            st.success(f"✅ Successfully processed {len(results)} results")
            return results
            
        except Exception as e:
            error_msg = f"Error searching documents: {str(e)}"
            logging.error(error_msg)
            st.error(f"❌ Search error: {str(e)}")
            st.write(f"Error type: {type(e)}")
            
            # Try to get more details
            import traceback
            st.write("Full traceback:")
            st.code(traceback.format_exc())
            return []
    
    def get_query_engine(self, similarity_top_k: int = None):
        """
        Get query engine for direct querying
        
        Args:
            similarity_top_k: Number of similar documents to retrieve
            
        Returns:
            Query engine or None if initialization failed
        """
        if not self._ensure_initialized():
            return None
            
        try:
            if similarity_top_k is None:
                similarity_top_k = settings.DEFAULT_SIMILARITY_TOP_K
                
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k
            )
            return query_engine
            
        except Exception as e:
            logging.error(f"Error creating query engine: {str(e)}")
            return None
    
    def query_direct(self, query: str, similarity_top_k: int = None) -> Optional[str]:
        """
        Direct query using LlamaIndex query engine
        
        Args:
            query: Query string
            similarity_top_k: Number of documents to consider
            
        Returns:
            Response string or None if failed
        """
        query_engine = self.get_query_engine(similarity_top_k)
        if query_engine is None:
            return None
            
        try:
            response = query_engine.query(query)
            return str(response)
            
        except Exception as e:
            logging.error(f"Error in direct query: {str(e)}")
            return None
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a readable string
        
        Args:
            results: List of search results
            
        Returns:
            Formatted string with all results
        """
        if not results:
            return "No results found."
            
        formatted_parts = []
        for result in results:
            metadata = result.get('metadata', {})
            title = metadata.get('title', 'Unknown Title')
            authors = metadata.get('authors', 'Unknown Authors')
            
            part = f"""
Source {result['rank']}:
Title: {title}
Authors: {authors}
Relevance Score: {result['score']:.3f}
Content: {result['content'][:500]}...
---
"""
            formatted_parts.append(part)
            
        return "\n".join(formatted_parts)
    
    def get_source_metadata(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source metadata from search results
        
        Args:
            results: Search results
            
        Returns:
            List of metadata dictionaries
        """
        metadata_list = []
        for result in results:
            metadata = result.get('metadata', {})
            source_info = {
                'title': metadata.get('title', 'Unknown Title'),
                'authors': metadata.get('authors', 'Unknown Authors'),
                'year': metadata.get('year'),
                'source': metadata.get('source', 'Unknown Source'),
                'doi': metadata.get('doi'),
                'relevance_score': result.get('score', 0.0),
                'excerpt': result.get('content', '')[:300] + '...'
            }
            metadata_list.append(source_info)
            
        return metadata_list
    
    def _ensure_initialized(self) -> bool:
        """
        Ensure the service is initialized
        
        Returns:
            bool: True if initialized successfully
        """
        if not self._initialized:
            return self.initialize_index()
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check service health
        
        Returns:
            Dictionary with health status
        """
        status = {
            'service': 'LlamaService',
            'initialized': self._initialized,
            'index_available': self.index is not None
        }
        
        if self._initialized:
            try:
                # Try a simple test query
                test_results = self.search_documents("test", top_k=1)
                status['search_functional'] = len(test_results) >= 0
            except Exception as e:
                status['search_functional'] = False
                status['error'] = str(e)
        else:
            status['search_functional'] = False
            
        return status
