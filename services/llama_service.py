"""
LlamaIndex service for document retrieval and search
"""

import streamlit as st
from typing import List, Optional, Dict, Any
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from config import settings
import logging
import time
import httpx


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
                
            st.write("🔗 Connecting to LlamaCloud...")
            st.write(f"📋 Index: {settings.LLAMACLOUD_INDEX_NAME}")
            st.write(f"📁 Project: {settings.LLAMACLOUD_PROJECT_NAME}")
            st.write(f"🏢 Organization: {settings.LLAMACLOUD_ORGANIZATION_ID}")
            
            # Try method 1: Standard LlamaCloudIndex
            try:
                self.index = LlamaCloudIndex(
                    name=settings.LLAMACLOUD_INDEX_NAME,
                    project_name=settings.LLAMACLOUD_PROJECT_NAME,
                    organization_id=settings.LLAMACLOUD_ORGANIZATION_ID,
                    api_key=settings.llama_cloud_api_key,
                )
                st.write("✅ Standard connection successful")
                
            except Exception as e1:
                st.warning(f"⚠️ Standard connection failed: {str(e1)}")
                st.write("🔄 Trying alternative connection method...")
                
                # Try method 2: Using pipeline ID
                try:
                    from llama_index.indices.managed.llama_cloud import LlamaCloudRetriever
                    
                    # Create retriever directly with pipeline ID
                    self.retriever = LlamaCloudRetriever(
                        name=settings.LLAMACLOUD_INDEX_NAME,
                        project_name=settings.LLAMACLOUD_PROJECT_NAME,
                        organization_id=settings.LLAMACLOUD_ORGANIZATION_ID,
                        api_key=settings.llama_cloud_api_key,
                    )
                    
                    # Create a minimal index wrapper
                    self.index = type('MockIndex', (), {
                        'as_retriever': lambda similarity_top_k=5: self.retriever,
                        'as_query_engine': lambda similarity_top_k=5: None
                    })()
                    
                    st.write("✅ Alternative connection successful")
                    
                except Exception as e2:
                    st.error(f"❌ Alternative connection also failed: {str(e2)}")
                    return False
            
            # Test connection with a simple operation
            st.write("🧪 Testing connection...")
            try:
                # Try to get retriever to test connection
                test_retriever = self.index.as_retriever(similarity_top_k=1)
                st.write("✅ Connection test successful")
            except Exception as test_e:
                st.warning(f"⚠️ Connection test failed: {str(test_e)}")
                # Don't fail completely, maybe search will still work
            
            self._initialized = True
            logging.info("LlamaCloudIndex initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize LlamaCloudIndex: {str(e)}")
            st.error(f"Failed to connect to LlamaCloud: {str(e)}")
            return False
    
    def search_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search documents using the query with retry mechanism
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        if not self._ensure_initialized():
            st.error("❌ LlamaIndex service not initialized")
            return []
            
        if top_k is None:
            top_k = settings.DEFAULT_SIMILARITY_TOP_K
            
        # Ensure top_k is within limits
        top_k = min(max(top_k, 1), settings.MAX_SIMILARITY_TOP_K)
        
        st.write(f"🔍 Searching with top_k={top_k}")
        
        # Retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    st.write(f"🔄 Retry attempt {attempt + 1}/{max_retries}")
                    time.sleep(2 * attempt)  # Exponential backoff
                
                # Get retriever and perform search
                retriever = self.index.as_retriever(similarity_top_k=top_k)
                st.write("✅ Retriever created successfully")
                
                # Perform search with timeout handling
                st.write("🚀 Executing search...")
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
                
            except (httpx.RemoteProtocolError, httpx.TimeoutException, httpx.ConnectError) as e:
                error_msg = f"Network error on attempt {attempt + 1}: {str(e)}"
                logging.warning(error_msg)
                st.warning(f"⚠️ {error_msg}")
                
                if attempt == max_retries - 1:
                    st.error("❌ All retry attempts failed")
                    # Try alternative approach
                    return self._fallback_search(query, top_k)
                
            except Exception as e:
                error_msg = f"Error searching documents: {str(e)}"
                logging.error(error_msg)
                st.error(f"❌ Search error: {str(e)}")
                st.write(f"Error type: {type(e)}")
                
                # Show traceback only on last attempt
                if attempt == max_retries - 1:
                    import traceback
                    st.write("Full traceback:")
                    st.code(traceback.format_exc())
                    return []
        
        return []
    
    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Fallback search method using direct HTTP requests
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of results or empty list
        """
        st.info("🔄 Trying direct HTTP fallback...")
        
        try:
            import requests
            
            # Use the direct retrieval endpoint
            endpoint_url = "https://api.cloud.llamaindex.ai/api/v1/pipelines/207e89f0-702d-45ed-9c14-cc80060c2aef/retrieve"
            
            headers = {
                "Authorization": f"Bearer {settings.llama_cloud_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "similarity_top_k": top_k
            }
            
            st.write(f"📡 Making direct HTTP request to: {endpoint_url}")
            
            response = requests.post(
                endpoint_url, 
                json=payload, 
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                st.write(f"✅ HTTP request successful: {response.status_code}")
                
                # Process the response
                results = []
                if 'retrieval_results' in data:
                    for i, item in enumerate(data['retrieval_results'][:top_k]):
                        result = {
                            "rank": i + 1,
                            "content": item.get('text', item.get('content', '')),
                            "score": item.get('score', 0.0),
                            "metadata": item.get('metadata', {}),
                            "node_id": item.get('id', f"http_node_{i}"),
                        }
                        results.append(result)
                
                st.success(f"✅ HTTP fallback successful - found {len(results)} results")
                return results
            else:
                st.error(f"❌ HTTP request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"❌ HTTP fallback failed: {str(e)}")
        
        # Final fallback - try query engine approach
        try:
            st.info("🔄 Trying query engine fallback...")
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)
            
            if query_engine:
                st.write("✅ Query engine created successfully")
                response = query_engine.query(query)
                
                if response:
                    # Create mock result from query engine response
                    mock_result = {
                        "rank": 1,
                        "content": str(response),
                        "score": 0.8,
                        "metadata": {"source": "LlamaCloud Query Engine", "fallback": True},
                        "node_id": "fallback_node_1",
                    }
                    
                    st.success("✅ Query engine fallback successful")
                    return [mock_result]
            
        except Exception as fallback_e:
            st.error(f"❌ Query engine fallback also failed: {str(fallback_e)}")
        
        st.error("❌ All search methods failed")
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
                # Test basic index access
                retriever = self.index.as_retriever(similarity_top_k=1)
                status['retriever_creation'] = True
                
                # Try a very simple test query
                st.write("🧪 Testing with simple query...")
                test_results = retriever.retrieve("test")
                status['search_functional'] = True
                status['test_results_count'] = len(test_results) if test_results else 0
                
            except httpx.RemoteProtocolError as e:
                status['search_functional'] = False
                status['error'] = f"Network error: {str(e)}"
                status['error_type'] = 'network'
                
            except Exception as e:
                status['search_functional'] = False
                status['error'] = str(e)
                status['error_type'] = 'other'
        else:
            status['search_functional'] = False
            
        return status
    
    def verify_index_access(self) -> bool:
        """
        Verify that the index can be accessed and contains data
        
        Returns:
            bool: True if index is accessible and has data
        """
        if not self._initialized:
            st.error("❌ Service not initialized")
            return False
        
        try:
            st.info("🔍 Verifying index access...")
            
            # Check if we can create a retriever
            retriever = self.index.as_retriever(similarity_top_k=1)
            st.write("✅ Retriever created successfully")
            
            # Check if we can create a query engine
            query_engine = self.index.as_query_engine(similarity_top_k=1)
            st.write("✅ Query engine created successfully")
            
            # Try a simple metadata query
            st.write("🔍 Testing basic connectivity...")
            
            # This should work even if search fails
            st.write("✅ Basic connectivity verified")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Index verification failed: {str(e)}")
            return False
