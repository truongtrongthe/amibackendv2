"""
Brain Vector Tool - Retrieves relevant brain vectors and knowledge during conversations

This tool allows Ami to access the brain vector system during conversations to:
1. Fetch relevant brain vectors based on conversation context
2. Link existing knowledge with what humans are talking about
3. Increase knowledge over conversation sessions
4. Provide context-aware responses using historical brain data
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
import json
import traceback

logger = logging.getLogger(__name__)


class BrainVectorTool:
    """Tool for accessing brain vectors and knowledge during conversations"""
    
    def __init__(self):
        """Initialize the brain vector tool"""
        self.production_index = None
        self.embeddings = None
        self.pccontroller = None
        self._initialize_brain_modules()
    
    def _initialize_brain_modules(self):
        """Initialize connections to brain modules"""
        try:
            # Import production index and utilities
            from pccontroller import get_production_index, query_knowledge
            from utilities import EMBEDDINGS
            
            self.production_index = get_production_index()
            self.embeddings = EMBEDDINGS
            self.pccontroller = {
                'query_knowledge': query_knowledge
            }
            logger.info("Brain vector modules initialized successfully - using production Pinecone index")
        except Exception as e:
            logger.error(f"Failed to initialize brain vector modules: {e}")
            logger.error(traceback.format_exc())
    
    async def fetch_relevant_vectors(
        self, 
        query: str, 
        user_id: str = "unknown", 
        org_id: str = "unknown",
        graph_version_id: str = "",
        top_k: int = 5,
        conversation_context: str = "",
        categories: List[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch relevant brain vectors based on query and conversation context
        
        Args:
            query: The search query or conversation topic
            user_id: User identifier
            org_id: Organization identifier  
            graph_version_id: Specific graph version to query (unused - for compatibility)
            top_k: Number of results to return
            conversation_context: Previous conversation context
            categories: Filter by categories (e.g. ["human_approved", "ai_synthesis"])
            
        Returns:
            Dict containing relevant vectors and metadata
        """
        try:
            if not self.production_index or not self.embeddings:
                return {
                    "success": False,
                    "error": "Brain vector system not available - production index or embeddings not initialized",
                    "query": query,
                    "results": [],
                    "metadata": {"user_id": user_id, "org_id": org_id}
                }
            
            # Prepare enhanced query using conversation context
            enhanced_query = self._enhance_query_with_context(query, conversation_context)
            
            # Use pccontroller query_knowledge for standardized access
            if self.pccontroller and self.pccontroller['query_knowledge']:
                logger.info(f"Querying brain vectors for: '{enhanced_query[:100]}...' (org: {org_id})")
                
                knowledge_results = await self.pccontroller['query_knowledge'](
                    query=enhanced_query,
                    org_id=org_id,
                    user_id=user_id,
                    top_k=top_k,
                    min_similarity=0.3  # Reasonable threshold for relevance
                )
                
                if knowledge_results:
                    # Filter by categories if specified
                    filtered_results = []
                    for result in knowledge_results:
                        if categories:
                            result_categories = result.get("categories", [])
                            if any(cat in result_categories for cat in categories):
                                filtered_results.append(result)
                        else:
                            filtered_results.append(result)
                    
                    formatted_results = self._format_knowledge_results(filtered_results, top_k)
                    return {
                        "success": True,
                        "query": query,
                        "enhanced_query": enhanced_query,
                        "namespace": f"ent-{org_id}",
                        "results": formatted_results,
                        "total_found": len(knowledge_results),
                        "total_returned": len(formatted_results),
                        "metadata": {
                            "user_id": user_id,
                            "org_id": org_id,
                            "source": "production_pinecone",
                            "categories_filter": categories
                        }
                    }
            
            # Direct Pinecone query as fallback
            logger.info(f"Using direct Pinecone query for: '{enhanced_query[:100]}...'")
            namespace = f"ent-{org_id}"
            
            # Generate embedding for the enhanced query
            embedding = await self.embeddings.aembed_query(enhanced_query)
            
            if len(embedding) != 1536:
                return {
                    "success": False,
                    "error": f"Invalid embedding dimension: {len(embedding)}",
                    "query": query,
                    "results": [],
                    "metadata": {"user_id": user_id, "org_id": org_id}
                }
            
            # Build metadata filter
            filter_dict = {}
            if categories:
                filter_dict["categories"] = {"$in": categories}
            
            # Query Pinecone directly
            results = await asyncio.to_thread(
                self.production_index.query,
                vector=embedding,
                top_k=top_k * 2,  # Get more results to allow for filtering
                include_metadata=True,
                namespace=namespace,
                filter=filter_dict if filter_dict else None
            )
            
            matches = results.get("matches", [])
            logger.info(f"Found {len(matches)} brain vector matches for query: '{query[:50]}...'")
            
            # Format and filter results
            formatted_results = []
            for match in matches[:top_k]:
                metadata = match.get("metadata", {})
                formatted_result = {
                    "id": match.get("id", "unknown"),
                    "content": metadata.get("raw", ""),
                    "title": metadata.get("title", ""),
                    "score": match.get("score", 0.0),
                    "similarity": match.get("score", 0.0),
                    "confidence": metadata.get("confidence", 0.0),
                    "created_at": metadata.get("created_at", ""),
                    "user_id": metadata.get("user_id", ""),
                    "thread_id": metadata.get("thread_id", ""),
                    "topic": metadata.get("topic", "unknown"),
                    "categories": metadata.get("categories", []),
                    "source": metadata.get("source", ""),
                    "metadata": metadata
                }
                formatted_results.append(formatted_result)
            
            return {
                "success": True,
                "query": query,
                "enhanced_query": enhanced_query,
                "namespace": namespace,
                "results": formatted_results,
                "total_found": len(matches),
                "total_returned": len(formatted_results),
                "metadata": {
                    "user_id": user_id,
                    "org_id": org_id,
                    "source": "direct_pinecone",
                    "categories_filter": categories
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching brain vectors: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": [],
                "metadata": {"user_id": user_id, "org_id": org_id}
            }
    
    def _enhance_query_with_context(self, query: str, conversation_context: str) -> str:
        """Enhance the query using conversation context for better brain vector retrieval"""
        if not conversation_context:
            return query
        
        try:
            # Extract key terms and topics from recent conversation
            import re
            
            # Get recent user messages for context
            user_messages = re.findall(r'User: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
            recent_topics = []
            
            # Get the last 3 user messages for context
            for msg in user_messages[-3:]:
                # Extract key nouns and topics (simple keyword extraction)
                words = re.findall(r'\b[A-Za-z]{3,}\b', msg.lower())
                recent_topics.extend(words[:5])  # Get top 5 words per message
            
            # Remove duplicates and common words
            stop_words = {'the', 'and', 'that', 'this', 'for', 'are', 'you', 'can', 'what', 'how', 'with'}
            unique_topics = list(set([t for t in recent_topics if t not in stop_words]))
            
            if unique_topics:
                # Enhance query with contextual keywords
                context_keywords = " ".join(unique_topics[:10])  # Limit to top 10 context words
                enhanced_query = f"{query} {context_keywords}"
                logger.info(f"Enhanced query with context: '{query}' -> '{enhanced_query[:150]}...'")
                return enhanced_query
            
        except Exception as e:
            logger.error(f"Error enhancing query with context: {e}")
        
        return query
    
    def _format_knowledge_results(self, raw_results: List[Dict], top_k: int) -> List[Dict]:
        """Format knowledge results from pccontroller for consistent output"""
        try:
            formatted = []
            for result in raw_results[:top_k]:
                formatted_result = {
                    "id": result.get("id", "unknown"),
                    "content": result.get("raw", ""),
                    "title": result.get("title", ""),
                    "score": result.get("score", 0.0),
                    "similarity": result.get("score", 0.0),  # Use score as similarity
                    "confidence": result.get("confidence", 0.0),
                    "created_at": result.get("created_at", ""),
                    "user_id": result.get("user_id", ""),
                    "thread_id": result.get("thread_id", ""),
                    "topic": result.get("topic", "unknown"),
                    "categories": result.get("categories", []),
                    "source": result.get("source", ""),
                    "metadata": result  # Include full result as metadata
                }
                formatted.append(formatted_result)
            
            logger.info(f"Formatted {len(formatted)} knowledge results")
            return formatted
        except Exception as e:
            logger.error(f"Error formatting knowledge results: {e}")
            return []
    
    async def get_contextual_knowledge(
        self, 
        topics: List[str], 
        user_id: str = "unknown",
        org_id: str = "unknown",
        graph_version_id: str = ""
    ) -> Dict[str, Any]:
        """
        Get contextual knowledge for multiple topics to build conversation awareness
        
        Args:
            topics: List of topics to search for
            user_id: User identifier
            org_id: Organization identifier
            graph_version_id: Graph version to query
            
        Returns:
            Aggregated knowledge context across topics
        """
        try:
            all_results = []
            
            for topic in topics[:5]:  # Limit to 5 topics to avoid overload
                topic_results = await self.fetch_relevant_vectors(
                    query=topic,
                    user_id=user_id,
                    org_id=org_id,
                    graph_version_id=graph_version_id,
                    top_k=3  # Get top 3 per topic
                )
                
                if topic_results.get("success") and topic_results.get("results"):
                    for result in topic_results["results"]:
                        result["topic"] = topic
                        all_results.append(result)
            
            # Sort by relevance score and deduplicate
            unique_results = self._deduplicate_results(all_results)
            sorted_results = sorted(unique_results, key=lambda x: x.get("score", 0), reverse=True)
            
            return {
                "success": True,
                "topics": topics,
                "total_results": len(sorted_results),
                "results": sorted_results[:10],  # Return top 10 overall
                "metadata": {
                    "user_id": user_id,
                    "org_id": org_id,
                    "graph_version_id": graph_version_id
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting contextual knowledge: {e}")
            return {
                "success": False,
                "error": str(e),
                "topics": topics,
                "results": []
            }
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content similarity"""
        try:
            unique_results = []
            seen_content = set()
            
            for result in results:
                content = result.get("content", "")
                # Create a simple hash of the content for deduplication
                content_hash = hash(content[:200])  # Use first 200 chars for hash
                
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(result)
            
            logger.info(f"Deduplicated {len(results)} results to {len(unique_results)} unique results")
            return unique_results
        except Exception as e:
            logger.error(f"Error deduplicating results: {e}")
            return results
    
    def get_brain_status(self) -> Dict[str, Any]:
        """Get current brain system status and capabilities"""
        try:
            status = {
                "production_index_available": bool(self.production_index),
                "embeddings_available": bool(self.embeddings),
                "pccontroller_available": bool(self.pccontroller),
                "capabilities": []
            }
            
            if self.production_index:
                status["capabilities"].extend([
                    "pinecone_vector_search",
                    "organization_namespaced_queries",
                    "metadata_filtering"
                ])
            
            if self.embeddings:
                status["capabilities"].extend([
                    "text_to_vector_embedding",
                    "semantic_similarity_search"
                ])
            
            if self.pccontroller:
                status["capabilities"].extend([
                    "standardized_knowledge_queries",
                    "production_ready_api",
                    "category_based_filtering"
                ])
            
            status["ready"] = bool(
                self.production_index and 
                self.embeddings and 
                self.pccontroller
            )
            
            return status
        except Exception as e:
            logger.error(f"Error getting brain status: {e}")
            return {"error": str(e), "available": False}

    async def query_knowledge(self, user_id: str, org_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query knowledge from the brain vector system
        
        This method provides compatibility with the skill discovery system
        by exposing the pccontroller.query_knowledge function as a class method.
        
        Args:
            user_id: User identifier
            org_id: Organization identifier  
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of knowledge results with content and metadata
        """
        try:
            if not self.pccontroller or 'query_knowledge' not in self.pccontroller:
                logger.warning("BrainVectorTool not properly initialized - pccontroller missing")
                return []
                
            # Call the actual async query_knowledge function
            results = await self.pccontroller['query_knowledge'](
                user_id=user_id,
                org_id=org_id,
                query=query,
                limit=limit
            )
            
            # Ensure we return a list
            if isinstance(results, dict):
                return results.get('results', [])
            elif isinstance(results, list):
                return results
            else:
                logger.warning(f"Unexpected query_knowledge response type: {type(results)}")
                return []
                
        except Exception as e:
            logger.error(f"Error in query_knowledge: {e}")
            logger.error(traceback.format_exc())
            return []


# Tool interface functions for LLM integration
def search_brain_knowledge(query: str, user_id: str = "unknown", org_id: str = "unknown", conversation_context: str = "") -> str:
    """
    Search brain knowledge vectors for relevant information
    
    Args:
        query: Search query or topic
        user_id: User identifier 
        org_id: Organization identifier
        conversation_context: Previous conversation for context
        
    Returns:
        Formatted knowledge results as string
    """
    try:
        # Create tool instance
        brain_tool = BrainVectorTool()
        
        # Run async function synchronously (for LLM tool compatibility)
        import asyncio
        try:
            # Try to get existing event loop
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we need to use a different approach
            future = asyncio.ensure_future(
                brain_tool.fetch_relevant_vectors(
                    query=query,
                    user_id=user_id,
                    org_id=org_id,
                    conversation_context=conversation_context
                )
            )
            # This is a workaround for being called from within an async context
            result = None
            while not future.done():
                continue
            result = future.result()
        except RuntimeError:
            # No event loop running, create a new one
            result = asyncio.run(
                brain_tool.fetch_relevant_vectors(
                    query=query,
                    user_id=user_id,
                    org_id=org_id,
                    conversation_context=conversation_context
                )
            )
        
        if result.get("success") and result.get("results"):
            # Format results for LLM consumption
            formatted_knowledge = []
            for item in result["results"][:5]:  # Limit to top 5 results
                content = item.get("content", "")[:300]  # Truncate long content
                score = item.get("score", 0)
                formatted_knowledge.append(f"[Score: {score:.2f}] {content}")
            
            knowledge_text = "\n\n".join(formatted_knowledge)
            return f"Found {len(result['results'])} relevant knowledge entries:\n\n{knowledge_text}"
        
        elif result.get("error"):
            return f"Error searching brain knowledge: {result['error']}"
        
        else:
            return f"No relevant brain knowledge found for query: {query}"
    
    except Exception as e:
        logger.error(f"Error in search_brain_knowledge tool function: {e}")
        return f"Error accessing brain knowledge: {str(e)}"


def get_brain_context(topics: List[str], user_id: str = "unknown", org_id: str = "unknown") -> str:
    """
    Get brain context for multiple conversation topics
    
    Args:
        topics: List of topics to get context for
        user_id: User identifier
        org_id: Organization identifier
        
    Returns:
        Formatted contextual knowledge as string
    """
    try:
        brain_tool = BrainVectorTool()
        
        # Run async function synchronously
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.ensure_future(
                brain_tool.get_contextual_knowledge(
                    topics=topics,
                    user_id=user_id,
                    org_id=org_id
                )
            )
            while not future.done():
                continue
            result = future.result()
        except RuntimeError:
            result = asyncio.run(
                brain_tool.get_contextual_knowledge(
                    topics=topics,
                    user_id=user_id,
                    org_id=org_id
                )
            )
        
        if result.get("success") and result.get("results"):
            context_summary = []
            for item in result["results"][:3]:  # Top 3 contextual items
                topic = item.get("topic", "unknown")
                content = item.get("content", "")[:200]
                context_summary.append(f"**{topic.upper()}**: {content}")
            
            return f"Brain context for topics {topics}:\n\n" + "\n\n".join(context_summary)
        
        return f"No brain context found for topics: {topics}"
    
    except Exception as e:
        logger.error(f"Error in get_brain_context tool function: {e}")
        return f"Error getting brain context: {str(e)}" 