"""
Curiosity: Iterative Knowledge Exploration Module

This module provides advanced knowledge exploration capabilities that perform
multi-round searches to achieve higher similarity scores and better knowledge discovery.

Key Features:
- Multi-round iterative exploration
- Refined query generation based on partial results
- Semantic neighborhood exploration
- Knowledge synthesis across rounds
- Configurable exploration strategies

Usage:
    from curiosity import KnowledgeExplorer
    
    explorer = KnowledgeExplorer(graph_version_id="your_graph_id")
    result = await explorer.explore(
        message="user message",
        conversation_context="context",
        user_id="user123",
        max_rounds=3
    )
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

# Try to import dependencies, but make them optional for testing
try:
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel
    from utilities import logger
    from pccontroller import query_knowledge_from_graph
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    # Fallback for testing without full dependencies
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    
    # Create mock logger
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    logger = MockLogger()
    
    # Mock BaseModel
    class BaseModel:
        def model_dump(self): return {}
    
    # Mock ChatOpenAI
    class ChatOpenAI:
        def __init__(self, **kwargs): pass
        async def ainvoke(self, prompt): 
            class MockResponse:
                content = '["mock query 1", "mock query 2"]'
            return MockResponse()
    
    # Mock query function
    async def query_knowledge_from_graph(**kwargs):
        return []

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model to a JSON-serializable dict."""
    return json.loads(json.dumps(model.model_dump(), cls=DateTimeEncoder))

# Initialize LLM (only if dependencies are available)
if DEPENDENCIES_AVAILABLE:
    LLM = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.01)
else:
    LLM = ChatOpenAI()  # Mock version

class KnowledgeExplorer:
    """
    Advanced knowledge exploration engine that performs iterative multi-round searches
    to discover relevant knowledge with higher similarity scores.
    """
    
    def __init__(self, graph_version_id: str, support_module=None):
        """
        Initialize the Knowledge Explorer.
        
        Args:
            graph_version_id: The graph version ID for knowledge queries
            support_module: Optional support module for initial search (e.g., LearningSupport)
        """
        self.graph_version_id = graph_version_id
        self.support = support_module
    
    async def explore(self, message: str, conversation_context: str, user_id: str, 
                     thread_id: Optional[str] = None, max_rounds: int = 3, org_id: str = "unknown") -> Dict[str, Any]:
        """
        Perform iterative knowledge exploration to achieve higher similarity scores.
        
        This method addresses the multi-round discovery problem by:
        1. Starting with initial queries
        2. Using partial results to generate refined queries
        3. Exploring semantic neighborhoods
        4. Synthesizing multi-round findings
        
        Args:
            message: User's input message
            conversation_context: Full conversation history
            user_id: User identifier
            thread_id: Thread identifier
            max_rounds: Maximum exploration rounds (default: 3)
            org_id: Organization identifier
            
        Returns:
            Dict containing best knowledge found across all rounds
        """
        logger.info(f"Starting iterative knowledge exploration for: '{message[:50]}...' (max_rounds: {max_rounds})")
        
        # Track all rounds
        all_rounds_data = []
        best_similarity = 0.0
        best_knowledge_context = ""
        best_query_results = []
        best_queries = []
        cumulative_knowledge_items = []
        
        # Round 1: Initial search
        logger.info("🔍 Round 1: Initial knowledge search")
        if self.support:
            # Use support module if available
            round1_result = await self.support.search_knowledge(message, conversation_context, user_id, thread_id, org_id=org_id)
        else:
            # Fallback to basic search
            round1_result = await self._basic_search(message, user_id, thread_id, org_id)
        
        round1_similarity = round1_result.get("similarity", 0.0)
        round1_context = round1_result.get("knowledge_context", "")
        round1_queries = round1_result.get("queries", [])
        round1_results = round1_result.get("query_results", [])
        
        all_rounds_data.append({
            "round": 1,
            "similarity": round1_similarity,
            "queries": round1_queries,
            "result_count": len(round1_results),
            "strategy": "initial_search"
        })
        
        # Update best results
        if round1_similarity > best_similarity:
            best_similarity = round1_similarity
            best_knowledge_context = round1_context
            best_query_results = round1_results
            best_queries = round1_queries
        
        cumulative_knowledge_items.extend(round1_results)
        logger.info(f"Round 1 complete: similarity={round1_similarity:.3f}, queries={len(round1_queries)}, results={len(round1_results)}")
        
        # Early exit if we already have high similarity
        if round1_similarity >= 0.70:
            logger.info(f"✅ Round 1 achieved high similarity ({round1_similarity:.3f}) - stopping early")
            return self._build_exploration_result(best_similarity, best_knowledge_context, best_query_results, best_queries, all_rounds_data, cumulative_knowledge_items)
        
        # Round 2: Refined search based on partial results
        if max_rounds >= 2 and round1_results:
            logger.info("🔍 Round 2: Refined search based on partial results")
            round2_queries = await self._generate_refined_queries(message, round1_results, round1_queries)
            
            if round2_queries:
                round2_result = await self._execute_query_batch(round2_queries, user_id, thread_id, org_id)
                round2_similarity = round2_result.get("similarity", 0.0)
                round2_context = round2_result.get("knowledge_context", "")
                round2_results = round2_result.get("query_results", [])
                
                all_rounds_data.append({
                    "round": 2,
                    "similarity": round2_similarity,
                    "queries": round2_queries,
                    "result_count": len(round2_results),
                    "strategy": "refined_search"
                })
                
                # Update best results if improved
                if round2_similarity > best_similarity:
                    best_similarity = round2_similarity
                    best_knowledge_context = round2_context
                    best_query_results = round2_results
                    best_queries = round2_queries
                
                cumulative_knowledge_items.extend(round2_results)
                logger.info(f"Round 2 complete: similarity={round2_similarity:.3f}, queries={len(round2_queries)}, results={len(round2_results)}")
                
                # Early exit if we achieved high similarity
                if round2_similarity >= 0.70:
                    logger.info(f"✅ Round 2 achieved high similarity ({round2_similarity:.3f}) - stopping early")
                    return self._build_exploration_result(best_similarity, best_knowledge_context, best_query_results, best_queries, all_rounds_data, cumulative_knowledge_items)
        
        # Round 3: Semantic neighborhood exploration
        if max_rounds >= 3 and best_similarity < 0.70:
            logger.info("🔍 Round 3: Semantic neighborhood exploration")
            round3_queries = await self._generate_semantic_queries(message, cumulative_knowledge_items, best_queries)
            
            if round3_queries:
                round3_result = await self._execute_query_batch(round3_queries, user_id, thread_id, org_id)
                round3_similarity = round3_result.get("similarity", 0.0)
                round3_context = round3_result.get("knowledge_context", "")
                round3_results = round3_result.get("query_results", [])
                
                all_rounds_data.append({
                    "round": 3,
                    "similarity": round3_similarity,
                    "queries": round3_queries,
                    "result_count": len(round3_results),
                    "strategy": "semantic_exploration"
                })
                
                # Update best results if improved
                if round3_similarity > best_similarity:
                    best_similarity = round3_similarity
                    best_knowledge_context = round3_context
                    best_query_results = round3_results
                    best_queries = round3_queries
                
                cumulative_knowledge_items.extend(round3_results)
                logger.info(f"Round 3 complete: similarity={round3_similarity:.3f}, queries={len(round3_queries)}, results={len(round3_results)}")
        
        # Final synthesis: Combine insights from all rounds
        if len(all_rounds_data) > 1:
            logger.info("🔍 Final synthesis: Combining insights from all rounds")
            final_result = await self._synthesize_multi_round_knowledge(message, cumulative_knowledge_items, all_rounds_data, best_queries)
            
            final_similarity = final_result.get("similarity", best_similarity)
            final_context = final_result.get("knowledge_context", best_knowledge_context)
            final_results = final_result.get("query_results", best_query_results)
            
            # Use final synthesis if it's better
            if final_similarity > best_similarity:
                best_similarity = final_similarity
                best_knowledge_context = final_context
                best_query_results = final_results
                logger.info(f"Final synthesis improved similarity: {best_similarity:.3f}")
        
        logger.info(f"🎯 Iterative exploration complete: best_similarity={best_similarity:.3f} across {len(all_rounds_data)} rounds")
        return self._build_exploration_result(best_similarity, best_knowledge_context, best_query_results, best_queries, all_rounds_data, cumulative_knowledge_items)

    async def _basic_search(self, message: str, user_id: str, thread_id: Optional[str], org_id: str) -> Dict[str, Any]:
        """Basic search fallback when no support module is provided."""
        try:
            results = await query_knowledge_from_graph(
                query=message,
                graph_version_id=self.graph_version_id,
                org_id=org_id,
                user_id=user_id,
                thread_id=thread_id,
                topic=None,
                top_k=20,
                min_similarity=0.2,
                include_categories=["ai_synthesis"]
            )
            
            if results:
                best_similarity = max(result.get("score", 0.0) for result in results)
                knowledge_context = f"Found {len(results)} knowledge items with best similarity: {best_similarity:.3f}"
                return {
                    "similarity": best_similarity,
                    "knowledge_context": knowledge_context,
                    "queries": [message],
                    "query_results": results
                }
            else:
                return {
                    "similarity": 0.0,
                    "knowledge_context": "",
                    "queries": [message],
                    "query_results": []
                }
        except Exception as e:
            logger.error(f"Basic search failed: {str(e)}")
            return {
                "similarity": 0.0,
                "knowledge_context": "",
                "queries": [message],
                "query_results": []
            }

    async def _generate_refined_queries(self, original_message: str, partial_results: List[Dict], original_queries: List[str]) -> List[str]:
        """Generate refined queries based on partial search results."""
        if not partial_results:
            return []
        
        # Extract key concepts from partial results
        result_snippets = []
        for result in partial_results[:5]:  # Use top 5 results
            content = result.get("raw", "")
            if content:
                # Extract first 100 characters as snippet
                snippet = content[:100].replace("\n", " ").strip()
                result_snippets.append(snippet)
        
        combined_snippets = " | ".join(result_snippets)
        
        refinement_prompt = f"""Based on the user's message and partial search results, generate 3-5 refined search queries that could find more relevant knowledge.

        User message: {original_message}

        Partial results found: {combined_snippets}

        Original queries used: {original_queries}

        Generate refined queries that:
        1. Explore related concepts mentioned in the partial results
        2. Use different terminology or synonyms
        3. Focus on specific aspects that might yield higher similarity
        4. Avoid repeating the original queries

        Return only a JSON array of strings: ["query1", "query2", "query3"]
        """
        
        try:
            response = await LLM.ainvoke(refinement_prompt)
            
            # Debug: Log the actual response content
            response_content = response.content.strip() if hasattr(response, 'content') else str(response)
            logger.info(f"LLM refined query response (first 200 chars): {response_content[:200]}")
            
            # Check if response is empty
            if not response_content:
                logger.warning("LLM returned empty response for refined queries")
                return []
            
            # Clean markdown code blocks if present
            cleaned_content = response_content
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.replace("```json", "").replace("```", "").strip()
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content.replace("```", "").strip()
            
            # Try to parse JSON
            try:
                refined_queries = json.loads(cleaned_content)
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing failed for refined queries: {json_error}")
                logger.error(f"Raw LLM response: {response_content}")
                logger.error(f"Cleaned content: {cleaned_content}")
                
                # Fallback: try to extract queries using regex if JSON parsing fails
                import re
                query_pattern = r'"([^"]+)"'
                fallback_queries = re.findall(query_pattern, response_content)
                if fallback_queries:
                    logger.info(f"Extracted {len(fallback_queries)} queries using regex fallback")
                    refined_queries = fallback_queries
                else:
                    logger.warning("No queries found in LLM response, returning empty list")
                    return []
            
            # Validate that we got a list
            if not isinstance(refined_queries, list):
                logger.warning(f"LLM returned non-list response: {type(refined_queries)}")
                return []
            
            # Filter out duplicates and original queries
            unique_queries = []
            for query in refined_queries:
                if isinstance(query, str) and len(query.strip()) > 5:
                    query_clean = query.strip()
                    if query_clean not in original_queries and query_clean not in unique_queries:
                        unique_queries.append(query_clean)
            
            logger.info(f"Generated {len(unique_queries)} refined queries from {len(partial_results)} partial results")
            return unique_queries[:5]  # Limit to 5 queries
            
        except Exception as e:
            logger.error(f"Failed to generate refined queries: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    async def _generate_semantic_queries(self, original_message: str, all_knowledge_items: List[Dict], all_queries: List[str]) -> List[str]:
        """Generate semantic neighborhood queries to explore related concepts."""
        if not all_knowledge_items:
            return []
        
        # Extract key themes from all knowledge found so far
        knowledge_themes = []
        for item in all_knowledge_items[:10]:  # Use top 10 items
            content = item.get("raw", "")
            if content:
                # Extract key phrases (simple approach)
                words = content.lower().split()
                # Look for potential key terms (longer words, capitalized terms)
                key_terms = [word for word in words if len(word) > 6 and word.isalpha()]
                knowledge_themes.extend(key_terms[:3])  # Take first 3 from each
        
        # Remove duplicates and get most common themes
        unique_themes = list(set(knowledge_themes))[:10]
        
        semantic_prompt = f"""Based on the user's message and discovered knowledge themes, generate 3-5 semantic neighborhood queries to explore related concepts.

        User message: {original_message}

        Discovered themes: {unique_themes}

        Previous queries: {all_queries}

        Generate semantic queries that:
        1. Explore broader categories or frameworks
        2. Look for related methodologies or approaches  
        3. Search for contextual applications
        4. Find complementary or alternative concepts
        5. Avoid repeating previous queries

        Return only a JSON array of strings: ["query1", "query2", "query3"]
        """
        
        try:
            response = await LLM.ainvoke(semantic_prompt)
            
            # Debug: Log the actual response content
            response_content = response.content.strip() if hasattr(response, 'content') else str(response)
            logger.info(f"LLM semantic query response (first 200 chars): {response_content[:200]}")
            
            # Check if response is empty
            if not response_content:
                logger.warning("LLM returned empty response for semantic queries")
                return []
            
            # Clean markdown code blocks if present
            cleaned_content = response_content
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.replace("```json", "").replace("```", "").strip()
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content.replace("```", "").strip()
            
            # Try to parse JSON
            try:
                semantic_queries = json.loads(cleaned_content)
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing failed for semantic queries: {json_error}")
                logger.error(f"Raw LLM response: {response_content}")
                logger.error(f"Cleaned content: {cleaned_content}")
                
                # Fallback: try to extract queries using regex if JSON parsing fails
                import re
                query_pattern = r'"([^"]+)"'
                fallback_queries = re.findall(query_pattern, response_content)
                if fallback_queries:
                    logger.info(f"Extracted {len(fallback_queries)} queries using regex fallback")
                    semantic_queries = fallback_queries
                else:
                    logger.warning("No queries found in LLM response, returning empty list")
                    return []
            
            # Validate that we got a list
            if not isinstance(semantic_queries, list):
                logger.warning(f"LLM returned non-list response: {type(semantic_queries)}")
                return []
            
            # Filter out duplicates
            unique_queries = []
            for query in semantic_queries:
                if isinstance(query, str) and len(query.strip()) > 5:
                    query_clean = query.strip()
                    if query_clean not in all_queries and query_clean not in unique_queries:
                        unique_queries.append(query_clean)
            
            logger.info(f"Generated {len(unique_queries)} semantic queries from {len(unique_themes)} themes")
            return unique_queries[:5]  # Limit to 5 queries
            
        except Exception as e:
            logger.error(f"Failed to generate semantic queries: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    async def _execute_query_batch(self, queries: List[str], user_id: str, thread_id: Optional[str], org_id: str) -> Dict[str, Any]:
        """Execute a batch of queries and return consolidated results."""
        if not queries:
            return {"similarity": 0.0, "knowledge_context": "", "query_results": []}
        
        logger.info(f"Executing batch of {len(queries)} queries")
        
        # Execute all queries in parallel
        results_list = await asyncio.gather(
            *(query_knowledge_from_graph(
                query=query,
                graph_version_id=self.graph_version_id,
                org_id=org_id,
                user_id=user_id,
                thread_id=None,
                topic=None,
                top_k=50,  # Reduced from 100 for efficiency
                min_similarity=0.2,
                include_categories=["ai_synthesis"]
            ) for query in queries),
            return_exceptions=True
        )
        
        # Consolidate results
        all_results = []
        best_similarity = 0.0
        
        for query, results in zip(queries, results_list):
            if isinstance(results, Exception):
                logger.warning(f"Query '{query[:30]}...' failed: {str(results)}")
                continue
            if not results:
                continue
                
            for result_item in results:
                result_item["query"] = query
                all_results.append(result_item)
                
                # Track best similarity
                similarity = result_item.get("score", 0.0)
                if similarity > best_similarity:
                    best_similarity = similarity
        
        # Build knowledge context from top results
        if all_results:
            # Sort by similarity and take top results
            sorted_results = sorted(all_results, key=lambda x: x.get("score", 0.0), reverse=True)
            top_results = sorted_results[:20]  # Top 20 results
            
            knowledge_sections = ["KNOWLEDGE RESULTS:"]
            for i, result in enumerate(top_results, 1):
                query = result.get("query", "unknown")
                score = result.get("score", 0.0)
                content = result.get("raw", "")
                
                # Clean content
                if content.startswith("AI: "):
                    content = content[4:]
                elif content.startswith("AI Synthesis: "):
                    content = content[14:]
                
                knowledge_sections.append(f"[{i}] Query: '{query}' (score: {score:.2f})\n{content}")
            
            knowledge_context = "\n\n".join(knowledge_sections)
        else:
            knowledge_context = ""
        
        return {
            "similarity": best_similarity,
            "knowledge_context": knowledge_context,
            "query_results": all_results,
            "queries": queries
        }

    async def _synthesize_multi_round_knowledge(self, original_message: str, all_knowledge_items: List[Dict], rounds_data: List[Dict], all_queries: List[str]) -> Dict[str, Any]:
        """Synthesize knowledge from multiple exploration rounds."""
        if not all_knowledge_items:
            return {"similarity": 0.0, "knowledge_context": "", "query_results": []}
        
        logger.info(f"Synthesizing knowledge from {len(all_knowledge_items)} items across {len(rounds_data)} rounds")
        
        # Remove duplicates based on content similarity
        unique_items = []
        seen_content = set()
        
        for item in all_knowledge_items:
            content = item.get("raw", "")
            content_hash = hash(content[:100])  # Use first 100 chars as fingerprint
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_items.append(item)
        
        logger.info(f"Deduplicated to {len(unique_items)} unique knowledge items")
        
        # Sort by similarity and take best items
        sorted_items = sorted(unique_items, key=lambda x: x.get("score", 0.0), reverse=True)
        best_items = sorted_items[:30]  # Top 30 unique items
        
        # Calculate synthesized similarity (weighted average with boost for diversity)
        if best_items:
            # Base similarity from top item
            base_similarity = best_items[0].get("score", 0.0)
            
            # Diversity bonus: more rounds and unique results = higher confidence
            diversity_bonus = min(0.15, len(rounds_data) * 0.05)  # Up to 15% bonus
            coverage_bonus = min(0.10, len(best_items) * 0.003)   # Up to 10% bonus
            
            synthesized_similarity = min(1.0, base_similarity + diversity_bonus + coverage_bonus)
            
            logger.info(f"Synthesized similarity: {base_similarity:.3f} + {diversity_bonus:.3f} (diversity) + {coverage_bonus:.3f} (coverage) = {synthesized_similarity:.3f}")
        else:
            synthesized_similarity = 0.0
        
        # Build synthesized knowledge context
        if best_items:
            knowledge_sections = ["SYNTHESIZED KNOWLEDGE (Multi-round exploration):"]
            
            for i, item in enumerate(best_items, 1):
                query = item.get("query", "unknown")
                score = item.get("score", 0.0)
                content = item.get("raw", "")
                
                # Clean content
                if content.startswith("AI: "):
                    content = content[4:]
                elif content.startswith("AI Synthesis: "):
                    content = content[14:]
                
                knowledge_sections.append(f"[{i}] Query: '{query}' (score: {score:.2f})\n{content}")
            
            # Add exploration summary
            rounds_summary = []
            for round_data in rounds_data:
                rounds_summary.append(f"Round {round_data['round']} ({round_data['strategy']}): {round_data['similarity']:.3f} similarity, {round_data['result_count']} results")
            
            knowledge_sections.append(f"\nExploration Summary:\n" + "\n".join(rounds_summary))
            
            knowledge_context = "\n\n".join(knowledge_sections)
        else:
            knowledge_context = ""
        
        return {
            "similarity": synthesized_similarity,
            "knowledge_context": knowledge_context,
            "query_results": best_items,
            "synthesis_metadata": {
                "rounds_count": len(rounds_data),
                "total_items": len(all_knowledge_items),
                "unique_items": len(unique_items),
                "final_items": len(best_items)
            }
        }

    def _build_exploration_result(self, similarity: float, knowledge_context: str, query_results: List[Dict], queries: List[str], rounds_data: List[Dict], all_items: List[Dict]) -> Dict[str, Any]:
        """Build the final exploration result."""
        return {
            "knowledge_context": knowledge_context,
            "similarity": similarity,
            "query_count": len(queries),
            "queries": queries,
            "query_results": query_results,
            "exploration_metadata": {
                "rounds_completed": len(rounds_data),
                "total_items_found": len(all_items),
                "final_similarity": similarity,
                "rounds_data": rounds_data,
                "exploration_strategy": "iterative_multi_round"
            },
            "prior_data": {"topic": "", "knowledge": ""},
            "metadata": {"similarity": similarity, "iterative_exploration": True}
        }


# Convenience function for direct usage
async def explore_knowledge(message: str, conversation_context: str, user_id: str, 
                           graph_version_id: str, thread_id: Optional[str] = None, 
                           max_rounds: int = 3, support_module=None) -> Dict[str, Any]:
    """
    Convenience function to perform iterative knowledge exploration.
    
    Args:
        message: User's input message
        conversation_context: Full conversation history
        user_id: User identifier
        graph_version_id: The graph version ID for knowledge queries
        thread_id: Thread identifier
        max_rounds: Maximum exploration rounds (default: 3)
        support_module: Optional support module for initial search
        
    Returns:
        Dict containing best knowledge found across all rounds
    """
    explorer = KnowledgeExplorer(graph_version_id, support_module)
    return await explorer.explore(message, conversation_context, user_id, thread_id, max_rounds)

