"""
AI Brain Module - Knowledge Aggregation and Summarization

This module provides functions to:
1. Fetch all AI synthesis vectors from Pinecone indexes
2. Summarize and analyze existing knowledge base
3. Generate comprehensive knowledge insights

Key Features:
- Bulk AI synthesis vector retrieval
- Knowledge base summarization using LLM
- Statistical analysis of knowledge distribution
- Category-based knowledge organization
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import defaultdict, Counter
import traceback

from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from utilities import logger, EMBEDDINGS
from pccontroller import get_org_index
import os
import re

# FastAPI imports for API endpoints
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal

# Initialize LLM for summarization
SUMMARIZATION_LLM = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=4000)

# FastAPI router for brain endpoints
router = APIRouter()

# Request models for API endpoints
class BrainPreviewRequest(BaseModel):
    namespace: str = "conversation"
    max_vectors: Optional[int] = 1000
    summary_type: Literal["comprehensive", "topics_only", "categories_only"] = "comprehensive"
    include_vectors: bool = False
    max_content_length: int = 50000
    org_id: str = "unknown"  # Add org_id field with default value

class PickupLineRequest(BaseModel):
    namespace: str = "conversation"
    max_vectors: Optional[int] = 500
    summary_type: Literal["comprehensive", "topics_only", "categories_only"] = "comprehensive"
    max_content_length: int = 30000
    pickup_style: Literal["witty", "charming", "nerdy", "confident", "humorous"] = "charming"
    org_id: str = "unknown"  # Add org_id field with default value

class BrainVectorsRequest(BaseModel):
    namespace: str = "conversation"
    max_vectors: Optional[int] = 1000
    batch_size: int = 100
    include_metadata: bool = True
    include_content: bool = True
    content_preview_length: int = 300
    org_id: str = "unknown"  # Add org_id field with default value

class AIBrainAnalyzer:
    """Analyzer for AI synthesis knowledge and comprehensive knowledge summarization."""
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self.index = get_org_index(org_id)
        
    async def fetch_all_ai_synthesis_vectors(
        self, 
        namespace: str = "conversation",
        batch_size: int = 100,
        max_vectors: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fetch ALL vectors with ai_synthesis category from the specified namespace.
        
        Args:
            namespace: Pinecone namespace to search (default: "conversation")
            batch_size: Number of vectors to fetch per batch
            max_vectors: Maximum number of vectors to fetch (None = all)
            
        Returns:
            Dict containing:
                - vectors: List of all ai_synthesis vectors
                - metadata: Summary statistics
                - total_count: Total number of vectors found
        """
        logger.info(f"Starting to fetch all ai_synthesis vectors from namespace '{namespace}'")
        
        try:
            # Since Pinecone doesn't support querying by metadata alone without a vector,
            # we'll use a dummy embedding and set top_k very high with metadata filtering
            dummy_query = "ai synthesis knowledge summary"
            embedding = await EMBEDDINGS.aembed_query(dummy_query)
            
            if len(embedding) != 1536:
                logger.error(f"Invalid embedding dimension: {len(embedding)}")
                return {"success": False, "error": "Invalid embedding dimension"}
            
            ai_synthesis_vectors = []
            total_fetched = 0
            
            # Use a high top_k to get as many results as possible
            # We'll filter by ai_synthesis category in the metadata filter
            query_top_k = min(10000, max_vectors) if max_vectors else 10000
            
            logger.info(f"Querying Pinecone with top_k={query_top_k} and ai_synthesis filter")
            
            # Query with metadata filter for ai_synthesis category
            results = await asyncio.to_thread(
                self.index.query,
                vector=embedding,
                top_k=query_top_k,
                include_metadata=True,
                namespace=namespace,
                filter={
                    "categories": {"$in": ["ai_synthesis"]}
                }
            )
            
            matches = results.get("matches", [])
            logger.info(f"Found {len(matches)} vectors with ai_synthesis category")
            
            # Process all matches
            for match in matches:
                if max_vectors and len(ai_synthesis_vectors) >= max_vectors:
                    break
                    
                metadata = match.get("metadata", {})
                categories = metadata.get("categories", [])
                
                # Double-check that ai_synthesis is in categories
                if "ai_synthesis" in categories:
                    vector_data = {
                        "id": match["id"],
                        "score": match.get("score", 0.0),
                        "raw": metadata.get("raw", ""),
                        "title": metadata.get("title", ""),
                        "created_at": metadata.get("created_at", ""),
                        "expires_at": metadata.get("expires_at"),
                        "confidence": metadata.get("confidence", 0.0),
                        "source": metadata.get("source", ""),
                        "user_id": metadata.get("user_id", ""),
                        "thread_id": metadata.get("thread_id", ""),
                        "topic": metadata.get("topic", "unknown"),
                        "categories": categories,
                        "metadata": metadata
                    }
                    ai_synthesis_vectors.append(vector_data)
                    total_fetched += 1
            
            # Generate statistics
            stats = self._generate_vector_statistics(ai_synthesis_vectors)
            
            logger.info(f"Successfully fetched {total_fetched} ai_synthesis vectors")
            
            return {
                "success": True,
                "vectors": ai_synthesis_vectors,
                "total_count": total_fetched,
                "namespace": namespace,
                "query_params": {
                    "batch_size": batch_size,
                    "max_vectors": max_vectors,
                    "actual_fetched": total_fetched
                },
                "statistics": stats,
                "fetched_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching ai_synthesis vectors: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "namespace": namespace,
                "total_count": 0,
                "vectors": []
            }
    
    def _generate_vector_statistics(self, vectors: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive statistics for the fetched vectors."""
        if not vectors:
            return {}
            
        # Basic counts
        total_vectors = len(vectors)
        
        # Category analysis
        all_categories = []
        for vector in vectors:
            all_categories.extend(vector.get("categories", []))
        category_counts = Counter(all_categories)
        
        # Topic analysis
        topics = [v.get("topic", "unknown") for v in vectors]
        topic_counts = Counter(topics)
        
        # User analysis
        users = [v.get("user_id", "unknown") for v in vectors]
        user_counts = Counter(users)
        
        # Thread analysis
        threads = [v.get("thread_id", "") for v in vectors if v.get("thread_id")]
        thread_counts = Counter(threads)
        
        # Content length analysis
        content_lengths = [len(v.get("raw", "")) for v in vectors]
        avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        
        # Confidence analysis
        confidences = [v.get("confidence", 0.0) for v in vectors]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Date analysis
        dates = []
        for vector in vectors:
            created_at = vector.get("created_at")
            if created_at:
                try:
                    dates.append(datetime.fromisoformat(created_at.replace('Z', '+00:00')))
                except:
                    continue
        
        date_range = {}
        if dates:
            dates.sort()
            date_range = {
                "earliest": dates[0].isoformat(),
                "latest": dates[-1].isoformat(),
                "span_days": (dates[-1] - dates[0]).days
            }
        
        return {
            "total_vectors": total_vectors,
            "categories": {
                "unique_count": len(category_counts),
                "top_10": dict(category_counts.most_common(10)),
                "total_category_instances": sum(category_counts.values())
            },
            "topics": {
                "unique_count": len(topic_counts),
                "top_10": dict(topic_counts.most_common(10))
            },
            "users": {
                "unique_count": len(user_counts),
                "top_10": dict(user_counts.most_common(10))
            },
            "threads": {
                "unique_count": len(thread_counts),
                "total_threads_with_ai_synthesis": len(threads)
            },
            "content": {
                "average_length": round(avg_content_length, 2),
                "min_length": min(content_lengths) if content_lengths else 0,
                "max_length": max(content_lengths) if content_lengths else 0
            },
            "confidence": {
                "average": round(avg_confidence, 3),
                "min": min(confidences) if confidences else 0,
                "max": max(confidences) if confidences else 0
            },
            "dates": date_range
        }

    async def summarize_all_knowledge(
        self,
        vectors: Optional[List[Dict]] = None,
        namespace: str = "conversation",
        max_content_length: int = 50000,
        summary_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of all existing knowledge.
        
        Args:
            vectors: Pre-fetched vectors (if None, will fetch ai_synthesis vectors)
            namespace: Pinecone namespace to analyze
            max_content_length: Maximum characters to include in LLM analysis
            summary_type: Type of summary ("comprehensive", "topics_only", "categories_only")
            
        Returns:
            Dict containing comprehensive knowledge summary
        """
        logger.info(f"Starting comprehensive knowledge summarization for namespace '{namespace}'")
        
        try:
            # Fetch vectors if not provided
            if vectors is None:
                logger.info("Fetching ai_synthesis vectors for summarization")
                fetch_result = await self.fetch_all_ai_synthesis_vectors(namespace=namespace)
                if not fetch_result.get("success"):
                    return {
                        "success": False,
                        "error": f"Failed to fetch vectors: {fetch_result.get('error')}"
                    }
                vectors = fetch_result.get("vectors", [])
                base_statistics = fetch_result.get("statistics", {})
            else:
                base_statistics = self._generate_vector_statistics(vectors)
            
            if not vectors:
                return {
                    "success": False,
                    "error": "No vectors available for summarization"
                }
            
            logger.info(f"Summarizing {len(vectors)} ai_synthesis vectors")
            
            # Organize content for LLM analysis
            content_summary = self._organize_content_for_llm(vectors, max_content_length)
            
            # Generate LLM-based knowledge summary
            llm_summary = await self._generate_llm_summary(content_summary, summary_type)
            
            # Combine all analysis
            comprehensive_summary = {
                "success": True,
                "namespace": namespace,
                "analysis_timestamp": datetime.now().isoformat(),
                "vector_count": len(vectors),
                "summary_type": summary_type,
                "statistics": base_statistics,
                "content_organization": content_summary["organization"],
                "llm_analysis": llm_summary,
                "processing_info": {
                    "max_content_length": max_content_length,
                    "actual_content_length": content_summary["total_length"],
                    "truncated": content_summary["truncated"]
                }
            }
            
            logger.info(f"Successfully generated comprehensive knowledge summary")
            return comprehensive_summary
            
        except Exception as e:
            logger.error(f"Error generating knowledge summary: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "namespace": namespace
            }
    
    def _organize_content_for_llm(self, vectors: List[Dict], max_length: int) -> Dict[str, Any]:
        """Organize vector content for efficient LLM analysis."""
        
        # Group by categories and topics
        category_groups = defaultdict(list)
        topic_groups = defaultdict(list)
        
        for vector in vectors:
            categories = vector.get("categories", [])
            topic = vector.get("topic", "unknown")
            content = vector.get("raw", "").strip()
            title = vector.get("title", "")
            
            # Add to category groups
            for category in categories:
                if category != "ai_synthesis":  # Skip the ai_synthesis category itself
                    category_groups[category].append({
                        "content": content,
                        "title": title,
                        "id": vector["id"],
                        "created_at": vector.get("created_at", "")
                    })
            
            # Add to topic groups
            topic_groups[topic].append({
                "content": content,
                "title": title,
                "id": vector["id"],
                "created_at": vector.get("created_at", "")
            })
        
        # Build organized content string
        organized_content = []
        current_length = 0
        truncated = False
        
        # Add category-based organization
        organized_content.append("=== KNOWLEDGE BY CATEGORIES ===\n")
        
        for category, items in sorted(category_groups.items()):
            section = f"\n--- {category.upper()} ({len(items)} items) ---\n"
            
            for item in items[:10]:  # Limit to top 10 per category
                entry = f"• {item['title'] or 'Untitled'}\n  {item['content'][:200]}...\n"
                
                if current_length + len(section) + len(entry) > max_length:
                    truncated = True
                    break
                    
                section += entry
            
            if current_length + len(section) > max_length:
                truncated = True
                break
                
            organized_content.append(section)
            current_length += len(section)
        
        # Add topic-based organization if space allows
        if not truncated and current_length < max_length * 0.7:
            organized_content.append("\n\n=== KNOWLEDGE BY TOPICS ===\n")
            
            for topic, items in sorted(topic_groups.items()):
                if topic == "unknown":
                    continue
                    
                section = f"\n--- {topic.upper()} ({len(items)} items) ---\n"
                
                for item in items[:5]:  # Limit to top 5 per topic
                    entry = f"• {item['title'] or 'Untitled'}\n  {item['content'][:150]}...\n"
                    
                    if current_length + len(section) + len(entry) > max_length:
                        truncated = True
                        break
                        
                    section += entry
                
                if current_length + len(section) > max_length:
                    truncated = True
                    break
                    
                organized_content.append(section)
                current_length += len(section)
        
        final_content = "".join(organized_content)
        
        return {
            "content": final_content,
            "total_length": len(final_content),
            "truncated": truncated,
            "organization": {
                "categories_count": len(category_groups),
                "topics_count": len(topic_groups),
                "content_sections": len(organized_content)
            },
            "detected_language": self._detect_content_language_fallback(final_content)
        }
    

    def _detect_content_language_fallback(self, content: str) -> Dict[str, Any]:
        """Fallback language detection method."""
        try:
            # Simple language detection based on character patterns
            # This is a basic fallback - could be enhanced with a proper language detection library
            
            # Check for common Vietnamese characters
            vietnamese_chars = set('àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựýỳỷỹỵ')
            content_lower = content.lower()
            vietnamese_count = sum(1 for char in content_lower if char in vietnamese_chars)
            vietnamese_ratio = vietnamese_count / len(content) if content else 0
            
            if vietnamese_ratio > 0.05:  # If more than 5% Vietnamese characters
                return {
                    "language": "vi",
                    "language_name": "Vietnamese", 
                    "confidence": min(0.9, vietnamese_ratio * 10),
                    "mixed_languages": vietnamese_ratio < 0.5,
                    "detected_by": "character_analysis"
                }
            else:
                return {
                    "language": "en",
                    "language_name": "English",
                    "confidence": 0.7,
                    "mixed_languages": False,
                    "detected_by": "fallback_default"
                }
                
        except Exception as e:
            logger.error(f"Error in fallback language detection: {str(e)}")
            return {
                "language": "en",
                "language_name": "English", 
                "confidence": 0.5,
                "mixed_languages": False,
                "detected_by": "error_fallback",
                "error": str(e)
            }


    async def _generate_llm_summary(self, content_summary: Dict[str, Any], summary_type: str) -> Dict[str, Any]:
        """Generate LLM-based analysis of the organized content with language detection."""
        
        content = content_summary["content"]
        
        # Base language detection instruction
        language_instruction = """
        
        IMPORTANT: First, detect the primary language of the knowledge base content and provide your analysis in that same language. Include language detection information at the beginning of your response in this format:
        
        **LANGUAGE DETECTION:**
        - Primary Language: [Language Name] ([language_code])
        - Confidence: [0.0-1.0]
        - Mixed Languages: [Yes/No]
        
        Then provide your analysis in the detected language."""
        
        if summary_type == "comprehensive":
            prompt = f"""Analyze this comprehensive knowledge base and provide a detailed summary.

                    KNOWLEDGE BASE CONTENT:
                    {content}

                    {language_instruction}

                    Please provide a comprehensive analysis covering:

                    1. **MAIN KNOWLEDGE DOMAINS**: What are the primary areas of knowledge represented?

                    2. **KEY THEMES & PATTERNS**: What common themes, patterns, or relationships do you see?

                    3. **KNOWLEDGE DEPTH**: Assess the depth and breadth of knowledge in different areas.

                    4. **CONTENT QUALITY**: Evaluate the overall quality and usefulness of the knowledge.

                    5. **KNOWLEDGE GAPS**: What areas might be missing or underrepresented?

                    6. **PRACTICAL APPLICATIONS**: How could this knowledge be best utilized?

                    7. **ORGANIZATION INSIGHTS**: How well is the knowledge organized and categorized?

                    8. **RECOMMENDATIONS**: Suggestions for improving or expanding this knowledge base.

                    Provide a detailed but structured response. Be specific about what you observe in the content."""

        elif summary_type == "topics_only":
            prompt = f"""Analyze the topics and subject areas in this knowledge base.

                    KNOWLEDGE BASE CONTENT:
                    {content}

                    {language_instruction}

                    Focus specifically on:
                    1. Main topic areas and their coverage
                    2. Relationships between different topics
                    3. Topic depth and breadth
                    4. Missing or underrepresented topics

                    Provide a concise topic-focused analysis."""

        else:  # categories_only
            prompt = f"""Analyze the categorization and organization of this knowledge base.

                    KNOWLEDGE BASE CONTENT:
                    {content}

                    {language_instruction}

                    Focus specifically on:
                    1. Category structure and effectiveness
                    2. Content distribution across categories
                    3. Category overlap and relationships
                    4. Suggestions for better categorization

                    Provide a category-focused analysis."""

        try:
            logger.info(f"Generating LLM summary with language detection for {len(content)} characters of content")
            
            response = await SUMMARIZATION_LLM.ainvoke(prompt)
            analysis_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract language information from the response if present
            language_info = self._extract_language_info_from_response(analysis_text)
            
            return {
                "success": True,
                "analysis": analysis_text,
                "summary_type": summary_type,
                "content_length_analyzed": len(content),
                "language_detection": language_info,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "summary_type": summary_type
            }
    
    def _extract_language_info_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract language detection information from the LLM response."""
        try:
            # Look for the language detection section
            import re
            
            # Pattern to match the language detection section
            pattern = r'\*\*LANGUAGE DETECTION:\*\*\s*\n?.*?Primary Language: ([^(]+)\(([^)]+)\).*?Confidence: ([0-9.]+).*?Mixed Languages: (Yes|No)'
            
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                language_name = match.group(1).strip()
                language_code = match.group(2).strip()
                confidence = float(match.group(3))
                mixed_languages = match.group(4).lower() == 'yes'
                
                return {
                    "language": language_code,
                    "language_name": language_name,
                    "confidence": confidence,
                    "mixed_languages": mixed_languages,
                    "detected_by": "llm_integrated"
                }
            else:
                logger.warning("Could not extract language information from LLM response")
                return {
                    "language": "en",
                    "language_name": "English",
                    "confidence": 0.5,
                    "mixed_languages": False,
                    "detected_by": "fallback"
                }
                
        except Exception as e:
            logger.error(f"Error extracting language info: {str(e)}")
            return {
                "language": "en",
                "language_name": "English",
                "confidence": 0.5,
                "mixed_languages": False,
                "detected_by": "error_fallback",
                "error": str(e)
            }


# Convenience functions for direct usage
async def fetch_all_ai_synthesis_vectors(namespace: str = "conversation", max_vectors: Optional[int] = None) -> Dict[str, Any]:
    """Convenience function to fetch all AI synthesis vectors."""
    analyzer = AIBrainAnalyzer()
    return await analyzer.fetch_all_ai_synthesis_vectors(namespace=namespace, max_vectors=max_vectors)


async def summarize_knowledge_base(namespace: str = "conversation", summary_type: str = "comprehensive") -> Dict[str, Any]:
    """Convenience function to summarize all knowledge in the knowledge base."""
    analyzer = AIBrainAnalyzer()
    return await analyzer.summarize_all_knowledge(namespace=namespace, summary_type=summary_type)


# FastAPI endpoint for brain preview
@router.post('/brain-preview')
async def brain_preview_endpoint(request: BrainPreviewRequest):
    """
    Generate a comprehensive preview of the AI knowledge base.
    
    This endpoint fetches all AI synthesis vectors and generates insights about the knowledge base.
    
    Args:
        request: BrainPreviewRequest containing preview parameters including org_id
        
    Returns:
        Comprehensive brain analysis including statistics, insights, and optional vectors
    """
    start_time = datetime.now()
    request_id = f"brain_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"[{request_id}] === BEGIN brain-preview request ===")
    logger.info(f"[{request_id}] Parameters: org_id={request.org_id}, namespace={request.namespace}, max_vectors={request.max_vectors}, summary_type={request.summary_type}")
    
    try:
        # Initialize the analyzer
        analyzer = AIBrainAnalyzer(request.org_id)
        
        # Fetch AI synthesis vectors
        logger.info(f"[{request_id}] Fetching AI synthesis vectors...")
        vectors_result = await analyzer.fetch_all_ai_synthesis_vectors(
            namespace=request.namespace,
            max_vectors=request.max_vectors
        )
        
        if not vectors_result.get("success"):
            logger.error(f"[{request_id}] Failed to fetch vectors: {vectors_result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch AI synthesis vectors: {vectors_result.get('error')}"
            )
        
        vectors = vectors_result.get("vectors", [])
        total_count = vectors_result.get("total_count", 0)
        
        logger.info(f"[{request_id}] Found {total_count} AI synthesis vectors")
        
        # Generate knowledge summary if we have vectors
        summary_result = None
        if vectors:
            logger.info(f"[{request_id}] Generating knowledge summary...")
            summary_result = await analyzer.summarize_all_knowledge(
                vectors=vectors,
                namespace=request.namespace,
                max_content_length=request.max_content_length,
                summary_type=request.summary_type
            )
            
            if not summary_result.get("success"):
                logger.warning(f"[{request_id}] Failed to generate summary: {summary_result.get('error')}")
                # Don't fail the whole request, just note the summary failure
        
        # Prepare response data
        response_data = {
            "success": True,
            "request_id": request_id,
            "namespace": request.namespace,
            "generated_at": datetime.now().isoformat(),
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "vector_analysis": {
                "total_ai_synthesis_vectors": total_count,
                "vectors_analyzed": len(vectors),
                "statistics": vectors_result.get("statistics", {}),
                "query_params": vectors_result.get("query_params", {})
            }
        }
        
        # Add summary if available
        if summary_result and summary_result.get("success"):
            response_data["knowledge_summary"] = {
                "summary_type": request.summary_type,
                "analysis": summary_result.get("llm_analysis", {}),
                "content_organization": summary_result.get("content_organization", {}),
                "processing_info": summary_result.get("processing_info", {})
            }
        elif summary_result:
            response_data["knowledge_summary"] = {
                "error": summary_result.get("error"),
                "summary_type": request.summary_type
            }
        
        # Include vectors if requested (be careful with large responses)
        if request.include_vectors and vectors:
            # Limit vector details to avoid huge responses
            limited_vectors = []
            for vector in vectors[:100]:  # Limit to first 100 vectors
                limited_vector = {
                    "id": vector["id"],
                    "title": vector.get("title", ""),
                    "created_at": vector.get("created_at", ""),
                    "confidence": vector.get("confidence", 0.0),
                    "categories": vector.get("categories", []),
                    "topic": vector.get("topic", "unknown"),
                    "content_preview": vector.get("raw", "")[:300] + "..." if len(vector.get("raw", "")) > 300 else vector.get("raw", "")
                }
                limited_vectors.append(limited_vector)
            
            response_data["vectors"] = {
                "count": len(limited_vectors),
                "total_available": len(vectors),
                "vectors": limited_vectors
            }
        
        # Add insights and recommendations
        if vectors:
            stats = vectors_result.get("statistics", {})
            response_data["insights"] = {
                "knowledge_health": {
                    "total_vectors": total_count,
                    "average_content_length": stats.get("content", {}).get("average_length", 0),
                    "average_confidence": stats.get("confidence", {}).get("average", 0),
                    "category_diversity": stats.get("categories", {}).get("unique_count", 0),
                    "topic_diversity": stats.get("topics", {}).get("unique_count", 0)
                },
                "recommendations": _generate_recommendations(stats, total_count)
            }
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{request_id}] Successfully generated brain preview in {elapsed:.2f}s")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        logger.error(f"[{request_id}] Error in brain-preview endpoint: {error_msg}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {error_msg}"
        )
    finally:
        logger.info(f"[{request_id}] === END brain-preview request ===")


def _generate_recommendations(stats: Dict[str, Any], total_count: int) -> List[str]:
    """Generate actionable recommendations based on knowledge base statistics."""
    recommendations = []
    
    # Content quality recommendations
    avg_confidence = stats.get("confidence", {}).get("average", 0)
    if avg_confidence < 0.7:
        recommendations.append("Consider improving AI confidence by providing more context in conversations")
    
    # Content length recommendations
    avg_length = stats.get("content", {}).get("average_length", 0)
    if avg_length < 500:
        recommendations.append("Knowledge entries are relatively short - consider encouraging more detailed explanations")
    elif avg_length > 2000:
        recommendations.append("Knowledge entries are quite long - consider breaking down complex topics")
    
    # Category diversity recommendations
    category_count = stats.get("categories", {}).get("unique_count", 0)
    if category_count < 5:
        recommendations.append("Limited category diversity - encourage discussions across more topic areas")
    elif category_count > 50:
        recommendations.append("High category diversity - consider consolidating similar categories")
    
    # Volume recommendations
    if total_count < 100:
        recommendations.append("Building knowledge base - continue engaging with the AI to accumulate more insights")
    elif total_count > 10000:
        recommendations.append("Large knowledge base - consider periodic cleanup of outdated information")
    
    # Date span recommendations
    date_info = stats.get("dates", {})
    if date_info:
        span_days = date_info.get("span_days", 0)
        if span_days > 365:
            recommendations.append("Knowledge spans over a year - consider archiving very old information")
    
    if not recommendations:
        recommendations.append("Knowledge base appears healthy - maintain current engagement patterns")
    
    return recommendations


@router.options('/brain-preview')
async def brain_preview_options():
    """Handle OPTIONS requests for brain-preview endpoint."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

# FastAPI endpoint for fetching brain vectors
@router.post('/brain-vectors')
async def brain_vectors_endpoint(request: BrainVectorsRequest):
    """
    Fetch all AI synthesis vectors from the knowledge base.
    
    This endpoint returns all vectors with ai_synthesis category from the specified namespace,
    with optional content and metadata filtering for frontend display.
    
    Args:
        request: BrainVectorsRequest containing fetch parameters including org_id
        
    Returns:
        List of AI synthesis vectors with statistics and metadata
    """
    start_time = datetime.now()
    request_id = f"vectors_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"[{request_id}] === BEGIN brain-vectors request ===")
    logger.info(f"[{request_id}] Parameters: org_id={request.org_id}, namespace={request.namespace}, max_vectors={request.max_vectors}")
    
    try:
        # Initialize the analyzer
        analyzer = AIBrainAnalyzer(request.org_id)
        
        # Fetch AI synthesis vectors
        logger.info(f"[{request_id}] Fetching AI synthesis vectors...")
        vectors_result = await analyzer.fetch_all_ai_synthesis_vectors(
            namespace=request.namespace,
            batch_size=request.batch_size,
            max_vectors=request.max_vectors
        )
        
        if not vectors_result.get("success"):
            logger.error(f"[{request_id}] Failed to fetch vectors: {vectors_result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch AI synthesis vectors: {vectors_result.get('error')}"
            )
        
        vectors = vectors_result.get("vectors", [])
        total_count = vectors_result.get("total_count", 0)
        
        logger.info(f"[{request_id}] Found {total_count} AI synthesis vectors")
        
        # Format vectors for frontend consumption
        formatted_vectors = []
        for vector in vectors:
            formatted_vector = {
                "id": vector["id"],
                "title": vector.get("title", ""),
                "created_at": vector.get("created_at", ""),
                "confidence": vector.get("confidence", 0.0),
                "source": vector.get("source", ""),
                "user_id": vector.get("user_id", ""),
                "thread_id": vector.get("thread_id", ""),
                "topic": vector.get("topic", "unknown"),
                "categories": vector.get("categories", []),
                "score": vector.get("score", 0.0)
            }
            
            # Add content preview if requested
            if request.include_content:
                raw_content = vector.get("raw", "")
                if request.content_preview_length > 0 and len(raw_content) > request.content_preview_length:
                    formatted_vector["content_preview"] = raw_content[:request.content_preview_length] + "..."
                    formatted_vector["content_truncated"] = True
                else:
                    formatted_vector["content_preview"] = raw_content
                    formatted_vector["content_truncated"] = False
                formatted_vector["content_length"] = len(raw_content)
            
            # Add full metadata if requested
            if request.include_metadata:
                formatted_vector["metadata"] = vector.get("metadata", {})
            
            formatted_vectors.append(formatted_vector)
        
        # Prepare response data
        response_data = {
            "success": True,
            "request_id": request_id,
            "namespace": request.namespace,
            "generated_at": datetime.now().isoformat(),
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "vectors": {
                "total_count": total_count,
                "returned_count": len(formatted_vectors),
                "data": formatted_vectors
            },
            "statistics": vectors_result.get("statistics", {}),
            "query_params": {
                "namespace": request.namespace,
                "max_vectors": request.max_vectors,
                "batch_size": request.batch_size,
                "include_metadata": request.include_metadata,
                "include_content": request.include_content,
                "content_preview_length": request.content_preview_length
            }
        }
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{request_id}] Successfully fetched {len(formatted_vectors)} brain vectors in {elapsed:.2f}s")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        logger.error(f"[{request_id}] Error in brain-vectors endpoint: {error_msg}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {error_msg}"
        )
    finally:
        logger.info(f"[{request_id}] === END brain-vectors request ===")


@router.options('/brain-vectors')
async def brain_vectors_options():
    """Handle OPTIONS requests for brain-vectors endpoint."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

@router.post('/pickup-line')
async def pickup_line_endpoint(request: PickupLineRequest):
    """
    Generate a personalized pickup line based on AI knowledge base recommendations.
    
    This endpoint analyzes the knowledge base, extracts recommendations, and creates
    a creative pickup line based on the user's knowledge patterns and preferences.
    
    Args:
        request: PickupLineRequest containing parameters for pickup line generation
        
    Returns:
        A response containing the generated pickup line and context
    """
    start_time = datetime.now()
    request_id = f"pickup_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"[{request_id}] === BEGIN pickup-line request ===")
    logger.info(f"[{request_id}] Parameters: org_id={request.org_id}, namespace={request.namespace}, pickup_style={request.pickup_style}")
    
    try:
        # Initialize the analyzer
        analyzer = AIBrainAnalyzer(request.org_id)
        
        # Fetch AI synthesis vectors (smaller set for faster processing)
        logger.info(f"[{request_id}] Fetching AI synthesis vectors for pickup line generation...")
        vectors_result = await analyzer.fetch_all_ai_synthesis_vectors(
            namespace=request.namespace,
            max_vectors=request.max_vectors
        )
        
        if not vectors_result.get("success"):
            logger.error(f"[{request_id}] Failed to fetch vectors: {vectors_result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch AI synthesis vectors: {vectors_result.get('error')}"
            )
        
        vectors = vectors_result.get("vectors", [])
        total_count = vectors_result.get("total_count", 0)
        
        logger.info(f"[{request_id}] Found {total_count} AI synthesis vectors")
        
        if not vectors:
            # No knowledge base content, generate a generic pickup line
            logger.info(f"[{request_id}] No vectors found, generating generic pickup line")
            pickup_line = await _generate_generic_pickup_line(request.pickup_style)
            
            return {
                "success": True,
                "request_id": request_id,
                "pickup_line": pickup_line,
                "style": request.pickup_style,
                "based_on": "generic_template",
                "knowledge_base_analyzed": "a fresh learning journey",
                "generated_at": datetime.now().isoformat(),
                "processing_time_seconds": (datetime.now() - start_time).total_seconds()
            }
        
        # Generate comprehensive LLM summary to extract recommendations
        logger.info(f"[{request_id}] Generating comprehensive LLM summary...")
        content_organization = analyzer._organize_content_for_llm(vectors, request.max_content_length)
        
        # Generate the comprehensive LLM summary
        llm_summary_result = await analyzer._generate_llm_summary(content_organization, "comprehensive")
        
        if not llm_summary_result.get("success"):
            logger.warning(f"[{request_id}] Failed to generate LLM summary, using fallback")
            # Fallback to generic pickup line
            pickup_line = await _generate_generic_pickup_line(request.pickup_style)
            return {
                "success": True,
                "request_id": request_id,
                "pickup_line": pickup_line,
                "style": request.pickup_style,
                "based_on": "generic_fallback",
                "knowledge_base_analyzed": _get_friendly_knowledge_description(total_count),
                "generated_at": datetime.now().isoformat(),
                "processing_time_seconds": (datetime.now() - start_time).total_seconds()
            }
        
        # Extract the full comprehensive analysis for better insights
        llm_analysis = llm_summary_result.get("analysis", "")
        language_info = content_organization.get("detected_language", {})
        
        logger.info(f"[{request_id}] Analyzing comprehensive summary for exploratory pickup line...")
        logger.info(f"[{request_id}] Detected language: {language_info.get('language_name', 'English')} (confidence: {language_info.get('confidence', 0.7):.2f})")
        
        # Generate exploratory pickup line with thought-provoking questions
        pickup_line_result = await _generate_focused_pickup_line(
            comprehensive_analysis=llm_analysis,
            language_info=language_info,
            pickup_style=request.pickup_style,
            total_vectors=total_count
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{request_id}] Successfully generated pickup line in {elapsed:.2f}s")
        
        return {
            "success": True,
            "request_id": request_id,
            "pickup_line": pickup_line_result["pickup_line"],
            "explanation": pickup_line_result["explanation"],
            "style": request.pickup_style,
            "based_on": "comprehensive_knowledge_analysis",
            "analysis_summary": {
                "knowledge_base_analyzed": _get_friendly_knowledge_description(total_count),
                "comprehensive_analysis_generated": True,
                "language_detected": language_info.get('language_name', 'English'),
                "detection_confidence": language_info.get('confidence', 0.7),
                "approach": "multi_domain_exploratory_questions"
            },
            "generated_at": datetime.now().isoformat(),
            "processing_time_seconds": elapsed
        }
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        logger.error(f"[{request_id}] Error in pickup-line endpoint: {error_msg}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {error_msg}"
        )
    finally:
        logger.info(f"[{request_id}] === END pickup-line request ===")

async def _generate_generic_pickup_line(style: str) -> str:
    """Generate a generic pickup line when no knowledge base is available."""
    generic_lines = {
        "witty": "Are you a neural network? Because you've got my parameters all optimized.",
        "charming": "I must be drawn to your intellectual energy, because I'm feeling a strong connection.",
        "nerdy": "If you were a function, you'd be f(x) = beautiful, and I'd want to solve for x.",
        "confident": "I don't need an AI to tell me you're worth getting to know.",
        "humorous": "Are you machine learning? Because you're making me smarter just by being here."
    }
    return generic_lines.get(style, generic_lines["charming"])

def _get_friendly_knowledge_description(total_vectors: int) -> str:
    """Convert technical vector count to user-friendly description."""
    if total_vectors == 0:
        return "a fresh learning journey"
    elif total_vectors < 50:
        return f"{total_vectors} learning moments"
    elif total_vectors < 100:
        return f"{total_vectors} knowledge insights"
    elif total_vectors < 200:
        return f"{total_vectors} intellectual discoveries"
    elif total_vectors < 500:
        return f"{total_vectors} wisdom pieces"
    else:
        return f"{total_vectors} knowledge gems"

def _extract_recommendations_section(llm_analysis: str) -> str:
    """Extract the complete RECOMMENDATIONS section from LLM comprehensive analysis."""
    recommendations_section = ""
    
    # Look for the recommendations section (section 8)
    lines = llm_analysis.split('\n')
    in_recommendations = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if we're entering recommendations section
        if ("8." in line or "**RECOMMENDATIONS**" in line.upper() or 
            (line_stripped.startswith("**") and "RECOMMENDATION" in line.upper())):
            in_recommendations = True
            recommendations_section += line + "\n"
            continue
        
        # Check if we're leaving recommendations section (next numbered section or end)
        if in_recommendations:
            # Stop if we hit another numbered section or conclusion
            if (line_stripped.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "9.", "10.")) or
                "**" in line and any(keyword in line.upper() for keyword in ["CONCLUSION", "SUMMARY", "ANALYSIS"])):
                break
            
            # Add the line to recommendations section
            if line_stripped:  # Only add non-empty lines
                recommendations_section += line + "\n"
    
    # If no structured section found, look for any recommendations content
    if not recommendations_section:
        for line in lines:
            if "recommend" in line.lower() or "suggest" in line.lower():
                recommendations_section += line + "\n"
    
    return recommendations_section.strip()

def _extract_knowledge_context(vectors: List[Dict], stats: Dict[str, Any], summary_result: Optional[Dict]) -> Dict[str, Any]:
    """Extract key context from knowledge base for pickup line generation."""
    context = {
        "dominant_topics": list(stats.get("topics", {}).get("top_10", {}).keys())[:3],
        "knowledge_depth": "deep" if stats.get("content", {}).get("average_length", 0) > 1000 else "broad",
        "confidence_level": "high" if stats.get("confidence", {}).get("average", 0) > 0.8 else "moderate",
        "conversation_style": "analytical" if "analysis" in str(stats.get("categories", {})) else "conversational"
    }
    
    # Add enhanced insights from LLM summary if available
    if summary_result and summary_result.get("success"):
        llm_analysis = summary_result.get("llm_analysis", {}).get("analysis", "")
        
        # Extract key domains and themes
        if "domain" in llm_analysis.lower():
            context["has_domain_expertise"] = True
        if "creative" in llm_analysis.lower():
            context["conversation_style"] = "creative"
        elif "technical" in llm_analysis.lower():
            context["conversation_style"] = "technical"
        elif "practical" in llm_analysis.lower():
            context["conversation_style"] = "practical"
            
        # Extract learning patterns
        if "learning" in llm_analysis.lower():
            context["shows_learning_growth"] = True
        if "diverse" in llm_analysis.lower() or "variety" in llm_analysis.lower():
            context["knowledge_diversity"] = "high"
    
    return context

async def _generate_focused_pickup_line(
    comprehensive_analysis: str,
    language_info: Dict[str, Any],
    pickup_style: str,
    total_vectors: int
) -> Dict[str, Any]:
    """Generate a comprehensive pickup line that shows understanding of their full knowledge profile and asks thought-provoking questions."""
    
    # Convert technical count to user-friendly terms
    knowledge_description = _get_friendly_knowledge_description(total_vectors)
    
    # Get the primary language from language_info
    primary_language = language_info.get('language_name', 'English')
    
    pickup_line_prompt = f"""
    You are analyzing someone's comprehensive knowledge summary to create a {pickup_style} pickup line that shows deep understanding of their FULL knowledge profile.
    
    📊 USER'S COMPREHENSIVE KNOWLEDGE ANALYSIS:
    {comprehensive_analysis}
    
    🚨 LANGUAGE REQUIREMENT: Write the pickup line in {primary_language} (user's primary language)! 🚨
    
    YOUR MISSION:
    1. Show understanding of their BROAD knowledge areas (mention 2-3 main domains they know about)
    2. Demonstrate you've analyzed their {knowledge_description} comprehensively 
    3. Connect their knowledge areas together to show the full picture
    4. Ask a THOUGHTFUL, OPEN-ENDED QUESTION that sparks curiosity and guides exploration
    5. Make it sound like you're impressed by their overall intellectual profile and want to explore ideas together
    
    GUIDELINES:
    - Reference multiple knowledge domains they have (e.g., "your expertise spans animal behavior, travel insights, and conservation")
    - Show you understand the connections between their interests
    - Ask an open-ended question that encourages brainstorming and exploration
    - Create curiosity about knowledge gaps or unexplored connections
    - Guide them to think about ambiguities or interesting possibilities
    - Write in {pickup_style} tone but show intellectual depth
    - Write in {primary_language} language!
    - Use friendly terms for their knowledge base ({knowledge_description})
    
    EXAMPLES of curious, exploratory approach:
    ❌ PRESCRIPTIVE: "You should explore eco-tourism research opportunities!"
    ✅ CURIOUS: "I've been analyzing your knowledge spanning animal behavior, travel insights, and conservation - what do you think would happen if we could design tourism experiences that actually helped wildlife instead of harming them?"
    ✅ EXPLORATORY: "Your understanding covers everything from primate behavior to local food culture - I'm curious, how do you think traditional food practices might be connected to wildlife conservation efforts?"
    ✅ BRAINSTORMING: "Looking at your insights on wolves, travel pricing, and environmental impact - what if there was a way to make conservation profitable for local communities? What would that look like?"
    
    The pickup line should feel like:
    "I've analyzed your {knowledge_description} covering [area 1], [area 2], and [area 3] - [thoughtful open-ended question that sparks curiosity and exploration]?"
    
    🔥 CRITICAL: Write the pickup line in {primary_language}, not English! 🔥
    
    Respond in JSON format:
    {{
        "pickup_line": "Your comprehensive pickup line in {primary_language}",
        "explanation": "Explain which knowledge areas you highlighted and what thoughtful question you asked to spark curiosity and exploration (this can be in English)"
    }}
    """
    
    try:
        response = await SUMMARIZATION_LLM.ainvoke(pickup_line_prompt)
        response_text = response.content.strip()
        
        # Try to parse JSON response
        import json
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "pickup_line": response_text,
                "explanation": f"Generated based on {pickup_style} style and comprehensive analysis"
            }
            
    except Exception as e:
        logger.error(f"Error generating exploratory pickup line: {e}")
        # Fallback to generic line
        generic_line = await _generate_generic_pickup_line(pickup_style)
        return {
            "pickup_line": generic_line,
            "explanation": "Generated using fallback method due to processing error"
        }

@router.options('/pickup-line')
async def pickup_line_options():
    """Handle OPTIONS requests for pickup-line endpoint."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

@router.post("/brain/preview")
async def preview_brain(request: BrainPreviewRequest, org_id: str):
    """Preview brain contents with optional summarization."""
    try:
        analyzer = AIBrainAnalyzer(org_id)
        return await analyzer.preview_brain(
            namespace=request.namespace,
            max_vectors=request.max_vectors,
            summary_type=request.summary_type,
            include_vectors=request.include_vectors,
            max_content_length=request.max_content_length
        )
    except Exception as e:
        logger.error(f"Error in brain preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/brain/summary/{namespace}")
async def get_brain_summary(namespace: str, org_id: str):
    """Get comprehensive summary of brain contents."""
    try:
        analyzer = AIBrainAnalyzer(org_id)
        return await analyzer.get_brain_summary(namespace)
    except Exception as e:
        logger.error(f"Error getting brain summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/brain/stats/{namespace}")
async def get_brain_stats(namespace: str, org_id: str):
    """Get statistical analysis of brain contents."""
    try:
        analyzer = AIBrainAnalyzer(org_id)
        return await analyzer.get_brain_stats(namespace)
    except Exception as e:
        logger.error(f"Error getting brain stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

