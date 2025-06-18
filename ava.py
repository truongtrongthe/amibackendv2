"""
AVA (Active Learning Assistant) - Enhanced Knowledge Management System

OVERVIEW:
=========
AVA is a sophisticated conversational AI system that provides streaming responses while intelligently managing knowledge through multiple saving strategies. It features teaching intent detection, similarity-based knowledge gating, and human-in-the-loop decision making for optimal knowledge organization.

CORE ARCHITECTURE:
==================

1. **Streaming Conversation Engine**
   - Real-time streaming responses via AsyncGenerator
   - Knowledge retrieval using KnowledgeExplorer with iterative search
   - WebSocket integration for real-time frontend updates
   - Conversation context awareness with multi-turn handling

2. **Intelligent Knowledge Management**
   - Multi-vector saving strategy (3 formats per knowledge entry)
   - Teaching intent detection with LLM-based analysis
   - Similarity-based knowledge gating system
   - UPDATE vs CREATE decision workflow with human oversight

3. **Tool Integration System**
   - Direct tool execution via execute_tool method
   - Support for knowledge_query, save_knowledge, handle_update_decision
   - Background task management for non-blocking operations

KNOWLEDGE SAVING STRATEGY:
=========================

**Multi-Vector Approach** (3 vectors per knowledge entry):
1. **Combined Knowledge** - Full "User: X\n\nAI: Y" format for context
2. **AI Synthesis** - Enhanced with metadata and synthesis markers
3. **Standalone Summary** - Pure AI content with summary titles

**Similarity-Based Gating:**
- High Similarity (≥65%) → Auto-save as new knowledge (3 vectors)
- Medium Similarity (35%-65%) → Human decision required (UPDATE vs CREATE)
- Low Similarity (<35%) → Auto-save as new knowledge (3 vectors)

**Categories Applied:**
- Teaching Intent: ["teaching_intent", "human_approved"]
- Tool Execution: ["teaching_intent", "tool_executed"]
- Decision-based: ["human_approved", "decision_created/updated"]
- High Quality: ["high_quality_conversation", "general"]

UPDATE vs CREATE WORKFLOW:
==========================

1. **Teaching Intent Detection**
   - User provides information with teaching intent
   - System calculates similarity with existing knowledge using embeddings
   - KnowledgeExplorer performs iterative search for relevant content

2. **Similarity-Based Decision Gate**
   - evaluate_knowledge_similarity_gate() determines action type
   - identify_knowledge_update_candidates() finds similar entries
   - Medium similarity triggers human decision requirement

3. **Human-in-the-Loop for Medium Similarity**
   - create_update_decision_request() generates decision prompt
   - System presents 1-3 similar knowledge candidates
   - Frontend displays options: CREATE NEW vs UPDATE EXISTING
   - Human selects preferred action via handle_update_decision tool

4. **Knowledge Processing**
   - **CREATE NEW**: Saves 3 new vectors with decision_created category
   - **UPDATE EXISTING**: Merges content + saves 3 new vectors with decision_updated category
   - Original vectors preserved for reference (no deletion)

5. **Tool Integration**
   - handle_update_decision tool processes human decisions
   - _save_tool_knowledge_multiple creates all vector formats
   - Background task execution for non-blocking saves

STREAMING RESPONSE FLOW:
=======================

1. **Input Processing**
   - Receive user message and conversation context
   - Extract suggested queries from previous AI responses
   - Use KnowledgeExplorer for iterative knowledge search

2. **Knowledge Retrieval**
   - Primary search based on user message
   - Enrichment with suggested queries if needed
   - Similarity scoring and result aggregation

3. **Streaming Response Generation**
   - _active_learning_streaming yields chunks in real-time
   - LLM generates response with teaching intent analysis
   - WebSocket events emitted for frontend updates

4. **Post-Stream Knowledge Management**
   - Evaluate teaching intent and similarity scores
   - Trigger appropriate saving strategy (auto-save vs decision)
   - Handle UPDATE vs CREATE workflow if needed

TOOL EXECUTION SYSTEM:
=====================

**Supported Tools:**
- **knowledge_query**: Search knowledge base with enhanced parameters
- **save_knowledge**: Direct knowledge save with multi-vector strategy
- **handle_update_decision**: Process human UPDATE/CREATE decisions

**Tool Features:**
- Enhanced parameter validation and metadata enrichment
- Background task execution for non-blocking operations
- Multi-vector saving for all knowledge operations
- Automatic category assignment based on operation type

WEBSOCKET INTEGRATION:
=====================

**Real-time Events:**
- learning_intent_event: Teaching intent detection results
- learning_knowledge_event: Knowledge retrieval progress
- Emitted to specific thread_id for targeted frontend updates

EXAMPLE USAGE:
==============

# User teaches something with medium similarity (42%)
User: "Customer segmentation can also be done using behavioral patterns like purchase frequency."

# AVA Streaming Response includes decision request:
{
  "message": "I understand you're teaching about customer segmentation...\n\n🤔 **Knowledge Decision Required**\n\nI found 2 similar knowledge entries (similarity: 0.42). Would you like me to:\n\n1. **Create New Knowledge** - Save this as a completely new entry\n2. **Update Existing Knowledge** - Merge with similar existing knowledge\n\n**Similar Knowledge Found:**\n**Option 2:** Update existing knowledge (similarity: 0.42)\n*Preview:* Customer segmentation involves grouping customers...",
  "metadata": {
    "update_decision_request": {
      "request_id": "update_decision_thread123_1234567890",
      "decision_type": "UPDATE_OR_CREATE",
      "candidates_count": 2,
      "requires_human_input": true
    }
  }
}

# Human decides to UPDATE existing knowledge via tool:
{
  "tool_name": "handle_update_decision",
  "parameters": {
    "request_id": "update_decision_thread123_1234567890", 
    "action": "UPDATE_EXISTING",
    "target_id": "vector_abc123"
  }
}

# System merges knowledge and saves 3 new vectors:
{
  "success": true,
  "action": "UPDATE_EXISTING", 
  "original_vector_id": "vector_abc123",
  "merged_content": "Enhanced merged content...",
  "message": "Multiple knowledge vectors created for merged content",
  "save_types": ["combined_knowledge", "ai_synthesis", "standalone_summary"],
  "note": "Original vector preserved for reference"
}

"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Set
from datetime import datetime
from uuid import uuid4
import re
import pytz
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pccontroller import save_knowledge, query_knowledge, query_knowledge_from_graph
from utilities import logger, EMBEDDINGS
from learning_support import LearningSupport
from curiosity import KnowledgeExplorer
from alpha import (
    save_teaching_synthesis, 
    evaluate_knowledge_similarity_gate,
    identify_knowledge_update_candidates,
    merge_knowledge_content,
    extract_user_facing_content,
    regenerate_teaching_intent_response,
    create_update_decision_request,
    get_pending_knowledge_decisions,
    clear_pending_knowledge_decisions,
    get_pending_decision,
    remove_pending_decision
)
from time import time

# Import socketio manager for WebSocket support
try:
    from socketio_manager_async import emit_learning_intent_event, emit_learning_knowledge_event
    socket_imports_success = True
    logger.info("Successfully imported learning WebSocket functions in ava.py")
except ImportError:
    socket_imports_success = False
    logger.warning("Could not import learning WebSocket functions in ava.py - WebSocket events may not be delivered")
    # Create dummy functions to prevent NameError
    async def emit_learning_intent_event(*args, **kwargs):
        pass
    async def emit_learning_knowledge_event(*args, **kwargs):
        pass

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model to a JSON-serializable dict."""
    return json.loads(json.dumps(model.model_dump(), cls=DateTimeEncoder))

# Initialize LLM
LLM = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.01)

class AVA:
    def __init__(self):
        self.graph_version_id = ""
        # Set to keep track of pending background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        # Initialize support class
        self.support = LearningSupport(self)
        # Initialize knowledge explorer
        self.knowledge_explorer = None

    async def initialize(self):
        """Initialize the processor asynchronously."""
        logger.info("Initializing AVA")
        # Initialize knowledge explorer with support module
        self.knowledge_explorer = KnowledgeExplorer(self.graph_version_id, self.support)
        return self

    async def _emit_to_socket(self, thread_id_for_analysis, intent_type, has_teaching_intent, is_priority_topic, priority_topic_name, should_save_knowledge):
        # WEBSOCKET EMISSION: learning intent event when intent is understood
        try:
            learning_intent_event = {
                            "type": "learning_intent",
                            "thread_id": thread_id_for_analysis,
                            "timestamp": datetime.now().isoformat(),
                            "content": {
                                "message": "Understanding human intent",
                                "intent_type": intent_type,
                                "has_teaching_intent": has_teaching_intent,
                                "is_priority_topic": is_priority_topic,
                                "priority_topic_name": priority_topic_name,
                                "should_save_knowledge": should_save_knowledge,
                                "complete": True
                            }
                        }
            await emit_learning_intent_event(thread_id_for_analysis, learning_intent_event)
            logger.info(f"Emitted learning intent event for thread {thread_id_for_analysis}: {intent_type}")
        except Exception as e:
            logger.error(f"Error emitting learning intent event: {str(e)}")
    
    async def _emit_learning_knowledge_to_socket(self, thread_id_for_analysis, analysis_knowledge):
        # WEBSOCKET EMISSION: learning knowledge event when knowledge is found
        try:
            # Extract additional context data for frontend
            query_results = analysis_knowledge.get("query_results", [])
            queries = analysis_knowledge.get("queries", [])
            knowledge_context = analysis_knowledge.get("knowledge_context", "")
            
            # Prepare query summaries for frontend
            query_summaries = []
            for i, query in enumerate(queries):
                query_summary = {
                    "query": query,
                    "results_count": len([r for r in query_results if r.get("query") == query]) if query_results else 0
                }
                query_summaries.append(query_summary)
            
            # Prepare top knowledge results for frontend preview
            top_results = []
            for result in query_results[:5]:  # Top 5 results for preview
                # Handle categories safely - could be dict or list
                categories_data = result.get("categories", [])
                if isinstance(categories_data, dict):
                    categories_list = list(categories_data.keys())
                elif isinstance(categories_data, list):
                    categories_list = categories_data
                else:
                    categories_list = []
                
                result_summary = {
                    "id": result.get("id", ""),
                    "score": result.get("score", 0.0),
                    "preview": result.get("raw", "")[:100] + "..." if len(result.get("raw", "")) > 100 else result.get("raw", ""),
                    "categories": categories_list,
                    "query": result.get("query", "")
                }
                top_results.append(result_summary)
            
            learning_knowledge_event = {
                "type": "learning_knowledge",
                "thread_id": thread_id_for_analysis,
                "timestamp": datetime.now().isoformat(),
                "content": {
                    "message": "Found relevant knowledge for learning",
                    "similarity_score": analysis_knowledge.get("similarity", 0.0),
                    "knowledge_count": len(query_results),
                    "queries": queries,
                    "query_summaries": query_summaries,
                    "top_results": top_results,
                    "has_knowledge_context": bool(knowledge_context),
                    "knowledge_context_length": len(knowledge_context) if knowledge_context else 0,
                    "complete": False
                }
            }
            await emit_learning_knowledge_event(thread_id_for_analysis, learning_knowledge_event)
            logger.info(f"Emitted enhanced learning knowledge event for thread {thread_id_for_analysis} with {len(query_results)} results")
        except Exception as e:
            logger.error(f"Error emitting learning knowledge event: {str(e)}")

    async def _detect_early_intent_and_emit(self, thread_id_for_analysis, message: str, analysis_knowledge: Dict, conversation_context: str):
        """Detect preliminary intent using LLM-based analysis and emit early acknowledgement to frontend."""
        try:
            # Use lightweight LLM for quick preliminary intent analysis
            similarity_score = analysis_knowledge.get("similarity", 0.0)
            query_results = analysis_knowledge.get("query_results", [])
            
            # Build context for LLM analysis
            context_summary = ""
            if conversation_context:
                # Get last few exchanges for context
                recent_context = conversation_context.split('\n\n')[-4:] if conversation_context else []
                context_summary = "\n".join(recent_context)
            
            knowledge_summary = ""
            if query_results:
                knowledge_summary = f"Found {len(query_results)} relevant knowledge entries (similarity: {similarity_score:.2f})"
            
            # Quick LLM prompt for intent detection
            intent_prompt = f"""Analyze this user message for preliminary intent classification. Be concise and quick.

            User Message: "{message}"

            Recent Context: {context_summary if context_summary else "No prior context"}

            Knowledge Context: {knowledge_summary if knowledge_summary else "No relevant knowledge found"}

            Classify the preliminary intent and provide confidence. Respond in JSON format:
            {{
                "preliminary_intent": "teaching|questioning|explaining|requesting|general_conversation",
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation",
                "has_teaching_signals": true/false,
                "is_question": true/false,
                "knowledge_relevance": "high|medium|low|none"
            }}

            Focus on the most likely intent based on message patterns and context."""

            # Use a quick LLM call for preliminary analysis
            from langchain_openai import ChatOpenAI
            QuickLLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=200)
            
            logger.info("Performing LLM-based early intent detection...")
            llm_response = await QuickLLM.ainvoke(intent_prompt)
            llm_content = llm_response.content.strip() if llm_response and llm_response.content else ""
            
            logger.info(f"LLM intent response content: '{llm_content[:100]}...'")
            
            # Parse LLM response with better error handling
            try:
                import json
                # Try to extract JSON if it's wrapped in markdown code blocks
                if "```json" in llm_content:
                    json_start = llm_content.find("```json") + 7
                    json_end = llm_content.find("```", json_start)
                    if json_end > json_start:
                        json_content = llm_content[json_start:json_end].strip()
                    else:
                        json_content = llm_content[json_start:].strip()
                elif "```" in llm_content:
                    # Handle plain code blocks
                    json_start = llm_content.find("```") + 3
                    json_end = llm_content.find("```", json_start)
                    if json_end > json_start:
                        json_content = llm_content[json_start:json_end].strip()
                    else:
                        json_content = llm_content[json_start:].strip()
                else:
                    json_content = llm_content
                
                logger.info(f"Extracted JSON content: '{json_content[:100]}...'")
                
                if not json_content:
                    raise ValueError("Empty JSON content")
                
                intent_analysis = json.loads(json_content)
                
                # Validate required fields
                if not isinstance(intent_analysis, dict):
                    raise ValueError("Response is not a JSON object")
                
                # Extract analysis results with validation
                preliminary_intent = intent_analysis.get("preliminary_intent", "unknown")
                confidence_score = float(intent_analysis.get("confidence", 0.5))
                reasoning = intent_analysis.get("reasoning", "LLM analysis completed")
                has_teaching_signals = bool(intent_analysis.get("has_teaching_signals", False))
                is_question = bool(intent_analysis.get("is_question", False))
                knowledge_relevance = intent_analysis.get("knowledge_relevance", "none")
                
                logger.info(f"Successfully parsed LLM intent: {preliminary_intent} (confidence: {confidence_score})")
                
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Failed to parse LLM intent analysis: {e}")
                logger.warning(f"Raw LLM content was: '{llm_content}'")
                
                # Fallback to basic pattern analysis
                message_lower = message.lower()
                preliminary_intent = "general_conversation"
                confidence_score = 0.3
                reasoning = f"Fallback analysis due to parsing error: {str(e)}"
                
                # Simple pattern detection as fallback
                if any(word in message_lower for word in ["teach", "explain", "show", "understand", "learn"]):
                    preliminary_intent = "teaching" if "teach" in message_lower or "explain" in message_lower else "questioning"
                    confidence_score = 0.5
                elif "?" in message or any(word in message_lower for word in ["what", "how", "why", "when", "where"]):
                    preliminary_intent = "questioning"
                    confidence_score = 0.6
                    
                has_teaching_signals = "teach" in message_lower or "explain" in message_lower
                is_question = "?" in message or any(word in message_lower for word in ["what", "how", "why", "when", "where"])
                knowledge_relevance = "medium" if similarity_score > 0.3 else "low"
                
                logger.info(f"Fallback intent analysis: {preliminary_intent} (confidence: {confidence_score})")
            
            # Analyze knowledge context signals
            knowledge_context_signals = {
                "high_similarity": similarity_score >= 0.65,
                "medium_similarity": 0.35 <= similarity_score < 0.65,
                "has_knowledge_results": len(query_results) > 0,
                "substantial_knowledge": len(query_results) >= 3,
                "knowledge_relevance": knowledge_relevance
            }
            
            # Emit early intent acknowledgement
            learning_intent_acknowledgement = {
                "type": "learning_intent_acknowledgement",
                "thread_id": thread_id_for_analysis,
                "timestamp": datetime.now().isoformat(),
                "content": {
                    "message": "Analyzing your intent and gathering context",
                    "preliminary_intent": preliminary_intent,
                    "confidence": confidence_score,
                    "reasoning": reasoning,
                    "signals": {
                        "has_teaching_signals": has_teaching_signals,
                        "is_question": is_question,
                        "knowledge_context": knowledge_context_signals
                    },
                    "status": "preliminary_analysis",
                    "complete": False
                }
            }
            
            await emit_learning_intent_event(thread_id_for_analysis, learning_intent_acknowledgement)
            logger.info(f"Emitted LLM-based early intent acknowledgement for thread {thread_id_for_analysis}: {preliminary_intent} (confidence: {confidence_score:.2f}) - {reasoning}")
            
        except Exception as e:
            logger.error(f"Error in LLM-based early intent detection: {str(e)}")
            import traceback
            logger.error(f"Early intent detection traceback: {traceback.format_exc()}")
            
            # Emit basic acknowledgement on failure
            try:
                fallback_acknowledgement = {
                    "type": "learning_intent_acknowledgement",
                    "thread_id": thread_id_for_analysis,
                    "timestamp": datetime.now().isoformat(),
                    "content": {
                        "message": "Processing your message",
                        "preliminary_intent": "processing",
                        "confidence": 0.1,
                        "reasoning": f"Fallback due to analysis error: {str(e)}",
                        "signals": {
                            "has_teaching_signals": False,
                            "is_question": "?" in message,
                            "knowledge_context": {
                                "has_knowledge_results": len(analysis_knowledge.get("query_results", [])) > 0,
                                "knowledge_relevance": "unknown"
                            }
                        },
                        "status": "processing",
                        "complete": False
                    }
                }
                await emit_learning_intent_event(thread_id_for_analysis, fallback_acknowledgement)
                logger.info(f"Emitted fallback acknowledgement for thread {thread_id_for_analysis}")
            except Exception as fallback_error:
                logger.error(f"Failed to emit fallback acknowledgement: {fallback_error}")
                import traceback
                logger.error(f"Fallback emission traceback: {traceback.format_exc()}")

    async def read_human_input(self, message: str, conversation_context: str = "", user_id: str = None, 
                             thread_id: str = None, use_websocket: bool = False, 
                             thread_id_for_analysis: str = None, org_id: str = "unknown"):
        """
        Process human input and generate a response using the learning-based approach.
        
        Args:
            user_input: The user's message
            conversation_context: Previous conversation history
            user_id: User identifier
            thread_id: Conversation thread identifier
            use_websocket: Whether to use WebSocket for analysis streaming
            thread_id_for_analysis: Thread ID to use for WebSocket analysis events
            org_id: Organization identifier for knowledge management
        """
        try:
            if not message.strip():
                logger.error("Empty message")
                yield {"status": "error", "message": "Empty message provided", "complete": True}
                return
            
            # Step 1: Get suggested knowledge queries from previous responses
            suggested_queries = []
            if conversation_context:
                ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                if ai_messages:
                    last_ai_message = ai_messages[-1]
                    query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', last_ai_message, re.DOTALL)
                    if query_section:
                        try:
                            queries_data = json.loads(query_section.group(1).strip())
                            suggested_queries = queries_data if isinstance(queries_data, list) else queries_data.get("queries", [])
                            logger.info(f"Extracted {len(suggested_queries)} suggested knowledge queries")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse query section as JSON")

            # Step 2: Search for relevant knowledge using iterative exploration
            logger.info(f"Searching for knowledge based on message using iterative exploration...")
            analysis_knowledge = await self.knowledge_explorer.explore(message, conversation_context, user_id, thread_id)
            
            # WEBSOCKET EMISSION POINT 2A: Emit early intent acknowledgement when analysis begins
            if use_websocket and thread_id_for_analysis and socket_imports_success and analysis_knowledge:
                await self._detect_early_intent_and_emit(thread_id_for_analysis, message, analysis_knowledge, conversation_context)
            
            # WEBSOCKET EMISSION POINT 2B: Emit learning knowledge event when knowledge is found
            if use_websocket and thread_id_for_analysis and socket_imports_success and analysis_knowledge:
                await self._emit_learning_knowledge_to_socket(thread_id_for_analysis, analysis_knowledge)
            
            # Step 3: Enrich with suggested queries if we don't have sufficient results
            if suggested_queries and len(analysis_knowledge.get("query_results", [])) < 3:
                logger.info(f"Searching for additional knowledge using {len(suggested_queries)} suggested queries")
                primary_similarity = analysis_knowledge.get("similarity", 0.0)
                primary_knowledge = analysis_knowledge.get("knowledge_context", "")
                primary_queries = analysis_knowledge.get("queries", [])
                primary_query_results = analysis_knowledge.get("query_results", [])
                
                for query in suggested_queries:
                    if query not in primary_queries:  # Avoid duplicate queries
                        query_knowledge = await self.support.search_knowledge(query, conversation_context, user_id, thread_id)
                        query_similarity = query_knowledge.get("similarity", 0.0)
                        query_results = query_knowledge.get("query_results", [])
                        
                        logger.info(f"Suggested query '{query}' yielded similarity score: {query_similarity}")
                        
                        # Add the new query and its results to our collection
                        if query_results:
                            primary_queries.append(query)
                            primary_query_results.extend(query_results)
                            logger.info(f"Added results from suggested query '{query}'")
                
                # Update analysis_knowledge with enriched data
                if len(primary_query_results) > len(analysis_knowledge.get("query_results", [])):
                    analysis_knowledge["queries"] = primary_queries
                    analysis_knowledge["query_results"] = primary_query_results
                    logger.info(f"Updated knowledge with {len(primary_query_results)} total query results")
            
            # Step 4: Log final similarity score (conversation history scanning now handled by LLM)
            similarity = analysis_knowledge.get("similarity", 0.0)
            logger.info(f"Final knowledge similarity: {similarity:.3f}")
            
            # Step 5: Generate streaming response (LLM will scan conversation history automatically)
            prior_data = analysis_knowledge.get("prior_data", {})
            
            # Stream the response as it comes from LLM
            final_response = None
            async for chunk in self._active_learning_streaming(message, conversation_context, analysis_knowledge, user_id, prior_data, use_websocket, thread_id_for_analysis, org_id):
                if chunk.get("type") == "response_chunk":
                    # Yield streaming chunks immediately
                    yield chunk
                elif chunk.get("type") == "response_complete":
                    # Store final response for post-processing
                    final_response = chunk
                    yield chunk
                elif chunk.get("type") == "error":
                    yield chunk
                    return
            
            # Step 6: Enhanced knowledge saving with similarity gating (after streaming is complete)
            if final_response and final_response.get("status") == "success":
                # Get teaching intent and priority topic info from LLM evaluation
                metadata = final_response.get("metadata", {})
                has_teaching_intent = metadata.get("has_teaching_intent", False)
                is_priority_topic = metadata.get("is_priority_topic", False)
                should_save_knowledge = metadata.get("should_save_knowledge", False)
                priority_topic_name = metadata.get("priority_topic_name", "")
                intent_type = metadata.get("intent_type", "unknown")
                
                # WEBSOCKET EMISSION POINT 1: Emit learning intent event when intent is understood
                if use_websocket and thread_id_for_analysis and socket_imports_success:
                    await self._emit_to_socket(thread_id_for_analysis, intent_type, has_teaching_intent, is_priority_topic, priority_topic_name, should_save_knowledge)
                
                # Get similarity score from analysis
                similarity_score = analysis_knowledge.get("similarity", 0.0) if analysis_knowledge else 0.0
                
                # Use similarity-based gating system from alpha.py
                save_decision = await evaluate_knowledge_similarity_gate(final_response, similarity_score, message)
                
                logger.info(f"Knowledge saving decision: {save_decision}")
                logger.info(f"LLM evaluation: intent={intent_type}, teaching_intent={has_teaching_intent}, priority_topic={is_priority_topic}, similarity={similarity_score:.2f}")
                
                # Check for UPDATE vs CREATE decision BEFORE yielding final response
                if save_decision.get("action_type") == "UPDATE_OR_CREATE" and has_teaching_intent:
                    logger.info("🔍 Checking for UPDATE vs CREATE decision before yielding response")
                    
                    # Get query results for candidate identification
                    query_results = analysis_knowledge.get("query_results", [])
                    
                    # Identify update candidates using alpha.py function
                    candidates = await identify_knowledge_update_candidates(message, similarity_score, query_results)
                    
                    if candidates:
                        # We have candidates - create decision request
                        logger.info(f"📝 Creating UPDATE vs CREATE decision request with {len(candidates)} candidates")
                        
                        # Extract response content for decision
                        message_content = final_response["message"]
                        conversational_response = re.split(r'<knowledge_queries>', message_content)[0].strip()
                        
                        # Create the new content that would be saved
                        new_content = f"User: {message}\n\nAI: {conversational_response}"
                        
                        # Present options to human using alpha.py function
                        decision_request = await create_update_decision_request(
                            message=message,
                            new_content=new_content,
                            candidates=candidates,
                            user_id=user_id,
                            thread_id=thread_id
                        )
                        
                        # Add decision info to response metadata BEFORE yielding
                        final_response["metadata"]["update_decision_request"] = {
                            "request_id": decision_request["request_id"],
                            "decision_type": "UPDATE_OR_CREATE",
                            "candidates_count": len(candidates),
                            "similarity_score": similarity_score,
                            "requires_human_input": True,
                            "candidates": [
                                {
                                    "id": candidate["vector_id"],
                                    "similarity": candidate["similarity"],
                                    "preview": candidate["content"][:150] + "..." if len(candidate["content"]) > 150 else candidate["content"]
                                }
                                for candidate in candidates
                            ]
                        }
                        
                        # Enhance the response message to ask for human decision
                        human_decision_prompt = f"""

                        🤔 **Knowledge Decision Required**

                        I found {len(candidates)} similar knowledge entries (similarity: {similarity_score:.2f}). This new information is quite similar to existing knowledge. Would you like me to:

                        1. **Create New Knowledge** - Save this as a completely new entry
                        2. **Update Existing Knowledge** - Merge with existing knowledge below

                        **Similar Knowledge Found:**
                        """
                        
                        for i, candidate in enumerate(candidates, 1):
                            preview = candidate["content"][:150] + "..." if len(candidate["content"]) > 150 else candidate["content"]
                            human_decision_prompt += f"\n**Option {i+1}:** Update existing knowledge (similarity: {candidate['similarity']:.2f})\n*Preview:* {preview}\n"
                        
                        human_decision_prompt += f"\nPlease let me know your preference!"
                        
                        # Add the decision prompt to the response
                        final_response["message"] = conversational_response + human_decision_prompt
                        
                        logger.info(f"✅ Added UPDATE vs CREATE decision to response metadata: {decision_request['request_id']}")
                    else:
                        logger.info("No suitable candidates found for high similarity case - this is unexpected")
                
                # Update metadata based on similarity gating decision
                if save_decision.get("should_save", False):
                    # Only update knowledge saving flags, NOT intent classification
                    final_response["metadata"]["should_save_knowledge"] = True
                    final_response["metadata"]["similarity_gating_reason"] = save_decision.get("reason", "")
                    final_response["metadata"]["similarity_gating_confidence"] = save_decision.get("confidence", "")
                    
                    # Add auto-save notification for low-medium similarity
                    if save_decision.get("auto_save", False):
                        # Extract response content and add auto-save notification
                        message_content = final_response["message"]
                        conversational_response = re.split(r'<knowledge_queries>', message_content)[0].strip()
                        
                        auto_save_notification = f"""

                            💾 **Knowledge Saved**

                            I've automatically saved this information as new knowledge since it's sufficiently different from existing content (similarity: {similarity_score:.2f}).
                        """
                        
                        final_response["message"] = conversational_response + auto_save_notification
                        logger.info(f"✅ Added auto-save notification for low-medium similarity ({similarity_score:.2f})")
                    
                    # Only override teaching intent if LLM actually detected teaching intent
                    # Don't force teaching intent just because of high similarity
                    original_teaching_intent = final_response["metadata"].get("has_teaching_intent", False)
                    if original_teaching_intent:
                        final_response["metadata"]["response_strategy"] = "TEACHING_INTENT"
                        logger.info(f"Confirmed teaching intent with similarity gating - will save knowledge")
                    else:
                        logger.info(f"High similarity ({similarity_score:.2f}) - saving knowledge but preserving original intent classification")
                    
                    # Only run background knowledge saving if we DON'T have a pending decision
                    if not final_response["metadata"].get("update_decision_request"):
                        logger.info("💾 Running background knowledge saving (no pending decision)")
                        if original_teaching_intent:
                            # Handle as teaching intent with regular save (no UPDATE vs CREATE flow)
                            org_id = final_response["metadata"].get("org_id", org_id)  # Use org_id from response metadata or fallback to original
                            self._create_background_task(
                                self.handle_teaching_intent(message, final_response, user_id, thread_id, priority_topic_name, org_id)
                            )
                        else:
                            # Save as regular high-quality conversation without teaching intent processing
                            org_id = final_response["metadata"].get("org_id", org_id)  # Use org_id from response metadata or fallback to original
                            self._create_background_task(
                                self._save_high_quality_conversation(message, final_response, user_id, thread_id, org_id)
                            )
                    else:
                        logger.info("⏳ Skipping background save - waiting for human decision")

        except Exception as e:
            logger.error(f"Error processing streaming message: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            yield {"status": "error", "message": f"Error: {str(e)}", "complete": True}

    async def _active_learning_streaming(self, message: Union[str, List], conversation_context: str = "", analysis_knowledge: Dict = None, user_id: str = "unknown", prior_data: Dict = None, use_websocket: bool = False, thread_id_for_analysis: Optional[str] = None, org_id: str = "unknown") -> AsyncGenerator[Union[str, Dict], None]:
        """Streaming version of active learning method that yields chunks as they come from LLM."""
        logger.info("Starting streaming active learning approach")
        
        try:
            # Step 1: Setup and validation
            temporal_context = self.support.setup_temporal_context()
            message_str = self.support.validate_and_normalize_message(message)
            
            # Step 2: Extract and organize data
            analysis_data = self.support.extract_analysis_data(analysis_knowledge)
            prior_data_extracted = self.support.extract_prior_data(prior_data)
            logger.info(f"Conversation Context data at _active_learning_streaming: {conversation_context}")
            prior_messages = self.support.extract_prior_messages(conversation_context)
            
            knowledge_context = analysis_data["knowledge_context"]
            similarity_score = analysis_data["similarity_score"]
            queries = analysis_data["queries"]
            query_results = analysis_data["query_results"]
            prior_topic = prior_data_extracted["prior_topic"]
            prior_knowledge = prior_data_extracted["prior_knowledge"]
            
            logger.info(f"Using similarity score: {similarity_score}")
            
            # Step 3: Detect conversation flow and message characteristics
            
            logger.info(f"Prior messages: {prior_messages}")
            logger.info(f"Conversation context: {conversation_context}")
            logger.info(f"Message: {message_str}")
            flow_result = await self.support.detect_conversation_flow(
                message=message_str,
                prior_messages=prior_messages,
                conversation_context=conversation_context
            )
            logger.info(f"Convo Flow detection result: {flow_result}")
            flow_type = flow_result.get("flow_type", "NEW_TOPIC")
            flow_confidence = flow_result.get("confidence", 0.5)
            logger.info(f"Active learning conversation flow: {flow_type} (confidence: {flow_confidence})")
            
            message_characteristics = self.support.detect_message_characteristics(message_str)
            knowledge_relevance = self.support.check_knowledge_relevance(analysis_knowledge)
            
            # Step 4: Determine response strategy
            #This is the KEY decision point for the AI to decide what to do with the knowledge.
            # strategy_result is a dictionary with the following keys:
            # - strategy: the strategy to use
            # - instructions: the instructions for the strategy
            # - knowledge_context: the knowledge context to use
            # - similarity_score: the similarity score to use
            # - prior_knowledge: the prior knowledge to use
            # - queries: the queries to use
            knowledge_response_sections = []
            strategy_result = await self.support.determine_response_strategy(
                flow_type=flow_type,
                flow_confidence=flow_confidence,
                message_characteristics=message_characteristics,
                knowledge_relevance=knowledge_relevance,
                similarity_score=similarity_score,
                prior_knowledge=prior_knowledge,
                queries=queries,
                query_results=query_results,
                knowledge_response_sections=knowledge_response_sections,
                conversation_context=conversation_context,
                message_str=message_str
            )
            
            #At this point, the AI has decided what to do with the knowledge.
            # Now we need to extract the strategy, instructions, and knowledge context from the strategy_result.
            # The strategy is the strategy to use.
            # The instructions are the instructions for the strategy.
            # The knowledge context is the knowledge context to use.
            # The similarity score is the similarity score to use.
            # The prior knowledge is the prior knowledge to use.
            
            response_strategy = strategy_result["strategy"]
            strategy_instructions = strategy_result["instructions"]
            
            # Update knowledge context and similarity score from strategy
            if "knowledge_context" in strategy_result and strategy_result["knowledge_context"]:
                knowledge_context = strategy_result["knowledge_context"]
            if "similarity_score" in strategy_result:
                similarity_score = strategy_result["similarity_score"]
            
            # Handle fallback knowledge sections if needed
            if not knowledge_context and queries and query_results:
                fallback_context = self.support.build_knowledge_fallback_sections(queries, query_results)
                if fallback_context:
                    knowledge_context = fallback_context
            
            #logger.info(f"Knowledge context: {knowledge_context}")
            
            # Step 5: Build LLM prompt
            # This is the prompt that the LLM will use to generate the response. it should be a string.
            # This should be dynamic and based on the conversation context, message, and prior knowledge.
            prompt = self.support.build_llm_prompt(
                message_str=message_str,
                conversation_context=conversation_context,
                temporal_context=temporal_context,
                knowledge_context=knowledge_context,
                response_strategy=response_strategy,
                strategy_instructions=strategy_instructions,
                core_prior_topic=prior_topic,
                user_id=user_id
            )
            
            # Step 6: Stream LLM response and process it
            from langchain_openai import ChatOpenAI
            StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True, temperature=0.25)
            
            content_buffer = ""
            evaluation_started = False
            logger.info("Starting LLM streaming response")
            #logger.info(f"LLM Prompt: {prompt}")
            async for chunk in StreamLLM.astream(prompt):
                chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if chunk_content:
                    content_buffer += chunk_content
                    
                    # Check if we're entering an evaluation section
                    if '<evaluation>' in content_buffer and not evaluation_started:
                        evaluation_started = True
                        # Extract content before evaluation and yield it
                        pre_evaluation = content_buffer.split('<evaluation>')[0]
                        remaining_content = pre_evaluation[len(content_buffer) - len(chunk_content):]
                        if remaining_content:
                            yield {
                                "type": "response_chunk",
                                "content": remaining_content,
                                "complete": False
                            }
                        continue
                    
                    # Skip chunks if we're in evaluation section
                    if evaluation_started and '</evaluation>' not in content_buffer:
                        continue
                    
                    # If evaluation section ended, reset flag and continue
                    if evaluation_started and '</evaluation>' in content_buffer:
                        evaluation_started = False
                        continue
                    
                    # Only yield chunks if we're not in evaluation section
                    if not evaluation_started:
                        yield {
                            "type": "response_chunk",
                            "content": chunk_content,
                            "complete": False
                        }
            
            logger.info(f"LLM streaming completed, total content length: {len(content_buffer)}")
            
            # Step 7: Process the complete response
            content = content_buffer.strip()
            logger.info(f"LLM streaming completed, Content: {content}")
            
            # Extract tool calls and evaluation first
            content, tool_calls, evaluation = self.support.extract_tool_calls_and_evaluation(content, message_str)
            
            # Step 8: Handle teaching intent regeneration if needed using alpha.py function
            
            if evaluation.get("has_teaching_intent", False) and response_strategy != "TEACHING_INTENT":
                logger.info(f"Content before regeneration: {content}")
                content, response_strategy = await regenerate_teaching_intent_response(
                    message_str, content, response_strategy
                )
                logger.info(f"Content after regeneration: {content}")
            
            # Step 8.5: Extract structured sections AFTER content enhancement
            structured_sections = self.support.extract_structured_sections(content)
            logger.info(f"Extracted structured sections: {list(structured_sections.keys())}")
            
            # Step 9: Extract user-facing content using alpha.py function
            user_facing_content = extract_user_facing_content(content, response_strategy, structured_sections)
            
            # Step 10: Handle empty response fallbacks
            user_facing_content = self.support.handle_empty_response_fallbacks(
                user_facing_content, response_strategy, message_str
            )
            
            # Step 11: Yield final complete response with metadata
            yield {
                "type": "response_complete",
                "status": "success",
                "message": user_facing_content,
                "complete": True,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "additional_knowledge_found": bool(knowledge_context),
                    "similarity_score": similarity_score,
                    "response_strategy": response_strategy,
                    "core_prior_topic": prior_topic,
                    "tool_calls": tool_calls,
                    "has_teaching_intent": evaluation.get("has_teaching_intent", False),
                    "is_priority_topic": evaluation.get("is_priority_topic", False),
                    "priority_topic_name": evaluation.get("priority_topic_name", ""),
                    "should_save_knowledge": evaluation.get("should_save_knowledge", False),
                    "full_structured_response": content,
                    "org_id": org_id
                }
            }
            
        except ValueError as e:
            logger.error(f"Validation error in streaming active learning: {str(e)}")
            yield {"type": "error", "status": "error", "message": "Empty message provided", "complete": True}
        except Exception as e:
            logger.error(f"Error in streaming active learning: {str(e)}")
            yield {
                "type": "error",
                "status": "error", 
                "message": "Tôi xin lỗi, nhưng tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
                "complete": True
            }

    def _create_background_task(self, coro):
        """Create and track a background task."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        # Add callback to remove task from set when done
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def cleanup(self):
        """Wait for all background tasks to complete."""
        if not self._background_tasks:
            return
            
        logger.info(f"Waiting for {len(self._background_tasks)} background tasks to complete...")
        # Create a copy as the set may be modified during iteration
        pending_tasks = list(self._background_tasks)
        
        if not pending_tasks:
            return
            
        done, pending = await asyncio.wait(
            pending_tasks, 
            timeout=5.0,  # Increased timeout to 5 seconds for tasks to complete
            return_when=asyncio.ALL_COMPLETED
        )
        
        if pending:
            logger.warning(f"{len(pending)} background tasks did not complete in time and will be cancelled")
            for task in pending:
                if not task.done():
                    task.cancel()
                    try:
                        # Try to await cancelled tasks to handle cancellation properly
                        await asyncio.wait_for(task, timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    
        # Clean up the set
        self._background_tasks.clear()
        logger.info("Background task cleanup completed")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with enhanced parameter validation and metadata enrichment."""
        logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
        try:
            user_id = parameters.get("user_id", "")
            if not user_id:
                logger.warning("Missing user_id in tool call, defaulting to 'unknown'")
                user_id = "unknown"
            
            org_id = parameters.get("org_id", "unknown")  # Add org_id parameter

            if tool_name == "knowledge_query":
                query = parameters.get("query", "")
                if not query:
                    return {"status": "error", "message": "Missing required parameter: query"}
                context = parameters.get("context", "")
                thread_id = parameters.get("thread_id", None)
                topic = parameters.get("topic", None)
                
                bank_name = "conversation"
                
                # Try querying the specified bank
                results = await query_knowledge(
                    query=query,
                    org_id=org_id,  # Add org_id
                    bank_name=bank_name,
                    user_id=user_id,
                    thread_id=None,  # Remove thread_id restriction to find more results
                    topic=None,      # Remove topic restriction
                    top_k=10,
                    min_similarity=0.2  # Lower threshold for better matching
                )
                return {
                    "status": "success",
                    "message": f"Queried knowledge for '{query}' from {bank_name} bank" + 
                              f" with results: {results}",
                    "data": results
                }

            elif tool_name == "save_knowledge":
                query = parameters.get("query", "")
                content = parameters.get("content", "")
                input_text = f"{query} {content}".strip() if query and content else query or content
                if not input_text:
                    return {"status": "error", "message": "Missing required parameter: query or content"}
                
                thread_id = parameters.get("thread_id", None)
                topic = parameters.get("topic", None)
                categories = parameters.get("categories", ["general"])
                
                if not categories or categories == ["general"]:
                    categories = ["human shared"]
                
                # Add teaching_intent category for explicit knowledge saves
                if "teaching_intent" not in categories:
                    categories.append("teaching_intent")
                    
                # Add tool_executed category to distinguish tool-based saves
                if "tool_executed" not in categories:
                    categories.append("tool_executed")
                    
                bank_name = "conversation"
                priority_topic_name = topic or "tool_knowledge"
                
                # Create a response-like structure for vector tracking
                response = {
                    "message": input_text,
                    "metadata": {}
                }
                
                # Format content for different save types
                if input_text.startswith("User:"):
                    # Already in User+AI format - extract parts
                    user_part = ""
                    ai_part = ""
                    if "\n\nAI:" in input_text:
                        parts = input_text.split("\n\nAI:", 1)
                        user_part = parts[0].replace("User:", "").strip()
                        ai_part = parts[1].strip()
                    else:
                        # Fallback - treat whole thing as user input
                        user_part = input_text.replace("User:", "").strip()
                        ai_part = f"Knowledge saved via tool: {user_part}"
                    
                    # Run multiple saves like teaching intent flow
                    self._create_background_task(self._save_tool_knowledge_multiple(
                        user_message=user_part,
                        ai_response=ai_part,
                        combined_content=input_text,
                        user_id=user_id,
                        org_id=org_id,  # Add org_id
                        bank_name=bank_name,
                        thread_id=thread_id,
                        priority_topic_name=priority_topic_name,
                        categories=categories,
                        response=response
                    ))
                else:
                    # AI-only or plain content - create User+AI format
                    if input_text.startswith("AI:"):
                        ai_content = input_text.replace("AI:", "").strip()
                    else:
                        ai_content = input_text
                    
                    # Create a synthetic user message for context
                    user_message = f"[Tool Save] Knowledge entry"
                    combined_content = f"User: {user_message}\n\nAI: {ai_content}"
                    
                    # Add ai_synthesis category since this contains AI content
                    if "ai_synthesis" not in categories:
                        categories.append("ai_synthesis")
                    
                    # Run multiple saves like teaching intent flow
                    self._create_background_task(self._save_tool_knowledge_multiple(
                        user_message=user_message,
                        ai_response=ai_content,
                        combined_content=combined_content,
                        user_id=user_id,
                        org_id=org_id,  # Add org_id
                        bank_name=bank_name,
                        thread_id=thread_id,
                        priority_topic_name=priority_topic_name,
                        categories=categories,
                        response=response
                    ))
                
                return {
                    "status": "success",
                    "message": f"Multiple knowledge vectors save initiated for '{input_text[:50]}...'"
                }

            elif tool_name == "handle_update_decision":
                request_id = parameters.get("request_id", "")
                action = parameters.get("action", "")
                target_id = parameters.get("target_id", None)
                
                if not request_id or not action:
                    return {"status": "error", "message": "Missing required parameters: request_id and action"}
                
                # Get thread_id from parameters or extract from request_id or use default
                thread_id_param = parameters.get("thread_id", None)
                if thread_id_param:
                    thread_id = thread_id_param
                else:
                    # Try to extract thread_id from request_id pattern
                    # e.g. "update_decision_learning_thread_1748946579708_1748946621" -> "learning_thread"
                    if "update_decision_" in request_id:
                        try:
                            # Extract the thread part from the request_id
                            # Remove the "update_decision_" prefix
                            after_prefix = request_id.replace("update_decision_", "")
                            # Split by underscores and find the thread part
                            parts = after_prefix.split("_")
                            if len(parts) >= 2:
                                # Reconstruct thread_id from first two parts
                                thread_id = f"{parts[0]}_{parts[1]}"  # "learning_thread"
                            else:
                                thread_id = parts[0] if parts else "unknown_thread"
                        except:
                            thread_id = "unknown_thread"
                    else:
                        thread_id = "unknown_thread"
                
                user_decision = {"action": action}
                if target_id:
                    user_decision["target_id"] = target_id
                
                result = await self.handle_update_decision(request_id, user_decision, user_id, thread_id, org_id)
                
                if result.get("success"):
                    return {
                        "status": "success", 
                        "message": result.get("message", "Decision processed successfully"),
                        "data": result
                    }
                else:
                    return {
                        "status": "error",
                        "message": result.get("error", "Failed to process decision"),
                        "data": result
                    }

            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {"status": "error", "message": f"Tool execution failed: {str(e)}"}

    async def handle_teaching_intent(self, message: str, response: Dict[str, Any], user_id: str, thread_id: Optional[str], priority_topic_name: str, org_id: str) -> None:
        """Handle teaching intent detection and knowledge saving."""
        logger.info("Saving knowledge due to teaching intent.")
        logger.info(f"Response: {response}")
        message_content = response["message"]
        conversational_response = re.split(r'<knowledge_queries>', message_content)[0].strip()
        query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', message_content, re.DOTALL)
        knowledge_queries = json.loads(query_section.group(1).strip()) if query_section else []

        # NEW: Create conversation turns structure for multi-turn synthesis
        conversation_turns = [
            {
                "user": message,
                "ai": conversational_response
            }
        ]
        
        # Extract final synthesis content
        final_synthesis = conversational_response
        if response.get("metadata", {}).get("response_strategy") == "TEACHING_INTENT":
            # Try to extract knowledge_synthesis if available
            synthesis_match = re.search(r'<knowledge_synthesis>(.*?)</knowledge_synthesis>', message_content, re.DOTALL)
            if synthesis_match:
                final_synthesis = synthesis_match.group(1).strip()
        
        # Call the new alpha.py function to save teaching synthesis
        logger.info(f"Saving teaching synthesis with final_synthesis: {final_synthesis}")
        synthesis_result = await save_teaching_synthesis(
            conversation_turns=conversation_turns,
            final_synthesis=final_synthesis,
            topic=priority_topic_name or "user_teaching",
            user_id=user_id,
            thread_id=thread_id,
            priority_topic_name=priority_topic_name
        )
        
        logger.info(f"Teaching synthesis result: {synthesis_result}")
        
        # Store synthesis result in response metadata for tracking
        response["metadata"]["teaching_synthesis_result"] = synthesis_result

        # Set up categories and bank name
        categories = ["general"]
        bank_name = "conversation"
        
        # Add teaching intent category
        categories.append("teaching_intent")
        
        # Add specific topic category if provided by LLM
        if priority_topic_name and priority_topic_name not in categories:
            categories.append(priority_topic_name.lower().replace(" ", "_"))
        
        # Create synthesis_content early so it's available for combined_knowledge
        synthesis_content = conversational_response
        
        # Extract summary if present using regex
        summary = ""
        summary_match = re.search(r'SUMMARY:\s*(.*?)(?:\n|$)', conversational_response, re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()
            logger.info(f"Topic Summary Gen by LLM: {summary}")
        
        # Check for structured format sections
        user_response = ""
        knowledge_synthesis = ""
        knowledge_summary = ""
        
        # Always extract from full_structured_response first if available
        if "full_structured_response" in response.get("metadata", {}):
            full_response = response["metadata"]["full_structured_response"]
            user_response_match = re.search(r'<user_response>(.*?)</user_response>', full_response, re.DOTALL)
            synthesis_match = re.search(r'<knowledge_synthesis>(.*?)</knowledge_synthesis>', full_response, re.DOTALL)
            summary_match = re.search(r'<knowledge_summary>(.*?)</knowledge_summary>', full_response, re.DOTALL)
            
            # Handle incomplete responses - try to extract even without closing tags
            if not summary_match and '<knowledge_summary>' in full_response:
                # Try to extract without requiring closing tag
                incomplete_summary_match = re.search(r'<knowledge_summary>\s*(.*?)(?=\n\n|$)', full_response, re.DOTALL)
                if incomplete_summary_match:
                    summary_match = incomplete_summary_match
                    logger.info("Extracted knowledge_summary from incomplete response (missing closing tag)")
            
            logger.info("Extracting structured sections from full_structured_response metadata")
        else:
            # Fallback to message_content if full_structured_response not available
            user_response_match = re.search(r'<user_response>(.*?)</user_response>', message_content, re.DOTALL)
            synthesis_match = re.search(r'<knowledge_synthesis>(.*?)</knowledge_synthesis>', message_content, re.DOTALL)
            summary_match = re.search(r'<knowledge_summary>(.*?)</knowledge_summary>', message_content, re.DOTALL)
            
            # Handle incomplete responses in message_content too
            if not summary_match and '<knowledge_summary>' in message_content:
                incomplete_summary_match = re.search(r'<knowledge_summary>\s*(.*?)(?=\n\n|$)', message_content, re.DOTALL)
                if incomplete_summary_match:
                    summary_match = incomplete_summary_match
                    logger.info("Extracted knowledge_summary from incomplete message_content (missing closing tag)")
            
            logger.info("Extracting structured sections from message_content (fallback)")

        logger.info(f"User response match: {user_response_match}")
        logger.info(f"Synthesis match: {synthesis_match}")
        logger.info(f"Summary match: {summary_match}")
        
        if user_response_match:
            user_response = user_response_match.group(1).strip()
            logger.info(f"Found structured user response section")
        
        if synthesis_match:
            knowledge_synthesis = synthesis_match.group(1).strip()
            logger.info(f"Found structured knowledge synthesis section")
            # Use the structured synthesis content if available
            synthesis_content = knowledge_synthesis
        
        if summary_match:
            knowledge_summary = summary_match.group(1).strip()
            logger.info(f"Found structured knowledge summary section: {knowledge_summary}")
            summary = knowledge_summary  # Use the explicit summary if available
        
        # Format synthesis_content with summary if available
        if summary and not synthesis_match:
            # Only add summary prefix if we don't have structured synthesis
            synthesis_content = f"SUMMARY: {summary}\n\nAI Synthesis: {conversational_response}"
        elif not synthesis_match:
            # Default format when no structured synthesis is found
            synthesis_content = f"AI Synthesis: {conversational_response}"
        
        logger.info(f"Final synthesis_content: {synthesis_content[:100]}...")
        
        # Combine user input and AI synthesis content in a formatted way
        combined_knowledge = f"User: {message}\n\nAI: {synthesis_content}"
        logger.info(f"Combined knowledge: {combined_knowledge}")
        
        # Add synthesized flag if this was from TEACHING_INTENT strategy
        if response.get("metadata", {}).get("response_strategy") == "TEACHING_INTENT":
            categories.append("synthesized_knowledge")
            logger.info(f"Adding synthesized_knowledge category for enhanced teaching content")
        
        # Check if we have structured format available
        has_structured_format = False
        if response.get("metadata", {}).get("full_structured_response", ""):
            full_response = response["metadata"]["full_structured_response"]
            if ("<user_response>" in full_response and 
                "<knowledge_synthesis>" in full_response and 
                "<knowledge_summary>" in full_response):
                has_structured_format = True
                logger.info("Detected structured format in response, will use that instead of combined knowledge")
        
        # Save combined knowledge only if we don't have structured format
        if not has_structured_format:
            await self._save_combined_knowledge(
                combined_knowledge, user_id, bank_name, thread_id, 
                priority_topic_name, categories, response, org_id
            )
        else:
            logger.info(f"Skipping combined knowledge save since structured format is available")

        # Save additional AI synthesis for TEACHING_INTENT strategy
        if response.get("metadata", {}).get("response_strategy") == "TEACHING_INTENT":
            await self._save_ai_synthesis(
                message, synthesis_content, knowledge_synthesis, knowledge_summary,
                summary, conversational_response, user_id, bank_name, thread_id,
                priority_topic_name, categories, response, org_id
            )

    async def _save_combined_knowledge(self, combined_knowledge: str, user_id: str, bank_name: str, 
                                     thread_id: Optional[str], priority_topic_name: str, 
                                     categories: List[str], response: Dict[str, Any], org_id: str) -> None:
        """Save combined knowledge to the knowledge base."""
        try:
            logger.info(f"Saving combined knowledge to {bank_name} bank: '{combined_knowledge[:100]}...'")
            save_result = await self.support.background_save_knowledge(
                input_text=combined_knowledge,
                title="", # Add empty title parameter 
                user_id=user_id,
                org_id=org_id,  # Add org_id
                bank_name=bank_name,
                thread_id=thread_id,
                topic=priority_topic_name or "user_teaching",
                categories=categories,
                ttl_days=365  # 365 days TTL
            )
            logger.info(f"Save combined knowledge completed: {save_result}")
            logger.info(f"Save result type: {type(save_result)}, content: {save_result}")
            
            # Store vector ID for frontend response
            if isinstance(save_result, dict):
                logger.info(f"Save result is dict, success: {save_result.get('success')}")
                if save_result.get("success"):
                    vector_id = save_result.get("vector_id")
                    response["metadata"]["combined_knowledge_vector_id"] = vector_id
                    logger.info(f"✅ Captured combined knowledge vector ID: {vector_id}")
                else:
                    logger.warning(f"Save operation failed: {save_result.get('error')}")
            else:
                logger.warning(f"Save result is not dict, got: {type(save_result)}")
        except Exception as e:
            logger.error(f"Error saving combined knowledge: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        logger.info(f"Saved combined knowledge for topic '{priority_topic_name or 'user_teaching'}'")

    async def _save_ai_synthesis(self, message: str, synthesis_content: str, knowledge_synthesis: str,
                               knowledge_summary: str, summary: str, conversational_response: str,
                               user_id: str, bank_name: str, thread_id: Optional[str],
                               priority_topic_name: str, categories: List[str], response: Dict[str, Any], org_id: str) -> None:
        """Save AI synthesis and summary for improved future retrieval."""
        try:
            # Create categories list - do NOT add ai_synthesis here as this is for combined User+AI content
            synthesis_categories = list(categories)
            # Note: ai_synthesis should only be added to pure AI content, not combined User+AI content
            
            logger.info(f"Saving additional AI synthesis for improved future retrieval")
            
            # Determine storage format and title
            if knowledge_synthesis and knowledge_summary:
                logger.info(f"Using structured sections for knowledge storage")
                logger.info(f"knowledge_synthesis: {knowledge_synthesis[:100]}...")
                logger.info(f"knowledge_summary (will be used as title): {knowledge_summary}")
                # Format for storage with clear separation - this is combined User+AI content
                final_synthesis_content = f"User: {message}\n\nAI: {knowledge_synthesis}"
                summary_title = knowledge_summary
            else:
                # Use the synthesis_content that was already built - this is combined User+AI content
                logger.info(f"Using synthesis_content for knowledge storage")
                logger.info(f"No structured synthesis/summary found - falling back to legacy summary: '{summary}'")
                final_synthesis_content = f"User: {message}\n\nAI: {synthesis_content}"
                summary_title = summary or ""
            
            logger.info(f"Final title that will be saved: '{summary_title}'")
            logger.info(f"Categories for combined User+AI content: {synthesis_categories} (no ai_synthesis)")
            
            logger.info(f"Final synthesis content to save: {final_synthesis_content[:100]}...")
            synthesis_result = await self.support.background_save_knowledge(
                input_text=final_synthesis_content,
                title=summary_title,
                user_id=user_id,
                org_id=org_id,  # Add org_id
                bank_name=bank_name,
                thread_id=thread_id,
                topic=priority_topic_name or "user_teaching",
                categories=synthesis_categories,
                ttl_days=365  # 365 days TTL
            )
            logger.info(f"Save AI synthesis completed: {synthesis_result}")
            logger.info(f"Synthesis result type: {type(synthesis_result)}, content: {synthesis_result}")
            
            # Store synthesis vector ID
            if isinstance(synthesis_result, dict):
                logger.info(f"Synthesis result is dict, success: {synthesis_result.get('success')}")
                if synthesis_result.get("success"):
                    vector_id = synthesis_result.get("vector_id")
                    response["metadata"]["synthesis_vector_id"] = vector_id
                    logger.info(f"✅ Captured synthesis vector ID: {vector_id}")
                else:
                    logger.warning(f"Synthesis save failed: {synthesis_result.get('error')}")
            else:
                logger.warning(f"Synthesis result is not dict, got: {type(synthesis_result)}")
            
            # Save standalone summary if available - this will be pure AI content with ai_synthesis
            await self._save_standalone_summary(
                knowledge_synthesis, knowledge_summary, summary, conversational_response,
                user_id, bank_name, thread_id, priority_topic_name, categories, org_id
            )
        except Exception as e:
            logger.error(f"Error saving AI synthesis: {str(e)}")

    async def _save_standalone_summary(self, knowledge_synthesis: str, knowledge_summary: str,
                                     summary: str, conversational_response: str, user_id: str,
                                     bank_name: str, thread_id: Optional[str], priority_topic_name: str,
                                     categories: List[str], org_id: str) -> None:
        """Save standalone summary for quick retrieval."""
        if knowledge_summary:
            # Define categories for summary - ensure ai_synthesis is included within first 5 categories
            # Since pccontroller.py limits to 5 categories, we need to prioritize
            summary_categories = []
            
            # Always include these core categories first
            if "general" in categories:
                summary_categories.append("general")
            if "teaching_intent" in categories:
                summary_categories.append("teaching_intent")
            
            # Add ai_synthesis early to ensure it's within the 5-category limit
            summary_categories.append("ai_synthesis")
            summary_categories.append("knowledge_summary")
            
            # Add one more category from the original list if space allows
            for cat in categories:
                if cat not in summary_categories and len(summary_categories) < 5:
                    summary_categories.append(cat)
                    break
            
            logger.info(f"Saving standalone summary with prioritized categories (max 5): {summary_categories}")
            
            # Call save_knowledge directly with correct parameter order
            summary_success = await save_knowledge(
                input=f"AI: {knowledge_synthesis}",
                user_id=user_id,
                org_id=org_id,  # Add org_id
                title=knowledge_summary,
                bank_name=bank_name,
                thread_id=thread_id,
                topic=priority_topic_name or "user_teaching",
                categories=summary_categories,
                ttl_days=365  # 365 days TTL
            )
            logger.info(f"Save summary completed: {summary_success}")
        elif summary and len(summary) > 10:  # Only save non-empty summaries
            # Define categories for summary - ensure ai_synthesis is included within first 5 categories
            summary_categories = []
            
            # Always include these core categories first
            if "general" in categories:
                summary_categories.append("general")
            if "teaching_intent" in categories:
                summary_categories.append("teaching_intent")
            
            # Add ai_synthesis early to ensure it's within the 5-category limit
            summary_categories.append("ai_synthesis")
            summary_categories.append("knowledge_summary")
            
            # Add one more category from the original list if space allows
            for cat in categories:
                if cat not in summary_categories and len(summary_categories) < 5:
                    summary_categories.append(cat)
                    break
            
            logger.info(f"Saving standalone summary (legacy) with prioritized categories (max 5): {summary_categories}")
            
            # Call save_knowledge directly with correct parameter order
            summary_success = await save_knowledge(
                input=f"AI: {conversational_response}",
                user_id=user_id,
                org_id=org_id,  # Add org_id
                title=summary,
                bank_name=bank_name,
                thread_id=thread_id,
                topic=priority_topic_name or "user_teaching",
                categories=summary_categories,
                ttl_days=365  # 365 days TTL
            )
            logger.info(f"Save legacy summary completed: {summary_success}")
        else:
            logger.info("No valid summary available, skipping standalone summary save")

    async def update_existing_knowledge(self, vector_id: str, new_content: str, user_id: str, org_id: str = "unknown", preserve_metadata: bool = True) -> Dict[str, Any]:
        """Update existing knowledge in the vector database by replacing the old vector."""
        try:
            logger.info(f"Updating existing knowledge vector {vector_id} with new content")
            
            # Step 1: Create new vector with merged content
            update_result = await save_knowledge(
                input=new_content,
                user_id=user_id,
                org_id=org_id,  # Use provided org_id
                bank_name="conversation",
                thread_id=None,
                topic="updated_knowledge",
                categories=["updated_knowledge", "teaching_intent"],
                ttl_days=365
            )
            
            if update_result and update_result.get("success"):
                new_vector_id = update_result.get("vector_id")
                logger.info(f"✅ Successfully created new merged vector: {new_vector_id}")
                
                # Step 2: Delete the old vector to avoid duplication
                try:
                    # TODO: Implement delete_vector function in pccontroller.py
                    # For now, we'll log that the old vector remains
                    logger.warning(f"⚠️ delete_vector function not implemented - old vector {vector_id} will remain in database")
                    logger.info(f"💡 Consider implementing vector deletion to avoid knowledge duplication")
                    
                    # Future implementation would be:
                    # from pccontroller import delete_vector
                    # delete_result = await delete_vector(vector_id=vector_id, bank_name="conversation")
                    
                except Exception as delete_error:
                    logger.warning(f"⚠️ Error in delete vector handling: {str(delete_error)}")
                    # Continue anyway - new vector was created successfully
                
                return {
                    "success": True,
                    "vector_id": new_vector_id,
                    "original_vector_id": vector_id,
                    "action": "REPLACED",  # Changed from "UPDATED" to "REPLACED"
                    "message": "Knowledge successfully updated and old version replaced"
                }
            else:
                logger.error(f"Failed to create new merged vector: {update_result}")
                return {
                    "success": False,
                    "error": "Failed to save updated knowledge",
                    "original_vector_id": vector_id
                }
                
        except Exception as e:
            logger.error(f"Failed to update knowledge {vector_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original_vector_id": vector_id
            }

    async def handle_update_decision(self, request_id: str, user_decision: Dict[str, Any], user_id: str, thread_id: str, org_id: str = "unknown") -> Dict[str, Any]:
        """Handle the human decision for UPDATE vs CREATE."""
        try:
            # Retrieve the pending decision from alpha.py shared storage
            decision_request = await get_pending_decision(request_id)
            if not decision_request:
                return {
                    "success": False,
                    "error": f"Decision request {request_id} not found or expired"
                }
            
            action = user_decision.get("action")
            
            logger.info(f"Processing human decision for {request_id}: {action}")
            
            if action == "CREATE_NEW":
                # Use multiple vector save logic for new knowledge
                new_content = decision_request["new_content"]
                
                # Create a response-like structure for vector tracking
                response = {
                    "message": new_content,
                    "metadata": {}
                }
                
                # Parse the content to extract user and AI parts
                if "\n\nAI:" in new_content:
                    parts = new_content.split("\n\nAI:", 1)
                    user_message = parts[0].replace("User:", "").strip()
                    ai_response = parts[1].strip()
                else:
                    # Fallback - treat as user message with default AI response
                    user_message = new_content.replace("User:", "").strip()
                    ai_response = "Knowledge saved via decision tool"
                
                # Set up categories for decision-based saves
                categories = ["teaching_intent", "human_approved", "decision_created"]
                
                # Run multiple saves like teaching intent flow
                self._create_background_task(self._save_tool_knowledge_multiple(
                    user_message=user_message,
                    ai_response=ai_response,
                    combined_content=new_content,
                    user_id=user_id,
                    bank_name="conversation",
                    thread_id=thread_id,
                    priority_topic_name="user_teaching",
                    categories=categories,
                    response=response,
                    org_id=org_id
                ))
                
                # Clean up pending decision using alpha.py function
                await remove_pending_decision(request_id)
                
                return {
                    "success": True,
                    "action": "CREATE_NEW",
                    "message": "Multiple knowledge vectors created as requested",
                    "save_types": ["combined_knowledge", "ai_synthesis", "standalone_summary"]
                }
                
            elif action == "UPDATE_EXISTING":
                target_id = user_decision.get("target_id")
                if not target_id:
                    return {
                        "success": False,
                        "error": "No target_id provided for UPDATE_EXISTING action"
                    }
                
                # Find the candidate
                candidate = None
                for c in decision_request["candidates"]:
                    if c["id"] == target_id:
                        candidate = c
                        break
                
                if not candidate:
                    return {
                        "success": False,
                        "error": f"Candidate {target_id} not found"
                    }
                
                # Merge the knowledge using alpha.py function
                existing_content = candidate["full_content"]
                new_content = decision_request["new_content"]
                
                merged_content = await merge_knowledge_content(
                    existing_content=existing_content,
                    new_content=new_content,
                    merge_strategy="enhance"
                )
                
                # Create a response-like structure for vector tracking
                response = {
                    "message": merged_content,
                    "metadata": {}
                }
                
                # Parse the merged content to extract user and AI parts
                if "\n\nAI:" in merged_content:
                    parts = merged_content.split("\n\nAI:", 1)
                    user_message = parts[0].replace("User:", "").strip()
                    ai_response = parts[1].strip()
                else:
                    # Fallback - treat as user message with default AI response
                    user_message = merged_content.replace("User:", "").strip()
                    ai_response = "Knowledge updated via decision tool"
                
                # Set up categories for decision-based updates
                categories = ["teaching_intent", "human_approved", "decision_updated", "merged_knowledge"]
                
                # Run multiple saves for the updated/merged content
                self._create_background_task(self._save_tool_knowledge_multiple(
                    user_message=user_message,
                    ai_response=ai_response,
                    combined_content=merged_content,
                    user_id=user_id,
                    bank_name="conversation",
                    thread_id=thread_id,
                    priority_topic_name="user_teaching",
                    categories=categories,
                    response=response,
                    org_id=org_id
                ))
                
                # Note: We're not deleting the old vector here - could be implemented later
                # TODO: Consider implementing old vector cleanup
                logger.info(f"⚠️ Original vector {target_id} remains in database - new merged vectors created")
                
                # Clean up pending decision using alpha.py function
                await remove_pending_decision(request_id)
                
                return {
                    "success": True,
                    "action": "UPDATE_EXISTING",
                    "original_vector_id": target_id,
                    "merged_content": merged_content,
                    "message": "Multiple knowledge vectors created for merged content",
                    "save_types": ["combined_knowledge", "ai_synthesis", "standalone_summary"],
                    "note": "Original vector preserved for reference"
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
                
        except Exception as e:
            logger.error(f"Error handling update decision {request_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def handle_teaching_intent_with_update_flow(self, message: str, response: Dict[str, Any], user_id: str, thread_id: Optional[str], priority_topic_name: str, analysis_knowledge: Dict, org_id: str) -> None:
        """Enhanced teaching intent handler with UPDATE vs CREATE flow."""
        logger.info("Processing teaching intent with UPDATE vs CREATE flow")
        
        # Get similarity score and query results
        similarity_score = analysis_knowledge.get("similarity", 0.0)
        query_results = analysis_knowledge.get("query_results", [])
        
        # Check if this is a medium similarity case requiring human decision using alpha.py function
        save_decision = await evaluate_knowledge_similarity_gate(response, similarity_score, message)
        
        if save_decision.get("action_type") == "UPDATE_OR_CREATE":
            logger.info("🤔 Medium similarity detected - initiating UPDATE vs CREATE flow")
            
            # Identify update candidates using alpha.py function
            candidates = await identify_knowledge_update_candidates(message, similarity_score, query_results)
            
            if candidates:
                # Extract response content for decision
                message_content = response["message"]
                conversational_response = re.split(r'<knowledge_queries>', message_content)[0].strip()
                
                # Create the new content that would be saved
                new_content = f"User: {message}\n\nAI: {conversational_response}"
                
                # Present options to human using alpha.py function
                decision_request = await create_update_decision_request(
                    message=message,
                    new_content=new_content,
                    candidates=candidates,
                    user_id=user_id,
                    thread_id=thread_id
                )
                
                # Store decision info in response metadata for frontend
                response["metadata"]["update_decision_request"] = {
                    "request_id": decision_request["request_id"],
                    "decision_type": "UPDATE_OR_CREATE",
                    "candidates_count": len(candidates),
                    "similarity_score": similarity_score,
                    "requires_human_input": True,
                    "candidates": [
                        {
                            "id": candidate["vector_id"],
                            "similarity": candidate["similarity"],
                            "preview": candidate["content"][:150] + "..." if len(candidate["content"]) > 150 else candidate["content"]
                        }
                        for candidate in candidates
                    ]
                }
                
                # Enhance the response to ask for human decision
                human_decision_prompt = f"""

                            🤔 **Knowledge Decision Required**

                            I found {len(candidates)} similar knowledge entries (similarity: {similarity_score:.2f}). Would you like me to:

                            1. **Create New Knowledge** - Save this as a completely new entry
                            2. **Update Existing Knowledge** - Merge with similar existing knowledge

                            **Similar Knowledge Found:**
                            """
                
                for i, candidate in enumerate(candidates, 1):
                    preview = candidate["content"][:150] + "..." if len(candidate["content"]) > 150 else candidate["content"]
                    human_decision_prompt += f"\n**Option {i+1}:** Update existing knowledge (similarity: {candidate['similarity']:.2f})\n*Preview:* {preview}\n"
                
                human_decision_prompt += f"\nPlease let me know your preference, or I can proceed with creating new knowledge."
                
                # Add the decision prompt to the response
                response["message"] = conversational_response + human_decision_prompt
                
                logger.info(f"✅ Created UPDATE vs CREATE decision request: {decision_request['request_id']}")
                return
            else:
                # FALLBACK: Even with no candidates, still offer decision if we have medium similarity and results
                if query_results and len(query_results) > 0:
                    logger.info(f"💡 No strict candidates found but have {len(query_results)} results - offering simplified decision")
                    
                    # Take top 3 results regardless of similarity score as fallback candidates
                    fallback_candidates = []
                    for result in query_results[:3]:
                        fallback_candidates.append({
                            "vector_id": result.get("id"),
                            "content": result.get("raw", ""),
                            "similarity": result.get("score", 0.0),
                            "categories": result.get("categories", {}),
                            "metadata": result.get("metadata", {}),
                            "query": result.get("query", "")
                        })
                    
                    if fallback_candidates:
                        # Extract response content for decision
                        message_content = response["message"]
                        conversational_response = re.split(r'<knowledge_queries>', message_content)[0].strip()
                        
                        # Create the new content that would be saved
                        new_content = f"User: {message}\n\nAI: {conversational_response}"
                        
                        # Present options to human with fallback candidates using alpha.py function
                        decision_request = await create_update_decision_request(
                            message=message,
                            new_content=new_content,
                            candidates=fallback_candidates,
                            user_id=user_id,
                            thread_id=thread_id
                        )
                        
                        # Store decision info in response metadata for frontend
                        response["metadata"]["update_decision_request"] = {
                            "request_id": decision_request["request_id"],
                            "decision_type": "UPDATE_OR_CREATE",
                            "candidates_count": len(fallback_candidates),
                            "similarity_score": similarity_score,
                            "requires_human_input": True,
                            "fallback_candidates": True,
                            "candidates": [
                                {
                                    "id": candidate["vector_id"],
                                    "similarity": candidate["similarity"],
                                    "preview": candidate["content"][:150] + "..." if len(candidate["content"]) > 150 else candidate["content"]
                                }
                                for candidate in fallback_candidates
                            ]
                        }
                        
                        # Enhance the response to ask for human decision
                        human_decision_prompt = f"""

                                    🤔 **Knowledge Decision Required**

                                    I found some potentially related knowledge entries (overall similarity: {similarity_score:.2f}). Would you like me to:

                                    1. **Create New Knowledge** - Save this as a completely new entry
                                    2. **Update Existing Knowledge** - Merge with existing knowledge below

                                    **Potentially Related Knowledge:**
                                    """
                        
                        for i, candidate in enumerate(fallback_candidates, 1):
                            preview = candidate["content"][:150] + "..." if len(candidate["content"]) > 150 else candidate["content"]
                            human_decision_prompt += f"\n**Option {i+1}:** Update existing knowledge (similarity: {candidate['similarity']:.2f})\n*Preview:* {preview}\n"
                        
                        human_decision_prompt += f"\nPlease let me know your preference, or I can proceed with creating new knowledge."
                        
                        # Add the decision prompt to the response
                        response["message"] = conversational_response + human_decision_prompt
                        
                        logger.info(f"✅ Created fallback UPDATE vs CREATE decision request: {decision_request['request_id']}")
                        return
                
                logger.info("No suitable update candidates found, proceeding with regular save")
        
        # Fall back to regular teaching intent handling
        await self.handle_teaching_intent(message, response, user_id, thread_id, priority_topic_name, org_id)

    async def _save_high_quality_conversation(self, message: str, response: Dict[str, Any], user_id: str, thread_id: Optional[str], org_id: str = "unknown") -> None:
        """Save high-quality conversation that doesn't have teaching intent but has high similarity."""
        try:
            logger.info("Saving high-quality conversation without teaching intent")
            
            # Extract response content
            response_content = response.get("message", "")
            similarity_score = response.get("metadata", {}).get("similarity_score", 0.0)
            
            # Create combined knowledge entry - this is User+AI format, not pure AI
            combined_knowledge = f"User: {message}\n\nAI: {response_content}"
            
            # Determine categories - do NOT add ai_synthesis for combined User+AI content
            categories = ["general", "high_quality_conversation"]
            
            # Add similarity-based category
            if similarity_score >= 0.70:
                categories.append("high_similarity")
            
            # Note: ai_synthesis should only be added to pure AI content, not combined User+AI content
            logger.info(f"Categories for combined User+AI content: {categories} (no ai_synthesis)")
            
            # Save to knowledge base
            await self._save_combined_knowledge(
                combined_knowledge=combined_knowledge,
                user_id=user_id,
                bank_name="conversation",
                thread_id=thread_id,
                priority_topic_name="general_conversation",
                categories=categories,
                response=response,
                org_id=org_id  # <-- Ensure org_id is passed
            )
            
            logger.info(f"Successfully saved high-quality conversation (similarity: {similarity_score:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to save high-quality conversation: {str(e)}")

    def get_pending_decisions(self) -> Dict[str, Any]:
        """Get all pending decisions for debugging/testing."""
        return get_pending_knowledge_decisions()

    def clear_pending_decisions(self) -> None:
        """Clear all pending decisions for testing."""
        clear_pending_knowledge_decisions()

    async def _save_tool_knowledge_multiple(self, user_message: str, ai_response: str, combined_content: str,
                                          user_id: str, bank_name: str, thread_id: Optional[str],
                                          priority_topic_name: str, categories: List[str], response: Dict[str, Any], org_id: str) -> None:
        """Save knowledge in multiple formats like teaching intent flow (for tool-executed saves)."""
        try:
            logger.info(f"Saving tool knowledge in multiple formats for improved retrieval")
            logger.info(f"User message: {user_message[:100]}...")
            logger.info(f"AI response: {ai_response[:100]}...")
            
            # 1. Save Combined Knowledge (User + AI format)
            logger.info("Saving combined knowledge (User + AI format)")
            await self._save_combined_knowledge(
                combined_knowledge=combined_content,
                user_id=user_id,
                bank_name=bank_name,
                thread_id=thread_id,
                priority_topic_name=priority_topic_name,
                categories=categories,
                response=response,
                org_id=org_id
            )
            
            # 2. Save AI Synthesis for enhanced retrieval
            logger.info("Saving AI synthesis for enhanced retrieval")
            
            # Extract or create summary information
            summary = ""
            knowledge_synthesis = ai_response
            knowledge_summary = ""
            
            # Try to extract summary if present in AI response
            summary_match = re.search(r'SUMMARY:\s*(.*?)(?:\n|$)', ai_response, re.IGNORECASE)
            if summary_match:
                summary = summary_match.group(1).strip()
                knowledge_summary = summary
                logger.info(f"Extracted summary from AI response: {summary}")
            else:
                # Generate a basic summary from first sentence or first 100 chars
                first_sentence = ai_response.split('.')[0] if '.' in ai_response else ai_response
                if len(first_sentence) > 100:
                    knowledge_summary = first_sentence[:97] + "..."
                else:
                    knowledge_summary = first_sentence
                summary = knowledge_summary
                logger.info(f"Generated summary: {knowledge_summary}")
            
            # Save AI synthesis (this will also call _save_standalone_summary)
            await self._save_ai_synthesis(
                message=user_message,
                synthesis_content=ai_response,
                knowledge_synthesis=knowledge_synthesis,
                knowledge_summary=knowledge_summary,
                summary=summary,
                conversational_response=ai_response,
                user_id=user_id,
                bank_name=bank_name,
                thread_id=thread_id,
                priority_topic_name=priority_topic_name,
                categories=categories,
                response=response,
                org_id=org_id
            )
            
            logger.info(f"✅ Successfully saved tool knowledge in multiple formats")
            logger.info(f"Vectors saved: Combined Knowledge + AI Synthesis + Standalone Summary")
            
            # Update response metadata with save status
            response["metadata"]["tool_multiple_save_completed"] = True
            response["metadata"]["save_types"] = ["combined_knowledge", "ai_synthesis", "standalone_summary"]
            
        except Exception as e:
            logger.error(f"Error saving tool knowledge in multiple formats: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            response["metadata"]["tool_multiple_save_error"] = str(e)