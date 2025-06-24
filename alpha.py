import json
import asyncio
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from time import time
from langchain_openai import ChatOpenAI
from utilities import logger

# Global state for decision management (replaces AVA class-level storage)
_pending_decisions: Dict[str, Any] = {}
_decisions_lock = asyncio.Lock()

# Initialize LLM for knowledge merging
LLM = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.01)

#This is the main function that will be called to save the teaching synthesis into local storage.
async def save_teaching_synthesis(
    conversation_turns: List[Dict[str, str]], 
    final_synthesis: str,
    topic: str,
    user_id: str,
    thread_id: Optional[str] = None,
    priority_topic_name: str = ""
) -> Dict[str, Any]:
    """
    Save teaching synthesis with conversation turns in a structured JSON format.
    
    Args:
        conversation_turns: List of {"user": "message", "ai": "response"} dicts
        final_synthesis: AI-generated synthesis of the teaching content
        topic: The main topic being taught
        user_id: User identifier
        thread_id: Optional thread identifier
        priority_topic_name: Optional priority topic name
    
    Returns:
        Dict with save result and metadata
    """
    try:
        # Create the structured synthesis object
        synthesis_data = {
            "conversation_turns": conversation_turns,
            "final_synthesis": final_synthesis,
            "metadata": {
                "topic": topic,
                "priority_topic_name": priority_topic_name,
                "user_id": user_id,
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat(),
                "turn_count": len(conversation_turns),
                "synthesis_type": "teaching_multi_turn"
            }
        }
        
        # Log the synthesis data structure
        logger.info(f"Creating teaching synthesis for topic '{topic}' with {len(conversation_turns)} turns")
        logger.info(f"Final synthesis preview: {final_synthesis[:100]}...")
        
        # For now, just return the structured data (we'll implement actual saving later)
        # This allows us to see the structure and test the flow first
        result = {
            "success": True,
            "synthesis_id": f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}",
            "data": synthesis_data,
            "message": f"Teaching synthesis prepared for topic: {topic}"
        }
        
        logger.info(f"Teaching synthesis prepared successfully: {result['synthesis_id']}")
        return result
        
    except Exception as e:
        logger.error(f"Error creating teaching synthesis: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to create teaching synthesis"
        }

# =============================================================================
# TIER 1: Pure Logic Functions (Knowledge Management Utilities)
# =============================================================================

async def evaluate_knowledge_similarity_gate(response: Dict[str, Any], similarity_score: float, message: str) -> Dict[str, Any]:
    """
    Determine if knowledge should be saved based on similarity score and conversation flow.
    CRITICAL: Teaching intent is REQUIRED for all knowledge saving to prevent pollution.
    
    UPDATED LOGIC:
    - High Similarity (>70%) + Teaching Intent → Ask human UPDATE vs CREATE
    - Low Similarity (≤70%) + Teaching Intent → Auto-save and inform human
    """
    has_teaching_intent = response.get("metadata", {}).get("has_teaching_intent", False)
    is_priority_topic = response.get("metadata", {}).get("is_priority_topic", False)
    
    # High similarity threshold for human decision
    HIGH_SIMILARITY_THRESHOLD = 0.70
    
    logger.info(f"Evaluating knowledge saving: similarity={similarity_score}, teaching_intent={has_teaching_intent}, priority_topic={is_priority_topic}")
    
    # CRITICAL: No teaching intent = NO SAVING (prevents knowledge base pollution)
    if not has_teaching_intent:
        logger.info(f"❌ No teaching intent detected - not saving knowledge (similarity={similarity_score:.2f})")
        return {
            "should_save": False,
            "reason": "no_teaching_intent",
            "confidence": "high",
            "encourage_context": True
        }
    
    # From here on, teaching intent is confirmed - evaluate based on similarity
    
    # Case 1: High similarity + teaching intent - ASK HUMAN (UPDATE vs CREATE decision)
    if similarity_score > HIGH_SIMILARITY_THRESHOLD:
        logger.info(f"🤔 High similarity ({similarity_score:.2f}) + teaching intent - asking human UPDATE vs CREATE")
        return {
            "should_save": True,
            "action_type": "UPDATE_OR_CREATE",
            "reason": "high_similarity_teaching_decision",
            "confidence": "high",
            "requires_human_decision": True,
            "similarity_score": similarity_score
        }
    
    # Case 2: Low-medium similarity + teaching intent - AUTO-SAVE as new knowledge
    if similarity_score <= HIGH_SIMILARITY_THRESHOLD:
        logger.info(f"✅ Low-medium similarity ({similarity_score:.2f}) + teaching intent - auto-saving as new knowledge")
        return {
            "should_save": True,
            "reason": "low_medium_similarity_teaching",
            "confidence": "high",
            "is_new_knowledge": True,
            "auto_save": True
        }
    
    # Default case - don't save
    logger.info(f"❌ Default case - not saving knowledge (similarity={similarity_score:.2f})")
    return {
        "should_save": False,
        "reason": "default_no_save",
        "confidence": "low"
    }

async def identify_knowledge_update_candidates(message: str, similarity_score: float, query_results: List[Dict]) -> List[Dict]:
    """Identify existing knowledge that could be updated."""
    candidates = []
    
    # For high overall similarity (>70%), be more flexible with individual result filtering
    if similarity_score > 0.70:
        logger.info(f"High overall similarity ({similarity_score:.3f}) - using flexible candidate selection")
        
        # For high overall similarity, take top results with reasonable similarity
        for result in query_results:
            result_similarity = result.get("score", 0.0)
            # Accept any result with similarity >0.3 for high overall similarity cases
            if result_similarity >= 0.30:
                candidates.append({
                    "vector_id": result.get("id"),
                    "content": result.get("raw", ""),
                    "similarity": result_similarity,
                    "categories": result.get("categories", {}),
                    "metadata": result.get("metadata", {}),
                    "query": result.get("query", "")
                })
    else:
        # For lower overall similarity, this method shouldn't be called, but handle gracefully
        logger.info(f"Lower overall similarity ({similarity_score:.3f}) - this shouldn't trigger UPDATE vs CREATE")
        return []
    
    # Sort by similarity (highest first) and take top 3 candidates
    sorted_candidates = sorted(candidates, key=lambda x: x["similarity"], reverse=True)[:3]
    
    logger.info(f"Identified {len(sorted_candidates)} update candidates from {len(query_results)} query results")
    logger.info(f"Candidate similarities: {[c['similarity'] for c in sorted_candidates]}")
    return sorted_candidates

async def merge_knowledge_content(existing_content: str, new_content: str, merge_strategy: str = "enhance") -> str:
    """Intelligently merge new knowledge with existing knowledge."""
    
    merge_prompt = f"""You are merging knowledge. Combine the EXISTING knowledge with NEW information intelligently.

                EXISTING KNOWLEDGE:
                {existing_content}

                NEW INFORMATION:
                {new_content}

                MERGE STRATEGY: {merge_strategy}

                Instructions:
                1. Preserve all valuable information from EXISTING knowledge
                2. Integrate NEW information where it adds value
                3. If there are contradictions, note both perspectives clearly
                4. Organize the merged content logically with clear structure
                5. Maintain the same language as the original content
                6. Remove any redundant information
                7. Enhance clarity and completeness

                Format the output as:
                User: [Combined user inputs if applicable]

                AI: [Merged and enhanced knowledge content]

                MERGED KNOWLEDGE:
                """
    
    try:
        response = await LLM.ainvoke(merge_prompt)
        merged_content = response.content.strip()
        logger.info(f"Successfully merged knowledge using LLM (original: {len(existing_content)} chars, new: {len(new_content)} chars, merged: {len(merged_content)} chars)")
        return merged_content
    except Exception as e:
        logger.error(f"Knowledge merge failed: {str(e)}")
        # Fallback: structured concatenation
        fallback_content = f"{existing_content}\n\n--- UPDATED WITH NEW INFORMATION ---\n\n{new_content}"
        logger.info(f"Using fallback merge strategy")
        return fallback_content

# =============================================================================
# TIER 2: Content Processing Functions
# =============================================================================

def extract_user_facing_content(content: str, response_strategy: str, structured_sections: Dict) -> str:
    """Extract the user-facing content from the response."""
    user_facing_content = content
    
    if response_strategy == "TEACHING_INTENT":
        # Extract user response (this is what should be sent to the user)
        if structured_sections.get("user_response"):
            user_facing_content = structured_sections["user_response"]
            logger.info(f"✅ Extracted user-facing content from structured response (length: {len(user_facing_content)})")
        else:
            # Remove knowledge_synthesis and knowledge_summary sections if they exist
            user_facing_content = re.sub(r'<knowledge_synthesis>.*?</knowledge_synthesis>', '', content, flags=re.DOTALL).strip()
            user_facing_content = re.sub(r'<knowledge_summary>.*?</knowledge_summary>', '', user_facing_content, flags=re.DOTALL).strip()
            logger.info(f"⚠️ No structured user_response found - cleaned non-user sections from response (length: {len(user_facing_content)})")
    
    # Handle JSON responses
    if content.startswith('{') and '"message"' in content:
        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, dict) and "message" in parsed_json:
                user_facing_content = parsed_json["message"]
                logger.info("Extracted message from JSON response")
        except Exception as json_error:
            logger.warning(f"Failed to parse JSON response: {json_error}")
    
    return user_facing_content

async def regenerate_teaching_intent_response(message_str: str, original_content: str, response_strategy: str) -> tuple:
    """Handle regeneration of response when teaching intent is detected."""
    original_strategy = response_strategy
    response_strategy = "TEACHING_INTENT"
    logger.info(f"LLM detected teaching intent, changing response_strategy from {original_strategy} to TEACHING_INTENT")
    
    # Step 1: Enhance the original content with smart follow-up questions
    enhancement_prompt = f"""You have this excellent response to preserve:

ORIGINAL RESPONSE:
{original_content}

TASK: Keep the original response EXACTLY as is, but add a natural transition and EXACTLY 1-2 thoughtful follow-up questions (no more than 2 questions total).

Requirements:
1. **Preserve original content**: Keep every word of the original response exactly as provided
2. **Add natural bridge**: Add a smooth transition sentence that naturally leads to the questions
3. **Add questions**: Include EXACTLY 1-2 thoughtful follow-up questions (MAXIMUM 2 questions) that:
   - Are open-ended and insightful
   - Directly relate to the specific topic being taught
   - Encourage sharing of practical details, examples, or experiences
   - Ask about challenges, edge cases, or advanced aspects
   - Keep the same language and tone as the original response

CRITICAL: You MUST provide the complete response. Do not truncate or cut off your output.

Structure (COMPLETE THIS ENTIRE STRUCTURE):
[ORIGINAL RESPONSE EXACTLY AS PROVIDED]

[One natural transition sentence that bridges to questions]

[EXACTLY 1-2 follow-up questions - NO MORE THAN 2]

Example transition phrases (adapt to context and language):
- "Để hiểu rõ hơn về..." / "To understand more about..."
- "Dựa trên những thông tin này..." / "Based on this information..."
- "Trong thực tế..." / "In practice..."

Your complete enhanced response:"""

    try:
        # Use a dedicated LLM instance with higher token limit for enhancement
        from langchain_openai import ChatOpenAI
        enhancement_llm = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.1)
        enhancement_response = await enhancement_llm.ainvoke(enhancement_prompt)
        enhanced_user_content = enhancement_response.content.strip()
        logger.info(f"Successfully enhanced original response with follow-up questions (length: {len(enhanced_user_content)})")
        logger.info(f"Enhanced content preview: {enhanced_user_content[:200]}...")
        logger.info(f"Enhanced content full: {enhanced_user_content}")
    except Exception as e:
        logger.error(f"Failed to enhance original response: {str(e)}")
        # Fallback: use original content
        enhanced_user_content = original_content

    # Step 2: Generate knowledge sections based on the original user message
    teaching_prompt = f"""IMPORTANT: The user is TEACHING you something. Extract pure knowledge from their message.
    
    Original user message: {message_str}
    
    Instructions:
    Generate TWO separate outputs for knowledge storage:
    
    <knowledge_synthesis>
       This is for knowledge storage - include ONLY:
       - Factual information extracted from the user's message
       - Structured, clear explanation of the concepts
       - NO greeting phrases, acknowledgments, or questions
       - NO conversational elements - pure knowledge only
       - Organized in logical sections if appropriate
    </knowledge_synthesis>
    
    <knowledge_summary>
       A concise 2-3 sentence summary capturing the core teaching point
       This should be factual and descriptive, not conversational
    </knowledge_summary>
    
    CRITICAL: RESPOND IN THE SAME LANGUAGE AS THE USER'S MESSAGE.
    - If the user wrote in Vietnamese, respond entirely in Vietnamese
    - If the user wrote in English, respond entirely in English
    - Match the language exactly - do not mix languages
    
    Your knowledge extraction:
    """
    
    try:
        knowledge_response = await LLM.ainvoke(teaching_prompt)
        knowledge_content = knowledge_response.content.strip()
        logger.info("Successfully generated knowledge sections")
    except Exception as e:
        logger.error(f"Failed to generate knowledge sections: {str(e)}")
        # Fallback: create basic sections
        knowledge_content = f"""<knowledge_synthesis>
        {message_str}
        </knowledge_synthesis>

        <knowledge_summary>
        Knowledge shared by user about the topic.
        </knowledge_summary>"""

    # Step 3: Combine enhanced user response with knowledge sections
    final_content = f"""<user_response>
            {enhanced_user_content}
            </user_response>

            {knowledge_content}"""

    return final_content, response_strategy

# =============================================================================
# TIER 3: Decision Management Functions
# =============================================================================

async def create_update_decision_request(message: str, new_content: str, candidates: List[Dict], user_id: str, thread_id: str) -> Dict[str, Any]:
    """Present UPDATE vs CREATE options to human via tool call."""
    
    # Prepare the decision request
    decision_request = {
        "decision_type": "UPDATE_OR_CREATE",
        "user_message": message,
        "new_content": new_content,
        "similarity_info": f"Found {len(candidates)} similar knowledge entries",
        "options": [
            {
                "action": "CREATE_NEW",
                "description": "Save as completely new knowledge",
                "reasoning": "This information is sufficiently different to warrant a new entry"
            }
        ],
        "candidates": []
    }
    
    # Add UPDATE options for each candidate
    for i, candidate in enumerate(candidates, 1):
        # Clean content preview
        content_preview = candidate["content"]
        if content_preview.startswith("User:") and "\n\nAI:" in content_preview:
            ai_part = content_preview.split("\n\nAI:", 1)[1] if "\n\nAI:" in content_preview else content_preview
            content_preview = ai_part.strip()
        
        # Truncate for preview
        preview = content_preview[:200] + "..." if len(content_preview) > 200 else content_preview
        
        decision_request["options"].append({
            "action": "UPDATE_EXISTING",
            "target_id": candidate["vector_id"],
            "description": f"Update existing knowledge #{i} (similarity: {candidate['similarity']:.2f})",
            "preview": preview,
            "reasoning": f"Enhance existing knowledge with new information (similarity: {candidate['similarity']:.2f})"
        })
        
        # Add candidate details for reference
        logger.info(f"ADDING candidates: {candidate["content"]}")
        decision_request["candidates"].append({
            "id": candidate["vector_id"],
            "similarity": candidate["similarity"],
            "preview": preview,
            "full_content": candidate["content"]
        })

    
    # Store the pending decision for later retrieval
    request_id = f"update_decision_{thread_id}_{int(time())}"
    decision_request["request_id"] = request_id
    
    # Store in shared global storage
    async with _decisions_lock:
        _pending_decisions[request_id] = decision_request
    
    logger.info(f"Created UPDATE vs CREATE decision request {request_id} with {len(candidates)} candidates")
    return decision_request

def get_pending_knowledge_decisions() -> Dict[str, Any]:
    """Get all pending decisions for debugging/testing."""
    return _pending_decisions.copy()

def clear_pending_knowledge_decisions() -> None:
    """Clear all pending decisions for testing."""
    _pending_decisions.clear()
    logger.info("Cleared all pending knowledge decisions")

async def get_pending_decision(request_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific pending decision by request_id."""
    async with _decisions_lock:
        return _pending_decisions.get(request_id)

async def remove_pending_decision(request_id: str) -> bool:
    """Remove a pending decision by request_id."""
    async with _decisions_lock:
        if request_id in _pending_decisions:
            del _pending_decisions[request_id]
            return True
        return False
