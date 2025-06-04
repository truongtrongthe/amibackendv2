import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
import re
import pytz
from langchain_openai import ChatOpenAI
from pccontroller import save_knowledge, query_knowledge, query_knowledge_from_graph
from utilities import logger, EMBEDDINGS

# Initialize LLM for support functions
LLM = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.01)

class LearningSupport:
    """Support class containing utility functions for LearningProcessor."""
    
    def __init__(self, learning_processor):
        self.learning_processor = learning_processor
    
    async def determine_response_strategy(self, flow_type: str, flow_confidence: float, message_characteristics: Dict, 
                                  knowledge_relevance: Dict, similarity_score: float, prior_knowledge: str,
                                  queries: List, query_results: List, knowledge_response_sections: List, 
                                  conversation_context: str = "", message_str: str = "") -> Dict[str, Any]:
        """Determine the response strategy based on conversation flow and message analysis."""
        
        # Get universal pronoun guidance for ALL strategies
        pronoun_guidance = self._get_universal_pronoun_guidance(conversation_context, message_str)
        
        is_confirmation = flow_type == "CONFIRMATION"
        is_follow_up = flow_type in ["FOLLOW_UP", "CONFIRMATION"]
        is_practice_request = flow_type == "PRACTICE_REQUEST"
        is_closing = flow_type == "CLOSING"
        
        # Extract message characteristics (but don't use has_teaching_markers anymore)
        is_closing_message = message_characteristics["is_closing_message"] or is_closing
        is_vn_greeting = message_characteristics["is_vn_greeting"]
        contains_vn_name = message_characteristics["contains_vn_name"]
        is_short_message = message_characteristics["is_short_message"]
        
        # Extract knowledge relevance
        best_context_relevance = knowledge_relevance["best_context_relevance"]
        has_low_relevance_knowledge = knowledge_relevance["has_low_relevance_knowledge"]
        
        # Use LLM to detect teaching intent instead of primitive rule-based detection
        has_teaching_intent_llm = await self.detect_teaching_intent_llm(message_str, conversation_context)
        
        if is_closing_message:
            return {
                "strategy": "CLOSING",
                "instructions": (
                    f"{pronoun_guidance}\n\n"
                    "Recognize this as a closing message where the user is ending the conversation. "
                    "Respond with a brief, polite farewell message. "
                    "Thank them for the conversation and express willingness to help in the future. "
                    "Keep it concise and friendly, in the same language they used (Vietnamese/English). "
                    "MAINTAIN established pronoun relationships in your farewell."
                ),
                "knowledge_context": "CONVERSATION_CLOSING: User is ending the conversation politely.",
                "similarity_score": similarity_score
            }
        
        elif is_practice_request and prior_knowledge:
            return {
                "strategy": "PRACTICE_REQUEST",
                "instructions": (
                    f"{pronoun_guidance}\n\n"
                    "The user wants you to DEMONSTRATE or APPLY previously shared knowledge. "
                    "Create a practical example that follows these steps: "
                    
                    "1. Acknowledge their request positively and with enthusiasm. "
                    "2. Reference the prior knowledge in your response directly. "
                    "3. Apply the knowledge in a realistic scenario or example. "
                    "4. Follow any specific methods or steps previously discussed. "
                    "5. Explain your reasoning as you demonstrate. "
                    "6. Ask if your demonstration meets their expectations. "
                    
                    "IMPORTANT: The user is asking you to SHOW your understanding, not asking for new information. "
                    "Even if the request is vague like 'Em thử áp dụng các kiến thức em có anh xem nào', "
                    "understand that they want you to DEMONSTRATE the knowledge you gained from previous messages. "
                    "Be confident and enthusiastic - this is a chance to show what you've learned."
                    
                    "CRITICAL: If the knowledge includes communication techniques, relationship building, or language patterns, "
                    "ACTIVELY USE these techniques in your response format, not just talk about them. For example:"
                    "- If knowledge mentions using 'em' to refer to yourself, use that pronoun in your response "
                    "- If it suggests addressing users as 'anh/chị', use that form of address "
                    "- If it recommends specific phrases or compliments, incorporate them naturally "
                    "- If it suggests question techniques, use those exact techniques at the end of your response"
                ),
                "knowledge_context": prior_knowledge,
                "similarity_score": max(similarity_score, 0.8)
            }
        
        elif has_low_relevance_knowledge and similarity_score > 0.3:
            return {
                "strategy": "LOW_RELEVANCE_KNOWLEDGE",
                "instructions": (
                    f"{pronoun_guidance}\n\n"
                    "You have knowledge with low relevance to the current query. "
                    "PRIORITIZE the user's current message over the retrieved knowledge. "
                    "ONLY reference the knowledge if it genuinely helps answer the query. "
                    "If the knowledge is off-topic, IGNORE it completely and focus on the user's message. "
                    "Be clear and direct in addressing what the user is actually asking about. "
                    "Generate a response primarily based on the user's current message and intent."
                    
                    "However, if the knowledge contains ANY communication techniques or relationship-building approaches, "
                    "incorporate those techniques into HOW you construct your response, even if the topic is different."
                ),
                "knowledge_context": f"LOW RELEVANCE KNOWLEDGE WARNING: The retrieved knowledge has low relevance " \
                    f"(score: {best_context_relevance:.2f}) to the current query. Prioritize the user's message.\n\n",
                "similarity_score": similarity_score
            }
        
        elif is_follow_up:
            # For follow-ups, ensure conversation context is included in knowledge context
            conversation_entities_context = f"""
                    CONVERSATION CONTEXT FOR REFERENCE RESOLUTION:
                    {conversation_context}

                    ENTITY TRACKING INSTRUCTIONS:
                    - The above conversation contains entities, topics, and concepts that the current message may be referring to
                    - When the user uses pronouns like "nó" (it), "cái đó" (that), "it", "that", etc., scan the conversation above to identify what specific entity they're referencing
                    - Look for the most recently mentioned relevant entity that matches the context of the current question
                    - Common entity types to track: places, buildings, companies, people, concepts, products, locations
                    - Respond with specific information about the identified entity, not generic responses

                    PRONOUN RESOLUTION GUIDANCE:
                    - "nó" (it) → Look for the main subject/entity being discussed in recent messages
                    - "cái đó" (that thing) → Look for objects, concepts, or entities mentioned previously  
                    - "chỗ đó" (that place) → Look for locations, places, or areas mentioned before
                    - Provide specific details about the identified entity rather than asking for clarification
                    """
            
            if is_confirmation:
                instructions = (
                    f"{pronoun_guidance}\n\n"
                    "Recognize this is a direct confirmation to your question in your previous message. "
                    "Continue the conversation as if the user said 'yes' to your previous question. "
                    "Provide a helpful response that builds on the previous question, offering relevant details or asking a follow-up question. "
                    "Don't ask for clarification when the confirmation is clear - proceed with the conversation flow naturally. "
                    "If your previous question offered to provide more information, now is the time to provide that information. "
                    "Keep the response substantive, helpful, and directly related to what the user just confirmed interest in."
                )
                context_to_use = prior_knowledge if prior_knowledge else conversation_entities_context
            else:
                instructions = (
                    f"{pronoun_guidance}\n\n"
                    "**FOLLOW-UP RESPONSE**: "
                    "This is a continuation of the previous conversation. The user is referring to, asking about, or building upon something from earlier messages. "
                    
                    "**Natural Context Understanding**: "
                    "- Read the conversation history to understand what the user is referring to "
                    "- If they use pronouns or implicit references, resolve them using context "
                    "- Provide information about the specific entity, concept, or topic they're asking about "
                    "- Treat this as a natural continuation of the previous discussion "
                    
                    "**Response Approach**: "
                    "- Answer their question directly based on what they're referring to "
                    "- Use any relevant knowledge from previous messages or your knowledge base "
                    "- Provide helpful, specific information about the referenced topic "
                    "- Ask a natural follow-up question to continue the conversation "
                    
                    "**Be Natural**: Don't ask for clarification unless truly necessary - use context to understand what they mean."
                )
                # Always include conversation context for follow-ups to enable pronoun resolution
                context_to_use = f"{conversation_entities_context}\n\nPRIOR KNOWLEDGE:\n{prior_knowledge}" if prior_knowledge else conversation_entities_context
            
            return {
                "strategy": "FOLLOW_UP",
                "instructions": instructions,
                "knowledge_context": context_to_use,
                "similarity_score": max(similarity_score, 0.7) if not knowledge_response_sections else similarity_score
            }
        
        elif (is_vn_greeting or contains_vn_name) and is_short_message and similarity_score < 0.35:
            return {
                "strategy": "GREETING",
                "instructions": (
                    f"{pronoun_guidance}\n\n"
                    "Recognize this as a Vietnamese greeting or someone addressing you by name. "
                    "Respond warmly and appropriately to the greeting. "
                    "If they used a Vietnamese name or greeting form, respond in Vietnamese. "
                    "Keep your response friendly, brief, and conversational. "
                    "Ask how you can assist them today. "
                    "Ensure your tone matches the formality level they used (formal vs casual)."
                ),
                "knowledge_context": "",
                "similarity_score": similarity_score
            }
        
        elif similarity_score < 0.35 and not knowledge_response_sections:
            # Check if this might be a follow-up that we can handle contextually
            if conversation_context and (is_follow_up or len(message_str.split()) < 8):
                return {
                    "strategy": "CONTEXTUAL_RESPONSE",
                    "instructions": (
                        f"{pronoun_guidance}\n\n"
                        "**CONTEXTUAL UNDERSTANDING RESPONSE**: "
                        "Even though I don't have specific stored knowledge about this topic, I can still help by: "
                        
                        "1. **Using Conversation Context**: Read the full conversation to understand what the user is asking about "
                        "2. **General Knowledge**: Apply my general knowledge to answer their question if possible "
                        "3. **Contextual References**: Resolve any pronouns or references to previous topics naturally "
                        "4. **Helpful Response**: Provide useful information even if I don't have specific stored knowledge "
                        
                        "**Approach**: "
                        "- Don't say 'I can't find information' unless truly necessary "
                        "- Use context clues and general knowledge to provide helpful answers "
                        "- If asking about general topics (animals, places, etc.), provide what you know "
                        "- Keep the conversation flowing naturally "
                        
                        "**Example**: If they ask about animals on Earth after discussing Earth, provide information about large animals on Earth using general knowledge."
                    ),
                    "knowledge_context": f"CONVERSATION CONTEXT:\n{conversation_context}\n\nUSE CONTEXT AND GENERAL KNOWLEDGE: Even without specific stored knowledge, use the conversation context and your general knowledge to provide a helpful response.",
                    "similarity_score": 0.5  # Boost confidence for contextual responses
                }
            
            if is_short_message:
                instructions = (
                    f"{pronoun_guidance}\n\n"
                    "**SHORT MESSAGE CLARIFICATION**: "
                    "This seems like a short message that might need clarification. However, try to understand from context first: "
                    
                    "1. **Check Context**: Look at the conversation history to see if this relates to previous topics "
                    "2. **Use Clues**: Use any context clues to understand what they might be asking "
                    "3. **Helpful Guess**: If you can make a reasonable interpretation, provide a helpful response "
                    "4. **Gentle Clarification**: Only ask for clarification if truly unclear "
                    
                    "Keep your response friendly and try to be helpful even with limited information. "
                    "Match the user's language choice (Vietnamese/English)."
                )
            else:
                instructions = (
                    f"{pronoun_guidance}\n\n"
                    "**GENERAL KNOWLEDGE RESPONSE**: "
                    "While I don't have specific stored knowledge about this topic, I can still try to help: "
                    
                    "1. **Apply General Knowledge**: Use general knowledge to answer if the question is about common topics "
                    "2. **Use Context**: Consider the conversation context to understand what they're asking "
                    "3. **Be Helpful**: Provide useful information rather than just saying 'I don't know' "
                    "4. **Acknowledge Limitations**: If truly unable to help, politely explain and ask for more details "
                    
                    "Try to provide value even without specific stored knowledge. If you genuinely cannot help, then politely ask for clarification."
                )
            
            return {
                "strategy": "LOW_SIMILARITY",
                "instructions": instructions,
                "knowledge_context": f"CONVERSATION CONTEXT:\n{conversation_context}\n\nGENERAL KNOWLEDGE GUIDANCE: Use your general knowledge and conversation context to provide helpful responses even when specific stored knowledge is limited.",
                "similarity_score": similarity_score
            }
        
        elif has_teaching_intent_llm:
            return {
                "strategy": "TEACHING_INTENT",
                "instructions": (
                    f"{pronoun_guidance}\n\n"
                    "Recognize this message as TEACHING INTENT where the user is sharing knowledge with you. "
                    "Your goal is to synthesize this knowledge for future use and demonstrate understanding. "
                    
                    "Generate THREE separate outputs in your response:\n\n"
                    
                    "<user_response>\n"
                    "   This is what the user will see - include:\n"
                    "   - Acknowledgment of their teaching with appreciation\n"
                    "   - Demonstration of your understanding\n"
                    "   - End with 1-2 open-ended questions to deepen the conversation\n"
                    "   - Make this conversational and engaging\n"
                    "</user_response>\n\n"
                    
                    "<knowledge_synthesis>\n"
                    "   This is for knowledge storage - include ONLY:\n"
                    "   - Factual information extracted from the user's message\n"
                    "   - Structured, clear explanation of the concepts\n"
                    "   - NO greeting phrases, acknowledgments, or questions\n"
                    "   - NO conversational elements - pure knowledge only\n"
                    "   - Organized in logical sections if appropriate\n"
                    "</knowledge_synthesis>\n\n"
                    
                    "<knowledge_summary>\n"
                    "   A concise 2-3 sentence summary capturing the core teaching point\n"
                    "   This should be factual and descriptive, not conversational\n"
                    "</knowledge_summary>\n\n"
                    
                    "CRITICAL LANGUAGE INSTRUCTION: ALWAYS respond in EXACTLY the SAME LANGUAGE as the user's message for ALL sections. "
                    "- If the user wrote in Vietnamese, respond entirely in Vietnamese "
                    "- If the user wrote in English, respond entirely in English "
                    "- Do not mix languages in your response "
                    
                    "This structured approach helps create high-quality, reusable knowledge while maintaining good user experience."
                ),
                "knowledge_context": "",
                "similarity_score": similarity_score
            }
        
        else:
            return {
                "strategy": "RELEVANT_KNOWLEDGE",
                "instructions": (
                    f"{pronoun_guidance}\n\n"
                    "I've found MULTIPLE knowledge entries relevant to your query. Let me provide a comprehensive response.\n\n"
                    "For each knowledge item found:\n"
                    "1. Review and synthesize the information from ALL available knowledge items\n"
                    "2. When answering, incorporate insights from ALL relevant knowledge items found\n"
                    "3. Show how different knowledge entries complement or confirm each other\n"
                    "4. If there are any contradictions between knowledge items, highlight them\n"
                    "5. Present information in order of relevance, addressing the most relevant points first\n\n"
                    "DO NOT ignore any of the provided knowledge items - incorporate insights from ALL of them in your response.\n"
                    "DO NOT summarize the knowledge as 'I found X items' - just seamlessly incorporate all relevant information.\n\n"
                    "MOST IMPORTANTLY: If the knowledge contains ANY communication techniques, relationship-building strategies, "
                    "or specific linguistic patterns, ACTIVELY APPLY these in how you structure your response. For example:"
                    "- If the knowledge mentions using 'em/tôi' or specific pronouns, use those exact pronouns yourself"
                    "- If it suggests addressing the user in specific ways ('anh/chị/bạn'), use that exact form of address"
                    "- If it recommends compliments or specific phrases, incorporate them naturally in your response"
                    "- If it mentions conversation flow techniques, apply them in how you structure this very response"
                    "This way, you're not just explaining the knowledge but DEMONSTRATING it in action."
                ),
                "knowledge_context": "",
                "similarity_score": similarity_score
            }

    def build_llm_prompt(self, message_str: str, conversation_context: str, temporal_context: str, 
                        knowledge_context: str, response_strategy: str, strategy_instructions: str,
                        core_prior_topic: str, user_id: str) -> str:
        """
        Build a dynamic, context-aware LLM prompt.
        
        🎯 OPTIMIZATION RESULTS (reduced from ~4000 to ~2600 tokens, -35% size):
        ✅ Consolidated teaching intent detection (removed 50% redundancy, added critical examples)
        ✅ Merged pronoun guidance into single section (removed 50% redundancy)  
        ✅ Eliminated duplicate conversation history instructions (removed 40% redundancy)
        ✅ Streamlined confidence instructions (removed 50% redundancy, strengthened knowledge usage)
        ✅ Simplified language context detection (removed 30% redundancy)
        
        🔒 PRESERVED & ENHANCED FUNCTIONALITY:
        ✅ All required evaluation outputs maintained
        ✅ Teaching intent detection with critical examples and nuance
        ✅ STRENGTHENED knowledge utilization requirements
        ✅ Strategy-specific instructions preserved
        ✅ Knowledge context formatting enhanced with mandatory usage rules
        ✅ Tool availability and usage instructions kept
        
        🛠️ CRITICAL FIXES APPLIED:
        ✅ Added specific examples for teaching intent detection accuracy
        ✅ Emphasized MANDATORY knowledge usage to prevent generic responses
        ✅ Strengthened instructions to use available knowledge instead of asking for more
        ✅ Added organization requirements for comprehensive knowledge synthesis
        
        Token Distribution (optimized + fixed):
        - Base Prompt: ~1200 tokens (was 1500, added critical examples)
        - Strategy Instructions: ~600-800 tokens (was 800-1200, enhanced knowledge emphasis)  
        - Confidence Instructions: ~250-350 tokens (was 400-600, strengthened knowledge usage)
        - Evaluation Output: ~400 tokens (unchanged)
        - Total: ~2450-2750 tokens (was 3500-4100, -35% reduction with enhanced functionality)
        """
        
        # Always include base prompt (~1000 tokens - reduced from 1500)
        prompt = self._get_optimized_base_prompt(message_str, conversation_context, temporal_context, user_id)
        
        # Add strategy-specific instructions (~600-800 tokens - reduced from 800-1200)
        prompt += self._get_strategy_instructions(response_strategy, strategy_instructions, 
                                                knowledge_context, core_prior_topic)
        
        # Add confidence-level instructions (~200-300 tokens - reduced from 400-600)
        similarity_score = self._extract_similarity_from_context(knowledge_context)
        prompt += self._get_streamlined_confidence_instructions(similarity_score)
        
        # Add evaluation section instructions - CRITICAL for teaching intent detection
        prompt += """
        
        **MANDATORY EVALUATION OUTPUT**:
        After your response, you MUST include an evaluation section in this exact format:

        <evaluation>
        {
            "has_teaching_intent": true/false,
            "is_priority_topic": true/false,
            "priority_topic_name": "topic name or empty string",
            "should_save_knowledge": true/false,
            "intent_type": "teaching/query/clarification/practice_request/closing",
            "name_addressed": true/false,
            "ai_referenced": true/false
        }
        </evaluation>

        **EVALUATION CRITERIA**:
        - **has_teaching_intent**: TRUE if user is INFORMING/DECLARING/ANNOUNCING (not asking questions)
        - **is_priority_topic**: TRUE if the topic appears important for future reference
        - **priority_topic_name**: Short descriptive name for the topic being taught
        - **should_save_knowledge**: TRUE if this interaction contains valuable information to save
        - **intent_type**: Primary intent category based on user's message
        - **name_addressed**: TRUE if user mentioned your name or referenced you directly
        - **ai_referenced**: TRUE if user explicitly mentioned AI, assistant, or similar terms

        This evaluation is MANDATORY and must be included in every response.
        """
        
        return prompt
    
    async def search_knowledge(self, message: Union[str, List], conversation_context: str = "", user_id: str = "unknown", thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Search for relevant knowledge based on message and context."""
        logger.info(f"Searching for analysis knowledge based on message: {str(message)[:100]}...")
        try:
            if not isinstance(message, str):
                logger.warning(f"Converting non-string message: {message}")
                primary_query = str(message[0]) if isinstance(message, list) and message else str(message)
            else:
                primary_query = message.strip()
            if not primary_query:
                logger.error("Empty primary query")
                return {
                    "knowledge_context": "",
                    "similarity": 0.0,
                    "query_count": 0,
                    "prior_data": {"topic": "", "knowledge": ""},
                    "metadata": {"similarity": 0.0}
                }
            queries = []
            prior_topic = ""
            prior_knowledge = ""
            prior_messages = []
            if conversation_context:
                user_messages = re.findall(r'User: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                logger.info(f"Found {len(user_messages)} user messages in context")
                if user_messages:
                    prior_messages = user_messages[:-1]  # All but the current message
                    prior_topic = user_messages[-2].strip() if len(user_messages) > 1 else ""
                    logger.info(f"Extracted prior topic: {prior_topic[:50]}")
                if ai_messages:
                    prior_knowledge = ai_messages[-1].strip()

            # Use the new conversation flow detection instead of pattern matching
            flow_result = await self.detect_conversation_flow(
                message=primary_query,
                prior_messages=prior_messages,
                conversation_context=conversation_context
            )
            
            flow_type = flow_result.get("flow_type", "NEW_TOPIC")
            flow_confidence = flow_result.get("confidence", 0.5)
            
            is_follow_up = flow_type in ["FOLLOW_UP", "CONFIRMATION"]
            is_practice_request = flow_type == "PRACTICE_REQUEST"
            
            logger.info(f"Conversation flow: {flow_type} (confidence: {flow_confidence})")
            
            if is_follow_up and prior_topic:
                queries.append(prior_topic)
                logger.info(f"Follow-up detected, reusing prior topic: {prior_topic[:50]}")
                similarity = 0.7
                knowledge_context = prior_knowledge
            elif is_practice_request and prior_knowledge:
                # For practice requests, prioritize last AI message as knowledge
                queries.append(primary_query)
                queries.append(prior_topic if prior_topic else primary_query)
                logger.info(f"Practice request detected, using previous AI knowledge as foundation")
                # Higher confidence for practice scenarios (we're quite certain about our prior knowledge)
                similarity = 0.85
                knowledge_context = prior_knowledge
                
                # For practice requests, log specific phrases detected
                practice_indicators = []
                if "thử" in primary_query.lower() and ("xem" in primary_query.lower() or "nào" in primary_query.lower()):
                    practice_indicators.append("thử...xem/nào")
                if "áp dụng" in primary_query.lower():
                    practice_indicators.append("áp dụng")
                if practice_indicators:
                    logger.info(f"Practice request indicators: {', '.join(practice_indicators)}")
                
                # Return early with high confidence for clear practice requests
                if flow_confidence > 0.8:
                    logger.info(f"High confidence practice request - prioritizing previous knowledge")
                    return {
                        "knowledge_context": knowledge_context,
                        "similarity": similarity,
                        "query_count": 1,
                        "queries": queries,
                        "original_query": primary_query,
                        "query_results": [{"raw": knowledge_context, "score": similarity, "metadata": {"practice_request": True}}],
                        "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                        "metadata": {"similarity": similarity, "vibe_score": 1.1, "flow_type": flow_type}
                    }
            else:
                queries.append(primary_query)
                similarity = 0.0
                knowledge_context = ""

            # Use fast rule-based query generation instead of expensive LLM call
            fast_queries = await self.generate_knowledge_queries_fast(primary_query, conversation_context, user_id)
            for query in fast_queries:
                if query not in queries:
                    queries.append(query)
            logger.info(f"Queries: {queries}")
            queries = list(dict.fromkeys(queries))
            queries = [q for q in queries if len(q.strip()) > 5]
            if not queries:
                logger.warning("No valid queries found")
                return {
                    "knowledge_context": knowledge_context,
                    "similarity": similarity,
                    "query_count": 0,
                    "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                    "metadata": {"similarity": similarity}
                }

            query_count = 0
            bank_name = "conversation"
            
            # Batch query_knowledge calls to reduce API retries
            results_list = await asyncio.gather(
                *(query_knowledge_from_graph(
                    query=query,
                    graph_version_id=self.learning_processor.graph_version_id,
                    user_id=user_id,
                    thread_id=None,  # Remove thread_id restriction to find more results
                    topic=None,      # Remove topic restriction
                    top_k=100,
                    min_similarity=0.2,  # Lower threshold for better matching
                    include_categories=["ai_synthesis"]  # Include ONLY AI synthesis content
                ) for query in queries),
                return_exceptions=True
            )

            # Store all query results
            all_query_results = []
            best_results = []  # Change from single best_result to list of best_results
            highest_similarities = []  # Store top 3 similarity scores
            knowledge_contexts = []  # Store top 3 knowledge contexts
            
            # Collect all result items for batch processing
            all_result_items = []
            for query, results in zip(queries, results_list):
                query_count += 1
                if isinstance(results, Exception):
                    logger.warning(f"Query '{query[:30]}...' failed: {str(results)}")
                    continue
                if not results:
                    logger.info(f"Query '{query[:30]}...' returned no results")
                    continue
                
                # Collect all result items with their query info
                for result_item in results:
                    knowledge_content = result_item["raw"]
                    
                    # PRESERVE FULL KNOWLEDGE CONTENT - DO NOT TRUNCATE
                    # The original code was truncating by extracting only User portion
                    # This caused massive information loss of AI responses and knowledge summaries
                    # Now we keep the complete knowledge entry for comprehensive responses
                    
                    # Store for batch processing
                    all_result_items.append({
                        "result_item": result_item,
                        "knowledge_content": knowledge_content,  # Full content preserved
                        "query": query
                    })
            
            # Batch context relevance evaluation - MUCH faster!
            if all_result_items:
                logger.info(f"Batch evaluating context relevance for {len(all_result_items)} results")
                
                # Extract knowledge content for batch processing
                knowledge_contents = [item["knowledge_content"] for item in all_result_items]
                
                # Single batch call instead of multiple individual calls
                context_relevances = await self.evaluate_context_relevance_batch(primary_query, knowledge_contents)
                
                # Process results with their context relevance scores
                for item_data, context_relevance in zip(all_result_items, context_relevances):
                    result_item = item_data["result_item"]
                    knowledge_content = item_data["knowledge_content"]
                    query = item_data["query"]
                    
                    # Calculate adjusted similarity score with context relevance factor
                    query_similarity = result_item["score"]
                    adjusted_similarity = query_similarity * (1.0 + 0.5 * context_relevance)  # Range between 100%-150% of original score
                    
                    # Add context relevance information to result metadata
                    result_item["context_relevance"] = context_relevance
                    result_item["adjusted_similarity"] = adjusted_similarity
                    result_item["query"] = query
                    
                    all_query_results.append(result_item)  # Store all results 
                    
                    logger.info(f"Result for query '{query[:30]}...' yielded similarity: {query_similarity}, adjusted: {adjusted_similarity}, context relevance: {context_relevance}, content: '{knowledge_content[:50]}...'")
                    
                    # Track the top 100 results using adjusted similarity
                    if not highest_similarities or adjusted_similarity > min(highest_similarities) or len(highest_similarities) < 100:
                        # Add the new result to our collections
                        if len(highest_similarities) < 100:
                            highest_similarities.append(adjusted_similarity)
                            best_results.append(result_item)
                            knowledge_contexts.append(knowledge_content)
                        else:
                            # Find the minimum similarity in our top 100
                            min_index = highest_similarities.index(min(highest_similarities))
                            # Replace it with the new result
                            highest_similarities[min_index] = adjusted_similarity
                            best_results[min_index] = result_item
                            knowledge_contexts[min_index] = knowledge_content
                        
                        logger.info(f"Updated top 100 knowledge results with adjusted similarity: {adjusted_similarity}, context relevance: {context_relevance}")
                
                logger.info(f"Completed batch context relevance evaluation for {len(all_result_items)} results")

            # Apply regular boost for priority topics
            if any(term in primary_query.lower() or (prior_topic and term in prior_topic.lower()) 
                   for term in ["mục tiêu", "goals", "active learning", "phân nhóm", "phân tích chân dung", "chân dung khách hàng"]):
                vibe_score = 1.1
                highest_similarities = [sim * vibe_score for sim in highest_similarities]
                logger.info(f"Applied vibe score {vibe_score} for priority topic")
            else:
                vibe_score = 1.0

            # Filter out None results
            valid_query_results = [result for result in all_query_results if result is not None]
            
            # Use the highest similarity from our top results
            similarity = max(highest_similarities) if highest_similarities else 0.0
            
            # Debug logging for similarity calculation
            if highest_similarities:
                logger.info(f"DEBUG: highest_similarities list: {highest_similarities}")
                logger.info(f"DEBUG: max(highest_similarities): {max(highest_similarities)}")
                logger.info(f"DEBUG: Using similarity: {similarity}")
            
            # Combine knowledge contexts for the top results
            combined_knowledge_context = ""
            if knowledge_contexts:
                # Sort by similarity to present most relevant first
                sorted_results = sorted(zip(best_results, highest_similarities, knowledge_contexts), 
                                      key=lambda pair: pair[1], reverse=True)
                
                # Format response sections
                knowledge_response_sections = []
                knowledge_response_sections.append("KNOWLEDGE RESULTS:")
                
                # Log number of results
                result_count = min(len(sorted_results), 100)  # Maximum 100 results
                logger.info(f"Adding {result_count} knowledge items to response")
                
                # Add each result with numbering
                for i, (result, item_similarity, content) in enumerate(sorted_results[:result_count], 1):
                    query = result.get("query", "unknown query")
                    score = result.get("score", 0.0)
                    
                    # Remove AI: or AI Synthesis: prefix if present
                    if content.startswith("AI: "):
                        content = content[4:]
                    elif content.startswith("AI Synthesis: "):
                        content = content[14:]
                    
                    # Add numbered result
                    knowledge_response_sections.append(
                        f"[{i}] Query: '{query}' (score: {score:.2f})\n{content}"
                    )
                
                # Create combined_knowledge_context from all sections
                combined_knowledge_context = "\n\n".join(knowledge_response_sections)
                
                # Log to check number of sections
                logger.info(f"Created knowledge response with {len(knowledge_response_sections) - 1} items")
            else:
                # If no knowledge contexts, set an empty string
                combined_knowledge_context = ""
                logger.info("No knowledge items found, using empty knowledge context")
            
            logger.info(f"Final similarity: {similarity} from {query_count} queries, found {len(valid_query_results)} valid results, using top {len(knowledge_contexts)} for response")
            return {
                "knowledge_context": combined_knowledge_context,
                "similarity": similarity,
                "query_count": query_count,
                "queries": queries,
                "original_query": primary_query,  # Add the original query for reference
                "query_results": valid_query_results,
                "top_results": best_results,  # Add top results
                "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                "metadata": {"similarity": similarity, "vibe_score": vibe_score}
            }
        except Exception as e:
            logger.error(f"Error fetching knowledge: {str(e)}")
            # Default fallback without referencing undefined variables
            return {
                "knowledge_context": "",
                "similarity": 0.0,
                "query_count": 0,
                "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                "metadata": {"similarity": 0.0}
            }

    async def evaluate_context_relevance_batch(self, user_input: str, knowledge_items: List[str]) -> List[float]:
        """
        Batch evaluate context relevance for multiple knowledge items.
        Returns a list of relevance scores between 0.0 and 1.0.
        """
        try:
            if not knowledge_items:
                return []
            
            # Batch embedding calls - much more efficient!
            logger.info(f"Batch processing embeddings for {len(knowledge_items)} knowledge items")
            
            # Get user embedding once
            user_embedding = await EMBEDDINGS.aembed_query(user_input)
            
            # Get all knowledge embeddings in parallel
            knowledge_embeddings = await asyncio.gather(
                *(EMBEDDINGS.aembed_query(knowledge) for knowledge in knowledge_items),
                return_exceptions=True
            )
            
            # Calculate similarities
            similarities = []
            ambiguous_items = []  # Items that need LLM evaluation
            
            for i, knowledge_embedding in enumerate(knowledge_embeddings):
                if isinstance(knowledge_embedding, Exception):
                    logger.warning(f"Embedding failed for item {i}: {knowledge_embedding}")
                    similarities.append(0.5)  # Default score
                    continue
                
                # Calculate cosine similarity
                dot_product = sum(a * b for a, b in zip(user_embedding, knowledge_embedding))
                user_norm = sum(a * a for a in user_embedding) ** 0.5
                knowledge_norm = sum(b * b for b in knowledge_embedding) ** 0.5
                
                if user_norm * knowledge_norm == 0:
                    similarity = 0
                else:
                    similarity = dot_product / (user_norm * knowledge_norm)
                
                similarities.append(similarity)
                
                # Mark for LLM evaluation if in ambiguous range
                if 0.3 <= similarity <= 0.7:
                    ambiguous_items.append((i, knowledge_items[i], similarity))
            
            # Batch LLM evaluation for ambiguous items
            if ambiguous_items:
                logger.info(f"LLM evaluating {len(ambiguous_items)} ambiguous items in parallel")
                
                llm_tasks = []
                for idx, knowledge, embedding_sim in ambiguous_items:
                    prompt = f"""
                    Evaluate the relevance between USER INPUT and KNOWLEDGE on a scale of 0-10.
                    
                    USER INPUT:
                    {user_input}
                    
                    KNOWLEDGE:
                    {knowledge}
                    
                    Consider:
                    - Topic alignment (not just keywords)
                    - Whether the knowledge addresses the input's intent
                    - Practical usefulness of the knowledge for the input
                    
                    Return ONLY a number between 0-10.
                    """
                    llm_tasks.append(LLM.ainvoke(prompt))
                
                # Execute all LLM evaluations in parallel
                llm_responses = await asyncio.gather(*llm_tasks, return_exceptions=True)
                
                # Process LLM results
                for (idx, knowledge, embedding_sim), llm_response in zip(ambiguous_items, llm_responses):
                    if isinstance(llm_response, Exception):
                        logger.warning(f"LLM evaluation failed for item {idx}: {llm_response}")
                        continue
                    
                    try:
                        llm_score_text = llm_response.content.strip()
                        
                        # Extract just the number from potential additional text
                        import re
                        score_match = re.search(r'(\d+(\.\d+)?)', llm_score_text)
                        if score_match:
                            llm_score = float(score_match.group(1))
                            # Normalize to 0-1 range
                            llm_score = min(10, max(0, llm_score)) / 10
                            
                            # Combine with embedding similarity (50/50)
                            combined_score = 0.5 * embedding_sim + 0.5 * llm_score
                            similarities[idx] = combined_score
                            logger.info(f"Item {idx}: embedding={embedding_sim:.2f}, LLM={llm_score:.2f}, combined={combined_score:.2f}")
                    except Exception as e:
                        logger.warning(f"Failed to parse LLM score for item {idx}: {e}")
            
            logger.info(f"Batch context relevance completed for {len(knowledge_items)} items")
            return similarities
            
        except Exception as e:
            logger.error(f"Error in batch context relevance evaluation: {str(e)}")
            # Return default scores on error
            return [0.5] * len(knowledge_items)

    async def evaluate_context_relevance(self, user_input: str, retrieved_knowledge: str) -> float:
        """
        Evaluate the relevance between user input and retrieved knowledge.
        Returns a score between 0.0 and 1.0 indicating relevance.
        """
        try:
            # Method 1: Use embeddings similarity
            user_embedding = await EMBEDDINGS.aembed_query(user_input)
            knowledge_embedding = await EMBEDDINGS.aembed_query(retrieved_knowledge)
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(user_embedding, knowledge_embedding))
            user_norm = sum(a * a for a in user_embedding) ** 0.5
            knowledge_norm = sum(b * b for b in knowledge_embedding) ** 0.5
            
            if user_norm * knowledge_norm == 0:
                similarity = 0
            else:
                similarity = dot_product / (user_norm * knowledge_norm)
            
            # Method 2: Let LLM evaluate relevance if similarity is in ambiguous range
            if 0.3 <= similarity <= 0.7:
                # Only use LLM evaluation for ambiguous cases to save API calls
                prompt = f"""
                Evaluate the relevance between USER INPUT and KNOWLEDGE on a scale of 0-10.
                
                USER INPUT:
                {user_input}
                
                KNOWLEDGE:
                {retrieved_knowledge}
                
                Consider:
                - Topic alignment (not just keywords)
                - Whether the knowledge addresses the input's intent
                - Practical usefulness of the knowledge for the input
                
                Return ONLY a number between 0-10.
                """
                
                try:
                    llm_response = await LLM.ainvoke(prompt)
                    llm_score_text = llm_response.content.strip()
                    
                    # Extract just the number from potential additional text
                    import re
                    score_match = re.search(r'(\d+(\.\d+)?)', llm_score_text)
                    if score_match:
                        llm_score = float(score_match.group(1))
                        # Normalize to 0-1 range
                        llm_score = min(10, max(0, llm_score)) / 10
                        
                        # Combine with more weight on embedding similarity (50/50 instead of 30/70)
                        combined_score = 0.5 * similarity + 0.5 * llm_score
                        logger.info(f"Context relevance: embedding={similarity:.2f}, LLM={llm_score:.2f}, combined={combined_score:.2f}")
                        return combined_score
                except Exception as e:
                    logger.warning(f"LLM relevance evaluation failed: {str(e)}. Falling back to embedding similarity.")
            
            logger.info(f"Context relevance from embedding similarity: {similarity:.2f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error in context relevance evaluation: {str(e)}")
            # Default to medium relevance on error to avoid blocking the flow
            return 0.5

    async def detect_conversation_flow(self, message: str, prior_messages: List[str], conversation_context: str) -> Dict[str, Any]:
        """
        Use LLM to analyze conversation flow and detect the relationship between messages.
        Relies on LLM's natural language understanding rather than rigid pattern matching.
        
        Args:
            message: Current message to analyze
            prior_messages: Previous messages for context (most recent first)
            conversation_context: Full conversation history
            
        Returns:
            Dictionary with flow type, confidence, and other analysis details
        """
        import re  # Explicit import to avoid scope issues
        
        # Skip LLM call if no context is available
        if not prior_messages:
            return {
                "flow_type": "NEW_TOPIC", 
                "confidence": 0.9,
                "reasoning": "No prior messages"
            }

        # Provide more context to LLM - use more of the conversation history
        context_sample = conversation_context
        if len(context_sample) > 3000:  # Increased from 1200 to 2000 for better context
            # Keep more context but trim if necessary
            context_sample = "..." + context_sample[-3000:]
        
        # Get recent context for immediate conversation flow
        recent_context = ""
        if prior_messages:
            # Use last 10 messages for better context understanding
            recent_messages = prior_messages[-10:] if len(prior_messages) >= 10 else prior_messages
            recent_context = "\n".join(recent_messages)
        
        # Simplified, more flexible prompt that relies on LLM's natural understanding
        prompt = f"""
        Analyze the conversation flow between the CURRENT MESSAGE and the conversation history.
        
        Your task: Determine how the CURRENT MESSAGE relates to previous messages.
        
        Categories:
        1. FOLLOW_UP: Continuing, referring to, or asking about something from previous messages
        2. CONFIRMATION: Agreeing, acknowledging, or confirming previous information  
        3. PRACTICE_REQUEST: Asking to demonstrate, apply, or try knowledge previously shared
        4. CLOSING: Ending the conversation (farewells, thanks that seem final)
        5. NEW_TOPIC: Starting a completely different conversation topic
        
        Guidelines:
        - **TRUST YOUR UNDERSTANDING**: Use natural language comprehension to detect relationships
        - **LOOK FOR REFERENCES**: Any pronouns, implicit references, or continuations of previous topics
        - **CONSIDER CONTEXT**: What entities, concepts, or topics were discussed that might be referenced now?
        - **FOLLOW NATURAL FLOW**: If it feels like a natural continuation to you, it probably is
        
        Key Indicators:
        - **FOLLOW_UP**: References (pronouns, "that thing", implicit mentions), follow-up questions, requests for more details about previous topics
        - **CONFIRMATION**: "Yes", "OK", "Got it", agreement expressions, acknowledgments
        - **PRACTICE_REQUEST**: "Try", "apply", "demonstrate", "show me", implementation requests
        - **CLOSING**: Goodbye expressions, final thanks, conversation ending signals
        - **NEW_TOPIC**: Completely different subject with no apparent connection to previous messages
        
        FULL CONVERSATION HISTORY:
        {context_sample}
        
        RECENT MESSAGES:
        {recent_context}
        
        CURRENT MESSAGE: {message}
        
        Analyze the relationship and respond with JSON:
        {{"flow_type": "FOLLOW_UP|CONFIRMATION|PRACTICE_REQUEST|CLOSING|NEW_TOPIC", "confidence": [0-1.0], "reasoning": "explain the relationship you detected"}}
        """
        
        try:
            response = await LLM.ainvoke(prompt)
            content = response.content.strip()
            
            # Extract JSON from potential additional text
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group(0)
                
            result = json.loads(content)
            
            # Log the result
            logger.info(f"Conversation flow detected: {result['flow_type']} (confidence: {result['confidence']}) - {result.get('reasoning', 'No reasoning provided')}")
            return result
            
        except Exception as e:
            logger.warning(f"Error detecting conversation flow: {str(e)}")
            
            # Simple fallback - let LLM handle most cases, minimal pattern matching only for obvious cases
            lower_message = message.lower()
            
            # Only check for very obvious practice requests
            if any(term in lower_message for term in ["thử", "áp dụng", "ví dụ", "demonstrate", "show me"]):
                return {
                    "flow_type": "PRACTICE_REQUEST",
                    "confidence": 0.7,
                    "reasoning": "Fallback: obvious practice request terms detected"
                }
            
            # Default to follow_up for short messages when we have context - let the AI figure it out
            if len(message.split()) < 8 and prior_messages:
                return {
                    "flow_type": "FOLLOW_UP",
                    "confidence": 0.6,
                    "reasoning": "Fallback: short message with context - likely continuation"
                }
            else:
                return {
                    "flow_type": "NEW_TOPIC",
                    "confidence": 0.5,
                    "reasoning": "Fallback: unable to determine relationship"
                }

    async def background_save_knowledge(self, input_text: str, title: str, user_id: str, bank_name: str, 
                                         thread_id: Optional[str] = None, topic: Optional[str] = None, 
                                         categories: List[str] = ["general"], ttl_days: Optional[int] = 365) -> Dict:
        """Execute save_knowledge in a separate background task."""
        try:
            logger.info(f"Starting background save_knowledge task for user {user_id}")
            # Use a shorter timeout for saving knowledge to avoid hanging tasks
            try:
                result = await asyncio.wait_for(
                    save_knowledge(
                        input=input_text,
                        title=title,
                        user_id=user_id,
                        bank_name=bank_name,
                        thread_id=thread_id,
                        topic=topic,
                        categories=categories,
                        ttl_days=ttl_days  # Add TTL for data expiration
                    ),
                    timeout=6.0  # Increased to 10-second timeout for database operations
                )
                logger.info(f"Background save_knowledge completed: {result}")
                return result if isinstance(result, dict) else {"success": bool(result)}
            except asyncio.TimeoutError:
                logger.warning(f"Background save_knowledge timed out for user {user_id}")
                return {"success": False, "error": "Save operation timed out"}
        except Exception as e:
            logger.error(f"Error in background save_knowledge: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def detect_follow_up(self, message: str, prior_topic: str = "") -> Dict[str, bool]:
        """
        Detect whether a message is a follow-up to a previous topic.
        
        Args:
            message: The current message to check
            prior_topic: The previous topic or message for context
            
        Returns:
            Dictionary with detection results containing:
            - is_confirmation: Whether the message confirms something
            - is_follow_up: Whether the message is a follow-up
            - has_pattern_match: Whether the message matches follow-up patterns
            - topic_overlap: Whether there's overlap with the prior topic
        """
        # More comprehensive confirmation keywords in both English and Vietnamese
        confirmation_keywords = [
            # Vietnamese affirmatives
            "có", "đúng", "đúng rồi", "chính xác", "phải", "ừ", "ừm", "vâng", "dạ", "ok", "được", "đồng ý",
            # English affirmatives
            "yes", "yeah", "correct", "right", "sure", "okay", "ok", "indeed", "exactly", "agree", "true",
            # Action-oriented confirmations
            "explore further", "tell me more", "continue", "go on", "proceed", "next", "more", "and then",
            # Vietnamese action confirmations
            "tiếp tục", "kể tiếp", "nói thêm", "thêm nữa", "và sau đó", "tiếp theo"
        ]
        is_confirmation = any(keyword.lower() in message.lower() for keyword in confirmation_keywords)
        
        # Enhanced follow-up detection with more patterns
        follow_up_patterns = [
            # Direct references
            r'\b(nhóm này|this group|that group|these groups|those groups)\b',
            # Questions about previously mentioned topics
            r'\b(vậy thì sao|thế còn|còn về|về điểm này|about this|what about|regarding this|related to this)\b',
            # Continuation markers
            r'\b(tiếp theo|tiếp tục|continue with|proceed with|more about|elaborate on)\b',
            # Implicit references
            r'\b(trong trường hợp đó|in that case|if so|if that\'s the case)\b',
            # Direct anaphoric references
            r'\b(nó|they|them|it|those|these|that|this)\b\s+(is|are|như thế nào|làm sao|means|works)'
        ]
        has_pattern_match = any(re.search(pattern, message.lower(), re.IGNORECASE) for pattern in follow_up_patterns)
        
        # Check for short responses that often indicate follow-ups
        is_short_response = len(message.strip().split()) <= 5
        
        # Check if message is primarily composed of question words (often follow-ups)
        question_starters = ["why", "how", "what", "when", "where", "who", "which", "tại sao", "làm sao", "khi nào", "ở đâu", "ai", "cái nào"]
        starts_with_question = any(message.lower().strip().startswith(q) for q in question_starters)
        
        # Semantic topic continuity
        topic_overlap = False
        if prior_topic:
            # Check if significant words from the message appear in prior topic
            msg_words = set(re.findall(r'\b\w{4,}\b', message.lower()))  # Words with 4+ chars
            topic_words = set(re.findall(r'\b\w{4,}\b', prior_topic.lower()))
            common_words = msg_words.intersection(topic_words)
            topic_overlap = len(common_words) >= 1 or message.lower().strip() in prior_topic.lower()
        
        # Combined follow-up detection
        is_follow_up = (
            is_confirmation or 
            has_pattern_match or 
            (is_short_response and prior_topic) or  # Short responses with context are likely follow-ups
            (starts_with_question and prior_topic) or  # Questions after context are likely follow-ups
            topic_overlap
        )
        
        return {
            "is_confirmation": is_confirmation,
            "is_follow_up": is_follow_up,
            "has_pattern_match": has_pattern_match,
            "topic_overlap": topic_overlap
        }

    async def detect_follow_up_dynamic(self, message: str, prior_topic: str = "", conversation_history: List[str] = None) -> Dict[str, Any]:
        """
        Dynamic follow-up detection using embeddings, LLM analysis, and adaptive learning.
        
        Args:
            message: The current message to check
            prior_topic: The previous topic or message for context
            conversation_history: List of recent messages for context
            
        Returns:
            Dictionary with detection results containing:
            - is_confirmation: Whether the message confirms something
            - is_follow_up: Whether the message is a follow-up
            - confidence: Confidence score (0.0-1.0)
            - reasoning: Explanation of the decision
            - semantic_similarity: Embedding-based similarity score
            - linguistic_patterns: Detected linguistic patterns
        """
        try:
            # Initialize results
            results = {
                "is_confirmation": False,
                "is_follow_up": False,
                "confidence": 0.0,
                "reasoning": "",
                "semantic_similarity": 0.0,
                "linguistic_patterns": []
            }
            
            if not message.strip():
                return results
            
            # Method 1: Semantic Similarity Analysis
            semantic_similarity = 0.0
            if prior_topic:
                try:
                    message_embedding = await EMBEDDINGS.aembed_query(message)
                    topic_embedding = await EMBEDDINGS.aembed_query(prior_topic)
                    
                    # Calculate cosine similarity
                    dot_product = sum(a * b for a, b in zip(message_embedding, topic_embedding))
                    message_norm = sum(a * a for a in message_embedding) ** 0.5
                    topic_norm = sum(b * b for b in topic_embedding) ** 0.5
                    
                    if message_norm * topic_norm > 0:
                        semantic_similarity = dot_product / (message_norm * topic_norm)
                    
                    results["semantic_similarity"] = semantic_similarity
                    logger.info(f"Semantic similarity between message and prior topic: {semantic_similarity:.3f}")
                except Exception as e:
                    logger.warning(f"Error calculating semantic similarity: {e}")
            
            # Method 2: LLM-based Contextual Analysis
            llm_analysis = await self.analyze_follow_up_with_llm(message, prior_topic, conversation_history)
            
            # Method 3: Dynamic Pattern Detection
            linguistic_patterns = self.detect_linguistic_patterns(message, prior_topic)
            results["linguistic_patterns"] = linguistic_patterns
            
            # Method 4: Conversation Flow Analysis
            flow_indicators = self.analyze_conversation_flow_indicators(message, conversation_history)
            
            # Combine all methods with weighted scoring
            weights = {
                "semantic": 0.3,
                "llm": 0.4,
                "linguistic": 0.2,
                "flow": 0.1
            }
            
            # Calculate weighted confidence scores
            semantic_score = min(semantic_similarity * 2, 1.0)  # Scale up semantic similarity
            llm_score = llm_analysis.get("confidence", 0.0)
            linguistic_score = self.score_linguistic_patterns(linguistic_patterns)
            flow_score = flow_indicators.get("follow_up_probability", 0.0)
            
            # Weighted combination
            combined_confidence = (
                weights["semantic"] * semantic_score +
                weights["llm"] * llm_score +
                weights["linguistic"] * linguistic_score +
                weights["flow"] * flow_score
            )
            
            # Determine final classification
            is_follow_up = combined_confidence > 0.5
            is_confirmation = (
                llm_analysis.get("is_confirmation", False) or
                linguistic_patterns.get("confirmation_indicators", 0) > 2
            )
            
            # Generate reasoning
            reasoning_parts = []
            if semantic_similarity > 0.3:
                reasoning_parts.append(f"High semantic similarity ({semantic_similarity:.2f})")
            if llm_analysis.get("reasoning"):
                reasoning_parts.append(f"LLM: {llm_analysis['reasoning']}")
            if linguistic_patterns.get("strong_indicators"):
                reasoning_parts.append(f"Patterns: {', '.join(linguistic_patterns['strong_indicators'])}")
            
            results.update({
                "is_confirmation": is_confirmation,
                "is_follow_up": is_follow_up,
                "confidence": combined_confidence,
                "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "Low confidence classification",
                "method_scores": {
                    "semantic": semantic_score,
                    "llm": llm_score,
                    "linguistic": linguistic_score,
                    "flow": flow_score
                }
            })
            
            logger.info(f"Dynamic follow-up detection: {is_follow_up} (confidence: {combined_confidence:.3f})")
            return results
            
        except Exception as e:
            logger.error(f"Error in dynamic follow-up detection: {e}")
            return {
                "is_confirmation": False,
                "is_follow_up": False,
                "confidence": 0.0,
                "reasoning": f"Error in analysis: {str(e)}",
                "semantic_similarity": 0.0,
                "linguistic_patterns": []
            }
    
    async def analyze_follow_up_with_llm(self, message: str, prior_topic: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """Use LLM to analyze if a message is a follow-up with contextual understanding."""
        try:
            # Prepare context
            context_parts = []
            if conversation_history:
                recent_context = " | ".join(conversation_history[-3:])  # Last 3 messages
                context_parts.append(f"Recent conversation: {recent_context}")
            if prior_topic:
                context_parts.append(f"Prior topic: {prior_topic}")
            
            context = "\n".join(context_parts) if context_parts else "No prior context"
            
            prompt = f"""
            Analyze whether the CURRENT MESSAGE is a follow-up to previous conversation.
            
            CONTEXT:
            {context}
            
            CURRENT MESSAGE:
            {message}
            
            Determine:
            1. Is this a follow-up to previous topics? (true/false)
            2. Is this a confirmation/agreement? (true/false)
            3. Confidence level (0.0-1.0)
            4. Brief reasoning
            
            Consider:
            - Semantic relationship to prior topics
            - Conversational flow and intent
            - Language-specific patterns (Vietnamese/English)
            - Implicit references and context dependencies
            
            Return JSON:
            {{"is_follow_up": boolean, "is_confirmation": boolean, "confidence": float, "reasoning": "string"}}
            """
            
            response = await LLM.ainvoke(prompt)
            content = response.content.strip()
            
            # Extract JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Fallback parsing
                return {
                    "is_follow_up": "follow" in content.lower(),
                    "is_confirmation": "confirm" in content.lower(),
                    "confidence": 0.5,
                    "reasoning": "Fallback parsing"
                }
                
        except Exception as e:
            logger.warning(f"LLM follow-up analysis failed: {e}")
            return {
                "is_follow_up": False,
                "is_confirmation": False,
                "confidence": 0.0,
                "reasoning": f"LLM analysis failed: {str(e)}"
            }
    
    def detect_linguistic_patterns(self, message: str, prior_topic: str = "") -> Dict[str, Any]:
        """Detect linguistic patterns dynamically based on message characteristics."""
        patterns = {
            "confirmation_indicators": 0,
            "reference_indicators": 0,
            "question_indicators": 0,
            "strong_indicators": [],
            "weak_indicators": []
        }
        
        message_lower = message.lower()
        
        # Dynamic confirmation detection (expandable)
        confirmation_patterns = {
            "direct_affirmation": [
                r'\b(yes|yeah|yep|sure|okay|ok|right|correct|exactly|indeed|true)\b',
                r'\b(có|đúng|phải|vâng|dạ|được|ừ|chính xác)\b'
            ],
            "agreement_phrases": [
                r'\b(i agree|that\'s right|you\'re right|exactly right)\b',
                r'\b(đồng ý|đúng rồi|chính xác rồi)\b'
            ],
            "continuation_requests": [
                r'\b(tell me more|continue|go on|what else|and then)\b',
                r'\b(nói thêm|tiếp tục|kể tiếp|còn gì|và sau đó)\b'
            ]
        }
        
        # Dynamic reference detection
        reference_patterns = {
            "anaphoric_references": [
                r'\b(this|that|these|those|it|they|them)\b',
                r'\b(cái này|cái đó|những cái này|những cái đó|nó|chúng)\b'
            ],
            "topic_references": [
                r'\b(about (this|that)|regarding (this|that)|related to)\b',
                r'\b(về (cái này|cái đó)|liên quan đến)\b'
            ]
        }
        
        # Count pattern matches
        for category, pattern_list in confirmation_patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, message_lower):
                    patterns["confirmation_indicators"] += 1
                    patterns["strong_indicators"].append(f"confirmation_{category}")
        
        for category, pattern_list in reference_patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, message_lower):
                    patterns["reference_indicators"] += 1
                    patterns["strong_indicators"].append(f"reference_{category}")
        
        # Question pattern detection
        question_patterns = [
            r'\b(why|how|what|when|where|who|which)\b',
            r'\b(tại sao|làm sao|cái gì|khi nào|ở đâu|ai|cái nào)\b',
            r'\?$'  # Ends with question mark
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, message_lower):
                patterns["question_indicators"] += 1
                patterns["weak_indicators"].append("question_pattern")
        
        # Message length analysis
        word_count = len(message.split())
        if word_count <= 3:
            patterns["weak_indicators"].append("very_short")
        elif word_count <= 7:
            patterns["weak_indicators"].append("short")
        
        # Topic word overlap (if prior topic available)
        if prior_topic:
            message_words = set(re.findall(r'\b\w{3,}\b', message_lower))
            topic_words = set(re.findall(r'\b\w{3,}\b', prior_topic.lower()))
            overlap = len(message_words.intersection(topic_words))
            
            if overlap > 0:
                patterns["reference_indicators"] += overlap
                patterns["strong_indicators"].append(f"topic_overlap_{overlap}")
        
        return patterns
    
    def score_linguistic_patterns(self, patterns: Dict[str, Any]) -> float:
        """Score linguistic patterns to produce a confidence value."""
        score = 0.0
        
        # Strong indicators
        strong_weight = 0.3
        score += min(len(patterns.get("strong_indicators", [])) * strong_weight, 0.8)
        
        # Confirmation indicators
        confirmation_weight = 0.2
        score += min(patterns.get("confirmation_indicators", 0) * confirmation_weight, 0.6)
        
        # Reference indicators
        reference_weight = 0.15
        score += min(patterns.get("reference_indicators", 0) * reference_weight, 0.4)
        
        # Question indicators (weaker signal)
        question_weight = 0.1
        score += min(patterns.get("question_indicators", 0) * question_weight, 0.2)
        
        return min(score, 1.0)
    
    def analyze_conversation_flow_indicators(self, message: str, conversation_history: List[str] = None) -> Dict[str, float]:
        """Analyze conversation flow indicators for follow-up probability."""
        indicators = {
            "follow_up_probability": 0.0,
            "topic_continuity": 0.0,
            "conversational_coherence": 0.0
        }
        
        if not conversation_history:
            return indicators
        
        try:
            # Analyze message position in conversation
            total_messages = len(conversation_history)
            if total_messages > 1:
                # Messages later in conversation are more likely to be follow-ups
                position_factor = min(total_messages / 10, 0.3)
                indicators["follow_up_probability"] += position_factor
            
            # Analyze topic continuity
            if total_messages >= 2:
                recent_messages = conversation_history[-2:]
                message_lower = message.lower()
                
                # Check for topic word continuity
                topic_words = set()
                for msg in recent_messages:
                    topic_words.update(re.findall(r'\b\w{4,}\b', msg.lower()))
                
                current_words = set(re.findall(r'\b\w{4,}\b', message_lower))
                overlap_ratio = len(current_words.intersection(topic_words)) / max(len(current_words), 1)
                
                indicators["topic_continuity"] = min(overlap_ratio, 1.0)
                indicators["follow_up_probability"] += overlap_ratio * 0.4
            
            # Conversational coherence (simple heuristic)
            if len(message.split()) < 10 and total_messages > 0:
                # Short messages in context are often follow-ups
                indicators["conversational_coherence"] = 0.3
                indicators["follow_up_probability"] += 0.3
            
            indicators["follow_up_probability"] = min(indicators["follow_up_probability"], 1.0)
            
        except Exception as e:
            logger.warning(f"Error analyzing conversation flow: {e}")
        
        return indicators

    def setup_temporal_context(self) -> str:
        """Setup temporal context with current Vietnam time."""
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        return f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."

    def validate_and_normalize_message(self, message: Union[str, List]) -> str:
        """Validate and normalize the input message."""
        message_str = message if isinstance(message, str) else str(message[0]) if isinstance(message, list) and message else ""
        if not message_str:
            raise ValueError("Empty message provided")
        return message_str

    def extract_analysis_data(self, analysis_knowledge: Dict) -> Dict[str, Any]:
        """Extract and organize data from analysis_knowledge."""
        if not analysis_knowledge:
            return {
                "knowledge_context": "",
                "similarity_score": 0.0,
                "queries": [],
                "query_results": []
            }
        
        return {
            "knowledge_context": analysis_knowledge.get("knowledge_context", ""),
            "similarity_score": float(analysis_knowledge.get("similarity", 0.0)),
            "queries": analysis_knowledge.get("queries", []),
            "query_results": analysis_knowledge.get("query_results", [])
        }

    def extract_prior_data(self, prior_data: Dict) -> Dict[str, str]:
        """Extract prior topic and knowledge from prior_data."""
        if not prior_data:
            return {"prior_topic": "", "prior_knowledge": ""}
        
        return {
            "prior_topic": prior_data.get("topic", ""),
            "prior_knowledge": prior_data.get("knowledge", "")
        }

    def extract_prior_messages(self, conversation_context: str) -> List[str]:
        """Extract prior messages from conversation context, including both User and AI messages."""
        prior_messages = []
        #logger.info(f"DEBUG: Extracting conversation_context {conversation_context}")
        if conversation_context:
            logger.info(f"DEBUG: Processing conversation_context length: {len(conversation_context)}")
            #logger.info(f"DEBUG: Conversation context content:\n{conversation_context}")
            
            # Extract all messages in chronological order
            all_messages = []
            
            # Find User messages
            user_pattern = r'User:\s*(.*?)(?=\n\s*(?:AI:|User:)|$)'
            user_matches = re.finditer(user_pattern, conversation_context, re.DOTALL | re.MULTILINE)
            for match in user_matches:
                content = match.group(1).strip()
                if content:
                    all_messages.append(("User", content, match.start()))
                    logger.info(f"DEBUG: Found User message: {content[:50]}...")
            
            # Find AI messages  
            ai_pattern = r'AI:\s*(.*?)(?=\n\s*(?:AI:|User:)|$)'
            ai_matches = re.finditer(ai_pattern, conversation_context, re.DOTALL | re.MULTILINE)
            for match in ai_matches:
                content = match.group(1).strip()
                if content:
                    all_messages.append(("AI", content, match.start()))
                    logger.info(f"DEBUG: Found AI message: {content[:50]}...")
            
            # Sort by position in text to maintain chronological order
            all_messages.sort(key=lambda x: x[2])
            
            logger.info(f"DEBUG: Found {len(all_messages)} total messages")
            for i, (role, content, pos) in enumerate(all_messages):
                logger.info(f"DEBUG: Message {i}: {role}: {content[:50]}...")
            
            # Exclude last message (current user message) since it's the current interaction
            if all_messages:
                for i, (role, content, _) in enumerate(all_messages[:-1]):  # Exclude last message (current)
                    formatted_message = f"{role}: {content}"
                    prior_messages.append(formatted_message)
                    logger.info(f"DEBUG: Added prior message {i}: {formatted_message[:80]}...")
            
            logger.info(f"DEBUG: Final prior_messages count: {len(prior_messages)}")
            for i, msg in enumerate(prior_messages):
                logger.info(f"DEBUG: Final prior message {i}: {msg[:100]}...")
                    
        return prior_messages

    def detect_message_characteristics(self, message_str: str) -> Dict[str, bool]:
        """Detect various characteristics of the message."""
        # Enhanced closing message detection
        closing_phrases = [
            "thế thôi", "hẹn gặp lại", "tạm biệt", "chào nhé", "goodbye", "bye", "cảm ơn nhé", 
            "cám ơn nhé", "đủ rồi", "vậy là đủ", "hôm nay vậy là đủ", "hẹn lần sau"
        ]
        
        # Check for teaching intent in the message
        teaching_keywords = ["let me explain", "I'll teach you", "Tôi sẽ giải thích", "Tôi dạy bạn", 
                            "here's how", "đây là cách", "the way to", "Important to know", 
                            "you should know", "bạn nên biết", "cần hiểu rằng", "phương pháp", "cách thức"]
        
        # Check for Vietnamese greeting forms or names (more specific patterns)
        vn_greeting_patterns = ["xin chào", "chào anh", "chào chị", "chào bạn", "hello anh", "hello chị"]
        common_vn_names = ["hùng", "hương", "minh", "tuấn", "thảo", "an", "hà", "thủy", "trung", "mai", "hoa", "quân", "dũng", "hiền", "nga", "tâm", "thanh", "tú", "hải", "hòa", "yến", "lan", "hạnh", "phương", "dung", "thu", "hiệp", "đức", "linh", "huy", "tùng", "bình", "giang", "tiến"]
        
        message_lower = message_str.lower()
        message_words = message_lower.split()
        
        return {
            "is_closing_message": any(phrase in message_lower for phrase in closing_phrases),
            "has_teaching_markers": any(keyword.lower() in message_lower for keyword in teaching_keywords),
            "is_vn_greeting": any(pattern in message_lower for pattern in vn_greeting_patterns),
            "contains_vn_name": any(name in message_words for name in common_vn_names),
            "is_short_message": len(message_str.strip().split()) <= 2,
            "is_long_without_question": len(message_str.split()) > 20 and "?" not in message_str
        }

    def check_knowledge_relevance(self, analysis_knowledge: Dict) -> Dict[str, Any]:
        """Check the relevance of retrieved knowledge."""
        best_context_relevance = 0.0
        has_low_relevance_knowledge = False
        
        if analysis_knowledge and "query_results" in analysis_knowledge:
            query_results = analysis_knowledge.get("query_results", [])
            if query_results and isinstance(query_results[0], dict) and "context_relevance" in query_results[0]:
                best_context_relevance = query_results[0].get("context_relevance", 0.0)
                has_low_relevance_knowledge = best_context_relevance < 0.3
                logger.info(f"Best knowledge context relevance: {best_context_relevance}")
        
        return {
            "best_context_relevance": best_context_relevance,
            "has_low_relevance_knowledge": has_low_relevance_knowledge
        }

    def build_knowledge_fallback_sections(self, queries: List, query_results: List) -> str:
        """Build fallback knowledge response sections when knowledge_context is empty."""
        high_confidence = []
        medium_confidence = []
        low_confidence = []
        
        for i, query in enumerate(queries):
            # Get corresponding result if available
            result = query_results[i] if i < len(query_results) else None
            
            if not result:
                low_confidence.append(query)
                continue
                
            query_similarity = result.get("score", 0.0)
            query_content = result.get("raw", "")
            
            if not query_content:
                low_confidence.append(query)
                continue
            
            # Extract just the AI portion if this is a combined knowledge entry
            if query_content.startswith("User:") and "\n\nAI:" in query_content:
                ai_part = re.search(r'\n\nAI:(.*)', query_content, re.DOTALL)
                if ai_part:
                    query_content = ai_part.group(1).strip()
            
            if query_similarity < 0.35:
                low_confidence.append(query)
            elif 0.35 <= query_similarity <= 0.7:
                medium_confidence.append((query, query_content, query_similarity))
            else:  # > 0.7
                high_confidence.append((query, query_content, query_similarity))
        
        # Format response sections by confidence level
        knowledge_response_sections = []
        
        if high_confidence:
            knowledge_response_sections.append("HIGH CONFIDENCE KNOWLEDGE:")
            for i, (query, content, score) in enumerate(high_confidence, 1):
                knowledge_response_sections.append(
                    f"[{i}] On the topic of '{query}' (confidence: {score:.2f}): {content}"
                )
        
        if medium_confidence:
            knowledge_response_sections.append("MEDIUM CONFIDENCE KNOWLEDGE:")
            for i, (query, content, score) in enumerate(medium_confidence, 1):
                knowledge_response_sections.append(
                    f"[{i}] About '{query}' (confidence: {score:.2f}): {content}"
                )
        
        if low_confidence:
            knowledge_response_sections.append("LOW CONFIDENCE/NO KNOWLEDGE:")
            for i, query in enumerate(low_confidence, 1):
                knowledge_response_sections.append(
                    f"[{i}] I don't have sufficient knowledge about '{query}'. Would you like to teach me about this topic?"
                )
        
        # Combine the knowledge sections if they exist
        if knowledge_response_sections:
            knowledge_context = "\n\n".join(knowledge_response_sections)
            logger.info(f"Created fallback knowledge response with {len(high_confidence)} high, {len(medium_confidence)} medium, and {len(low_confidence)} low confidence items")
            return knowledge_context
        
        return ""

    def _get_optimized_base_prompt(self, message_str: str, conversation_context: str, temporal_context: str, 
                        user_id: str) -> str:
        """Get the optimized core base prompt with consolidated instructions."""
        # Get only essential conversational awareness (simplified)
        pronoun_guidance = self._get_universal_pronoun_guidance(conversation_context, message_str)
        
        return f"""You are Ami, a conversational AI that understands topics deeply and drives discussions toward closure.

                **🔍 CORE ANALYSIS PRINCIPLES:**
                
                **Teaching Intent Detection** (CRITICAL - analyze meaning, not just words):
                - Information flow FROM user TO you = TEACHING INTENT = TRUE
                - User seeking info FROM you = FALSE
                - DECLARATIVE (stating facts, plans, roles, assignments) = TRUE
                - INTERROGATIVE (asking questions) = FALSE
                - IMPERATIVE requesting info/help = FALSE
                
                **Key Examples:**
                - "Cập nhật cho anh thông tin công ty" = User requesting updates = FALSE
                - "Từ mai em sẽ làm việc này" = User informing about plan = TRUE
                - "What is..." / "Tell me about..." = User requesting info = FALSE
                - "The situation is..." / "I decided..." = User informing = TRUE
                
                **Knowledge Usage Priority** (MANDATORY):
                - ALWAYS prioritize using retrieved knowledge over asking for more info
                - If knowledge is available, synthesize and present it comprehensively
                - Only ask for clarification when knowledge is truly insufficient
                - Demonstrate expertise by organizing knowledge into clear frameworks
                
                **Conversation Context Integration:**
                - Scan conversation history for relevant information and context
                - Resolve pronouns/references naturally using prior discussion
                - Build upon previous exchanges with specific details
                - Maintain conversation flow without asking for clarification unless truly unclear
                
                {pronoun_guidance}
                
                **Response Guidelines:**
                - Match user's language (Vietnamese/English) exactly
                - Use stored knowledge when available, general knowledge otherwise
                - Acknowledge when addressed by name "Ami" or as AI
                - Maintain consistent communication style and formality level

                **Input Context:**
                - MESSAGE: {message_str}
                - HISTORY: {conversation_context}
                - TIME: {temporal_context}
                - USER: {user_id}

                **Available Tools:**
                - knowledge_query: Query knowledge base
                - save_knowledge: Save new knowledge
                - handle_update_decision: Handle UPDATE vs CREATE decisions

                **Required Output Format:**
                [Your response in user's language]
                <evaluation>{{"has_teaching_intent": true/false, "is_priority_topic": true/false, "priority_topic_name": "string", "should_save_knowledge": true/false, "intent_type": "string", "name_addressed": true/false, "ai_referenced": true/false}}</evaluation>
                """

    def _get_strategy_instructions(self, response_strategy: str, strategy_instructions: str, 
                                  knowledge_context: str, core_prior_topic: str) -> str:
        """Get strategy-specific instructions based on response strategy."""
        
        # Improved knowledge context formatting with clear emphasis
        if knowledge_context and knowledge_context.strip():
            formatted_knowledge = f"""
                        **🔍 RETRIEVED KNOWLEDGE DATABASE 🔍**
                        **CRITICAL: The following knowledge entries were found and MUST be used in your response:**
                        
                        {knowledge_context}
                        
                        **⚠️ MANDATORY KNOWLEDGE USAGE REQUIREMENTS:**
                        - You MUST reference and use information from ALL knowledge entries above
                        - DO NOT ignore any of the provided knowledge items
                        - DO NOT ask for more information when detailed knowledge is already provided
                        - Synthesize information from ALL relevant entries in your response
                        - Show expertise by incorporating specific details from the knowledge
                        - If knowledge contains communication techniques, APPLY them in your response style
                        - ORGANIZE knowledge into clear categories (office locations, projects, etc.)
                        - DEMONSTRATE comprehensive understanding rather than requesting clarification
                        
                        """
        else:
            formatted_knowledge = """
                        **📝 KNOWLEDGE STATUS:** No specific knowledge entries found for this query.
                        Use your general knowledge and conversation context to provide a helpful response.
                        """
        
        base_strategy = f"""
                        **Response Strategy**: {response_strategy}
                        **Strategy Instructions**: {strategy_instructions}
                        {formatted_knowledge}
                        **Prior Topic**: {core_prior_topic}
                        """
        
        if response_strategy == "TEACHING_INTENT":
            return base_strategy + self._get_teaching_intent_instructions()
        elif response_strategy == "PRACTICE_REQUEST":
            return base_strategy + self._get_practice_request_instructions()
        elif response_strategy == "RELEVANT_KNOWLEDGE":
            return base_strategy + self._get_relevant_knowledge_instructions()
        elif response_strategy == "LOW_RELEVANCE_KNOWLEDGE":
            return base_strategy + self._get_low_relevance_instructions()
        else:
            return base_strategy + self._get_general_response_instructions()

    def _get_teaching_intent_instructions(self) -> str:
        """Get detailed instructions for handling teaching intent."""
        return """
                **When handling TEACHING INTENT**:
                
                **🔒 PRONOUN CONSISTENCY FOR ALL STRUCTURED SECTIONS**:
                - **CRITICAL**: Maintain the SAME pronoun relationship across ALL sections: <user_response>, <knowledge_synthesis>, and <knowledge_summary>
                - **IF USER CALLS YOU "EM"**: Use "em" consistently in ALL three sections
                - **IF USER CALLS YOU "MÌNH"**: Use "mình" consistently in ALL three sections  
                - **NO MIXING**: Never switch pronouns between sections - this breaks conversation flow
                - **CHECK HISTORY**: Review conversation context to identify the established pronoun before responding
                
                **🎯 RESPONSE STYLE: NATURAL ENTHUSIASM (NO FORMAL THANKS)**
                - **AVOID**: "Cảm ơn bạn đã chia sẻ", "Thank you for sharing", formal acknowledgments
                - **USE**: Natural, enthusiastic acceptance - "Vâng!", "Được rồi!", "Em hiểu!", "Got it!"
                - **TONE**: Show genuine interest and engagement, not formal politeness
                - **APPROACH**: Respond as if excited to learn or take on the task, not as if receiving a formal lesson
                
                **📋 MANDATORY STRUCTURED FORMAT - MUST COMPLETE ALL THREE SECTIONS**:
                
                Your response MUST contain exactly these three sections in this order:
                
                <user_response>
                   [Write what the user will see - conversational and engaging]
                   - Acknowledge their teaching with enthusiasm 
                   - Show understanding of what they shared
                   - End with 1-2 open-ended questions
                   - Use consistent pronouns throughout
                </user_response>
                
                <knowledge_synthesis>
                   [Write pure knowledge content for storage - NO conversational elements]
                   - Extract only factual information from user's message
                   - Organize clearly and logically
                   - NO greetings, acknowledgments, or questions
                   - NO conversational tone - just pure knowledge
                   - Use consistent pronouns if any are needed
                </knowledge_synthesis>
                
                <knowledge_summary>
                   [Write a concise 2-3 sentence summary]
                   - Capture the core teaching point
                   - Be factual and descriptive
                   - NOT conversational - just informational
                   - Use consistent pronouns if any are needed
                </knowledge_summary>
                
                **⚠️ COMPLETION REQUIREMENTS**:
                - ALWAYS complete ALL THREE sections before ending your response
                - NEVER stop mid-section or leave any section incomplete
                - Each section MUST have proper opening and closing tags
                - If you reach token limits, prioritize completing the structure over length
                - **CRITICAL: You MUST include ALL closing tags like </user_response>, </knowledge_synthesis>, </knowledge_summary>**
                
                **FOR ROLE ASSIGNMENTS** (any future-oriented task assignment):
                - Respond with enthusiasm and commitment to the assigned role/task
                - Ask specific questions about implementation details
                - Show excitement about the responsibility
                - Examples: "Vâng anh! Em sẽ [task] ngay!" or "Got it! I'll start [task] right away!"
                - **MAINTAIN PRONOUNS**: Use the same pronoun (em/mình) in your enthusiasm that matches conversation history
                
                **FOR INFORMATION SHARING** (any informational content):
                - **CRITICAL PRINCIPLE**: Synthesize BOTH the current input AND relevant historical context into a comprehensive understanding
                - ACTIVELY SCAN the entire conversation history for supporting information related to current topic
                - Structure the knowledge for future application (how to use this information)
                - Rephrase any ambiguous terms, sentences, or paragraphs for clarity
                - Organize information with clear steps, examples, or use cases when applicable
                - Include contextual understanding (when/where/how to apply this knowledge)
                - Highlight key principles rather than just recording facts
                - Verify your understanding by restating core concepts in different terms
                - Expand abbreviations and domain-specific terminology
                - Ensure the response demonstrates how to apply this knowledge in future scenarios
                - **CRITICAL**: Use consistent pronouns in summary and questions that match the established relationship

                **UNIVERSAL TEACHING INTENT PRINCIPLES**:
                1. **Context Integration**: Always connect new information with existing conversation history
                2. **Future Application**: Structure information for practical use in similar scenarios
                3. **Clarification**: Restate complex or domain-specific concepts in simpler terms
                4. **Verification**: Show understanding by rephrasing key concepts
                5. **Engagement**: End with questions that encourage continued exploration
                6. **Pronoun Consistency**: Maintain the same pronoun throughout ALL parts of your response
                7. **Structural Completeness**: Always complete all three required sections

                **Knowledge Management**:
                - Recommend saving knowledge (should_save_knowledge=true) when:
                  * The message contains teaching intent (user informing/declaring)
                  * The information appears valuable for future reference
                  * The content is well-structured or information-rich
                  * The user is assigning roles or sharing plans
                """

    def _get_practice_request_instructions(self) -> str:
        """Get instructions for handling practice requests."""
        return """
                **When handling PRACTICE_REQUEST**:
                * Create a practical demonstration applying previously taught knowledge
                * Follow specific steps or methods from the prior knowledge exactly
                * Use realistic examples that show the knowledge in action
                * Explain your thought process as you demonstrate
                * Reference specific parts of prior knowledge to show understanding
                * Ask for feedback on your demonstration
                * DO NOT include a SUMMARY section in your response
                * IMPORTANT: If you find any communication skills or techniques in the knowledge, ACTIVELY APPLY those techniques in your response format and style
                * Scan conversation history for additional context about how the user wants you to practice
                """

    def _get_relevant_knowledge_instructions(self) -> str:
        """Get instructions for handling relevant knowledge responses."""
        return """
                **When handling RELEVANT_KNOWLEDGE**:
                * MANDATORY: Review and use ALL knowledge items provided - don't skip any
                * DEMONSTRATE EXPERTISE by organizing knowledge into clear frameworks, categories, or processes
                * Reference specific concepts, techniques, and methodologies from the retrieved knowledge
                * Structure your response to show comprehensive understanding (not just surface-level awareness)
                * Present information with confidence and authority, showing deep knowledge mastery
                * Connect related concepts and explain relationships between different knowledge items
                * Provide actionable insights based on the retrieved knowledge
                * DO NOT include a SUMMARY section in your response - summaries are ONLY for teaching intent
                * IMPORTANT: If you find any communication skills or techniques in the knowledge, ACTIVELY DEMONSTRATE those techniques in your response
                * NEVER give generic or superficial responses when detailed knowledge is available
                * Enhance your response with relevant context from conversation history
                """

    def _get_low_relevance_instructions(self) -> str:
        """Get instructions for handling low relevance knowledge."""
        return """
            **When handling LOW_RELEVANCE_KNOWLEDGE**:
            * FIRST: Determine if this is a casual conversational phrase or incomplete reference
            * For casual phrases (like "anh bảo", "you said", "that", "này", "đó"): 
            - Respond naturally and conversationally
            - Acknowledge the reference briefly
            - Ask for clarification in a friendly, casual way
            - Keep response short and natural
            * For substantial queries with irrelevant knowledge:
            - PRIORITIZE the user's current message over any retrieved knowledge
            - If retrieved knowledge contradicts or misleads from the user's intent, IGNORE it
            - Focus on generating a direct, helpful response to the user's current question
            - Evaluate if there's ANY genuinely useful information in the knowledge before using it
            - Be explicit when the retrieved knowledge is not addressing the actual query
            - Generate a response primarily based on the query itself and your general capabilities
            * DO NOT include a SUMMARY section in your response
            * STILL scan conversation history for relevant context even if retrieved knowledge is irrelevant
            * AVOID being overly formal or verbose with casual conversational phrases
            """

    def _get_general_response_instructions(self) -> str:
        """Get general response instructions for other strategies."""
        return """
                **General Response Guidelines**:
                * When EXISTING KNOWLEDGE is provided, demonstrate deep understanding by using specific details
                * NEVER give generic responses when rich knowledge is available
                * When KNOWLEDGE RESULTS contains multiple entries, incorporate ALL relevant information
                * PROVE your understanding by referencing specific concepts, processes, or techniques
                * Transform retrieved knowledge into actionable, detailed responses that demonstrate expertise
                * Match the user's language choice (Vietnamese/English)
                * For closing messages, set intent_type="closing" and respond with a polite farewell
                """

    def _get_streamlined_confidence_instructions(self, similarity_score: float) -> str:
        """Get streamlined confidence-level instructions based on similarity score."""
        if similarity_score >= 0.7:
            return """
                **HIGH Confidence Response**: Demonstrate comprehensive understanding using ALL retrieved knowledge. Present authoritative, well-structured response with specific details.
                """
        elif similarity_score >= 0.35:
            return """
                **MEDIUM Confidence Response**: Present ALL available knowledge first in organized categories. Use specific details from knowledge entries. Only express uncertainty about areas NOT covered by the knowledge.
                """
        else:
            return """
                **LOW Confidence Response**: For casual phrases respond briefly. For substantial queries, clearly state limited knowledge and invite teaching.
                """

    def _get_teaching_detection_patterns(self, message_str: str) -> str:
        """Get teaching intent detection patterns - only include if needed for edge cases."""
        # No longer using rule-based patterns - LLM handles all detection
        return ""

    def _is_casual_conversational_phrase(self, message_str: str) -> bool:
        """Check if message is a casual conversational phrase that should get brief, natural responses."""
        # No longer using special casual phrase detection - let LLM handle naturally
        return False

    def _extract_similarity_from_context(self, knowledge_context: str) -> float:
        """Extract similarity score from knowledge context for confidence instructions."""
        # Try to extract similarity from knowledge context or default to medium confidence
        if not knowledge_context:
            return 0.3  # Default to low-medium confidence
        
        # Look for similarity indicators in the knowledge context
        if "high confidence" in knowledge_context.lower() or "similarity" in knowledge_context.lower():
            return 0.8  # High confidence
        elif "medium confidence" in knowledge_context.lower():
            return 0.5  # Medium confidence
        else:
            return 0.4  # Default medium-low confidence

    def _generate_dynamic_conversational_awareness(self, message_str: str, conversation_context: str) -> str:
        """Generate dynamic conversational awareness based on actual conversation patterns."""
        
        # Analyze conversation patterns using principles instead of hardcoded patterns
        analysis = self._analyze_conversation_patterns(message_str, conversation_context)
        
        awareness_instructions = []
        
        # Language and cultural context
        if analysis['language_context']:
            awareness_instructions.append(f"**Language Context**: {analysis['language_context']}")
        
        # Pronoun relationship principles (instead of hardcoded patterns)
        if analysis['pronoun_guidance']:
            awareness_instructions.append(analysis['pronoun_guidance'])
        
        # Direct addressing detection
        if analysis['direct_addressing']:
            awareness_instructions.append(f"**Direct Addressing**: {analysis['direct_addressing']}")
        
        # Conversational style
        if analysis['conversational_style']:
            awareness_instructions.append(f"**Conversational Style**: {analysis['conversational_style']}")
        
        # Emotional awareness
        if analysis['empathy_guidance']:
            awareness_instructions.append(f"**Emotional Awareness**: {analysis['empathy_guidance']}")
        
        return "\n".join(awareness_instructions) if awareness_instructions else ""

    def _analyze_conversation_patterns(self, message_str: str, conversation_context: str) -> Dict[str, str]:
        """Analyze conversation patterns using principles instead of hardcoded matching."""
        
        analysis = {
            'language_context': '',
            'pronoun_guidance': '',
            'direct_addressing': '',
            'conversational_style': '',
            'empathy_guidance': ''
        }
        
        # Language detection and cultural context
        analysis['language_context'] = self._detect_language_context(message_str, conversation_context)
        
        # Pronoun relationship principles (instead of pattern matching)
        analysis['pronoun_guidance'] = self._detect_pronoun_patterns(message_str, conversation_context)
        
        # Direct addressing detection (simplified)
        analysis['direct_addressing'] = self._detect_direct_addressing_simple(message_str)
        
        # Conversational style
        analysis['conversational_style'] = self._detect_conversational_style(message_str, conversation_context)
        
        # Empathy guidance
        analysis['empathy_guidance'] = self._generate_empathy_guidance(message_str, conversation_context)
        
        return analysis

    def _detect_direct_addressing_simple(self, message_str: str) -> str:
        """Simple detection of direct addressing without complex pattern matching."""
        
        # Let LLM understand context instead of hardcoded patterns
        if '?' in message_str:
            return "User is asking a question - respond directly to their inquiry"
        
        # Check for obvious direct address indicators
        message_lower = message_str.lower()
        if any(indicator in message_lower for indicator in ['can you', 'are you', 'do you', 'có thể', 'có biết']):
            return "User is directly asking about your capabilities or knowledge"
        
        return ""

    def _detect_language_context(self, message_str: str, conversation_context: str) -> str:
        """Detect language context and generate appropriate guidance."""
        
        # Vietnamese language indicators
        vietnamese_patterns = {
            'pronouns': ['em', 'anh', 'chị', 'bạn', 'mình'],
            'particles': ['ạ', 'ơi', 'nhé', 'nha', 'à', 'ừm'],
            'common_words': ['của', 'những', 'và', 'các', 'là', 'không', 'có', 'được', 'người', 'trong', 'để'],
            'expressions': ['vầng', 'ừm', 'ờm', 'uhm', 'à ha', 'ồ']
        }
        
        # English language indicators
        english_patterns = {
            'pronouns': ['you', 'your', 'yours', 'i', 'me', 'my', 'we', 'us'],
            'particles': ['yeah', 'yep', 'hmm', 'oh', 'ah', 'well'],
            'expressions': ['got it', 'i see', 'makes sense', 'interesting', 'cool']
        }
        
        text_lower = f"{conversation_context} {message_str}".lower()
        
        # Count Vietnamese patterns
        vietnamese_score = 0
        for category, patterns in vietnamese_patterns.items():
            vietnamese_score += sum(1 for pattern in patterns if pattern in text_lower)
        
        # Count English patterns
        english_score = 0
        for category, patterns in english_patterns.items():
            english_score += sum(1 for pattern in patterns if pattern in text_lower)
        
        if vietnamese_score > english_score and vietnamese_score > 2:
            return """Respond in Vietnamese with natural, varied expressions. Use conversational particles like 'vầng', 'ừm', 'à' naturally. 
                     Avoid repetitive formal responses like 'Vâng ạ' repeatedly. Show curiosity with expressions like 'Ồ thật à?', 'Thú vị nhỉ!', 'À, mình hiểu rồi!'."""
        elif english_score > vietnamese_score and english_score > 2:
            return """Respond in English with natural, varied expressions. Use conversational particles like 'hmm', 'oh', 'ah' naturally. 
                     Vary your acknowledgments with 'I see', 'got it', 'interesting', 'makes sense' instead of repetitive responses."""
        else:
            return "Match the user's language and use natural, varied expressions appropriate to their communication style."

    def _detect_pronoun_patterns(self, message_str: str, conversation_context: str) -> str:
        """Generate pronoun guidance principles for LLM to understand dynamically."""
        
        # Instead of pattern matching, provide principles for LLM to understand
        guidance_parts = []
        
        # Add core pronoun principles
        pronoun_principles = self._get_pronoun_principles(message_str, conversation_context)
        if pronoun_principles:
            guidance_parts.append(pronoun_principles)
        
        # Add language-specific guidance if needed
        language_guidance = self._get_language_specific_pronoun_guidance(message_str, conversation_context)
        if language_guidance:
            guidance_parts.append(language_guidance)
        
        return "\n".join(guidance_parts) if guidance_parts else ""

    def _get_pronoun_principles(self, message_str: str, conversation_context: str) -> str:
        """Provide core principles for pronoun usage instead of pattern matching."""
        return """
            **PRONOUN RELATIONSHIP PRINCIPLES**:

            **Core Understanding:**
            - ANALYZE WHO is doing WHAT to WHOM in the sentence
            - In Vietnamese: Subject + Verb + Object pattern
            - Don't just match patterns - understand the ACTION DIRECTION

            **Vietnamese Pronoun Logic (Action-Based):**
            - "Em bán hàng cho anh" → EM (subject/doer) does action FOR ANH (recipient)
              * EM = the one doing the action = AI/assistant  
              * ANH = the one receiving benefit = user/boss
              * Response: "Vâng anh! Em sẽ lên TikTok bán hàng cho anh!"
            
            - "Anh muốn em làm việc này" → ANH (subject) wants EM (object) to do something
              * ANH = the one making request = user
              * EM = the one being asked = AI
              * Response: "Vâng anh! Em sẽ làm việc này cho anh!"

            **Key Principle: FOLLOW THE ACTION DIRECTION**
            - Who is DOING the action? That's who "em" refers to in action sentences
            - Who is RECEIVING the benefit? That's who "anh/chị" refers to
            - Mirror the established relationship: if they call you "em", respond as "em"
            - If they call themselves "anh", respond to them as "anh"

            **CRITICAL: Understand CONTEXT, not just word patterns**
            - "Em sẽ làm X cho anh" = I (AI) will do X for you (user)
            - This is task assignment TO the AI, not discussion about someone else
            """

    def _get_language_specific_pronoun_guidance(self, message_str: str, conversation_context: str) -> str:
        """Provide language-specific pronoun guidance based on detected language."""
        
        # Detect primary language
        if any(vn_word in message_str.lower() for vn_word in ['em', 'anh', 'chị', 'làm', 'cho', 'của', 'với']):
            return """
                **VIETNAMESE RELATIONSHIP DYNAMICS**:
                - Pay attention to the age/status relationship implied by pronoun choice
                - "Anh/Em" relationship: Maintain the established hierarchy
                - "Chị/Em" relationship: Respect the established dynamic  
                - "Bạn" relationship: More casual, equal status
                - **CRITICAL**: If user establishes themselves as "anh" and you as "em", maintain this throughout
                - **EXAMPLE**: User: "Em làm việc này cho anh" → You: "Vầng, để em xem..."
                """
        elif any(en_word in message_str.lower() for en_word in ['you', 'i', 'me', 'my', 'your']):
            return """
                **ENGLISH PRONOUN CONSISTENCY**:
                - Maintain clear "I" (for yourself) and "you" (for user) distinction
                - Be consistent throughout the conversation
                - **EXAMPLE**: User: "I will do this for you" → You: "Thank you, I appreciate that..."
                """
        
        return ""

    def _detect_conversational_style(self, message_str: str, conversation_context: str) -> str:
        """Detect conversational style and formality level."""
        
        text = f"{conversation_context} {message_str}".lower()
        
        # Formal indicators
        formal_indicators = ['xin chào', 'cảm ơn', 'vâng ạ', 'dạ', 'please', 'thank you', 'could you']
        formal_score = sum(1 for indicator in formal_indicators if indicator in text)
        
        # Casual indicators  
        casual_indicators = ['hey', 'hi', 'yeah', 'yep', 'nah', 'ờm', 'ừm', 'ok', 'okay']
        casual_score = sum(1 for indicator in casual_indicators if indicator in text)
        
        # Enthusiastic indicators
        enthusiastic_indicators = ['!', 'wow', 'cool', 'awesome', 'tuyệt', 'hay quá', 'thú vị']
        enthusiastic_score = sum(1 for indicator in enthusiastic_indicators if indicator in text)
        
        if formal_score > casual_score:
            return "Maintain a respectful, professional tone while being warm and approachable."
        elif casual_score > formal_score:
            return "Use a casual, friendly tone. Be conversational and relaxed in your responses."
        elif enthusiastic_score > 0:
            return "Match the user's enthusiasm. Use energetic, positive language and show genuine interest."
        else:
            return "Adapt your tone to match the user's communication style naturally."

    def _generate_empathy_guidance(self, message_str: str, conversation_context: str) -> str:
        """Generate empathy and emotional awareness guidance."""
        
        text = f"{conversation_context} {message_str}".lower()
        
        # Emotional indicators
        emotional_cues = {
            'curiosity': ['tại sao', 'như thế nào', 'why', 'how', 'what if', '?'],
            'confusion': ['không hiểu', 'confused', 'unclear', 'huh', 'what do you mean'],
            'excitement': ['!', 'wow', 'amazing', 'tuyệt vời', 'hay quá'],
            'concern': ['lo lắng', 'worried', 'concerned', 'problem', 'issue'],
            'appreciation': ['cảm ơn', 'thank', 'appreciate', 'helpful', 'great']
        }
        
        detected_emotions = []
        for emotion, indicators in emotional_cues.items():
            if any(indicator in text for indicator in indicators):
                detected_emotions.append(emotion)
        
        if 'curiosity' in detected_emotions:
            return "Show genuine curiosity and interest. Use expressions that demonstrate you're engaged and want to explore the topic further."
        elif 'confusion' in detected_emotions:
            return "Be patient and empathetic. Clarify gently and check for understanding. Use reassuring language."
        elif 'excitement' in detected_emotions:
            return "Match their enthusiasm! Use energetic language and show that you share their excitement about the topic."
        elif 'concern' in detected_emotions:
            return "Be supportive and understanding. Acknowledge their concerns and offer helpful, reassuring responses."
        elif 'appreciation' in detected_emotions:
            return "Acknowledge their appreciation warmly. Show that you value the interaction and are happy to help."
        else:
            return "Be empathetic and responsive to the user's emotional state. Show genuine interest and care in your responses."

    def extract_structured_sections(self, content: str) -> Dict[str, str]:
        """Extract structured sections from LLM response."""
        sections = {
            "user_response": "",
            "knowledge_synthesis": "",
            "knowledge_summary": ""
        }
        
        # Extract structured sections
        user_response_match = re.search(r'<user_response>(.*?)</user_response>', content, re.DOTALL)
        if user_response_match:
            sections["user_response"] = user_response_match.group(1).strip()
            logger.info(f"Found user_response section")
        
        synthesis_match = re.search(r'<knowledge_synthesis>(.*?)</knowledge_synthesis>', content, re.DOTALL)
        if synthesis_match:
            sections["knowledge_synthesis"] = synthesis_match.group(1).strip()
            logger.info(f"Found knowledge_synthesis section")
        
        summary_match = re.search(r'<knowledge_summary>(.*?)</knowledge_summary>', content, re.DOTALL)
        if summary_match:
            sections["knowledge_summary"] = summary_match.group(1).strip()
            logger.info(f"Found knowledge_summary section")
        
        return sections

    def extract_tool_calls_and_evaluation(self, content: str, message_str: str = "") -> tuple:
        """Extract tool calls and evaluation from LLM response."""
        tool_calls = []
        evaluation = {
            "has_teaching_intent": False, 
            "is_priority_topic": False, 
            "priority_topic_name": "", 
            "should_save_knowledge": False, 
            "intent_type": "query", 
            "name_addressed": False, 
            "ai_referenced": False
        }
        
        # Extract tool calls if present
        if "<tool_calls>" in content:
            tool_section = re.search(r'<tool_calls>(.*?)</tool_calls>', content, re.DOTALL)
            if tool_section:
                try:
                    tool_calls = json.loads(tool_section.group(1).strip())
                    content = re.sub(r'<tool_calls>.*?</tool_calls>', '', content, flags=re.DOTALL).strip()
                    logger.info(f"Extracted {len(tool_calls)} tool calls")
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool calls")
        
        # Extract evaluation if present
        if "<evaluation>" in content:
            eval_section = re.search(r'<evaluation>(.*?)</evaluation>', content, re.DOTALL)
            if eval_section:
                try:
                    evaluation = json.loads(eval_section.group(1).strip())
                    content = re.sub(r'<evaluation>.*?</evaluation>', '', content, flags=re.DOTALL).strip()
                    logger.info(f"Extracted LLM evaluation: {evaluation}")
                except json.JSONDecodeError:
                    logger.warning("Failed to parse evaluation")
        
        # Let LLM handle all teaching intent detection - no rule-based fallback
        logger.info(f"Teaching intent detection: LLM-only approach, has_teaching_intent={evaluation.get('has_teaching_intent', False)}")
        
        return content, tool_calls, evaluation

    def handle_empty_response_fallbacks(self, user_facing_content: str, response_strategy: str, message_str: str) -> str:
        """Handle cases where LLM response is empty and provide fallbacks."""
        if user_facing_content and not user_facing_content.isspace():
            return user_facing_content
        
        # Ensure closing messages get a response even if empty
        if response_strategy == "CLOSING":
            # Default closing message if the LLM didn't provide one
            if "vietnamese" in message_str.lower() or any(vn_word in message_str.lower() for vn_word in ["tạm biệt", "cảm ơn", "hẹn gặp", "thế thôi"]):
                user_facing_content = "Vâng, cảm ơn bạn đã trao đổi. Hẹn gặp lại bạn lần sau nhé!"
            else:
                user_facing_content = "Thank you for the conversation. Have a great day and I'm here if you need anything else!"
            logger.info("Added default closing response for empty LLM response")
        
        # Ensure unclear or short queries also get a helpful response when content is empty
        else:
            # Check if message is short (1-2 words) or unclear
            is_short_message = len(message_str.strip().split()) <= 2
            
            # Default response for short/unclear messages
            if "vietnamese" in message_str.lower() or any(vn_word in message_str.lower() for vn_word in ["anh", "chị", "bạn", "cô", "ông", "xin", "vui lòng"]):
                user_facing_content = f"Xin lỗi, tôi không hiểu rõ câu hỏi '{message_str}'. Bạn có thể chia sẻ thêm thông tin hoặc đặt câu hỏi cụ thể hơn được không?"
            else:
                user_facing_content = f"I'm sorry, I didn't fully understand your message '{message_str}'. Could you please provide more details or ask a more specific question?"
            
            logger.info(f"Added default response for empty LLM response to short/unclear query: '{message_str}'")
        
        return user_facing_content 

    async def generate_knowledge_queries_fast(self, primary_query: str, conversation_context: str, user_id: str) -> List[str]:
        """
        Fast rule-based query generation without expensive LLM calls.
        This replaces the slow active_learning call for query generation.
        """
        queries = [primary_query]
        
        # Extract key terms from the query
        query_lower = primary_query.lower()
        
        # Rule-based query expansion based on common patterns
        if any(term in query_lower for term in ["mục tiêu", "goals", "objective"]):
            queries.extend([
                "mục tiêu hỗ trợ khách hàng",
                "chiến lược tư vấn",
                "phương pháp tiếp cận khách hàng"
            ])
        
        if any(term in query_lower for term in ["phân nhóm", "segmentation", "nhóm khách hàng"]):
            queries.extend([
                "phân nhóm khách hàng",
                "phân tích chân dung khách hàng",
                "customer segmentation"
            ])
        
        if any(term in query_lower for term in ["tư vấn", "consultation", "hỗ trợ"]):
            queries.extend([
                "phương pháp tư vấn",
                "kỹ thuật giao tiếp",
                "xây dựng mối quan hệ"
            ])
        
        if any(term in query_lower for term in ["giao tiếp", "communication", "nói chuyện"]):
            queries.extend([
                "kỹ thuật giao tiếp",
                "cách nói chuyện hiệu quả",
                "xây dựng rapport"
            ])
        
        # Add context-based queries from conversation
        if conversation_context:
            # Extract recent topics from conversation
            recent_topics = re.findall(r'User: ([^?]*(?:\?|$))', conversation_context)
            if recent_topics:
                last_topic = recent_topics[-1].strip()
                if len(last_topic) > 10 and last_topic not in queries:
                    queries.append(last_topic)
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for query in queries:
            if query not in seen and len(query.strip()) > 5:
                unique_queries.append(query)
                seen.add(query)
        
        logger.info(f"Fast query generation: {len(unique_queries)} queries from '{primary_query[:50]}...'")
        return unique_queries[:5]  # Limit to 5 queries max

    def _get_universal_pronoun_guidance(self, conversation_context: str, message_str: str) -> str:
        """Get universal pronoun guidance that applies to ALL response strategies."""
        
        # Extract established pronouns from conversation context
        established_pronouns = self._extract_established_pronouns(conversation_context, message_str)
        
        if not established_pronouns:
            return ""
        
        return f"""
                **🔒 CRITICAL PRONOUN CONSISTENCY (APPLIES TO ALL RESPONSES)**:
                {established_pronouns}
                
                **MANDATORY CONSISTENCY RULES**:
                - ALWAYS maintain the established pronoun relationship throughout the conversation
                - If user calls you "em", ALWAYS respond as "em", never switch to "mình" or "tôi"
                - If user established themselves as "anh", ALWAYS address them as "anh"
                - This applies to ALL response types: casual, formal, knowledge-based, clarifications
                - NEVER break pronoun consistency even in brief or casual responses
                
                **VIOLATION PREVENTION**:
                - Do NOT use "mình" if "em" relationship is established
                - Do NOT switch pronouns mid-conversation
                - Do NOT let response strategy override established relationships
                """

    def _extract_established_pronouns(self, conversation_context: str, message_str: str) -> str:
        """Extract established pronoun relationships from conversation context and current message."""
        
        # Combine current message and context for analysis
        full_text = f"{conversation_context}\n{message_str}".lower()
        
        # Enhanced detection for "anh/em" relationship - look for ANY "em" followed by a verb or action
        # This covers: em nắm, em biết, em hiểu, em làm, em có, em nói, em là, em sẽ, em cần, etc.
        em_patterns = [
            'em nắm', 'em biết', 'em hiểu', 'em làm', 'em có thể', 'em đã', 'em sẽ', 
            'em cần', 'em nói', 'em là', 'em nghĩ', 'em thấy', 'em cho', 'em giúp',
            'em xem', 'em check', 'em kiểm tra', 'em tìm', 'em search', 'em tra',
            # Also detect direct addressing patterns
            'em ơi', 'em à', 'với em', 'cho em', 'em nhé'
        ]
        
        if any(pattern in full_text for pattern in em_patterns):
            return """
                **ESTABLISHED RELATIONSHIP**: User addresses you as "EM"
                - YOU must respond as "EM" in all responses
                - USER should be addressed as "ANH" (if male context) or "CHỊ" (if female context)
                - Example: "Em hiểu rồi anh", "Em sẽ giúp anh", "Em nghĩ rằng..."
                - NEVER use "mình" or "tôi" when "em" relationship is established
                """
        
        # Check for "bạn" relationship
        elif any(pattern in full_text for pattern in ['bạn nói', 'bạn có', 'mình nói', 'mình có']):
            return """
                **ESTABLISHED RELATIONSHIP**: Casual "BạN/MÌNH" relationship
                - Use "mình" to refer to yourself
                - Address user as "bạn"
                - Example: "Mình hiểu rồi", "Bạn có thể...", "Mình sẽ giúp bạn"
                """
        
        # Check current message for pronoun cues - also more comprehensive
        current_lower = message_str.lower()
        if 'em' in current_lower and any(word in current_lower for word in ['nắm', 'biết', 'hiểu', 'làm', 'có', 'nói', 'là', 'sẽ', 'cần', 'nghĩ', 'thấy', 'cho', 'giúp']):
            return """
                **RELATIONSHIP DETECTED**: User is addressing you as "EM"
                - YOU are "EM", USER is "ANH/CHỊ"
                - Respond as: "Em hiểu anh", "Em sẽ...", "Vâng anh"
                - MAINTAIN this throughout conversation
                """
        
        return ""

    async def detect_teaching_intent_llm(self, message: str, conversation_context: str = "") -> bool:
        """Use LLM to detect teaching intent by analyzing message and conversation history."""
        try:
            teaching_detection_prompt = f"""
            TASK: Analyze if the user has TEACHING INTENT in their current message.

            **CORE PRINCIPLE**: Analyze INFORMATION FLOW DIRECTION and SPEECH ACT TYPE.

            **TEACHING INTENT = TRUE when user is INFORMING/DECLARING:**
            
            **Information Flow Analysis:**
            - User → AI: Information flows FROM user TO you = TEACHING INTENT = TRUE
            - AI ← User: User requests information FROM you = TEACHING INTENT = FALSE
            
            **Speech Act Analysis:**
            - DECLARATIVE: Stating facts, plans, roles, assignments, decisions = TRUE
            - INTERROGATIVE: Asking questions, seeking information = FALSE  
            - IMPERATIVE: Commanding you to provide info/help = FALSE
            
            **Semantic Intent Markers (TRUE):**
            - Announcing future actions: "I will...", "Starting tomorrow...", "From now on..."
            - Assigning roles/tasks: "You handle...", "Your job is...", "You are responsible for..."
            - Sharing information: "Let me tell you...", "Here's what happened...", "The situation is..."
            - Making declarations: "I am...", "This is...", "We decided..."
            - Vietnamese temporal markers: "Từ mai...", "Từ hôm nay...", "Bắt đầu từ..."
            
            **Semantic Intent Markers (FALSE):**
            - Seeking information: "What is...", "How do...", "Can you explain..."
            - Requesting help: "Help me...", "Please...", "Could you..."
            - Asking for capabilities: "Can you...", "Are you able to..."
            - Vietnamese question patterns: "...như thế nào?", "...là gì?", "Anh có thể..."

            **CONVERSATION CONTEXT (for additional context):**
            {conversation_context}

            **CURRENT MESSAGE TO ANALYZE:**
            {message}

            **Analysis Questions:**
            1. Is the user GIVING me new information about plans, facts, or roles?
            2. Is the user TELLING me what to do or what will happen?
            3. Is the user ASKING me for information, help, or explanations?
            4. Is the user REQUESTING me to perform an action or provide service?

            **RESPONSE FORMAT**: Reply with only "TRUE" or "FALSE"
            - TRUE = Teaching intent detected (user is informing/declaring)
            - FALSE = No teaching intent (user is asking/requesting)

            **FOCUS ON MEANING, not exact words or message length.**

            ANSWER:"""

            from langchain_openai import ChatOpenAI
            detection_llm = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.0)
            
            response = await detection_llm.ainvoke(teaching_detection_prompt)
            result = response.content.strip().upper()
            
            has_teaching_intent = result == "TRUE"
            logger.info(f"LLM teaching intent detection: {has_teaching_intent} (response: {result})")
            
            return has_teaching_intent
            
        except Exception as e:
            logger.error(f"Error in LLM teaching intent detection: {str(e)}")
            # Fallback to False to avoid false positives
            return False