"""
Chain of Thought (CoT) handler definition for the MC system.
This file contains the implementation of the CoT handler that was accidentally deleted.
"""

import re
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Tuple
import traceback
from utilities import logger
from brain_singleton import get_brain, is_brain_loaded, load_brain_vectors, get_current_graph_version

# Access variables from tools_new.py
from tools_new import StreamLLM, PERF_METRICS, _last_cot_results, extract_structured_data_from_raw

async def cot_knowledge_analysis_actions_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Chain-of-Thought handler that combines knowledge retrieval with analysis and next actions.
    Enhanced with user profiling for more targeted knowledge retrieval and analysis.
    
    Args:
        params: Dictionary containing conversation_context, graph_version_id
                
    Yields:
        Dict events with streaming results for knowledge, analysis, and next actions
    """
    start_time = time.time()
    PERF_METRICS["cot_handler_start"] = start_time
    
    # Access the global variable to store results
    global _last_cot_results
    _last_cot_results = {
        "analysis_content": "",
        "next_actions_content": "",
        "knowledge_entries": [],
        "knowledge_context": "",
        "user_profile": {}  # Add user profile to results
    }
    
    conversation_context = params.get("conversation_context", "")
    graph_version_id = params.get("graph_version_id", "")
    thread_id = params.get("_thread_id")
    
    # Extract user message for knowledge retrieval - IMPROVED to capture up to 30 recent messages
    # This better captures the conversation context and history
    recent_messages = []
    for line in conversation_context.strip().split('\n'):
        if line.startswith("User:") or line.startswith("AI:"):
            recent_messages.append(line)
    
    # Keep last 30 messages for richer context
    recent_messages = recent_messages[-30:] if len(recent_messages) > 30 else recent_messages
    
    # Get most recent user message for initial knowledge search
    last_user_message = ""
    for line in reversed(recent_messages):
        if line.startswith("User:"):
            last_user_message = line[5:].strip()
            break
    
    # Create a comprehensive context string from recent messages
    context_window = "\n".join(recent_messages)
    
    logger.info(f"Starting CoT processing with context of {len(recent_messages)} messages")
    logger.info(f"Most recent query: '{last_user_message[:30]}...'")
    
    # 1. First, emit initial events to show progress
    # Send analysis starting event
    analysis_start_event = {
        "type": "analysis",
        "content": "Analyzing your message and building profile...",
        "complete": False,
        "thread_id": thread_id,
        "status": "analyzing"
    }
    logger.info(f"Sending initial CoT analysis event: {analysis_start_event}")
    yield analysis_start_event
    
    # NEW STEP: Build user profile first
    logger.info("Building user profile before knowledge retrieval")
    from tools_new import build_user_profile
    user_profile = build_user_profile(context_window, last_user_message)
    _last_cot_results["user_profile"] = user_profile  # Store for future use
    
    # Generate enhanced search queries based on user profile
    from tools_new import generate_profile_enhanced_queries
    enhanced_queries = generate_profile_enhanced_queries(last_user_message, user_profile)
    logger.info(f"Generated {len(enhanced_queries)} profile-enhanced queries: {enhanced_queries}")
    
    # Send knowledge search starting event
    knowledge_start_event = {
        "type": "knowledge",
        "content": [],
        "complete": False,
        "status": "searching",
        "thread_id": thread_id
    }
    logger.info(f"Sending knowledge start event for CoT: {knowledge_start_event}")
    yield knowledge_start_event
    
    # 2. Retrieve knowledge using profile-enhanced queries
    knowledge_entries = []
    structured_knowledge = []  # To store extracted structured data
    knowledge_context = ""
    
    try:
        # Make sure brain is loaded
        brain_loaded = False
        try:
            # Set short timeout for brain loading
            from tools_new import ensure_brain_loaded
            async with asyncio.timeout(5.0):  # Increase from 3 to 5 seconds
                brain_loaded = await ensure_brain_loaded(graph_version_id)
        except asyncio.TimeoutError:
            logger.warning("Brain loading timed out in CoT, continuing with limited functionality")
            brain_loaded = False
            
        if brain_loaded:
            # Get global brain instance
            from brain_singleton import get_brain
            brain = get_brain()
            
            # Use profile-enhanced queries for more targeted search
            # Use a short timeout for knowledge query
            KNOWLEDGE_TIMEOUT = 10  # Increase from 8 to 10 seconds
            
            try:
                # Process with timeout
                async with asyncio.timeout(KNOWLEDGE_TIMEOUT):
                    # HYBRID APPROACH - Phase 1: Profile-enhanced knowledge retrieval
                    logger.info(f"CoT PHASE 1: Profile-enhanced knowledge queries: {enhanced_queries[:2]}")
                    
                    # Use top 2 profile-enhanced queries for better results
                    for query_idx, enhanced_query in enumerate(enhanced_queries[:2]):
                        initial_results = await brain.get_similar_vectors_by_text(enhanced_query, top_k=3)
                        
                        # Process initial knowledge results
                        for vector_id, vector, metadata, similarity in initial_results:
                            # Skip duplicates
                            if any(entry.get("id") == vector_id for entry in knowledge_entries):
                                continue
                                
                            raw_text = metadata.get("raw", "")
                            structured_data = extract_structured_data_from_raw(raw_text)
                            
                            entry = {
                                "id": vector_id,
                                "similarity": float(similarity),
                                "raw": raw_text,
                                "structured": structured_data,
                                "query": enhanced_query,
                                "phase": "initial",
                                "profile_match": True  # Mark as profile-matched
                            }
                            knowledge_entries.append(entry)
                            
                            if structured_data:
                                structured_knowledge.append(structured_data)
                    
                    # Stream initial knowledge results
                    if knowledge_entries:
                        knowledge_event = {
                            "type": "knowledge",
                            "content": knowledge_entries,
                            "complete": False,
                            "thread_id": thread_id,
                            "status": "searching"
                        }
                        logger.info(f"CoT PHASE 1: Found {len(knowledge_entries)} initial knowledge entries with profile-enhanced queries")
                        yield knowledge_event
                    
                    # HYBRID APPROACH - Phase 2: Initial Analysis with Profile + Knowledge
                    # Perform quick analysis on initial results before expanded search
                    analysis_from_initial = ""
                    if knowledge_entries:
                        # Extract key insights based on user profile and initial knowledge
                        segment = user_profile["segment"]["category"]
                        emotion = user_profile["emotional_state"]["primary"]
                        
                        # Generate a brief analysis text based on profile and initial knowledge
                        analysis_from_initial = f"User appears to be in {segment} segment with {emotion} emotional state. "
                        
                        # Extract relevant topics from knowledge
                        topics = []
                        for entry in knowledge_entries:
                            if entry.get("structured", {}).get("title"):
                                topics.append(entry["structured"]["title"])
                        
                        if topics:
                            analysis_from_initial += f"Initial knowledge suggests interest in: {', '.join(topics[:3])}"
                        
                        logger.info(f"Generated initial analysis for Phase 2: {analysis_from_initial}")
                    
                    # HYBRID APPROACH - Phase 3: Expanded Knowledge with Analysis + Profile
                    if knowledge_entries:
                        # Extract key concepts from initial knowledge to inform expanded search
                        search_terms = []
                        key_concepts = set()
                        
                        # 1. First extract from cluster connections with enhanced parsing for Vietnamese
                        for entry in knowledge_entries:
                            structured = entry.get("structured", {})
                            if structured and "cross_cluster_connections" in structured:
                                connections_text = structured["cross_cluster_connections"]
                                
                                # Try to extract cluster references (e.g., "Cluster 8") for Vietnamese
                                cluster_refs = re.findall(r'Cluster (\d+)', connections_text)
                                for cluster_num in cluster_refs[:3]:
                                    key_concepts.add(f"Cluster {cluster_num}")
                                
                                # Extract phrases that might be between quotes
                                key_phrases = re.findall(r'"([^"]+)"|"([^"]+)"', connections_text)
                                for phrase in key_phrases[:2]:
                                    if isinstance(phrase, tuple):
                                        phrase = next((p for p in phrase if p), "")
                                    if phrase:
                                        key_concepts.add(phrase)
                                
                                # If normal splitting would work better for other languages
                                if not cluster_refs and not key_phrases:
                                    concepts = connections_text.split(", ")
                                    for concept in concepts[:3]:
                                        key_concepts.add(concept)
                        
                        # 2. Then extract titles and themes
                        for entry in knowledge_entries:
                            structured = entry.get("structured", {})
                            if structured and "title" in structured:
                                key_concepts.add(structured["title"])
                        
                        # 3. Add key concepts as search terms
                        search_terms = list(key_concepts)[:3]
                        
                        # 4. Create search term from initial analysis
                        if analysis_from_initial:
                            # Extract the most information-rich part
                            if "interest in:" in analysis_from_initial:
                                interests_part = analysis_from_initial.split("interest in:")[1].strip()
                                search_terms.append(interests_part)
                            
                        # 5. Add user profile information as search terms
                        profile_term = ""
                        knowledge_areas = user_profile.get("query_characteristics", {}).get("knowledge_areas", [])
                        if knowledge_areas:
                            profile_term += " ".join(knowledge_areas)
                        if user_profile["segment"]["category"] != "general":
                            profile_term += f" {user_profile['segment']['category']}"
                        if profile_term:
                            search_terms.append(f"{last_user_message} {profile_term}")
                        
                        # 6. Also generate a search term based on context window
                        if context_window:
                            context_query = " ".join(recent_messages[-3:])
                            if context_query and context_query != last_user_message:
                                search_terms.append(context_query)
                        
                        logger.info(f"CoT PHASE 3: Expanded search with terms: {search_terms}")
                        
                        # Track existing IDs to avoid duplicates
                        existing_ids = {entry["id"] for entry in knowledge_entries}
                        
                        # Execute expanded searches
                        for term in search_terms:
                            expanded_results = await brain.get_similar_vectors_by_text(term, top_k=1)
                            
                            # Process additional knowledge results
                            for vector_id, vector, metadata, similarity in expanded_results:
                                # Skip if we already have this entry
                                if vector_id in existing_ids:
                                    continue
                                    
                                existing_ids.add(vector_id)
                                raw_text = metadata.get("raw", "")
                                structured_data = extract_structured_data_from_raw(raw_text)
                                
                                entry = {
                                    "id": vector_id,
                                    "similarity": float(similarity),
                                    "raw": raw_text,
                                    "structured": structured_data,
                                    "query": term,
                                    "phase": "expanded"
                                }
                                knowledge_entries.append(entry)
                                
                                if structured_data:
                                    structured_knowledge.append(structured_data)
                        
                        # Stream the expanded results
                        if len(knowledge_entries) > 0:
                            expanded_event = {
                                "type": "knowledge",
                                "content": knowledge_entries,
                                "complete": False,
                                "thread_id": thread_id,
                                "status": "searching"
                            }
                            
                            logger.info(f"CoT PHASE 3: Additional knowledge found, now {len(knowledge_entries)} total entries")
                            yield expanded_event
            
            except asyncio.TimeoutError:
                logger.warning(f"CoT knowledge query timed out after {KNOWLEDGE_TIMEOUT}s")
                # Ensure Phase 3 properly completes even with timeout
                if knowledge_entries:
                    yield {
                        "type": "knowledge",
                        "content": knowledge_entries,
                        "complete": True,
                        "thread_id": thread_id,
                        "status": "complete",
                        "stats": {"total_results": len(knowledge_entries)},
                        "note": "Partial results due to timeout"
                    }
            except Exception as e:
                logger.error(f"Error in CoT knowledge query: {e}")
                # Ensure Phase 3 completes with error status
                yield {
                    "type": "knowledge",
                    "content": knowledge_entries,
                    "complete": True,
                    "thread_id": thread_id,
                    "status": "complete",
                    "stats": {"total_results": len(knowledge_entries)},
                    "error": f"Error in knowledge retrieval: {str(e)}"
                }
            
            # Send knowledge complete event
            yield {
                "type": "knowledge",
                "content": knowledge_entries,
                "complete": True,
                "thread_id": thread_id,
                "status": "complete",
                "stats": {"total_results": len(knowledge_entries)}
            }
            
            # Format knowledge context for the prompt using the optimized function
            if knowledge_entries:
                # Use the optimized formatter 
                from tools_new import optimize_knowledge_context
                knowledge_context = optimize_knowledge_context(knowledge_entries, last_user_message, max_chars=2500)
                logger.info(f"Created optimized knowledge context: {len(knowledge_context)} chars")
            else:
                # Fall back to empty knowledge context if no entries found
                knowledge_context = ""
                logger.warning("No knowledge entries found, knowledge context will be empty")
            
            # Now perform combined analysis and next actions with knowledge context and user profile
            current_section = None
            analysis_content = ""
            next_actions_content = ""
            
            # Build an enhanced CoT prompt with structured knowledge and user profile
            # Format the user profile for the prompt
            profile_summary = f"""
            USER PROFILE:
            - Identity: {user_profile['identity']['expertise_level']} level user, language: {user_profile['identity']['language_preference']}
            - Segment: {user_profile['segment']['category']}, interest: {user_profile['segment']['interest_level']}, stage: {user_profile['segment']['customer_stage']}
            - Communication: {user_profile['communication']['style']} style, {user_profile['communication']['tone']} tone, prefers {user_profile['communication']['detail_preference']} details
            - Emotional State: Primarily {user_profile['emotional_state']['primary']}, urgency: {user_profile['emotional_state']['urgency']}
            - Query Type: {user_profile['query_characteristics']['type']}, complexity: {user_profile['query_characteristics']['complexity']}
            """
            
            cot_prompt = f"""
            You are an AI assistant using Implicit Chain-of-Thought reasoning to help users effectively.
            
            CONVERSATION CONTEXT:
            {context_window}
            
            {profile_summary}
            
            {f"RELEVANT KNOWLEDGE:\n{knowledge_context}" if knowledge_context else "NO RELEVANT KNOWLEDGE FOUND."}
            
            Based on the conversation context, user profile, and provided knowledge:
            
            [ANALYSIS]
            Analyze the user's core needs through the lens of their profile. Consider their {user_profile['emotional_state']['primary']} emotional state, {user_profile['communication']['style']} communication style, and {user_profile['identity']['expertise_level']} expertise level.
            
            Address how the knowledge can specifically help this user segment ({user_profile['segment']['category']}) with their {user_profile['query_characteristics']['complexity']} complexity {user_profile['query_characteristics']['type']} query.
            [/ANALYSIS]
            
            [NEXT_ACTIONS]
            1. Most important action tailored to this user profile
            2. Secondary action considering their communication preferences  
            3. Additional action if needed based on their emotional state
            
            Each action should be concise, practical, and directly applicable to this specific user.
            [/NEXT_ACTIONS]
            """
            
            # Set timeout for combined analysis
            COT_TIMEOUT = 30  # seconds
            
            logger.info("Starting enhanced CoT LLM stream with structured knowledge and user profile")
            try:
                # Stream the CoT analysis with timeout
                stream_gen = StreamLLM.astream(cot_prompt)
                
                try:
                    # Process with timeout
                    async with asyncio.timeout(COT_TIMEOUT):
                        async for chunk in stream_gen:
                            content = chunk.content
                            
                            # Process content to separate sections
                            for line in content.split('\n'):
                                if '[ANALYSIS]' in line:
                                    current_section = 'analysis'
                                    # Send analysis section start event
                                    yield {
                                        "type": "analysis",
                                        "content": "",
                                        "complete": False,
                                        "thread_id": thread_id,
                                        "status": "analyzing"
                                    }
                                    continue
                                elif '[/ANALYSIS]' in line:
                                    # Complete analysis section
                                    analysis_complete_event = {
                                        "type": "analysis",
                                        "content": analysis_content,
                                        "complete": True,
                                        "thread_id": thread_id,
                                        "status": "complete",
                                        "search_terms": [],
                                        "user_profile": user_profile  # Include user profile in result
                                    }
                                    yield analysis_complete_event
                                    current_section = None
                                    continue
                                elif '[NEXT_ACTIONS]' in line:
                                    current_section = 'next_actions'
                                    # Send next_actions section start event
                                    yield {
                                        "type": "next_actions",
                                        "content": "",
                                        "complete": False,
                                        "thread_id": thread_id,
                                        "status": "analyzing"
                                    }
                                    continue
                                elif '[/NEXT_ACTIONS]' in line:
                                    # Complete next_actions section
                                    next_actions_complete_event = {
                                        "type": "next_actions",
                                        "content": next_actions_content,
                                        "complete": True,
                                        "thread_id": thread_id,
                                        "status": "complete",
                                        "user_profile": user_profile  # Include user profile in result
                                    }
                                    yield next_actions_complete_event
                                    current_section = None
                                    continue
                                
                                # Add content to appropriate section and stream
                                if current_section == 'analysis':
                                    analysis_content += line + "\n"
                                    # Stream analysis line
                                    yield {
                                        "type": "analysis",
                                        "content": line,
                                        "complete": False,
                                        "thread_id": thread_id,
                                        "status": "analyzing"
                                    }
                                elif current_section == 'next_actions':
                                    next_actions_content += line + "\n"
                                    # Stream next_actions line
                                    yield {
                                        "type": "next_actions",
                                        "content": line,
                                        "complete": False,
                                        "thread_id": thread_id,
                                        "status": "analyzing" 
                                    }
                
                except asyncio.TimeoutError:
                    logger.warning(f"CoT analysis timed out after {COT_TIMEOUT}s, using partial results")
                    logger.info("CoT processing timed out, continuing with partial results")
                    
                except Exception as e:
                    logger.error(f"Error in CoT analysis streaming: {e}")
                    logger.error(traceback.format_exc())
                
                # Now check the module-level variable for results if needed
                if not analysis_content:
                    # Create fallback analysis with knowledge reference and user profile
                    fallback_analysis = f"User with {user_profile['emotional_state']['primary']} emotional state and {user_profile['communication']['style']} communication style is asking about: {last_user_message[:50]}"
                    if knowledge_entries:
                        knowledge_topics = []
                        for entry in knowledge_entries[:2]:  # Use top 2 entries
                            structured = entry.get("structured", {})
                            if structured and "title" in structured:
                                knowledge_topics.append(structured["title"])
                        
                        if knowledge_topics:
                            fallback_analysis += f". This relates to the topics: {', '.join(knowledge_topics)}"
                        else:
                            fallback_analysis += f". This relates to {len(knowledge_entries)} knowledge entries."
                    
                    yield {
                        "type": "analysis",
                        "content": fallback_analysis,
                        "complete": True,
                        "thread_id": thread_id,
                        "status": "complete",
                        "search_terms": [],
                        "user_profile": user_profile
                    }
                    analysis_content = fallback_analysis
                
                if not next_actions_content:
                    # Create fallback next actions with knowledge reference and user profile
                    fallback_actions = []
                    
                    # Try to create profile-aware actions
                    emotional_state = user_profile["emotional_state"]["primary"]
                    comm_style = user_profile["communication"]["style"]
                    expertise = user_profile["identity"]["expertise_level"]
                    
                    # Emotional state handling
                    if emotional_state == "frustrated":
                        fallback_actions.append("1. Address frustration by offering clear solutions")
                    elif emotional_state == "curious":
                        fallback_actions.append("1. Satisfy curiosity with detailed explanations")
                    elif emotional_state == "urgent":
                        fallback_actions.append("1. Address urgent need with immediate actionable steps")
                    else:
                        fallback_actions.append("1. Provide balanced information and guidance")
                    
                    # Communication style handling
                    if comm_style == "formal":
                        fallback_actions.append("2. Maintain formal tone and structured response")
                    elif comm_style == "casual":
                        fallback_actions.append("2. Use conversational tone with practical examples")
                    elif comm_style == "technical":
                        fallback_actions.append("2. Include technical details and precise information")
                    else:
                        fallback_actions.append("2. Balance clarity with appropriate detail level")
                    
                    # Expertise level handling
                    if expertise == "beginner":
                        fallback_actions.append("3. Explain basic concepts clearly without jargon")
                    elif expertise == "advanced":
                        fallback_actions.append("3. Provide advanced insights and in-depth analysis")
                    else:
                        fallback_actions.append("3. Balance foundational concepts with practical applications")
                    
                    # Join the actions
                    fallback_actions_text = "\n".join(fallback_actions)
                    
                    yield {
                        "type": "next_actions",
                        "content": fallback_actions_text,
                        "complete": True,
                        "thread_id": thread_id,
                        "status": "complete",
                        "user_profile": user_profile
                    }
                    next_actions_content = fallback_actions_text
                
                # Before completing sections, store results in the module-level variable
                if analysis_content:
                    _last_cot_results["analysis_content"] = analysis_content
                
                if next_actions_content:
                    _last_cot_results["next_actions_content"] = next_actions_content
                
                if knowledge_entries:
                    _last_cot_results["knowledge_entries"] = knowledge_entries
                    _last_cot_results["knowledge_context"] = knowledge_context
                
                _last_cot_results["user_profile"] = user_profile
                
                # Track completion time
                PERF_METRICS["cot_handler_end"] = time.time()
                total_time = PERF_METRICS["cot_handler_end"] - PERF_METRICS["cot_handler_start"]
                logger.info(f"CoT handler completed in {total_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in CoT processing: {e}")
                logger.error(traceback.format_exc())
                
                # Create fallback content
                if not analysis_content:
                    analysis_content = f"User with {user_profile['emotional_state']['primary']} emotional state is asking about: {last_user_message[:30]}"
                    fallback_analysis = {
                        "type": "analysis", 
                        "content": analysis_content, 
                        "complete": True,
                        "thread_id": thread_id,
                        "status": "complete",
                        "search_terms": [],
                        "user_profile": user_profile
                    }
                    logger.info(f"Sending fallback analysis due to CoT error: {fallback_analysis}")
                    
                    # Use socketio_manager directly for WebSocket events
                    if thread_id:
                        try:
                            from socketio_manager import emit_analysis_event
                            was_delivered = emit_analysis_event(thread_id, fallback_analysis)
                        except Exception as e:
                            logger.error(f"Error emitting fallback analysis: {str(e)}")
                            yield fallback_analysis
                
                if not next_actions_content:
                    next_actions_results = "1. Provide helpful information aligned with user's communication style\n2. Use appropriate tone for their emotional state"
                    fallback_next_actions = {
                        "type": "next_actions", 
                        "content": next_actions_results, 
                        "complete": True,
                        "thread_id": thread_id,
                        "status": "complete",
                        "user_profile": user_profile
                    }
                    logger.info(f"Sending fallback next_actions due to CoT error")
                    
                    # Use socketio_manager directly for WebSocket events
                    if thread_id:
                        try:
                            from socketio_manager import emit_next_action_event
                            was_delivered = emit_next_action_event(thread_id, fallback_next_actions)
                        except Exception as e:
                            logger.error(f"Error emitting fallback next_actions: {str(e)}")
                            yield fallback_next_actions
                
                # Yield the final state
                yield {"state": _last_cot_results}
        else:
            # Send empty knowledge complete event if brain not loaded
            yield {
                "type": "knowledge",
                "content": [],
                "complete": True,
                "thread_id": thread_id,
                "status": "complete",
                "error": "Could not access knowledge database"
            }
    except Exception as e:
        logger.error(f"Error in CoT knowledge retrieval: {e}")
        # Send knowledge error event
        yield {
            "type": "knowledge",
            "content": [],
            "complete": True,
            "thread_id": thread_id,
            "status": "complete",
            "error": f"Error retrieving knowledge: {str(e)}"
        } 