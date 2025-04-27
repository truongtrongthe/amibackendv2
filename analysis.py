from typing import Dict, AsyncGenerator, Optional, Tuple, List
from utilities import logger
from langchain_openai import ChatOpenAI
import re

StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)
inferLLM = ChatOpenAI(model="gpt-4o", streaming=False)

__all__ = [
    'stream_analysis', 
    'stream_next_action',
    'build_context_analysis_prompt', 
    'build_next_actions_prompt',
    'process_analysis_result', 
    'process_next_actions_result',
    'extract_search_terms', 
    'extract_search_terms_from_next_actions'
]

async def stream_analysis(prompt: str, thread_id_for_analysis: Optional[str] = None, use_websocket: bool = False) -> AsyncGenerator[Dict, None]:
    """
    Stream the context analysis (parts 1-3) from the LLM.
    This function is specifically for the initial analysis, not for next actions.
    
    Args:
        prompt: The analysis prompt for parts 1-3
        thread_id_for_analysis: Thread ID to use for WebSocket analysis events
        use_websocket: Whether to use WebSocket for streaming
        
    Yields:
        Dict: Analysis events with streaming content
    """
    analysis_buffer = ""
    logger.info(f"Starting stream_analysis with use_websocket={use_websocket}, thread_id={thread_id_for_analysis}")
    
    try:
        async for chunk in StreamLLM.astream(prompt,temperature=0.1,max_tokens=5000):
            chunk_content = chunk.content
            analysis_buffer += chunk_content
            
            # Create analysis event with proper structure
            analysis_event = {
                "type": "analysis", 
                "content": chunk_content, 
                "complete": False
            }
            
            # If using WebSocket and thread ID is provided, emit to that room
            if use_websocket and thread_id_for_analysis:
                try:
                    from socketio_manager import emit_analysis_event
                    was_delivered = emit_analysis_event(thread_id_for_analysis, analysis_event)
                except Exception as e:
                    logger.error(f"Error in socketio_manager websocket delivery: {str(e)}")
                    was_delivered = False
            
            # Always yield for the standard flow too
            yield analysis_event
        
        # Process the analysis JUST ONCE to extract search terms and structure content
        processed_result = None
        try:
            # Process once and store the result for reuse
            processed_result = process_analysis_result(analysis_buffer)
            processed_content = processed_result.get("analysis_full", analysis_buffer)
        except Exception as process_error:
            logger.error(f"Error processing analysis: {str(process_error)}")
            processed_content = analysis_buffer
            processed_result = {
                "english": analysis_buffer,
                "vietnamese": "",
                "analysis_full": analysis_buffer,
                "search_terms": [
                    "health and lifestyle concerns",
                    "culturally sensitive communication in Vietnam",
                    "information gathering stage"
                ],
                "next_action_full": ""
            }
        
        # Final complete event for analysis - ensure it's a string
        if not isinstance(processed_content, str):
            processed_content = str(processed_content)
            
        analysis_complete_event = {
            "type": "analysis", 
            "content": processed_content, 
            "complete": True,
            # Include the extracted search terms in the event
            "search_terms": processed_result.get("search_terms", [])
        }
        
        # Send via WebSocket if configured
        if use_websocket and thread_id_for_analysis:
            try:
                logger.info(f"Emitting complete analysis via WebSocket to thread {thread_id_for_analysis}, length={len(processed_content)}")
                from socketio_manager import emit_analysis_event
                was_delivered_analysis = emit_analysis_event(thread_id_for_analysis, analysis_complete_event)
                if was_delivered_analysis:
                    logger.info(f"Successfully emitted complete analysis to WebSocket thread {thread_id_for_analysis}")
                else:
                    logger.warning(f"Complete analysis NOT delivered to WebSocket thread {thread_id_for_analysis} - No active sessions")
            except Exception as e:
                logger.error(f"Error in socketio_manager delivery of complete event: {str(e)}")
        
        # Yield final analysis complete event
        yield analysis_complete_event
            
    except Exception as e:
        logger.error(f"Analysis streaming failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Error event for analysis
        error_event = {
            "type": "analysis", 
            "content": f"Error in analysis process: {str(e)}", 
            "complete": True, 
            "error": True
        }
        
        # Send via WebSocket if configured
        if use_websocket and thread_id_for_analysis:
            try:
                from socketio_manager import emit_analysis_event
                was_delivered = emit_analysis_event(thread_id_for_analysis, error_event)
            except Exception as e:
                logger.error(f"Error in socketio_manager delivery of error event: {str(e)}")
        
        # Yield error event
        yield error_event

async def stream_next_action(prompt: str, thread_id_for_analysis: Optional[str] = None, use_websocket: bool = False) -> AsyncGenerator[Dict, None]:
    """
    Stream the next actions analysis from the LLM.
    This function is specifically for the next actions part, separate from the initial analysis.
    
    Args:
        prompt: The next actions prompt
        thread_id_for_analysis: Thread ID to use for WebSocket next_action events
        use_websocket: Whether to use WebSocket for streaming
        
    Yields:
        Dict: Next action events with streaming content
    """
    next_action_buffer = ""
    try:
        async for chunk in StreamLLM.astream(prompt,temperature=0.1,max_tokens=5000):
            chunk_content = chunk.content
            next_action_buffer += chunk_content
            
            # Create next_action event
            next_action_event = {
                "type": "next_actions", 
                "content": chunk_content, 
                "complete": False
            }
            
            # If using WebSocket and thread ID is provided, emit to that room
            if use_websocket and thread_id_for_analysis:
                try:
                    from socketio_manager import emit_next_action_event
                    emit_next_action_event(thread_id_for_analysis, next_action_event)
                except Exception as e:
                    logger.error(f"Error in socketio_manager websocket delivery: {str(e)}")
                    
            # Always yield for the standard flow too
            yield next_action_event
        
        # Send a final complete message with the full next actions
        logger.info(f"Streaming complete next actions, length: {len(next_action_buffer)}")
        
        # Process the next actions to structure content
        try:
            next_actions_data = process_next_actions_result(next_action_buffer)
            next_action_full = next_actions_data.get("next_action_full", next_action_buffer)
        except Exception as process_error:
            logger.error(f"Error processing next actions: {str(process_error)}")
            next_action_full = next_action_buffer
        
        # Ensure content is a string
        if not isinstance(next_action_full, str):
            next_action_full = str(next_action_full)
        
        # Final complete event for next_action
        next_action_complete_event = {
            "type": "next_actions", 
            "content": next_action_full, 
            "complete": True
        }
        
        # Send via WebSocket if configured
        if use_websocket and thread_id_for_analysis:
            try:
                from socketio_manager import emit_next_action_event
                __ = emit_next_action_event(thread_id_for_analysis, next_action_complete_event)
            except Exception as e:
                logger.error(f"Error in socketio_manager delivery of complete event: {str(e)}")
        
        # Yield final next_action complete event
        yield next_action_complete_event
            
    except Exception as e:
        logger.error(f"Next action streaming failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Error event for next_action
        error_event = {
            "type": "next_actions", 
            "content": f"Error in next action process: {str(e)}", 
            "complete": True, 
            "error": True
        }
        
        # Send via WebSocket if configured
        if use_websocket and thread_id_for_analysis:
            try:
                from socketio_manager import emit_next_action_event
                was_delivered = emit_next_action_event(thread_id_for_analysis, error_event)
            except Exception as e:
                logger.error(f"Error in socketio_manager delivery of error event: {str(e)}")
        
        # Yield error event
        yield error_event

def build_context_analysis_prompt(context: str, process_instructions: str) -> str:
    """
    Build a minimalist context analysis prompt focused on essential elements needed
    to understand the contact and guide the conversation, including customer classification.
    
    Args:
        context: The conversation context
        process_instructions: The knowledge base instructions
        
    Returns:
        str: The simplified essential analysis prompt
    """
    # Detect if this is the first message
    is_first_message = "User:" in context and context.count("User:") == 1 and "AI:" not in context
    first_message_note = "Note: This is the first message from the contact." if is_first_message else ""
    
    # Ultra-streamlined prompt focusing only on essentials
    return (
        f"Conversation:\n{context}\n\n"
        f"Process Instructions (Reference):\n{process_instructions}\n\n"
        f"{first_message_note}\n\n"
        
        f"IMPORTANT: Pay close attention to the structure of the process instructions. For each knowledge item, extract and use:"
        f"- The TITLE to understand the general topic"
        f"- The DESCRIPTION to grasp the purpose and context"
        f"- The TAKEAWAYS section containing specific APPLICATION METHODS with step-by-step instructions"
        f"- The CROSS-CLUSTER CONNECTIONS to understand how this knowledge relates to other concepts\n\n"
        
        f"Analyze this conversation using customer profiling techniques found in the process instructions. "
        f"Be concise and focus on evidence-based observations.\n\n"
        
        f"1. CONTACT IDENTITY:\n"
        f"   - DIRECTLY REFERENCE the customer classification frameworks from process instructions\n" 
        f"   - Who is this person? Identify demographics (age group, profession, role) using profiling techniques from process instructions\n"
        f"   - Apply both demographic AND psychographic profiling methods mentioned in process instructions\n"
        f"   - What is their current situation or context? Consider behavioral indicators mentioned in process instructions\n"
        f"   - CLASSIFY this contact using EXACTLY the categories/system provided in process instructions\n"
        f"   - If industry-specific classifications exist in process instructions, apply those frameworks\n\n"
        
        f"2. CORE NEEDS & DESIRES:\n"
        f"   - What specific problem or need does this person have? Consider both explicit statements and implicit signals\n"
        f"   - What outcome or solution are they seeking? Identify both practical and emotional objectives\n"
        f"   - What concerns or pain points are they expressing? Note priority and urgency signals\n"
        f"   - Assess their emotional state using the emotional assessment techniques referenced in process instructions\n\n"
        
        f"3. INTERACTION PATTERN:\n"
        f"   - What conversation stage are we in? Apply context-aware profiling techniques from process instructions\n"
        f"   - How do they communicate? (direct, detailed, brief, formal, indirect, etc.)\n"
        f"   - What communication preferences and channel preferences can you identify?\n"
        f"   - What cultural factors are relevant? Apply cultural assessment techniques from process instructions, especially for Vietnamese context\n\n"
        
        f"4. CONVERSATION FOCUS:\n"
        f"   - What is the main topic or purpose of this conversation? Prioritize based on customer type\n"
        f"   - What critical information is missing to move forward? Consider both stated and implied needs\n"
        f"   - What specific topics should be addressed next? Align with the customer profile you've identified\n\n"
        
        f"5. SUMMARY:\n"
        f"   - In 1-2 sentences, summarize who this person is, their classification according to process instructions, what they need, and how we should communicate with them.\n\n"
        f"Be direct, factual, and objective. Focus only on what's clearly evident in the conversation. "
        f"After completing the English analysis, provide a Vietnamese translation using terminology consistent with process instructions."
    )

def build_next_actions_prompt(context: str, initial_analysis: str, knowledge_content: str) -> str:
    """
    Build a next actions prompt that leverages the contact analysis data (identity, classification,
    needs, and communication style) to plan next steps using Chain of Thought reasoning.
    
    Args:
        context: The conversation context
        initial_analysis: The results from the initial analysis
        knowledge_content: The retrieved knowledge content
        
    Returns:
        str: The next actions prompt with CoT structure for effective planning
    """
    return (
        f"CONVERSATION CONTEXT:\n{context}\n\n"
        f"CONTACT ANALYSIS:\n{initial_analysis}\n\n"
        f"RETRIEVED KNOWLEDGE:\n{knowledge_content}\n\n"
        
        f"Based on the conversation and analysis above, determine the most appropriate next actions using Chain of Thought reasoning. "
        f"The analysis includes the contact's identity, classification, needs, and communication style. "
        f"Use the retrieved knowledge to select the most effective approaches.\n\n"
        
        f"IMPORTANT: Pay close attention to the structure of the retrieved knowledge. For each knowledge item, extract and use:"
        f"- The TITLE to understand the general topic"
        f"- The DESCRIPTION to grasp the purpose and context"
        f"- The TAKEAWAYS section containing specific APPLICATION METHODS with step-by-step instructions"
        f"- The CROSS-CLUSTER CONNECTIONS to understand how this knowledge relates to other concepts"
        f"These sections provide valuable structure and context for your recommendations.\n\n"
        
        f"Use the retrieved knowledge, build Chain of Thought reasoning: "
        f"1. Identify the appropriate knowledge elements relevant to this situation"
        f"2. Extract the specific value and application methods from the Takeaways section"
        f"3. Determine how to apply these methods in your conversation flow"
        f"4. Reference concrete examples from the knowledge to guide implementation\n\n"
        
        f"CHAIN OF THOUGHT REASONING:\n\n"
        
        f"1. CONTACT UNDERSTANDING:\n"
        f"   - Identify the contact's classification (potential customer, existing client, etc.)\n"
        f"   - Summarize their key needs or problems based on the analysis\n"
        f"   - Note their emotional state and how it should influence your approach\n"
        f"   - Consider their communication style preferences (direct, indirect, formal, etc.)\n"
        f"   - Identify cultural factors that should shape your response\n\n"
        
        f"2. CONVERSATION STAGE PLANNING:\n"
        f"   - Determine exactly where we are in the conversation journey\n"
        f"   - Identify what's typically needed at this specific stage\n"
        f"   - Evaluate what information is missing and must be gathered\n"
        f"   - Decide whether to focus on rapport building, information gathering, or solution presentation\n\n"
        
        f"3. KNOWLEDGE APPLICATION:\n"
        f"   - Identify 2-3 specific techniques from the knowledge content that apply to this situation\n"
        f"   - For each technique, extract specific methods from the Takeaways section\n"
        f"   - Reference step-by-step application instructions from the knowledge\n"
        f"   - Connect each knowledge element directly to this contact's specific needs and style\n\n"
        
        f"4. RESPONSE PLANNING:\n"
        f"   - Based on the above reasoning, determine the primary objective for the next message\n"
        f"   - Decide on the most appropriate tone and approach for this specific contact\n"
        f"   - Select the most effective techniques to apply from the knowledge content\n"
        f"   - Choose specific questions or statements that align with the contact's needs and style\n"
        f"   - Plan for different possible responses (positive engagement or resistance)\n\n"
        
        f"5. CLEAR NEXT ACTIONS:\n"
        f"   - Define precisely 3-5 specific actions that must be taken next\n"
        f"   - For each action, explain WHY it's important based on your reasoning\n"
        f"   - Prioritize these actions in order of importance\n"
        f"   - Connect each action directly to the contact's needs and the conversation stage\n"
        f"   - Specify which knowledge/techniques will be applied for each action\n\n"
        
        f"NEXT ACTIONS OUTPUT:\n"
        f"Based on your Chain of Thought reasoning, provide the recommended next actions in this structured format:\n\n"
        
        f"1. PRIMARY OBJECTIVE:\n"
        f"   [Single most important goal for the next response, considering the contact's classification and needs]\n\n"
        
        f"2. COMMUNICATION APPROACH:\n"
        f"   - TONE: [Specific emotional tone tailored to this contact's state and style]\n"
        f"   - STYLE: [Communication style matching the contact's preferences (direct/indirect, formal/casual, etc.)]\n"
        f"   - CULTURAL CONSIDERATIONS: [Specific cultural elements to incorporate or be mindful of]\n\n"
        
        f"3. KEY TECHNIQUES:\n"
        f"   - TECHNIQUE 1: [Name a specific technique from the knowledge]\n"
        f"     APPLICATION: [How to apply it to this specific contact and situation]\n"
        f"     SOURCE: [Brief relevant quote from knowledge content]\n"
        f"   - TECHNIQUE 2: [Name a second technique if appropriate]\n"
        f"     APPLICATION: [How to apply it to this specific contact and situation]\n"
        f"     SOURCE: [Brief relevant quote from knowledge content]\n\n"
        
        f"4. RECOMMENDED RESPONSE ELEMENTS:\n"
        f"   - OPENING: [How to start the response effectively]\n"
        f"   - KEY POINTS: [2-3 main points to include, aligned with contact's needs]\n"
        f"   - QUESTIONS: [Specific questions from knowledge content, if appropriate]\n"
        f"   - CLOSING: [How to effectively conclude this message]\n\n"
        
        f"5. ADAPTABILITY PLAN:\n"
        f"   - IF POSITIVE ENGAGEMENT: [Next step if they respond well]\n"
        f"   - IF RESISTANCE OR CONFUSION: [Alternative approach if needed]\n\n"
        
        f"6. PRIORITY NEXT ACTIONS:\n"
        f"   List the 3-5 most important specific actions to take next, in priority order:\n"
        f"   1. [Most important action] - WHY: [Brief explanation of importance]\n"
        f"   2. [Second action] - WHY: [Brief explanation of importance]\n"
        f"   3. [Third action] - WHY: [Brief explanation of importance]\n"
        f"   4. [Fourth action if needed] - WHY: [Brief explanation of importance]\n"
        f"   5. [Fifth action if needed] - WHY: [Brief explanation of importance]\n\n"
        
        f"After completing the English next actions, translate the full output to Vietnamese, maintaining the same structured format. "
        f"The reasoning should be thorough, but the final output should be practical and directly applicable."
    )

def process_analysis_result(full_analysis: str) -> Dict[str, str]:
    """
    Process the initial analysis result (parts 1-3).
    
    Args:
        full_analysis: The complete analysis string
        
    Returns:
        Dict[str, str]: Dictionary containing processed analysis parts
    """
    try:
        # Log the input for debugging
        logger.info(f"[DEBUG] process_analysis_result called with text length: {len(full_analysis)}")
        
        # Check if the analysis is too short to be meaningful
        if full_analysis is None or len(full_analysis.strip()) < 100:
            logger.warning(f"[DEBUG] Analysis text too short ({len(full_analysis) if full_analysis else 0} chars) for meaningful extraction")
            default_search_terms = [
                "health and lifestyle concerns",
                "culturally sensitive communication in Vietnam",
                "information gathering stage"
            ]
            return {
                "english": full_analysis or "",
                "vietnamese": "",
                "analysis_full": full_analysis or "",
                "search_terms": default_search_terms,
                "next_action_full": ""
            }
        
        # Try multiple possible delimiters for the Vietnamese section
        vietnamese_delimiters = [
            "VIETNAMESE ANALYSIS:",
            "VIETNAMESE TRANSLATION",
            "VIETNAMESE ANALYSIS",
            "VIETNAMESE NEXT ACTIONS:",
            "PHÂN TÍCH TIẾNG VIỆT:",
            "BẢN DỊCH TIẾNG VIỆT",
            "TIẾNG VIỆT:"
        ]
        
        # Split the analysis into English and Vietnamese parts
        english_analysis = full_analysis
        vietnamese_analysis = ""
        delimiter_found = False
        
        for delimiter in vietnamese_delimiters:
            if delimiter in full_analysis:
                lang_sections = full_analysis.split(delimiter, 1)
                english_analysis = lang_sections[0].strip()
                vietnamese_analysis = delimiter + lang_sections[1].strip()
                logger.info(f"[DEBUG] Successfully split analysis into English and Vietnamese parts using delimiter: {delimiter}")
                delimiter_found = True
                break
                
        if not delimiter_found:
            # Fallback if split fails
            english_analysis = full_analysis
            vietnamese_analysis = full_analysis
            logger.info(f"[DEBUG] Failed to split analysis into language sections, using full text as English")
        
        # Extract key information for knowledge queries
        logger.info(f"[DEBUG] Calling extract_search_terms to extract search terms from analysis")
        search_terms = extract_search_terms(vietnamese_analysis)
        
        # Log the result
        logger.info(f"[DEBUG] extract_search_terms returned {len(search_terms)} terms: {search_terms}")
        
        return {
            "english": english_analysis,
            "vietnamese": vietnamese_analysis,
            "analysis_full": full_analysis,
            "search_terms": search_terms,
            # Add an empty next_action_full to ensure this key always exists
            "next_action_full": ""
        }
    except Exception as e:
        logger.error(f"Error processing analysis result: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "english": full_analysis or "",
            "vietnamese": "",
            "analysis_full": full_analysis or "",
            "search_terms": [
                "health and lifestyle concerns",
                "culturally sensitive communication in Vietnam",
                "information gathering stage"
            ],
            # Add an empty next_action_full to ensure this key always exists even on error
            "next_action_full": ""
        }

def process_next_actions_result(next_actions_content: str) -> dict:
    """
    Process the next actions result to extract English and Vietnamese parts.
    Also checks for unattributed questions (questions not directly from knowledge).
    
    Args:
        next_actions_content: The next actions content string
        
    Returns:
        dict: Dictionary containing processed next actions parts and any warnings
    """
    try:
        # Define multiple possible patterns for English and Vietnamese sections
        english_section_patterns = [
            r"ENGLISH NEXT ACTIONS:(.*?)(?=VIETNAMESE NEXT ACTIONS:|VIETNAMESE TRANSLATION:|BẢN DỊCH TIẾNG VIỆT|TIẾNG VIỆT:|$)",
            r"ENGLISH CHAIN OF THOUGHT:(.*?)(?=ENGLISH NEXT ACTIONS:|$)",
            r"NEXT ACTIONS:(.*?)(?=VIETNAMESE NEXT ACTIONS:|VIETNAMESE TRANSLATION:|BẢN DỊCH TIẾNG VIỆT|TIẾNG VIỆT:|$)"
        ]
        
        vietnamese_section_patterns = [
            r"VIETNAMESE NEXT ACTIONS:(.*?)(?=$)",
            r"VIETNAMESE TRANSLATION:(.*?)(?=$)",
            r"BẢN DỊCH TIẾNG VIỆT:(.*?)(?=$)",
            r"(?:TIẾNG\s*VIỆT\s*:)(.*?)(?=\Z)"
        ]
        
        # Extract English next actions
        english_next_actions = ""
        for pattern in english_section_patterns:
            english_match = re.search(pattern, next_actions_content, re.DOTALL)
            if english_match:
                english_next_actions = english_match.group(1).strip()
                logger.info(f"[DEBUG] Found English next actions section with pattern: {pattern[:30]}...")
                break
        
        # Extract Vietnamese next actions
        vietnamese_next_actions = ""
        for pattern in vietnamese_section_patterns:
            vietnamese_match = re.search(pattern, next_actions_content, re.DOTALL)
            if vietnamese_match:
                vietnamese_next_actions = vietnamese_match.group(1).strip()
                logger.info(f"[DEBUG] Found Vietnamese next actions section with pattern: {pattern[:30]}...")
                break
        
        # If no sections were found, just use the entire content as English
        if not english_next_actions and not vietnamese_next_actions:
            logger.warning(f"[DEBUG] Could not find English or Vietnamese sections in next actions content")
            english_next_actions = next_actions_content
        
        # Check for unattributed questions (questions not in quotes or without citation)
        question_pattern = r'\?'
        quoted_question_pattern = r'"([^"]*\?)"'
        citation_pattern = r'from|in|according to|based on|cited in|as stated in|as mentioned in|as referenced in'
        
        # Find all questions (sentences ending with '?')
        questions = []
        unattributed_questions = []
        has_warning = False
        warning_message = ""
        
        for line in english_next_actions.split('\n'):
            if '?' in line:
                # Check if the line contains a question
                questions_in_line = re.findall(r'[^.!;]*\?', line)
                for question in questions_in_line:
                    questions.append(question.strip())
                    
                    # Check if the question is in quotes and has a citation
                    is_quoted = bool(re.search(quoted_question_pattern, line))
                    has_citation = bool(re.search(citation_pattern, line, re.IGNORECASE))
                    
                    if not (is_quoted and has_citation):
                        unattributed_questions.append(question.strip())
        
        # Check for statements indicating no questions were found
        no_questions_pattern = r'no (specific|suitable) questions found|no questions (exist|are available)|couldn\'t find (specific|suitable|any) questions'
        has_no_questions_statement = bool(re.search(no_questions_pattern, english_next_actions, re.IGNORECASE))
        
        # Set warning if unattributed questions were found
        if unattributed_questions and not has_no_questions_statement:
            has_warning = True
            warning_message = "Found questions that may not be directly from knowledge content"
        
        return {
            "next_action_english": english_next_actions,
            "next_action_vietnamese": vietnamese_next_actions,
            "next_action_full": next_actions_content,
            "unattributed_questions": unattributed_questions,
            "warning": warning_message if has_warning else "",
            "has_warning": has_warning
        }
    except Exception as e:
        logger.error(f"Error processing next actions: {str(e)}")
        return {
            "next_action_english": "",
            "next_action_vietnamese": "",
            "next_action_full": next_actions_content
        }

import re
import unicodedata
def is_vietnamese_text(analysis_text):
    # Normalize text to handle Unicode diacritics
    normalized_text = unicodedata.normalize('NFC', analysis_text.lower())
    
    # Expanded list of common Vietnamese words
    vietnamese_keywords = [
        "của", "những", "và", "các", "là", "không", "có", "được", 
        "người", "trong", "để", "anh", "chị", "em", "tiếng việt",
        "cho", "này", "tôi", "bạn", "với", "phải", "muốn", "cần",
        "khách hàng", "báo giá", "hỗ trợ", "kỹ thuật"
    ]
    
    # Check for Vietnamese keywords with word boundaries
    keyword_match = any(
        re.search(r'\b{}\b'.format(re.escape(word)), normalized_text)
        for word in vietnamese_keywords
    )
    
    # Check for Vietnamese-specific characters
    char_match = bool(
        re.search(r'[ạảẫắằẳâđêếềểẹẽíìỉĩọỏốồổơớờởụủứừử]', normalized_text)
    )
    
    return keyword_match or char_match

def extract_search_terms(analysis_text: str) -> List[str]:
    """
    Extract search terms from the analysis using LLM to generate relevant queries
    for understanding and processing the contact.
    
    Args:
        analysis_text: The analysis text (in English or Vietnamese)
        
    Returns:
        List[str]: List of search terms for knowledge queries
    """
    try:
        # Use a simple cache to avoid duplicate calls with the same content
        import hashlib
        cache_key = hashlib.md5(analysis_text.encode('utf-8')).hexdigest()
        
        # Check if we have a cached result for this exact content
        cached_terms = getattr(extract_search_terms, '_term_cache', {}).get(cache_key)
        if cached_terms:
            logger.info(f"Using cached search terms for analysis ({len(cached_terms)} terms)")
            return cached_terms
        
        # Log the analysis text length
        logger.info(f"Extracting search terms using LLM from analysis (length: {len(analysis_text)})")
        
        # Detect if Vietnamese content is present (simple check)
        is_vietnamese = is_vietnamese_text(analysis_text)
        
        # Create a focused prompt for the LLM to extract search terms
        llm_prompt = f"""
        The following is an analysis of a conversation with a contact. 

        ANALYSIS:
        {analysis_text}

        Generate 6-8 search queries to find knowledge on HOW TO understand and proceed with this contact effectively.

        Focus on queries that address the most relevant of the following areas, based on the analysis content:
        1. Understanding the contact's situation and needs
        2. Communicating effectively based on their style/context
        3. Addressing their expressed concerns or desires
        4. Gathering missing information appropriately
        5. Cultural considerations relevant to the contact
        6. Best approaches for the current conversation stage (e.g., initial outreach, trust-building, negotiation)
        7. Techniques for handling specific topics identified

        Each query should be:
        - Specific, actionable, and tied to the contact's unique situation
        - A concise phrase suitable for search engines (avoid full sentences)
        - In the SAME LANGUAGE as the analysis (e.g., Vietnamese for Vietnamese analysis, English for English analysis)
        - Prioritized based on the most prominent aspects of the analysis

        If the analysis is brief (<50 words), generate 3 queries. If detailed (>50 words), generate 6-8 queries. Only include queries for gathering missing information if the analysis notes specific unknowns.

        Format the response as a JSON array of strings.
        Example for English: ["building rapport with anxious clients", "effective questions for health concerns", "cultural communication patterns"]
        Example for Vietnamese: ["xây dựng mối quan hệ với khách hàng lo lắng", "câu hỏi hiệu quả về vấn đề sức khỏe", "mô hình giao tiếp văn hóa"]

        Return ONLY the JSON array, no additional text.
        """
        
        # Use the LLM to generate search terms
        response = inferLLM.invoke(llm_prompt,temperature=0.1,max_tokens=1000)
        content = response.content
        logger.info(f"LLM response for search term generation: {content}")
        # Extract JSON array from response
        import json
        import re
        
        json_pattern = r'\[.*\]'
        json_match = re.search(json_pattern, content, re.DOTALL)
        
        search_terms = []
        if json_match:
            try:
                search_terms = json.loads(json_match.group(0))
                logger.info(f"Successfully extracted {len(search_terms)} search terms from LLM")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from the LLM response match: {json_match.group(0)}")
                # Try to extract terms from the malformed JSON
                terms_pattern = r'"([^"]+)"'
                term_matches = re.findall(terms_pattern, json_match.group(0))
                if term_matches:
                    search_terms = term_matches
                    logger.info(f"Extracted {len(search_terms)} terms using regex fallback")
        
        # If no JSON array was found or parsed, try to extract from the full content
        if not search_terms:
            logger.warning("Attempting to extract search terms from full LLM response")
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                # Look for likely search terms (phrases in quotes or numbered/bulleted items)
                if (line.startswith('- "') or line.startswith('* "') or 
                    re.match(r'^\d+\.\s+"', line) or '"' in line):
                    term_match = re.search(r'"([^"]+)"', line)
                    if term_match:
                        search_terms.append(term_match.group(1))
        
        # Ensure we have at least a minimum set of terms
        if len(search_terms) < 3:
            logger.warning("Not enough search terms extracted. Adding fallback terms.")
            if is_vietnamese:
                fallback_terms = [
                    "cách hiểu ngữ cảnh của người liên hệ",
                    "kỹ thuật giao tiếp hiệu quả trong văn hóa Việt Nam",
                    "phương pháp đặt câu hỏi phù hợp",
                    "xây dựng mối quan hệ với người liên hệ mới"
                ]
            else:
                fallback_terms = [
                    "understanding contact context effectively",
                    "communication techniques for client needs",
                    "appropriate questioning methods",
                    "building relationship with new contacts"
                ]
            search_terms.extend(fallback_terms)
        
        # Clean and deduplicate terms
        cleaned_terms = []
        seen = set()
        for term in search_terms:
            if not isinstance(term, str):
                continue
                
            term = term.strip()
            if term and len(term) > 5:
                term_lower = term.lower()
                if term_lower not in seen:
                    seen.add(term_lower)
                    cleaned_terms.append(term)
        
        # Limit to reasonable number of terms
        #cleaned_terms = cleaned_terms[:5]
        
        # Store in cache
        if not hasattr(extract_search_terms, '_term_cache'):
            extract_search_terms._term_cache = {}
        extract_search_terms._term_cache[cache_key] = cleaned_terms
        
        logger.info(f"Final search terms ({len(cleaned_terms)}): {cleaned_terms}")
        return cleaned_terms
        
    except Exception as e:
        logger.error(f"Error extracting search terms: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Fallback terms in case of error
        logger.info("Returning fallback search terms due to error.")
        return [
            "understanding contact context",
            "effective communication techniques", 
            "relationship building strategies"
        ]

def extract_search_terms_from_next_actions(
    next_actions_text: str,
    min_words: int = 4,
    max_input_length: int = 10000
) -> List[str]:
    """
    Extract search terms from next actions text using LLM to generate
    specific search queries for finding knowledge needed to perform the planned actions.
    
    Args:
        next_actions_text: The next actions output text
        min_words: Minimum number of words for a term to be included (default: 4)
        max_input_length: Maximum allowed input length (default: 10000)
        
    Returns:
        List[str]: List of high-quality search terms for knowledge retrieval
    """
    # Use a simple cache to avoid duplicate calls with the same content
    import hashlib
    cache_key = hashlib.md5(next_actions_text.encode('utf-8')).hexdigest()
    
    # Check if we have a cached result for this exact content
    cached_terms = getattr(extract_search_terms_from_next_actions, '_term_cache', {}).get(cache_key)
    if cached_terms:
        logger.info(f"Using cached search terms for next actions ({len(cached_terms)} terms)")
        return cached_terms
    
    # Input validation
    if not isinstance(next_actions_text, str):
        logger.warning(f"Input must be a string, got {type(next_actions_text)}")
        return []
    if not next_actions_text.strip():
        logger.warning("Input is empty or whitespace")
        return []
    if len(next_actions_text) > max_input_length:
        logger.warning(f"Input exceeds max length ({max_input_length} chars), truncating")
        next_actions_text = next_actions_text[:max_input_length]
    if len(next_actions_text) < 10:
        logger.warning(f"Input too short (< 10 chars): {next_actions_text}")
        return []
    
    logger.info(f"Extracting search terms from next actions: {next_actions_text}")

    # Create a focused prompt for the LLM to extract search terms
    llm_prompt = f"""
    The following is a "next actions plan" for a conversation with a contact.
    
    NEXT ACTIONS PLAN:
    {next_actions_text}
    
    Generate 6-8 search queries that would help me find KNOWLEDGE needed to PERFORM these next actions effectively.
    
    Focus on queries that would help find resources about:
    
    1. HOW TO implement the specific techniques mentioned in the plan
    2. Methods for asking the recommended questions effectively
    3. How to create the communication approach specified in the plan
    4. Cultural considerations needed for implementation
    5. Specific procedures or frameworks mentioned in the techniques
    6. Ways to adapt the communication style to match the contact's preferences
    7. Strategies for handling potential responses from the contact
    8. Best practices for the specific conversation stage identified
    
    Each query should be highly specific and focused on IMPLEMENTATION knowledge.
    The queries should help find "how-to" guidance rather than conceptual information.
    
    IMPORTANT: Generate search terms in the SAME LANGUAGE as the provided next actions plan. If the plan contains Vietnamese, provide terms in Vietnamese. If it's in English, provide English terms.
    
    Format your response as a JSON array of strings, each being a search query.
    Example for English: ["how to implement active listening technique with anxious clients", "effective ways to ask about health concerns indirectly", "building rapport with Vietnamese customers"]
    Example for Vietnamese: ["cách áp dụng kỹ thuật lắng nghe chủ động với khách hàng lo lắng", "cách hỏi gián tiếp về vấn đề sức khỏe", "xây dựng mối quan hệ với khách hàng Việt Nam"]
    
    Return ONLY the JSON array, no additional text.
    """
    
    # Use the LLM to generate search terms
    try:
        response = inferLLM.invoke(llm_prompt, temperature=0.1, max_tokens=1000)
        content = response.content
        logger.info(f"LLM response for NEXT ACTION search term generation: {content}")
        
        # Extract JSON array from response
        import json
        import re
        
        json_pattern = r'\[.*\]'
        json_match = re.search(json_pattern, content, re.DOTALL)
        
        search_terms = []
        if json_match:
            try:
                search_terms = json.loads(json_match.group(0))
                logger.info(f"Successfully extracted {len(search_terms)} search terms from LLM for next actions")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from the LLM response match: {json_match.group(0)}")
                # Try to extract terms from the malformed JSON
                terms_pattern = r'"([^"]+)"'
                term_matches = re.findall(terms_pattern, json_match.group(0))
                if term_matches:
                    search_terms = term_matches
                    logger.info(f"Extracted {len(search_terms)} terms using regex fallback")
        
        # If no JSON array was found or parsed, try to extract from the full content
        if not search_terms:
            logger.warning("Attempting to extract search terms from full LLM response")
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                # Look for likely search terms (phrases in quotes or numbered/bulleted items)
                if (line.startswith('- "') or line.startswith('* "') or 
                    re.match(r'^\d+\.\s+"', line) or '"' in line):
                    term_match = re.search(r'"([^"]+)"', line)
                    if term_match:
                        search_terms.append(term_match.group(1))
        
        # Ensure we have at least a minimum set of terms
        if len(search_terms) < 2:
            logger.warning("Not enough search terms extracted for next actions. Adding fallback terms.")
            
            fallback_terms = [
                    "cách thực hiện kỹ thuật giao tiếp hiệu quả",
                    "phương pháp đặt câu hỏi trong cuộc trò chuyện",
                    "cách xử lý phản ứng của khách hàng",
                    "kỹ thuật xây dựng mối quan hệ trong văn hóa Việt Nam",
                    "phương pháp giao tiếp phù hợp với văn hóa Việt Nam",
                    "chiến lược thích ứng với phản hồi của khách hàng",
                    "kỹ thuật đặt câu hỏi hiệu quả cho giai đoạn trò chuyện"
                ]
            search_terms.extend(fallback_terms)
        
        # Clean and deduplicate terms
        cleaned_terms = []
        seen = set()
        for term in search_terms:
            if not isinstance(term, str):
                continue
                
            term = term.strip()
            if term and len(term) > 5 and len(term.split()) >= min_words:
                term_lower = term.lower()
                if term_lower not in seen:
                    seen.add(term_lower)
                    cleaned_terms.append(term)
        
        # Limit to reasonable number of terms (increased from 5 to 8)
        cleaned_terms = cleaned_terms[:8]
        
        # Store in cache
        if not hasattr(extract_search_terms_from_next_actions, '_term_cache'):
            extract_search_terms_from_next_actions._term_cache = {}
        extract_search_terms_from_next_actions._term_cache[cache_key] = cleaned_terms
        
        logger.info(f"Final next actions search terms ({len(cleaned_terms)}): {cleaned_terms}")
        return cleaned_terms
        
    except Exception as e:
        logger.error(f"Error extracting search terms from next actions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Fallback to pattern-based extraction
        logger.info("Falling back to pattern-based extraction for next actions")
        fallback_terms = _extract_pattern_based_terms(next_actions_text, min_words)
        
        # If still no terms, use very basic fallbacks
        if not fallback_terms or len(fallback_terms) < 3:
           
            fallback_terms = [
                    "kỹ thuật giao tiếp hiệu quả",
                    "phương pháp đặt câu hỏi",
                    "xây dựng mối quan hệ với khách hàng"
            ]
           
        # Limit number of terms
        fallback_terms = fallback_terms[:5]
        
        # Store in cache
        if not hasattr(extract_search_terms_from_next_actions, '_term_cache'):
            extract_search_terms_from_next_actions._term_cache = {}
        extract_search_terms_from_next_actions._term_cache[cache_key] = fallback_terms
        
        logger.info(f"Returning fallback search terms for next actions: {fallback_terms}")
        return fallback_terms

def _extract_pattern_based_terms(next_actions_text: str, min_words: int = 4, is_vietnamese: bool = False) -> List[str]:
    """
    Extract search terms using regex patterns as a fallback method.
    Supports both English and Vietnamese content.
    
    Args:
        next_actions_text: The next actions text
        min_words: Minimum number of words for a term to be included
        is_vietnamese: Whether the text is primarily in Vietnamese
        
    Returns:
        List[str]: Search terms extracted using patterns
    """
    search_terms = []
    
    try:
        # Adjust minimum word count threshold for Vietnamese
        if is_vietnamese:
            min_words = max(2, min_words - 1)  # Vietnamese often uses fewer words
        
        # Extract both language sections
        english_section = ""
        vietnamese_section = ""
        
        # Define multiple possible patterns for English and Vietnamese sections
        english_patterns = [
            r"(?:ENGLISH\s*(?:NEXT\s*)*ACTIONS\s*[:\n])(.*?)(?=(?:VIETNAMESE\s*(?:NEXT\s*)*ACTIONS|VIETNAMESE\s*TRANSLATION|BẢN\s*DỊCH\s*TIẾNG\s*VIỆT|\Z))",
            r"(?:ENGLISH\s*CHAIN\s*OF\s*THOUGHT\s*[:\n])(.*?)(?=(?:ENGLISH\s*NEXT\s*ACTIONS|VIETNAMESE\s*NEXT\s*ACTIONS|VIETNAMESE\s*TRANSLATION|BẢN\s*DỊCH\s*TIẾNG\s*VIỆT|\Z))",
            r"(?:NEXT\s*ACTIONS\s*[:\n])(.*?)(?=(?:VIETNAMESE\s*(?:NEXT\s*)*ACTIONS|VIETNAMESE\s*TRANSLATION|BẢN\s*DỊCH\s*TIẾNG\s*VIỆT|\Z))"
        ]
        
        vietnamese_patterns = [
            r"(?:VIETNAMESE\s*(?:NEXT\s*)*ACTIONS\s*[:\n])(.*?)(?=\Z)",
            r"(?:VIETNAMESE\s*TRANSLATION\s*[:\n])(.*?)(?=\Z)",
            r"(?:BẢN\s*DỊCH\s*TIẾNG\s*VIỆT\s*[:\n])(.*?)(?=\Z)",
            r"(?:TIẾNG\s*VIỆT\s*[:\n])(.*?)(?=\Z)"
        ]
        
        # Extract English section
        for pattern in english_patterns:
            english_match = re.search(pattern, next_actions_text, re.IGNORECASE | re.DOTALL)
            if english_match:
                english_section = english_match.group(1).strip()
                logger.info(f"Extracted English section (length: {len(english_section)}) using pattern: {pattern[:30]}...")
                break
        
        # Extract Vietnamese section
        for pattern in vietnamese_patterns:
            vietnamese_match = re.search(pattern, next_actions_text, re.IGNORECASE | re.DOTALL)
            if vietnamese_match:
                vietnamese_section = vietnamese_match.group(1).strip()
                logger.info(f"Extracted Vietnamese section (length: {len(vietnamese_section)}) using pattern: {pattern[:30]}...")
                break
        
        if not english_section and not vietnamese_section:
            logger.info("No language sections found, using full text")
            if is_vietnamese:
                vietnamese_section = next_actions_text
            else:
                english_section = next_actions_text

        # Define extraction patterns based on language
        if english_section:
            # Process English section with English patterns
            logger.info(f"Processing English section with pattern-based extraction")
            
            # Define extraction patterns (ordered by priority)
            extraction_patterns = [
                # Priority 1: Action sentences (with verbs like ask, confirm)
                {
                    "pattern": r'([^.!?\n]*(?:ask|inquire|request|provide|offer|explain|clarify|confirm|determine|find out|recommend|suggest|address|acknowledge)[^.!?\n]*[.!?])',
                    "type": "action_sentence",
                    "min_words": max(min_words, 4),
                },
                # Priority 2: Quoted questions
                {
                    "pattern": r'"([^"]*\?)"',
                    "type": "quoted_question",
                    "min_words": max(min_words, 3),
                },
                # Priority 3: General sentences
                {
                    "pattern": r'([^.!?\n]+[.!?])',
                    "type": "sentence",
                    "min_words": max(min_words, 5),
                },
                # Priority 4: Bullet points
                {
                    "pattern": r'[-•*]\s*(.*?)(?=\n|\Z)',
                    "type": "bullet_point",
                    "min_words": max(min_words, 3),
                },
            ]

            # Extract terms using patterns
            for extraction in extraction_patterns:
                matches = re.findall(extraction["pattern"], english_section, re.IGNORECASE)
                for match in matches:
                    term = match.strip()
                    if len(term.split()) >= extraction["min_words"]:
                        search_terms.append(term)
                        logger.debug(f"Added {extraction['type']} from English: {term}")

            # Priority 5: Specific information sections
            info_sections = [
                (r'framework(?:\s|:|is)+(.*?)(?=\n|\Z)', "framework"),
                (r'technique(?:\s|:|is)+(.*?)(?=\n|\Z)', "technique"),
                (r'specific questions(?:\s|:|are)+(.*?)(?=\n\n|\Z)', "specific_questions"),
            ]

            for pattern, section_type in info_sections:
                section_match = re.search(pattern, english_section, re.IGNORECASE | re.DOTALL)
                if section_match:
                    section_text = section_match.group(1).strip()
                    sentences = re.findall(r'([^.!?\n]+[.!?])', section_text)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence.split()) >= min_words and sentence not in search_terms:
                            search_terms.append(sentence)
                            logger.debug(f"Added {section_type} sentence from English: {sentence}")
        
        if vietnamese_section:
            # Process Vietnamese section with Vietnamese patterns
            logger.info(f"Processing Vietnamese section with pattern-based extraction")
            
            # Define Vietnamese extraction patterns
            vn_extraction_patterns = [
                # Priority 1: Action sentences with Vietnamese verbs
                {
                    "pattern": r'([^.!?\n]*(?:hỏi|yêu cầu|cung cấp|đề nghị|giải thích|làm rõ|xác nhận|xác định|tìm hiểu|đề xuất|gợi ý|giải quyết|ghi nhận)[^.!?\n]*[.!?])',
                    "type": "action_sentence",
                    "min_words": max(min_words, 3),  # Lower threshold for Vietnamese
                },
                # Priority 2: Quoted questions
                {
                    "pattern": r'"([^"]*\?)"',
                    "type": "quoted_question",
                    "min_words": max(min_words, 2),  # Lower threshold for Vietnamese
                },
                # Priority 3: General sentences
                {
                    "pattern": r'([^.!?\n]+[.!?])',
                    "type": "sentence",
                    "min_words": max(min_words, 4),  # Lower threshold for Vietnamese
                },
                # Priority 4: Bullet points
                {
                    "pattern": r'[-•*]\s*(.*?)(?=\n|\Z)',
                    "type": "bullet_point",
                    "min_words": max(min_words, 2),  # Lower threshold for Vietnamese
                },
            ]

            # Extract terms using Vietnamese patterns
            for extraction in vn_extraction_patterns:
                matches = re.findall(extraction["pattern"], vietnamese_section, re.IGNORECASE)
                for match in matches:
                    term = match.strip()
                    if len(term.split()) >= extraction["min_words"]:
                        search_terms.append(term)
                        logger.debug(f"Added {extraction['type']} from Vietnamese: {term}")

            # Priority 5: Specific Vietnamese information sections
            vn_info_sections = [
                (r'khung phân loại(?:\s|:|là)+(.*?)(?=\n|\Z)', "framework"),
                (r'kỹ thuật(?:\s|:|là)+(.*?)(?=\n|\Z)', "technique"),
                (r'câu hỏi cụ thể(?:\s|:|là)+(.*?)(?=\n\n|\Z)', "specific_questions"),
            ]

            for pattern, section_type in vn_info_sections:
                section_match = re.search(pattern, vietnamese_section, re.IGNORECASE | re.DOTALL)
                if section_match:
                    section_text = section_match.group(1).strip()
                    sentences = re.findall(r'([^.!?\n]+[.!?])', section_text)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence.split()) >= min_words and sentence not in search_terms:
                            search_terms.append(sentence)
                            logger.debug(f"Added {section_type} sentence from Vietnamese: {sentence}")

        # Add "how to" prefix to some terms to enhance retrieval of implementation knowledge
        enhanced_terms = []
        for term in search_terms:
            enhanced_terms.append(term)
            
            # Check if term is in English or Vietnamese and add appropriate prefix
            if any(word in term.lower() for word in ["implement", "use", "apply", "create", "develop", "establish", "manage", "handle", "communicate", "respond"]):
                if not term.lower().startswith("how to"):
                    enhanced_terms.append(f"how to {term}")
            elif any(word in term.lower() for word in ["triển khai", "sử dụng", "áp dụng", "tạo", "phát triển", "thiết lập", "quản lý", "xử lý", "giao tiếp", "phản hồi"]):
                if not term.lower().startswith("cách") and not term.lower().startswith("làm thế nào"):
                    enhanced_terms.append(f"cách {term}")

        # Deduplicate terms (case-insensitive, preserve original)
        unique_terms = []
        seen = set()
        for term in enhanced_terms:
            normalized = term.lower().strip()
            if normalized not in seen and len(normalized) > 0:
                seen.add(normalized)
                unique_terms.append(term)

        logger.info(f"Extracted {len(unique_terms)} unique search terms using pattern-based approach")
        return unique_terms

    except Exception as e:
        logger.error(f"Error extracting pattern-based terms: {str(e)}", exc_info=True)
        return []