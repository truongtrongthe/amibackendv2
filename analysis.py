from typing import Dict, AsyncGenerator, Optional, Tuple, List
from utilities import logger
from langchain_openai import ChatOpenAI
import re

StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

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
        async for chunk in StreamLLM.astream(prompt):
            chunk_content = chunk.content
            analysis_buffer += chunk_content
            
            # Create analysis event
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
            yield {"type": "analysis", "content": chunk_content, "complete": False}
        
        # Send a final complete message with the full analysis
        #logger.info(f"Streaming complete analysis, length: {len(analysis_buffer)}")
        
        # Process the analysis to extract search terms and structure content
        analysis_parts = process_analysis_result(analysis_buffer)
        
        # Final complete event for analysis
        analysis_complete_event = {
            "type": "analysis", 
            "content": analysis_parts.get("analysis_full", analysis_buffer), 
            "complete": True
        }
        
        # Send via WebSocket if configured
        if use_websocket and thread_id_for_analysis:
            try:
                logger.info(f"Emitting complete analysis via WebSocket to thread {thread_id_for_analysis}, length={len(analysis_buffer)}")
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
        async for chunk in StreamLLM.astream(prompt):
            chunk_content = chunk.content
            next_action_buffer += chunk_content
            
            # Create next_action event
            next_action_event = {
                "type": "next_action", 
                "content": chunk_content, 
                "complete": False
            }
            
            # If using WebSocket and thread ID is provided, emit to that room
            if use_websocket and thread_id_for_analysis:
                try:
                    from socketio_manager import emit_next_action_event
                    was_delivered = emit_next_action_event(thread_id_for_analysis, next_action_event)
                except Exception as e:
                    logger.error(f"Error in socketio_manager websocket delivery: {str(e)}")
                    was_delivered = False
            
            # Always yield for the standard flow too
            yield next_action_event
        
        # Send a final complete message with the full next actions
        logger.info(f"Streaming complete next actions, length: {len(next_action_buffer)}")
        
        # Process the next actions to structure content
        next_actions_data = process_next_actions_result(next_action_buffer)
        next_action_full = next_actions_data.get("next_action_full", next_action_buffer)
        
        # Final complete event for next_action
        next_action_complete_event = {
            "type": "next_action", 
            "content": next_action_full, 
            "complete": True
        }
        
        # Send via WebSocket if configured
        if use_websocket and thread_id_for_analysis:
            try:
                from socketio_manager import emit_next_action_event
                was_delivered = emit_next_action_event(thread_id_for_analysis, next_action_complete_event)
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
            "type": "next_action", 
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
    Build the context analysis prompt with only parts 1-3 (without next actions).
    
    Args:
        context: The conversation context
        process_instructions: The knowledge base instructions
        
    Returns:
        str: The analysis prompt for parts 1-3
    """
    return (
        f"Based on the conversation:\n{context}\n\n"
        f"KNOWLEDGE BASE INSTRUCTIONS:\n{process_instructions}\n\n"
        f"Analyze this conversation to determine context and information status. Provide your analysis in BOTH English and Vietnamese.\n\n"
        
        f"ENGLISH ANALYSIS:\n"
        f"1. CONTACT ANALYSIS:\n"
        f"   - Extract all relevant information provided by the contact in the entire conversation context\n"
        f"   - Identify any required information according to instructions that is missing\n"
        f"   - Assess completeness of required information (0-100%)\n\n"
        
        f"2. CONVERSATION CONTEXT:\n"
        f"   - What is the current topic or focus?\n"
        f"   - What stage is this conversation in? (initial contact, information gathering, etc.)\n"
        f"   - What signals about intent or needs has the contact provided?\n"
        f"   - Are there any strong reactions (agreement, disagreement, etc.)?\n\n"
        
        f"3. REQUIREMENTS ASSESSMENT:\n"
        f"   - What specific information requirements are in the knowledge base instructions?\n"
        f"   - Which requirements have been met and which are still missing?\n"
        f"   - What is the priority order for gathering missing information?\n\n"
        
        f"VIETNAMESE ANALYSIS:\n"
        f"1. PHÂN TÍCH THÔNG TIN LIÊN HỆ:\n"
        f"   - Trích xuất tất cả thông tin liên quan từ người dùng trong toàn bộ cuộc trò chuyện\n"
        f"   - Xác định thông tin bắt buộc còn thiếu theo hướng dẫn\n"
        f"   - Đánh giá mức độ hoàn thiện của thông tin bắt buộc (0-100%)\n\n"
        
        f"2. NGỮ CẢNH CUỘC TRÒ CHUYỆN:\n"
        f"   - Chủ đề hoặc trọng tâm hiện tại là gì?\n"
        f"   - Cuộc trò chuyện đang ở giai đoạn nào? (tiếp xúc ban đầu, thu thập thông tin, v.v.)\n"
        f"   - Người dùng đã cung cấp những tín hiệu nào về ý định hoặc nhu cầu?\n"
        f"   - Có phản ứng mạnh nào không (đồng ý, không đồng ý, v.v.)?\n\n"
        
        f"3. ĐÁNH GIÁ YÊU CẦU:\n"
        f"   - Yêu cầu thông tin cụ thể trong hướng dẫn cơ sở kiến thức là gì?\n"
        f"   - Yêu cầu nào đã được đáp ứng và yêu cầu nào còn thiếu?\n"
        f"   - Thứ tự ưu tiên thu thập thông tin còn thiếu là gì?\n\n"
        
        f"Be objective and factual. Only reference information explicitly present in either the conversation or knowledge base instructions."
    )

def build_next_actions_prompt(context: str, initial_analysis: str, knowledge_content: str) -> str:
    """
    Build the next actions prompt based on initial analysis and retrieved knowledge.
    
    Args:
        context: The conversation context
        initial_analysis: The results from the initial analysis (parts 1-3)
        knowledge_content: The retrieved knowledge content
        
    Returns:
        str: The next actions prompt
    """
    return (
        f"Based on the following information:\n\n"
        f"CONVERSATION:\n{context}\n\n"
        f"INITIAL ANALYSIS:\n{initial_analysis}\n\n"
        f"RETRIEVED KNOWLEDGE:\n{knowledge_content}\n\n"
        
        f"Determine the next appropriate actions in BOTH English and Vietnamese.\n\n"
        
        f"KNOWLEDGE SELECTION REASONING:\n"
        f"1. Reference the initial analysis which already identifies:\n"
        f"   - The current conversation stage (see 'What stage is this conversation in?')\n"
        f"   - User needs and signals of intent (see 'What signals about intent or needs has the contact provided?')\n"
        f"   - Missing information (see 'Which requirements have been met and which are still missing?')\n"
        f"2. For each knowledge item, evaluate:\n"
        f"   - How well it addresses the specific user needs identified in the initial analysis\n"
        f"   - Whether it contains usage instructions that match the identified conversation stage\n"
        f"   - If it helps gather the missing information prioritized in the initial analysis\n"
        f"3. Select knowledge that best addresses the highest priority needs/gaps while respecting the conversation stage\n"
        f"4. If relevant knowledge is limited, focus on the most stage-appropriate approach\n\n"
        
        f"KNOWLEDGE APPLICATION GUIDANCE:\n"
        f"- Review each knowledge piece and pay attention to any embedded usage instructions or conditions\n"
        f"- Follow any 'when to use' or staging guidance provided within the knowledge items themselves\n"
        f"- Connect knowledge directly to the needs and gaps identified in the initial analysis\n"
        f"- Apply knowledge only when its usage conditions align with the current conversation context\n"
        f"- CRITICAL: Only suggest specific questions that DIRECTLY APPEAR in the knowledge content\n"
        f"- If suggesting a question, you MUST mark it with quotes and indicate from which knowledge item it comes\n"
        f"- NEVER create generic questions (like 'How can I help you?' or 'What are you looking for?') when none exist in the knowledge\n"
        f"- If no suitable questions are found in the knowledge content, explicitly state this fact\n\n"
        
        f"ENGLISH NEXT ACTIONS:\n"
        f"- Briefly reference the key findings from the initial analysis (stage, needs, gaps)\n"
        f"- Explain which specific knowledge items you selected as most relevant and why\n"
        f"- Based on this reasoning, what is the most appropriate next action?\n"
        f"- If suggesting questions, ONLY include questions with the EXACT TEXT from the knowledge content, citing which knowledge item they come from\n"
        f"- If no suitable questions exist in the knowledge, state: 'No specific questions found in knowledge content for this scenario'\n"
        f"- How can you best address the priority information gaps identified in the initial analysis?\n\n"
        
        f"VIETNAMESE NEXT ACTIONS:\n"
        f"- Tóm tắt ngắn gọn các phát hiện chính từ phân tích ban đầu (giai đoạn, nhu cầu, thiếu sót)\n"
        f"- Giải thích những mục kiến thức cụ thể nào bạn đã chọn là phù hợp nhất và tại sao\n"
        f"- Dựa trên lập luận này, hành động tiếp theo phù hợp nhất là gì?\n"
        f"- Nếu đề xuất câu hỏi, CHỈ bao gồm các câu hỏi có VĂN BẢN CHÍNH XÁC từ nội dung kiến thức, trích dẫn câu hỏi đến từ mục kiến thức nào\n"
        f"- Nếu không có câu hỏi phù hợp trong kiến thức, hãy nêu rõ: 'Không tìm thấy câu hỏi cụ thể trong nội dung kiến thức cho tình huống này'\n" 
        f"- Làm thế nào để bạn giải quyết tốt nhất các thiếu sót thông tin ưu tiên được xác định trong phân tích ban đầu?\n\n"
        
        f"Be factual and precise. ONLY suggest questions that appear verbatim in the knowledge content. NEVER invent questions that don't exist in the knowledge, even if they seem appropriate."
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
        
        # Split the analysis into English and Vietnamese parts
        lang_sections = full_analysis.split("VIETNAMESE ANALYSIS:")
        if len(lang_sections) == 2:
            english_analysis = lang_sections[0].strip()
            vietnamese_analysis = "VIETNAMESE ANALYSIS:" + lang_sections[1].strip()
            logger.info(f"[DEBUG] Successfully split analysis into English and Vietnamese parts")
        else:
            # Fallback if split fails
            english_analysis = full_analysis
            vietnamese_analysis = ""
            logger.info(f"[DEBUG] Failed to split analysis into language sections, using full text as English")
        
        # Combined full analysis
        analysis_full = full_analysis
        
        # Extract key information for knowledge queries
        logger.info(f"[DEBUG] Calling extract_search_terms to extract search terms from analysis")
        search_terms = extract_search_terms(full_analysis)
        
        # Log the result
        logger.info(f"[DEBUG] extract_search_terms returned {len(search_terms)} terms: {search_terms}")
        
        return {
            "english": english_analysis,
            "vietnamese": vietnamese_analysis,
            "analysis_full": analysis_full,
            "search_terms": search_terms,
            # Add an empty next_action_full to ensure this key always exists
            "next_action_full": ""
        }
    except Exception as e:
        logger.error(f"Error processing analysis result: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "english": full_analysis,
            "vietnamese": "",
            "analysis_full": full_analysis,
            "search_terms": [],
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
        # Regular expressions to extract English and Vietnamese next actions
        english_pattern = r"ENGLISH NEXT ACTIONS:(.*?)(?=VIETNAMESE NEXT ACTIONS:|$)"
        vietnamese_pattern = r"VIETNAMESE NEXT ACTIONS:(.*?)(?=$)"
        
        # Extract English next actions
        english_match = re.search(english_pattern, next_actions_content, re.DOTALL)
        english_next_actions = english_match.group(1).strip() if english_match else ""
        
        # Extract Vietnamese next actions
        vietnamese_match = re.search(vietnamese_pattern, next_actions_content, re.DOTALL)
        vietnamese_next_actions = vietnamese_match.group(1).strip() if vietnamese_match else ""
        
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

def extract_search_terms(analysis_text: str) -> List[str]:
    """
    Extract search terms from the initial analysis to guide knowledge retrieval,
    using LLM instead of regex patterns for better semantic understanding.
    
    Args:
        analysis_text: The initial analysis text
        
    Returns:
        List[str]: List of search terms for knowledge queries
    """
    try:
        # Log the input for debugging
        logger.info(f"[DEBUG] extract_search_terms called with analysis_text length: {len(analysis_text)}")
        
        # Use existing LLM to extract search terms
        extraction_prompt = f"""
        Extract the most relevant search terms from the following analysis text. 
        Focus on extracting:
        
        1. The main topic or focus of the conversation
        2. Specific information requirements mentioned
        3. Missing requirements that need to be addressed
        4. Priority information that should be gathered
        5. The current conversation stage
        
        For each extracted term, ensure it is:
        - A complete phrase (at least 3 words where appropriate)
        - Directly relevant to knowledge retrieval
        - Non-redundant with other terms
        
        Format your response as a JSON array of strings, with each string being a search term.
        Example: ["customer inquiry about pricing", "product specifications", "delivery options"]
        
        Here's the analysis text:
        
        {analysis_text}
        """
        
        # Use the existing LLM model
        response = StreamLLM.invoke(extraction_prompt)
        
        # Check if the response is valid and extract the terms
        content = response.content
        logger.info(f"[DEBUG] LLM response for search term extraction: {content[:200]}...")
        
        # Try to extract JSON array from the response
        import json
        import re
        
        # Look for JSON-like array in the response
        json_pattern = r'\[.*\]'
        json_match = re.search(json_pattern, content, re.DOTALL)
        
        if json_match:
            try:
                search_terms = json.loads(json_match.group(0))
                logger.info(f"[DEBUG] Successfully extracted {len(search_terms)} search terms from LLM response")
            except json.JSONDecodeError:
                logger.warning(f"[DEBUG] Failed to parse JSON from the match: {json_match.group(0)}")
                search_terms = []
        else:
            # Fallback: split by lines and look for bullet points or numbered lists
            logger.warning("[DEBUG] No JSON array found in LLM response, trying fallback extraction")
            lines = content.split('\n')
            search_terms = []
            
            for line in lines:
                # Remove bullet points, numbers, etc.
                cleaned_line = re.sub(r'^[\s-•*\d.]+\s*', '', line).strip()
                if cleaned_line and len(cleaned_line.split()) >= 3:
                    search_terms.append(cleaned_line)
        
        # Add fallback for greetings
        if any(word in analysis_text.lower() for word in ["greeting", "hello", "welcome", "introduction", "initial contact", "first message"]):
            for greeting_term in ["greeting", "welcome", "introduction", "hello"]:
                if greeting_term not in search_terms:
                    search_terms.append(greeting_term)
                    logger.info(f"[DEBUG] Added greeting fallback term: {greeting_term}")
        
        # Ensure we have at least one term - add general stage fallback if needed
        if not search_terms:
            stage_match = re.search(r'(?:stage|conversation).*?(?:is|in)[^\n\.]+', analysis_text, re.IGNORECASE)
            if stage_match:
                stage_info = stage_match.group(0).strip()
                search_terms.append(stage_info)
                logger.info(f"[DEBUG] Added conversation stage as fallback term: {stage_info}")
            else:
                # Last-resort fallback
                search_terms = ["general inquiry", "customer interaction"]
                logger.info(f"[DEBUG] Using last-resort fallback terms: {search_terms}")
        
        # Deduplicate and clean terms
        unique_terms = []
        seen = set()
        for term in search_terms:
            if not isinstance(term, str):
                continue
            term = term.strip()
            normalized = term.lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_terms.append(term)
        
        logger.info(f"[DEBUG] Final extracted search terms ({len(unique_terms)}): {unique_terms}")
        return unique_terms
        
    except Exception as e:
        logger.error(f"Error extracting search terms: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def extract_search_terms_from_next_actions(
    next_actions_text: str,
    min_words: int = 4,
    max_input_length: int = 10000
) -> List[str]:
    """
    Extract search terms from next actions text to guide focused knowledge retrieval.
    Prioritizes context-rich terms (e.g., full sentences, quoted questions) for better query relevance.

    Args:
        next_actions_text: The input text containing next actions (str).
        min_words: Minimum number of words for a term to be included (default: 4).
        max_input_length: Maximum allowed input length to prevent performance issues (default: 10000).

    Returns:
        List[str]: Deduplicated list of search terms (e.g., sentences, questions, bullet points).

    Examples:
        >>> text = "ENGLISH NEXT ACTIONS:\\n- Confirm meeting.\\nAsk \\"What is the deadline?\\""
        >>> extract_search_terms_from_next_actions(text)
        ['Confirm meeting.', 'What is the deadline?']
    """
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

    search_terms = []
    try:
        # Extract English section (case-insensitive, flexible header)
        english_section = next_actions_text
        section_pattern = r"(?:ENGLISH\s*(?:NEXT\s*)*ACTIONS\s*[:\n])(.*?)(?=(?:VIETNAMESE\s*(?:NEXT\s*)*ACTIONS|\Z))"
        section_match = re.search(section_pattern, next_actions_text, re.IGNORECASE | re.DOTALL)
        if section_match:
            english_section = section_match.group(1).strip()
            logger.info(f"Extracted English section (length: {len(english_section)})")
        else:
            logger.info("No English section found, using full text")

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
                    logger.debug(f"Added {extraction['type']}: {term}")

        # Priority 5: Specific information sections
        info_sections = [
            (r'information gathering is needed[,\s]*(.*?)(?=\n\n|\Z)', "info_gathering"),
            (r'information is complete[,\s]*(.*?)(?=\n\n|\Z)', "response_focus"),
            (r'specific questions(?:\s|:|are)+(.*?)(?=\n\n|\Z)', "specific_questions"),
            (r'knowledge items selected(?:\s|:|as)+(.*?)(?=\n\n|\Z)', "selected_knowledge"),
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
                        logger.debug(f"Added {section_type} sentence: {sentence}")

        # Fallback: Extract long phrases if no terms found
        if not search_terms:
            logger.warning("No structured terms found, using fallback")
            sentences = re.split(r'[.!?]', english_section)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) >= max(min_words, 5):
                    search_terms.append(sentence)
                    logger.debug(f"Added fallback sentence: {sentence}")

        # Deduplicate terms (case-insensitive, preserve original)
        unique_terms = []
        seen = set()
        for term in search_terms:
            normalized = term.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_terms.append(term)

        logger.info(f"Extracted {len(unique_terms)} unique search terms")
        return unique_terms

    except Exception as e:
        logger.error(f"Error extracting search terms: {str(e)}", exc_info=True)
        return []