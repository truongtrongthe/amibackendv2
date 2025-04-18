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
        logger.info(f"Streaming complete analysis, length: {len(analysis_buffer)}")
        
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
                from socketio_manager import emit_analysis_event
                was_delivered_analysis = emit_analysis_event(thread_id_for_analysis, analysis_complete_event)
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
        f"   - Extract all relevant information provided by the contact in the entire conversation\n"
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
    Extract search terms from the initial analysis to guide knowledge retrieval.
    
    Args:
        analysis_text: The initial analysis text
        
    Returns:
        List[str]: List of search terms for knowledge queries
    """
    search_terms = []
    try:
        # Log the length and first part of the analysis text for debugging
        logger.info(f"[DEBUG] extract_search_terms called with analysis_text length: {len(analysis_text)}")
        logger.info(f"[DEBUG] First 200 chars of analysis_text: {analysis_text[:200]}")
        
        # Extract topic and focus
        topic_match = re.search(r'current topic or focus\?[:\s]*(.*?)(?:\n|\Z)', analysis_text, re.IGNORECASE)
        if topic_match:
            topic = topic_match.group(1).strip()
            if topic and len(topic) > 3:
                search_terms.append(topic)
                logger.info(f"Added topic as search term: {topic}")
        else:
            logger.info("[DEBUG] No match found for 'current topic or focus'")
            # Try an alternative pattern
            alt_topic_match = re.search(r'topic.*?(?:is|:)[^\n\.,]*', analysis_text, re.IGNORECASE)
            if alt_topic_match:
                alt_topic = alt_topic_match.group(0).strip()
                if alt_topic and len(alt_topic) > 3:
                    search_terms.append(alt_topic)
                    logger.info(f"[DEBUG] Added alternative topic as search term: {alt_topic}")
        
        # Extract requirements assessment information
        requirements_match = re.search(r'specific information requirements[:\s]*(.*?)(?:\n\n|\Z)', analysis_text, re.IGNORECASE | re.DOTALL)
        if requirements_match:
            requirements_text = requirements_match.group(1).strip()
            # Add key requirement phrases
            req_sentences = re.split(r'[.!?]', requirements_text)
            for sentence in req_sentences:
                if sentence.strip() and len(sentence.strip().split()) > 3:
                    search_terms.append(sentence.strip())
                    logger.info(f"Added requirement as search term: {sentence.strip()}")
        else:
            logger.info("[DEBUG] No match found for 'specific information requirements'")
        
        # Extract missing requirements information
        missing_match = re.search(r'requirements have been met and which are still missing[:\s]*(.*?)(?:\n\n|\Z)', analysis_text, re.IGNORECASE | re.DOTALL)
        if missing_match:
            missing_text = missing_match.group(1).strip()
            # Add key missing requirement phrases
            missing_sentences = re.split(r'[.!?]', missing_text)
            for sentence in missing_sentences:
                if sentence.strip() and len(sentence.strip().split()) > 3:
                    search_terms.append(sentence.strip())
                    logger.info(f"Added missing requirement as search term: {sentence.strip()}")
        else:
            logger.info("[DEBUG] No match found for 'requirements have been met and which are still missing'")
        
        # Extract priority information
        priority_match = re.search(r'priority order for gathering missing information[:\s]*(.*?)(?:\n\n|\Z)', analysis_text, re.IGNORECASE | re.DOTALL)
        if priority_match:
            priority_text = priority_match.group(1).strip()
            # Add priority phrases
            priority_sentences = re.split(r'[.!?]', priority_text)
            for sentence in priority_sentences:
                if sentence.strip() and len(sentence.strip().split()) > 3:
                    search_terms.append(sentence.strip())
                    logger.info(f"Added priority as search term: {sentence.strip()}")
        else:
            logger.info("[DEBUG] No match found for 'priority order for gathering missing information'")
            
        # ADD FALLBACK FOR GREETINGS - Extract greeting-related terms if the analysis mentions greeting
        if any(word in analysis_text.lower() for word in ["greeting", "hello", "welcome", "introduction", "initial contact", "first message"]):
            for greeting_term in ["greeting", "welcome", "introduction", "hello"]:
                search_terms.append(greeting_term)
                logger.info(f"[DEBUG] Added greeting fallback term: {greeting_term}")
                
        # General fallback - Extract any sentence containing "stage is" to identify conversation stage
        stage_match = re.search(r'(?:stage|conversation).*?(?:is|in)[^\n\.]+', analysis_text, re.IGNORECASE)
        if stage_match and len(search_terms) < 2:
            stage_info = stage_match.group(0).strip()
            search_terms.append(stage_info)
            logger.info(f"[DEBUG] Added conversation stage as fallback term: {stage_info}")
                    
        # Deduplicate search terms
        unique_terms = []
        seen = set()
        for term in search_terms:
            normalized = term.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_terms.append(term)
        
        logger.info(f"[DEBUG] Final extracted search terms ({len(unique_terms)}): {unique_terms}")        
        return unique_terms
        
    except Exception as e:
        logger.error(f"Error extracting search terms: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def extract_search_terms_from_next_actions(next_actions_text: str) -> List[str]:
    """
    Extract search terms from the next actions to guide focused knowledge retrieval.
    
    Args:
        next_actions_text: The next actions text
        
    Returns:
        List[str]: List of search terms for focused knowledge queries
    """
    # Safety check for empty or invalid input
    if not next_actions_text or not isinstance(next_actions_text, str) or len(next_actions_text) < 10:
        logger.warning(f"Invalid input to extract_search_terms_from_next_actions: {next_actions_text}")
        return []
        
    search_terms = []
    try:
        # Extract from English next actions (focus on the specific questions or actions)
        english_section = ""
        if "ENGLISH NEXT ACTIONS:" in next_actions_text:
            parts = next_actions_text.split("ENGLISH NEXT ACTIONS:")
            if len(parts) > 1:
                second_part = parts[1]
                if "VIETNAMESE NEXT ACTIONS" in second_part:
                    english_section = second_part.split("VIETNAMESE NEXT ACTIONS")[0].strip()
                else:
                    english_section = second_part.strip()
        elif "ENGLISH NEXT ACTIONS" in next_actions_text:
            parts = next_actions_text.split("ENGLISH NEXT ACTIONS")
            if len(parts) > 1:
                second_part = parts[1]
                if "VIETNAMESE NEXT ACTIONS" in second_part:
                    english_section = second_part.split("VIETNAMESE NEXT ACTIONS")[0].strip()
                else:
                    english_section = second_part.strip()
        
        # If no structured sections found, use the whole text
        if not english_section:
            logger.warning(f"No English section found in next_actions_text, using full text")
            english_section = next_actions_text
        
        # Extract specific questions (often in quotes)
        question_pattern = r'["\'](.*?)[\"\']'
        questions = re.findall(question_pattern, english_section)
        for question in questions:
            if question.strip() and len(question.strip().split()) > 3:
                search_terms.append(question.strip())
                logger.info(f"Added next action question as search term: {question.strip()}")
        
        # Extract key sentences related to information gathering or response focus
        if "information gathering is needed" in english_section.lower():
            info_gathering_section = re.search(r'information gathering is needed[,\s]*(.*?)(?:\n|\Z)', english_section, re.IGNORECASE | re.DOTALL)
            if info_gathering_section:
                sentences = re.split(r'[.!?]', info_gathering_section.group(1))
                for sentence in sentences:
                    if sentence.strip() and len(sentence.strip().split()) > 3:
                        search_terms.append(sentence.strip())
                        logger.info(f"Added info gathering focus as search term: {sentence.strip()}")
        
        if "information is complete" in english_section.lower():
            response_focus_section = re.search(r'information is complete[,\s]*(.*?)(?:\n|\Z)', english_section, re.IGNORECASE | re.DOTALL)
            if response_focus_section:
                sentences = re.split(r'[.!?]', response_focus_section.group(1))
                for sentence in sentences:
                    if sentence.strip() and len(sentence.strip().split()) > 3:
                        search_terms.append(sentence.strip())
                        logger.info(f"Added response focus as search term: {sentence.strip()}")
        
        # Extract bullet points (often contain key actions)
        bullet_points = re.findall(r'[-•*]\s*(.*?)(?:\n|\Z)', english_section)
        for point in bullet_points:
            if point.strip() and len(point.strip().split()) > 3:
                search_terms.append(point.strip())
                logger.info(f"Added bullet point as search term: {point.strip()}")
        
        # Extract any sentences containing action words or instructions
        action_words = ["ask", "inquire", "request", "provide", "offer", "explain", "clarify", "confirm", "determine", "find out"]
        for line in english_section.split("\n"):
            if any(action_word in line.lower() for action_word in action_words):
                if line.strip() and len(line.strip().split()) > 3:
                    search_terms.append(line.strip())
                    logger.info(f"Added action sentence as search term: {line.strip()}")
        
        # As a fallback, if no search terms found yet, extract key sentences
        if not search_terms:
            # Extract any sentence with more than 5 words as a potential search term
            sentences = re.split(r'[.!?]', english_section)
            for sentence in sentences:
                if sentence.strip() and len(sentence.strip().split()) > 5:
                    search_terms.append(sentence.strip())
                    logger.info(f"Added fallback sentence as search term: {sentence.strip()}")
        
        # Deduplicate search terms
        unique_terms = []
        seen = set()
        for term in search_terms:
            normalized = term.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_terms.append(term)
                
        return unique_terms
        
    except Exception as e:
        logger.error(f"Error extracting search terms from next actions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [] 