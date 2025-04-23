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
        
        # Process the analysis to extract search terms and structure content
        try:
            analysis_parts = process_analysis_result(analysis_buffer)
            processed_content = analysis_parts.get("analysis_full", analysis_buffer)
        except Exception as process_error:
            logger.error(f"Error processing analysis: {str(process_error)}")
            processed_content = analysis_buffer
        
        # Final complete event for analysis - ensure it's a string
        if not isinstance(processed_content, str):
            processed_content = str(processed_content)
            
        analysis_complete_event = {
            "type": "analysis", 
            "content": processed_content, 
            "complete": True
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
        async for chunk in StreamLLM.astream(prompt):
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
                    was_delivered = emit_next_action_event(thread_id_for_analysis, next_action_event)
                except Exception as e:
                    logger.error(f"Error in socketio_manager websocket delivery: {str(e)}")
                    was_delivered = False
            
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
    Build the context analysis prompt with only parts 1-3 (without next actions).
    
    Args:
        context: The conversation context
        process_instructions: The knowledge base instructions
        
    Returns:
        str: The analysis prompt for parts 1-3
    """
    logger.info(f"[DEBUG] build_context_analysis_prompt called with context: {context}")
    logger.info(f"[DEBUG] build_context_analysis_prompt called with process_instructions: {process_instructions}")
    
    # Check if this appears to be a first message
    is_first_message = "User:" in context and context.count("User:") <= 1 and context.count("AI:") <= 0
    first_message_instruction = """
IMPORTANT: This appears to be the FIRST MESSAGE from this contact. Apply first-message analysis techniques from the knowledge base instructions to make initial assessments with limited information. Look for subtle signals in word choice, phrasing, and the nature of their inquiry. When confidence is low, clearly indicate assumptions versus observations.
""" if is_first_message else ""
    
    return (
        f"Based on the conversation:\n{context}\n\n"
        f"KNOWLEDGE BASE INSTRUCTIONS:\n{process_instructions}\n\n"
        f"{first_message_instruction}"
        f"Analyze this conversation to determine contact profile and situational context. Provide your analysis in BOTH English and Vietnamese.\n\n"
        
        f"ENGLISH ANALYSIS:\n"
        f"1. AUDIENCE PORTRAIT:\n"
        f"   - EMOTIONAL STATE: Identify current emotions, mood, sentiment, and emotional triggers evident in their communication\n"
        f"   - NEEDS & GOALS: Determine objectives, immediate needs, longer-term goals, and urgency indicators\n"
        f"   - PAIN POINTS: Identify challenges, frustrations, concerns, or obstacles they're facing\n"
        f"   - BEHAVIORAL PATTERNS: Note patterns in decision-making, information processing, or response style\n"
        f"   - CLASSIFICATION: Apply specific classification frameworks from the knowledge base to categorize this person\n"
        f"   - ENGAGEMENT LEVEL: Assess their level of engagement, openness, and receptivity to information\n"
        f"   - CONFIDENCE LEVEL: Indicate your confidence level (0-100%) in your portrait assessment\n\n"
        
        f"2. SITUATIONAL ANALYSIS:\n"
        f"   - CONVERSATION STAGE: Identify precisely where this conversation is in its lifecycle (first contact, information gathering, solution presentation, objection handling, etc.)\n"
        f"   - CURRENT TOPIC/FOCUS: What specific subject matter or concern is currently being discussed?\n"
        f"   - INTENT SIGNALS: What explicit or implicit indicators of purpose, intent, or desired outcome has the contact provided?\n"
        f"   - KNOWLEDGE GAPS: Identify key information the person is missing or has misconceptions about\n"
        f"   - READINESS ASSESSMENT: Evaluate the contact's readiness to move forward in the conversation or decision process\n"
        f"   - RELATIONSHIP DYNAMICS: Assess the current rapport level and relationship quality between both parties\n"
        f"   - CONVERSATIONAL MOMENTUM: Is the conversation progressing, stalled, or regressing? What factors are driving this?\n\n"
        
        f"3. ACTION PREPARATION:\n"
        f"   - INFORMATION REQUIREMENTS: What specific information is required based on the knowledge base instructions?\n"
        f"   - INFORMATION STATUS: Which requirements have been met and which are still missing?\n"
        f"   - INFORMATION PRIORITIES: What is the priority order for gathering missing information?\n"
        f"   - APPROPRIATE FRAMEWORKS: Which classification frameworks or analytical models from the knowledge base are most relevant to this situation?\n"
        f"   - RECOMMENDED APPROACH: Based on the contact's classification and situation, what general approach is most likely to be effective?\n\n"
        
        f"VIETNAMESE ANALYSIS:\n"
        f"1. PHÂN TÍCH CHÂN DUNG ĐỐI TÁC:\n"
        f"   - TRẠNG THÁI CẢM XÚC: Xác định cảm xúc hiện tại, tâm trạng, tình cảm và yếu tố kích hoạt cảm xúc trong giao tiếp của họ\n"
        f"   - NHU CẦU & MỤC TIÊU: Xác định mục tiêu, nhu cầu ngay lập tức, mục tiêu dài hạn và chỉ báo tính khẩn cấp\n"
        f"   - ĐIỂM KHÓ KHĂN: Xác định thách thức, sự thất vọng, lo ngại hoặc trở ngại mà họ đang đối mặt\n"
        f"   - MÔ HÌNH HÀNH VI: Ghi nhận mô hình trong việc ra quyết định, xử lý thông tin hoặc phong cách phản hồi\n"
        f"   - PHÂN LOẠI: Áp dụng các khung phân loại cụ thể từ cơ sở kiến thức để phân loại người này\n"
        f"   - MỨC ĐỘ TƯƠNG TÁC: Đánh giá mức độ tham gia, cởi mở và khả năng tiếp nhận thông tin của họ\n"
        f"   - MỨC ĐỘ TIN CẬY: Chỉ ra mức độ tin cậy của bạn (0-100%) trong đánh giá chân dung\n\n"
        
        f"2. PHÂN TÍCH TÌNH HUỐNG:\n"
        f"   - GIAI ĐOẠN CUỘC TRÒ CHUYỆN: Xác định chính xác cuộc trò chuyện này đang ở giai đoạn nào (liên hệ đầu tiên, thu thập thông tin, trình bày giải pháp, xử lý phản đối, v.v.)\n"
        f"   - CHỦ ĐỀ/TRỌNG TÂM HIỆN TẠI: Vấn đề hoặc mối quan tâm cụ thể nào đang được thảo luận?\n"
        f"   - TÍN HIỆU VỀ Ý ĐỊNH: Người liên hệ đã cung cấp những chỉ báo rõ ràng hoặc ngầm định nào về mục đích, ý định hoặc kết quả mong muốn?\n"
        f"   - KHOẢNG TRỐNG KIẾN THỨC: Xác định thông tin quan trọng mà người đó đang thiếu hoặc có hiểu sai\n"
        f"   - ĐÁNH GIÁ MỨC ĐỘ SẴN SÀNG: Đánh giá mức độ sẵn sàng của người liên hệ để tiến tới trong cuộc trò chuyện hoặc quá trình ra quyết định\n"
        f"   - ĐỘNG LỰC MỐI QUAN HỆ: Đánh giá mức độ hòa hợp và chất lượng mối quan hệ hiện tại giữa hai bên\n"
        f"   - ĐÀ CUỘC TRÒ CHUYỆN: Cuộc trò chuyện đang tiến triển, bị đình trệ hay đang lùi lại? Những yếu tố nào đang thúc đẩy điều này?\n\n"
        
        f"3. CHUẨN BỊ HÀNH ĐỘNG:\n"
        f"   - YÊU CẦU THÔNG TIN: Những thông tin cụ thể nào cần có dựa trên hướng dẫn của cơ sở kiến thức?\n"
        f"   - TRẠNG THÁI THÔNG TIN: Những yêu cầu nào đã được đáp ứng và những yêu cầu nào vẫn còn thiếu?\n"
        f"   - ƯU TIÊN THÔNG TIN: Thứ tự ưu tiên để thu thập thông tin còn thiếu là gì?\n"
        f"   - KHUNG PHÂN TÍCH PHÙ HỢP: Những khung phân loại hoặc mô hình phân tích nào từ cơ sở kiến thức phù hợp nhất với tình huống này?\n"
        f"   - CÁCH TIẾP CẬN ĐƯỢC KHUYẾN NGHỊ: Dựa trên phân loại và tình huống của người liên hệ, cách tiếp cận chung nào có khả năng hiệu quả nhất?\n\n"
        
        f"Be objective and factual, but don't hesitate to apply relevant psychological analysis frameworks from the knowledge base. Reference all applicable frameworks and classifications by name when you use them."
    )

def build_next_actions_prompt(context: str, initial_analysis: str, knowledge_content: str) -> str:
    """
    Build the next actions prompt based on initial analysis and retrieved knowledge,
    using a Chain of Thought structure to determine optimal next steps.
    
    Args:
        context: The conversation context
        initial_analysis: The results from the initial analysis (parts 1-3)
        knowledge_content: The retrieved knowledge content
        
    Returns:
        str: The next actions prompt with Chain of Thought structure
    """
    return (
        f"Based on the following information:\n\n"
        f"CONVERSATION:\n{context}\n\n"
        f"INITIAL ANALYSIS:\n{initial_analysis}\n\n"
        f"RETRIEVED KNOWLEDGE:\n{knowledge_content}\n\n"
        
        f"Determine the next appropriate actions using this Chain of Thought process. Provide your analysis in BOTH English and Vietnamese.\n\n"
        
        f"ENGLISH NEXT ACTIONS THOUGHT:\n\n"
        
        f"1. AUDIENCE UNDERSTANDING:\n"
        f"   - Summarize the key aspects of the audience portrait (emotional state, needs/goals, pain points, classification)\n"
        f"   - Identify the most important classification framework from the analysis that should guide your approach\n"
        f"   - Note the engagement level and what it suggests about how receptive they'll be to different approaches\n"
        f"   - Consider how their emotional state should influence your tone and content\n\n"
        
        f"2. SITUATIONAL ASSESSMENT:\n"
        f"   - Identify the exact conversation stage and what typically happens at this stage\n"
        f"   - Recognize the knowledge gaps that need to be addressed\n"
        f"   - Consider the conversational momentum and what might maintain or improve it\n"
        f"   - Note the relationship dynamics and how they influence what approach will work best\n\n"
        
        f"3. KNOWLEDGE MAPPING:\n"
        f"   - Identify which specific knowledge items are most relevant to this audience's classification\n"
        f"   - Find knowledge that addresses their emotional state and needs\n"
        f"   - Locate specific guidance for the current conversation stage\n"
        f"   - Extract any scripts, questions, or frameworks from the knowledge that match this situation\n\n"
        
        f"4. APPROACH FORMULATION:\n"
        f"   - Based on the audience classification, determine the optimal communication approach\n"
        f"   - Match your strategy to their emotional state and engagement level\n"
        f"   - Decide what information needs to be gathered or provided next\n"
        f"   - Select specific questions or statements from the knowledge content that would be most effective\n"
        f"   - If multiple approaches exist in the knowledge, explain why you're selecting this particular one\n\n"
        
        f"5. IMPLEMENTATION PLAN:\n"
        f"   - Outline the specific next action(s) to take\n"
        f"   - If suggesting questions, ONLY include questions with EXACT TEXT from the knowledge content\n"
        f"   - If no suitable questions exist in the knowledge, state: 'No specific questions found in knowledge content for this scenario'\n"
        f"   - Specify how to adapt tone and approach based on audience classification and emotional state\n"
        f"   - Indicate what response patterns to anticipate based on the selected approach\n\n"
        
        f"VIETNAMESE CHAIN OF THOUGHT:\n\n"
        
        f"1. HIỂU VỀ ĐỐI TÁC:\n"
        f"   - Tóm tắt các khía cạnh chính của chân dung đối tác (trạng thái cảm xúc, nhu cầu/mục tiêu, điểm khó khăn, phân loại)\n"
        f"   - Xác định khung phân loại quan trọng nhất từ phân tích nên hướng dẫn cách tiếp cận của bạn\n"
        f"   - Lưu ý mức độ tương tác và những gợi ý về mức độ tiếp nhận các cách tiếp cận khác nhau\n"
        f"   - Xem xét trạng thái cảm xúc của họ nên ảnh hưởng như thế nào đến giọng điệu và nội dung của bạn\n\n"
        
        f"2. ĐÁNH GIÁ TÌNH HUỐNG:\n"
        f"   - Xác định chính xác giai đoạn cuộc trò chuyện và những gì thường xảy ra ở giai đoạn này\n"
        f"   - Nhận ra những khoảng trống kiến thức cần được giải quyết\n"
        f"   - Xem xét đà cuộc trò chuyện và những gì có thể duy trì hoặc cải thiện nó\n"
        f"   - Lưu ý động lực mối quan hệ và cách chúng ảnh hưởng đến cách tiếp cận nào sẽ hiệu quả nhất\n\n"
        
        f"3. XÁC ĐỊNH KIẾN THỨC PHÙ HỢP:\n"
        f"   - Xác định những kiến thức cụ thể nào phù hợp nhất với phân loại đối tác này\n"
        f"   - Tìm kiến thức giải quyết trạng thái cảm xúc và nhu cầu của họ\n"
        f"   - Xác định hướng dẫn cụ thể cho giai đoạn hội thoại hiện tại\n"
        f"   - Trích xuất mọi kịch bản, câu hỏi hoặc khung từ kiến thức phù hợp với tình huống này\n\n"
        
        f"4. XÂY DỰNG CÁCH TIẾP CẬN:\n"
        f"   - Dựa trên phân loại đối tác, xác định cách tiếp cận giao tiếp tối ưu\n"
        f"   - Phối hợp chiến lược của bạn với trạng thái cảm xúc và mức độ tương tác của họ\n"
        f"   - Quyết định thông tin nào cần được thu thập hoặc cung cấp tiếp theo\n"
        f"   - Chọn các câu hỏi hoặc phát biểu cụ thể từ nội dung kiến thức sẽ hiệu quả nhất\n"
        f"   - Nếu có nhiều cách tiếp cận tồn tại trong kiến thức, giải thích lý do bạn chọn cách tiếp cận cụ thể này\n\n"
        
        f"5. KẾ HOẠCH THỰC HIỆN:\n"
        f"   - Phác thảo (các) hành động cụ thể tiếp theo\n"
        f"   - Nếu đề xuất câu hỏi, CHỈ bao gồm các câu hỏi có VĂN BẢN CHÍNH XÁC từ nội dung kiến thức\n"
        f"   - Nếu không có câu hỏi phù hợp trong kiến thức, hãy nêu rõ: 'Không tìm thấy câu hỏi cụ thể trong nội dung kiến thức cho tình huống này'\n"
        f"   - Chỉ định cách điều chỉnh giọng điệu và cách tiếp cận dựa trên phân loại và trạng thái cảm xúc của đối tác\n"
        f"   - Chỉ ra các mẫu phản hồi nào dự đoán dựa trên cách tiếp cận đã chọn\n\n"
        
        f"ENGLISH NEXT ACTIONS:\n"
        f"Based on the Chain of Thought analysis, provide SPECIFIC AND CONCRETE next actions in this structured format:\n\n"
        
        f"1. PRIMARY OBJECTIVE: [State the single most important goal for the next response]\n\n"
        
        f"2. COMMUNICATION APPROACH:\n"
        f"   - TONE & STYLE: [Specify exact tone based on audience classification and emotional state]\n"
        f"   - KEY MESSAGES: [List 1-3 key points that must be communicated]\n\n"
        
        f"3. SPECIFIC QUESTIONS/STATEMENTS:\n"
        f"   [Include ONLY exact questions/statements from the knowledge content, with citations]\n"
        f"   If no suitable questions exist in the knowledge, state: 'No specific questions found'\n\n"
        
        f"4. KNOWLEDGE APPLICATION:\n"
        f"   - FRAMEWORK: [Name the specific classification framework being applied]\n"
        f"   - TECHNIQUE: [Identify the specific technique from knowledge to use]\n\n"
        
        f"5. CONTINGENCY PLANNING:\n"
        f"   - IF POSITIVE RESPONSE: [What to do next if they respond positively]\n"
        f"   - IF NEGATIVE RESPONSE: [How to adjust if they respond negatively]\n\n"
        
        f"VIETNAMESE NEXT ACTIONS:\n"
        f"Dựa trên phân tích Chuỗi Suy Luận, cung cấp các hành động tiếp theo CỤ THỂ VÀ RÕ RÀNG theo định dạng có cấu trúc này:\n\n"
        
        f"1. MỤC TIÊU CHÍNH: [Nêu mục tiêu quan trọng nhất cho phản hồi tiếp theo]\n\n"
        
        f"2. CÁCH TIẾP CẬN GIAO TIẾP:\n"
        f"   - GIỌNG ĐIỆU & PHONG CÁCH: [Xác định giọng điệu chính xác dựa trên phân loại đối tác và trạng thái cảm xúc]\n"
        f"   - THÔNG ĐIỆP CHÍNH: [Liệt kê 1-3 điểm chính phải được truyền đạt]\n\n"
        
        f"3. CÂU HỎI/PHÁT BIỂU CỤ THỂ:\n"
        f"   [Chỉ bao gồm các câu hỏi/phát biểu chính xác từ nội dung kiến thức, kèm trích dẫn]\n"
        f"   Nếu không có câu hỏi phù hợp trong kiến thức, hãy nêu rõ: 'Không tìm thấy câu hỏi cụ thể'\n\n"
        
        f"4. ÁP DỤNG KIẾN THỨC:\n"
        f"   - KHUNG PHÂN LOẠI: [Đặt tên khung phân loại cụ thể đang được áp dụng]\n"
        f"   - KỸ THUẬT: [Xác định kỹ thuật cụ thể từ kiến thức để sử dụng]\n\n"
        
        f"5. KẾ HOẠCH DỰ PHÒNG:\n"
        f"   - NẾU PHẢN HỒI TÍCH CỰC: [Làm gì tiếp theo nếu họ phản hồi tích cực]\n"
        f"   - NẾU PHẢN HỒI TIÊU CỰC: [Cách điều chỉnh nếu họ phản hồi tiêu cực]\n\n"
        
        f"IMPORTANT: This next action plan must be SPECIFIC, ACTIONABLE, and DIRECTLY BASED on the audience's classification and conversation stage. Never provide generic guidance - everything must connect to the audience's specific profile and needs."
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
        Extract the most relevant search terms from the following analysis text. Return search terms in content's language.
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
        logger.info(f"[DEBUG] LLM response for search term extraction: {content}")
        
        # Try to extract JSON array from the response
        import json
        import re
        
        # Look for JSON-like array in the response
        json_pattern = r'\[.*\]'
        json_match = re.search(json_pattern, content, re.DOTALL)
        
        if json_match:
            try:
                search_terms = json.loads(json_match.group(0))
                logger.info(f"[DEBUG] Successfully extracted {search_terms} search terms from LLM response")
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
    Enhanced to work with the Chain of Thought structure and leverage LLM for better search terms.
    Prioritizes action-specific knowledge needs identified in the structured next actions format.
    Preserves input language (English or Vietnamese) when extracting search terms.

    Args:
        next_actions_text: The input text containing next actions (str).
        min_words: Minimum number of words for a term to be included (default: 4).
        max_input_length: Maximum allowed input length to prevent performance issues (default: 10000).

    Returns:
        List[str]: Deduplicated list of search terms aligned with implementation needs.
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

    # Detect language in the input text
    is_vietnamese = False
    vietnamese_markers = ["của", "những", "và", "các", "là", "không", "có", "được", "người", "trong", "để", "anh", "chị", "em", "tiếng Việt", "VIETNAMESE"]
    if any(marker in next_actions_text for marker in vietnamese_markers):
        is_vietnamese = True
        logger.info("Detected Vietnamese content in next actions")

    # First, try to extract using regex patterns from the structured format
    structured_terms = _extract_structured_next_action_terms(next_actions_text)
    
    # If we found structured terms, use them directly
    if structured_terms and len(structured_terms) >= 2:
        logger.info(f"Using {len(structured_terms)} structured search terms from next actions")
        return structured_terms
    
    # If structured extraction didn't yield enough results, try the LLM approach
    try:
        logger.info("Using LLM to extract search terms from next actions")
        llm_terms = _extract_search_terms_with_llm(next_actions_text, is_vietnamese)
        
        # If we got good results from LLM, return those
        if llm_terms and len(llm_terms) >= 2:
            logger.info(f"Using {len(llm_terms)} LLM-generated search terms from next actions")
            return llm_terms
    except Exception as e:
        logger.error(f"Error using LLM for search term extraction: {str(e)}")
        # Continue to fallback pattern extraction
    
    # Fallback to pattern-based extraction if other methods failed
    logger.info("Falling back to pattern-based extraction for next actions search terms")
    return _extract_pattern_based_terms(next_actions_text, min_words, is_vietnamese)

def _extract_structured_next_action_terms(next_actions_text: str) -> List[str]:
    """
    Extract search terms from the structured next actions format based on our Chain of Thought approach.
    Focuses on Primary Objective, Communication Approach, Knowledge Application, and Specific Questions sections.
    Works with both English and Vietnamese sections.
    
    Args:
        next_actions_text: The next actions text in structured format
        
    Returns:
        List[str]: Extracted search terms from structured sections
    """
    search_terms = []
    
    try:
        # Extract English and Vietnamese sections
        english_section = ""
        vietnamese_section = ""
        
        # Extract English section
        english_pattern = r"(?:ENGLISH\s*(?:NEXT\s*)*ACTIONS\s*[:\n])(.*?)(?=(?:VIETNAMESE\s*(?:NEXT\s*)*ACTIONS|\Z))"
        english_match = re.search(english_pattern, next_actions_text, re.IGNORECASE | re.DOTALL)
        if english_match:
            english_section = english_match.group(1).strip()
            logger.info(f"Extracted English section (length: {len(english_section)})")
        
        # Extract Vietnamese section
        vietnamese_pattern = r"(?:VIETNAMESE\s*(?:NEXT\s*)*ACTIONS\s*[:\n])(.*?)(?=\Z)"
        vietnamese_match = re.search(vietnamese_pattern, next_actions_text, re.IGNORECASE | re.DOTALL)
        if vietnamese_match:
            vietnamese_section = vietnamese_match.group(1).strip()
            logger.info(f"Extracted Vietnamese section (length: {len(vietnamese_section)})")
        
        if not english_section and not vietnamese_section:
            logger.info("No language sections found, using full text")
            english_section = next_actions_text
        
        # Process both language sections
        for lang, section in [("English", english_section), ("Vietnamese", vietnamese_section)]:
            if not section:
                continue
                
            logger.info(f"Processing {lang} section for structured terms")
            
            # Define section patterns based on language
            if lang == "English":
                sections = {
                    "PRIMARY OBJECTIVE": r"(?:1\.\s*PRIMARY\s+OBJECTIVE\s*:)([^#\d]+?)(?=\d\.|\Z)",
                    "COMMUNICATION APPROACH": r"(?:2\.\s*COMMUNICATION\s+APPROACH\s*:)([^#\d]+?)(?=\d\.|\Z)",
                    "SPECIFIC QUESTIONS": r"(?:3\.\s*SPECIFIC\s+QUESTIONS/STATEMENTS\s*:)([^#\d]+?)(?=\d\.|\Z)",
                    "KNOWLEDGE APPLICATION": r"(?:4\.\s*KNOWLEDGE\s+APPLICATION\s*:)([^#\d]+?)(?=\d\.|\Z)",
                }
            else:  # Vietnamese
                sections = {
                    "MỤC TIÊU CHÍNH": r"(?:1\.\s*MỤC\s+TIÊU\s+CHÍNH\s*:)([^#\d]+?)(?=\d\.|\Z)",
                    "CÁCH TIẾP CẬN GIAO TIẾP": r"(?:2\.\s*CÁCH\s+TIẾP\s+CẬN\s+GIAO\s+TIẾP\s*:)([^#\d]+?)(?=\d\.|\Z)",
                    "CÂU HỎI CỤ THỂ": r"(?:3\.\s*CÂU\s+HỎI/PHÁT\s+BIỂU\s+CỤ\s+THỂ\s*:)([^#\d]+?)(?=\d\.|\Z)",
                    "ÁP DỤNG KIẾN THỨC": r"(?:4\.\s*ÁP\s+DỤNG\s+KIẾN\s+THỨC\s*:)([^#\d]+?)(?=\d\.|\Z)",
                }
            
            section_contents = {}
            for section_name, pattern in sections.items():
                match = re.search(pattern, section, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    section_contents[section_name] = content
                    logger.info(f"Found {section_name} section: {content[:50]}...")
            
            # Extract terms from each section
            if lang == "English":
                # Extract from PRIMARY OBJECTIVE
                if "PRIMARY OBJECTIVE" in section_contents:
                    objective = section_contents["PRIMARY OBJECTIVE"]
                    objective = re.sub(r'\[.*?\]', '', objective).strip()
                    if len(objective.split()) >= 4:
                        search_terms.append(f"how to {objective}")
                        search_terms.append(objective)
                
                # Extract from KNOWLEDGE APPLICATION
                if "KNOWLEDGE APPLICATION" in section_contents:
                    knowledge_app = section_contents["KNOWLEDGE APPLICATION"]
                    
                    # Extract framework
                    framework_match = re.search(r'(?:FRAMEWORK\s*:)([^-\n]+)', knowledge_app, re.IGNORECASE)
                    if framework_match:
                        framework = framework_match.group(1).strip()
                        framework = re.sub(r'\[.*?\]', '', framework).strip()
                        if framework and len(framework.split()) >= 2:
                            search_terms.append(framework)
            
                    # Extract technique
                    technique_match = re.search(r'(?:TECHNIQUE\s*:)([^-\n]+)', knowledge_app, re.IGNORECASE)
                    if technique_match:
                        technique = technique_match.group(1).strip()
                        technique = re.sub(r'\[.*?\]', '', technique).strip()
                        if technique and len(technique.split()) >= 2:
                            search_terms.append(f"how to use {technique}")
                            search_terms.append(technique)
                
                # Extract specific questions
                if "SPECIFIC QUESTIONS" in section_contents:
                    specific_q = section_contents["SPECIFIC QUESTIONS"]
                    
                    # Extract quoted questions
                    quoted_questions = re.findall(r'"([^"]*\?)"', specific_q)
                    for question in quoted_questions:
                        if len(question.split()) >= 4:
                            search_terms.append(question)
                
                # Extract from COMMUNICATION APPROACH
                if "COMMUNICATION APPROACH" in section_contents:
                    comm_approach = section_contents["COMMUNICATION APPROACH"]
                    
                    # Extract key messages
                    key_messages_match = re.search(r'(?:KEY\s+MESSAGES\s*:)([^-\n]+)', comm_approach, re.IGNORECASE | re.DOTALL)
                    if key_messages_match:
                        key_messages = key_messages_match.group(1).strip()
                        message_items = re.split(r'[-•*\d+\.\s]+', key_messages)
                        for message in message_items:
                            message = message.strip()
                            message = re.sub(r'\[.*?\]', '', message).strip()
                            if message and len(message.split()) >= 4:
                                search_terms.append(message)
            else:  # Vietnamese
                # Extract from MỤC TIÊU CHÍNH
                if "MỤC TIÊU CHÍNH" in section_contents:
                    objective = section_contents["MỤC TIÊU CHÍNH"]
                    objective = re.sub(r'\[.*?\]', '', objective).strip()
                    if len(objective.split()) >= 3:  # Vietnamese may need fewer words threshold
                        search_terms.append(f"cách {objective}")
                        search_terms.append(objective)
                
                # Extract from ÁP DỤNG KIẾN THỨC
                if "ÁP DỤNG KIẾN THỨC" in section_contents:
                    knowledge_app = section_contents["ÁP DỤNG KIẾN THỨC"]
                    
                    # Extract framework
                    framework_match = re.search(r'(?:KHUNG\s+PHÂN\s+LOẠI\s*:)([^-\n]+)', knowledge_app, re.IGNORECASE)
                    if framework_match:
                        framework = framework_match.group(1).strip()
                        framework = re.sub(r'\[.*?\]', '', framework).strip()
                        if framework and len(framework.split()) >= 2:
                            search_terms.append(framework)
                    
                    # Extract technique
                    technique_match = re.search(r'(?:KỸ\s+THUẬT\s*:)([^-\n]+)', knowledge_app, re.IGNORECASE)
                    if technique_match:
                        technique = technique_match.group(1).strip()
                        technique = re.sub(r'\[.*?\]', '', technique).strip()
                        if technique and len(technique.split()) >= 2:
                            search_terms.append(f"cách sử dụng {technique}")
                            search_terms.append(technique)
                
                # Extract specific questions
                if "CÂU HỎI CỤ THỂ" in section_contents:
                    specific_q = section_contents["CÂU HỎI CỤ THỂ"]
                    
                    # Extract quoted questions
                    quoted_questions = re.findall(r'"([^"]*\?)"', specific_q)
                    for question in quoted_questions:
                        if len(question.split()) >= 3:  # Vietnamese may need fewer words threshold
                            search_terms.append(question)
                
                # Extract from CÁCH TIẾP CẬN GIAO TIẾP
                if "CÁCH TIẾP CẬN GIAO TIẾP" in section_contents:
                    comm_approach = section_contents["CÁCH TIẾP CẬN GIAO TIẾP"]
                    
                    # Extract key messages (THÔNG ĐIỆP CHÍNH)
                    key_messages_match = re.search(r'(?:THÔNG\s+ĐIỆP\s+CHÍNH\s*:)([^-\n]+)', comm_approach, re.IGNORECASE | re.DOTALL)
                    if key_messages_match:
                        key_messages = key_messages_match.group(1).strip()
                        message_items = re.split(r'[-•*\d+\.\s]+', key_messages)
                        for message in message_items:
                            message = message.strip()
                            message = re.sub(r'\[.*?\]', '', message).strip()
                            if message and len(message.split()) >= 3:  # Vietnamese may need fewer words threshold
                                search_terms.append(message)
        
        # Deduplicate
        unique_terms = []
        seen = set()
        for term in search_terms:
            normalized = term.lower().strip()
            if normalized and normalized not in seen and len(normalized) > 0:
                seen.add(normalized)
                unique_terms.append(term)

        logger.info(f"Extracted {len(unique_terms)} unique search terms using pattern-based approach")
        return unique_terms

    except Exception as e:
        logger.error(f"Error extracting structured next action terms: {str(e)}", exc_info=True)
        return []

def _extract_search_terms_with_llm(next_actions_text: str, is_vietnamese: bool = False) -> List[str]:
    """
    Use LLM to extract search terms from next actions text.
    This provides more intelligent extraction based on semantic understanding.
    Supports both English and Vietnamese language.
    
    Args:
        next_actions_text: The next actions text
        is_vietnamese: Whether the text is primarily in Vietnamese
        
    Returns:
        List[str]: Search terms extracted by LLM
    """
    # Build prompt with language-specific instructions
    if is_vietnamese:
        extraction_prompt = f"""
        Trích xuất các từ khóa tìm kiếm tập trung vào "cách thực hiện" từ kế hoạch hành động tiếp theo này. Các từ khóa này sẽ được sử dụng để truy xuất kiến thức về CÁCH thực hiện các hành động đã lên kế hoạch.
        
        Tập trung vào trích xuất:
        1. Các kỹ thuật cụ thể được đề cập cần kiến thức triển khai
        2. Khung phân loại cần được áp dụng
        3. Cách tiếp cận giao tiếp yêu cầu kiến thức cụ thể
        4. Câu hỏi cụ thể nên được hỏi
        5. Động từ hành động kết hợp với các đối tượng của chúng (ví dụ: "quản lý phản đối", "xây dựng mối quan hệ")
        
        Đối với mỗi từ khóa được trích xuất, hãy tập trung vào khía cạnh "làm thế nào để làm điều đó", không chỉ xác định những gì cần làm.
        
        Định dạng phản hồi của bạn dưới dạng mảng JSON các chuỗi, với mỗi chuỗi là một từ khóa tìm kiếm tập trung vào kiến thức "cách thực hiện".
        Ví dụ: ["cách triển khai kỹ thuật đặt câu hỏi SPIN", "xây dựng mối quan hệ với các khách hàng phân tích", "kỹ thuật xử lý phản đối về giá"]
        
        Đây là các hành động tiếp theo:
        
        {next_actions_text}
        """
    else:
        extraction_prompt = f"""
        Extract "how-to" focused search terms from the following next actions plan. These search terms will be used to retrieve knowledge on HOW to implement the planned actions.
        
        Focus on extracting:
        1. Specific techniques mentioned that require implementation knowledge
        2. Frameworks that need to be applied
        3. Communication approaches that require specific knowledge
        4. Specific questions that should be asked
        5. Action verbs paired with their objects (e.g., "manage objections", "build rapport")
        
        For each extracted term, focus on the "how to do it" aspect, not just identifying what to do.
        
        Format your response as a JSON array of strings, with each string being a search term focused on "how to" knowledge.
        Example: ["how to implement SPIN questioning technique", "building rapport with analytical personalities", "techniques for managing price objections"]
        
        Here are the next actions:
        
        {next_actions_text}
        """
    
    response = StreamLLM.invoke(extraction_prompt)
    content = response.content
    
    # Try to extract JSON array from the response
    import json
    json_pattern = r'\[.*\]'
    json_match = re.search(json_pattern, content, re.DOTALL)
    
    if json_match:
        try:
            search_terms = json.loads(json_match.group(0))
            logger.info(f"Successfully extracted {len(search_terms)} search terms using LLM")
            
            # Ensure minimum quality criteria
            filtered_terms = [term for term in search_terms if isinstance(term, str) and len(term.split()) >= 2]
            return filtered_terms
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM match: {json_match.group(0)}")
            return []
    else:
        logger.warning("No JSON array found in LLM response")
        return []

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
        
        english_pattern = r"(?:ENGLISH\s*(?:NEXT\s*)*ACTIONS\s*[:\n])(.*?)(?=(?:VIETNAMESE\s*(?:NEXT\s*)*ACTIONS|\Z))"
        english_match = re.search(english_pattern, next_actions_text, re.IGNORECASE | re.DOTALL)
        if english_match:
            english_section = english_match.group(1).strip()
            logger.info(f"Extracted English section (length: {len(english_section)})")
        
        vietnamese_pattern = r"(?:VIETNAMESE\s*(?:NEXT\s*)*ACTIONS\s*[:\n])(.*?)(?=\Z)"
        vietnamese_match = re.search(vietnamese_pattern, next_actions_text, re.IGNORECASE | re.DOTALL)
        if vietnamese_match:
            vietnamese_section = vietnamese_match.group(1).strip()
            logger.info(f"Extracted Vietnamese section (length: {len(vietnamese_section)})")
        
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