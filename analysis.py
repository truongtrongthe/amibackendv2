from typing import Dict, AsyncGenerator, Optional
from utilities import logger
from langchain_openai import ChatOpenAI

StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

async def stream_analysis(prompt: str, thread_id_for_analysis: Optional[str] = None, use_websocket: bool = False) -> AsyncGenerator[Dict, None]:
    """
    Stream the context analysis from the LLM
    
    Args:
        prompt: The prompt to analyze
        thread_id_for_analysis: Thread ID to use for WebSocket analysis events
        use_websocket: Whether to use WebSocket for streaming
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
                    if was_delivered:
                        #logger.info(f"Sent analysis chunk via socketio_manager to room {thread_id_for_analysis}, length: {len(chunk_content)}")
                        pass
                    else:
                        #logger.warning(f"Analysis chunk NOT DELIVERED via socketio_manager to room {thread_id_for_analysis}, length: {len(chunk_content)} - No active sessions")
                        pass
                except Exception as e:
                    logger.error(f"Error in socketio_manager websocket delivery: {str(e)}")
                    was_delivered = False
            
            # Always yield for the standard flow too
            yield {"type": "analysis", "content": chunk_content, "complete": False}
        
        # Send a final complete message with the full analysis
        logger.info(f"Streaming complete analysis, length: {len(analysis_buffer)}")
        
        # Final complete event
        complete_event = {
            "type": "analysis", 
            "content": analysis_buffer, 
            "complete": True
        }
        
        # Send via WebSocket if configured
        if use_websocket and thread_id_for_analysis:
            try:
                from socketio_manager import emit_analysis_event
                was_delivered = emit_analysis_event(thread_id_for_analysis, complete_event)
                if was_delivered:
                    #logger.info(f"Sent complete analysis via socketio_manager to room {thread_id_for_analysis}")
                    pass
                else:
                    #logger.warning(f"Complete analysis NOT DELIVERED via socketio_manager to room {thread_id_for_analysis} - No active sessions")
                    pass
            except Exception as e:
                logger.error(f"Error in socketio_manager delivery of complete event: {str(e)}")
                was_delivered = False
        
        # Always yield for standard flow
        yield {"type": "analysis", "content": analysis_buffer, "complete": True}
        
    except Exception as e:
        logger.error(f"Analysis streaming failed: {e}")
        # Error event
        error_event = {
            "type": "analysis", 
            "content": "Error in analysis process", 
            "complete": True, 
            "error": True
        }
        
        # Send via WebSocket if configured
        if use_websocket and thread_id_for_analysis:
            try:
                from socketio_manager import emit_analysis_event
                was_delivered = emit_analysis_event(thread_id_for_analysis, error_event)
                if was_delivered:
                    #logger.info(f"Sent error event via socketio_manager to room {thread_id_for_analysis}")
                    pass
                else:
                    #logger.warning(f"Error event NOT DELIVERED via socketio_manager to room {thread_id_for_analysis} - No active sessions")
                    pass
            except Exception as e:
                logger.error(f"Error in socketio_manager delivery of error event: {str(e)}")
                was_delivered = False

            if was_delivered:
                #logger.info(f"Sent error event via WebSocket to room {thread_id_for_analysis}")
                pass
            else:
                #logger.warning(f"Error event NOT DELIVERED via WebSocket to room {thread_id_for_analysis} - No active sessions")
                pass
        
        # Always yield for standard flow
        yield {"type": "analysis", "content": "Error in analysis process", "complete": True, "error": True}

def build_context_analysis_prompt(context: str, process_instructions: str) -> str:
    """
    Build the context analysis prompt with the exact structure preserved.
    
    Args:
        context: The conversation context
        process_instructions: The knowledge base instructions
        
    Returns:
        str: The complete context analysis prompt
    """
    return (
        f"Based on the conversation:\n{context}\n\n"
        f"KNOWLEDGE BASE INSTRUCTIONS:\n{process_instructions}\n\n"
        f"Analyze this conversation to determine context and next steps. Provide your analysis in BOTH English and Vietnamese.\n\n"
        
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
        
        f"4. NEXT ACTIONS:\n"
        f"   - Based purely on the knowledge base instructions, what is the next appropriate action?\n"
        f"   - If information gathering is needed, what specific questions should be asked?\n"
        f"   - If information is complete, what should be the focus of the response?\n"
        f"   - If the user has expressed rejection/disagreement, how should you respond?\n\n"
        
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
        
        f"4. HÀNH ĐỘNG TIẾP THEO:\n"
        f"   - Dựa trên hướng dẫn cơ sở kiến thức, hành động tiếp theo phù hợp là gì?\n"
        f"   - Nếu cần thu thập thông tin, nên hỏi những câu hỏi cụ thể nào?\n"
        f"   - Nếu thông tin đã đầy đủ, trọng tâm của phản hồi nên là gì?\n"
        f"   - Nếu người dùng bày tỏ sự từ chối/không đồng ý, nên phản hồi như thế nào?\n\n"
        
        f"Be objective and factual. Only reference information explicitly present in either the conversation or knowledge base instructions."
    )

def process_analysis_result(full_analysis: str) -> Dict[str, str]:
    """
    Process the analysis result to separate English and Vietnamese parts.
    
    Args:
        full_analysis: The complete analysis string
        
    Returns:
        Dict[str, str]: Dictionary containing 'english' and 'vietnamese' analysis parts
    """
    try:
        # Split the analysis into English and Vietnamese parts
        sections = full_analysis.split("VIETNAMESE ANALYSIS:")
        if len(sections) == 2:
            english_analysis = sections[0].strip()
            vietnamese_analysis = sections[1].strip()
        else:
            # Fallback to original behavior if split fails
            english_analysis = full_analysis
            vietnamese_analysis = ""
        
        return {
            "english": english_analysis,
            "vietnamese": vietnamese_analysis
        }
    except Exception as e:
        logger.error(f"Error processing analysis result: {str(e)}")
        return {
            "english": full_analysis,
            "vietnamese": ""
        } 