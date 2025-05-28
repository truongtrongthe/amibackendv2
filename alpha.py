import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from utilities import logger

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
