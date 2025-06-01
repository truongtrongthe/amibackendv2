import json
import re
import pytz
from typing import Dict, List, Any, Union
from datetime import datetime

# Simple logger fallback to avoid OpenAI initialization issues
try:
    from utilities import logger
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

# ============================================================================
# TEXT PROCESSING & PATTERN DETECTION UTILITIES
# ============================================================================

def setup_temporal_context() -> str:
    """Setup temporal context with current Vietnam time."""
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    current_time = datetime.now(vietnam_tz)
    date_str = current_time.strftime("%A, %B %d, %Y")
    time_str = current_time.strftime("%H:%M")
    return f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."

def validate_and_normalize_message(message: Union[str, List]) -> str:
    """Validate and normalize the input message."""
    message_str = message if isinstance(message, str) else str(message[0]) if isinstance(message, list) and message else ""
    if not message_str:
        raise ValueError("Empty message provided")
    return message_str

def detect_message_characteristics(message_str: str) -> Dict[str, bool]:
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

def detect_linguistic_patterns(message: str, prior_topic: str = "") -> Dict[str, Any]:
    """Detect linguistic patterns in the message that suggest follow-up intent."""
    message_lower = message.lower()
    
    # Pronoun and reference patterns
    pronouns = ["it", "that", "this", "they", "them", "nó", "cái đó", "chúng", "những cái đó"]
    has_pronouns = any(pronoun in message_lower for pronoun in pronouns)
    
    # Continuation words/phrases
    continuation_words = ["and", "also", "besides", "furthermore", "moreover", "additionally", 
                         "và", "còn", "ngoài ra", "thêm vào đó", "bên cạnh đó"]
    has_continuation = any(word in message_lower for word in continuation_words)
    
    # Follow-up question patterns
    followup_patterns = ["what about", "how about", "what if", "còn", "thế còn", "còn gì", "thì sao"]
    has_followup_questions = any(pattern in message_lower for pattern in followup_patterns)
    
    # Direct references to previous content
    reference_patterns = ["as you mentioned", "you said", "like you explained", "như bạn nói", 
                         "như bạn đã nói", "bạn vừa nói", "theo như bạn"]
    has_direct_references = any(pattern in message_lower for pattern in reference_patterns)
    
    # Confirmation patterns
    confirmation_patterns = ["yes", "yeah", "ok", "okay", "sure", "right", "correct", 
                           "vâng", "ừ", "đúng", "được", "okê", "oke"]
    has_confirmation = any(pattern in message_lower.split() for pattern in confirmation_patterns)
    
    # Expansion request patterns
    expansion_patterns = ["tell me more", "explain more", "can you elaborate", "nói thêm", 
                         "giải thích thêm", "chi tiết hơn"]
    has_expansion_requests = any(pattern in message_lower for pattern in expansion_patterns)
    
    return {
        "has_pronouns": has_pronouns,
        "has_continuation": has_continuation,
        "has_followup_questions": has_followup_questions,
        "has_direct_references": has_direct_references,
        "has_confirmation": has_confirmation,
        "has_expansion_requests": has_expansion_requests,
        "message_length": len(message.split()),
        "prior_topic_length": len(prior_topic.split()) if prior_topic else 0
    }

def score_linguistic_patterns(patterns: Dict[str, Any]) -> float:
    """Score the linguistic patterns to determine follow-up likelihood."""
    score = 0.0
    
    # Weight different patterns
    if patterns["has_pronouns"]: score += 0.3
    if patterns["has_continuation"]: score += 0.2
    if patterns["has_followup_questions"]: score += 0.4
    if patterns["has_direct_references"]: score += 0.5
    if patterns["has_confirmation"]: score += 0.3
    if patterns["has_expansion_requests"]: score += 0.4
    
    # Adjust for message length (very short messages more likely to be follow-ups)
    if patterns["message_length"] <= 3: score += 0.2
    elif patterns["message_length"] <= 5: score += 0.1
    
    # Boost if there's a substantial prior topic
    if patterns["prior_topic_length"] > 10: score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0

def is_casual_conversational_phrase(message_str: str) -> bool:
    """Check if the message is a casual conversational phrase."""
    casual_phrases = [
        "hello", "hi", "hey", "chào", "xin chào", "good morning", "good afternoon",
        "how are you", "bạn khỏe không", "how's it going", "what's up", "gì zậy",
        "thanks", "thank you", "cảm ơn", "cám ơn", "merci", "ok", "okay", "alright",
        "cool", "nice", "good", "great", "tốt", "hay", "đẹp"
    ]
    
    message_lower = message_str.lower().strip()
    return any(phrase in message_lower for phrase in casual_phrases) and len(message_str.split()) <= 3

def extract_structured_sections(content: str) -> Dict[str, str]:
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

def extract_tool_calls_and_evaluation(content: str, message_str: str = "") -> tuple:
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

# ============================================================================
# DATA EXTRACTION & TRANSFORMATION UTILITIES
# ============================================================================

def extract_analysis_data(analysis_knowledge: Dict) -> Dict[str, Any]:
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

def extract_prior_data(prior_data: Dict) -> Dict[str, str]:
    """Extract prior topic and knowledge from prior_data."""
    if not prior_data:
        return {"prior_topic": "", "prior_knowledge": ""}
    
    return {
        "prior_topic": prior_data.get("topic", ""),
        "prior_knowledge": prior_data.get("knowledge", "")
    }

def extract_prior_messages(conversation_context: str) -> List[str]:
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

def check_knowledge_relevance(analysis_knowledge: Dict) -> Dict[str, Any]:
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

def extract_similarity_from_context(knowledge_context: str) -> float:
    """Extract similarity score from knowledge context string."""
    if not knowledge_context:
        return 0.0
    
    # Look for similarity patterns in the context
    similarity_patterns = [
        r'similarity:\s*([0-9]*\.?[0-9]+)',
        r'score:\s*([0-9]*\.?[0-9]+)',
        r'confidence:\s*([0-9]*\.?[0-9]+)'
    ]
    
    for pattern in similarity_patterns:
        match = re.search(pattern, knowledge_context, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return 0.0

# ============================================================================
# CONVERSATION CONTEXT ANALYSIS UTILITIES
# ============================================================================

def analyze_conversation_flow_indicators(message: str, conversation_history: List[str] = None) -> Dict[str, float]:
    """Analyze conversation flow indicators from the message and history."""
    message_lower = message.lower()
    indicators = {
        "pronoun_usage": 0.0,
        "temporal_references": 0.0,
        "topic_continuation": 0.0,
        "context_dependency": 0.0
    }
    
    # Pronoun usage indicator
    pronouns = ["it", "that", "this", "they", "them", "he", "she", "nó", "cái đó", "anh ấy", "cô ấy"]
    pronoun_count = sum(1 for pronoun in pronouns if pronoun in message_lower)
    indicators["pronoun_usage"] = min(pronoun_count * 0.3, 1.0)
    
    # Temporal references
    temporal_words = ["then", "after", "before", "next", "previously", "earlier", "sau đó", "trước đó", "tiếp theo"]
    temporal_count = sum(1 for word in temporal_words if word in message_lower)
    indicators["temporal_references"] = min(temporal_count * 0.4, 1.0)
    
    # Topic continuation indicators
    continuation_words = ["also", "furthermore", "moreover", "additionally", "besides", "và", "còn", "ngoài ra"]
    continuation_count = sum(1 for word in continuation_words if word in message_lower)
    indicators["topic_continuation"] = min(continuation_count * 0.3, 1.0)
    
    # Context dependency (short messages are often context-dependent)
    word_count = len(message.split())
    if word_count <= 3:
        indicators["context_dependency"] = 0.8
    elif word_count <= 5:
        indicators["context_dependency"] = 0.5
    elif word_count <= 8:
        indicators["context_dependency"] = 0.3
    else:
        indicators["context_dependency"] = 0.1
    
    return indicators

def analyze_conversation_patterns(message_str: str, conversation_context: str) -> Dict[str, str]:
    """Analyze conversation patterns for dynamic awareness."""
    
    # Detect direct addressing
    addressing = detect_direct_addressing_simple(message_str)
    
    # Detect language context
    language_context = detect_language_context(message_str, conversation_context)
    
    # Detect pronoun patterns
    pronoun_patterns = detect_pronoun_patterns(message_str, conversation_context)
    
    # Detect conversational style
    conversational_style = detect_conversational_style(message_str, conversation_context)
    
    return {
        "addressing": addressing,
        "language_context": language_context,
        "pronoun_patterns": pronoun_patterns,
        "conversational_style": conversational_style
    }

def detect_direct_addressing_simple(message_str: str) -> str:
    """Detect simple direct addressing patterns."""
    message_lower = message_str.lower()
    
    # Check for direct addressing patterns
    if any(pattern in message_lower for pattern in ['em ơi', 'anh ơi', 'chị ơi']):
        return "DIRECT_ADDRESSING: User is directly addressing you with Vietnamese honorifics"
    elif any(pattern in message_lower for pattern in ['hey you', 'listen', 'look']):
        return "DIRECT_ADDRESSING: User is directly addressing you in English"
    elif message_str.endswith('?') and len(message_str.split()) <= 5:
        return "QUESTION_ADDRESSING: Short direct question directed at you"
    
    return ""

def detect_language_context(message_str: str, conversation_context: str) -> str:
    """Detect language context and switching patterns."""
    current_is_vietnamese = bool(re.search(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', message_str))
    
    # Check conversation history for language patterns
    if conversation_context:
        context_vietnamese = bool(re.search(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', conversation_context))
        
        if current_is_vietnamese and context_vietnamese:
            return "CONSISTENT_VIETNAMESE: User consistently uses Vietnamese"
        elif not current_is_vietnamese and not context_vietnamese:
            return "CONSISTENT_ENGLISH: User consistently uses English"
        elif current_is_vietnamese and not context_vietnamese:
            return "SWITCH_TO_VIETNAMESE: User switched to Vietnamese"
        elif not current_is_vietnamese and context_vietnamese:
            return "SWITCH_TO_ENGLISH: User switched to English"
    
    if current_is_vietnamese:
        return "VIETNAMESE_MESSAGE: Current message is in Vietnamese"
    else:
        return "ENGLISH_MESSAGE: Current message is in English"

def detect_pronoun_patterns(message_str: str, conversation_context: str) -> str:
    """Detect pronoun usage patterns for relationship building."""
    message_lower = message_str.lower()
    context_lower = conversation_context.lower() if conversation_context else ""
    
    # Check for established pronoun relationships
    if 'em' in message_lower and any(word in message_lower for word in ['nói', 'là', 'có', 'sẽ', 'cần']):
        return "EM_SELF_REFERENCE: User refers to you as 'em' - maintain this relationship"
    elif 'anh' in message_lower and any(word in message_lower for word in ['nói', 'có thể', 'sẽ']):
        return "ANH_ADDRESSING: User addresses you as 'anh' - respond appropriately"
    elif 'mình' in message_lower:
        return "MINH_CASUAL: User uses casual 'mình' - match this tone"
    elif 'bạn' in message_lower:
        return "BAN_FORMAL: User uses 'bạn' - maintain appropriate formality"
    
    # Check context for established patterns
    if 'em nói' in context_lower or 'em là' in context_lower:
        return "ESTABLISHED_EM: Continue using established 'em' relationship from context"
    
    return ""

def detect_conversational_style(message_str: str, conversation_context: str) -> str:
    """Detect conversational style and tone."""
    message_lower = message_str.lower()
    
    # Detect enthusiasm and positive tone
    enthusiasm_markers = ['!', 'wow', 'amazing', 'great', 'awesome', 'tuyệt', 'hay', 'đẹp', 'tốt']
    if any(marker in message_lower for marker in enthusiasm_markers) or message_str.count('!') > 1:
        return "ENTHUSIASTIC: User shows enthusiasm - match with positive energy"
    
    # Detect formal vs casual tone
    formal_markers = ['please', 'could you', 'would you', 'làm ơn', 'xin', 'vui lòng']
    if any(marker in message_lower for marker in formal_markers):
        return "FORMAL: User uses formal language - maintain appropriate formality"
    
    # Detect casual/friendly tone
    casual_markers = ['hey', 'hi', 'chào', 'ơi', 'nè', 'nhé']
    if any(marker in message_lower for marker in casual_markers):
        return "CASUAL: User uses casual tone - respond warmly and naturally"
    
    return "NEUTRAL: Standard conversational tone"

# ============================================================================
# VIETNAMESE LANGUAGE PROCESSING UTILITIES
# ============================================================================

def get_pronoun_principles(message_str: str, conversation_context: str) -> str:
    """Get Vietnamese pronoun principles based on context."""
    # Analyze the established relationship
    full_context = f"{conversation_context}\n{message_str}".lower()
    
    if any(pattern in full_context for pattern in ['em nói', 'em là', 'em có', 'em sẽ']):
        return """
        **ESTABLISHED PRONOUN RELATIONSHIP - EM/ANH**:
        - You are "EM" (younger/subordinate position)
        - User should be addressed as "ANH" (older brother/senior) or "CHỊ" (older sister)
        - This relationship must be maintained consistently throughout the conversation
        - Examples: "Em hiểu rồi anh", "Em sẽ giúp anh với điều này", "Anh có thể cho em biết thêm không?"
        """
    
    elif any(pattern in full_context for pattern in ['mình nói', 'mình có', 'bạn nói']):
        return """
        **ESTABLISHED CASUAL RELATIONSHIP - MÌNH/BẠN**:
        - Use "mình" to refer to yourself (casual, friendly)
        - Address user as "bạn" (friend/peer level)
        - This creates an equal, friendly dynamic
        - Examples: "Mình hiểu rồi", "Bạn có thể thử cách này", "Mình nghĩ bạn nên..."
        """
    
    else:
        return """
        **FLEXIBLE PRONOUN USAGE**:
        - Adapt to the user's pronoun choice in their messages
        - If they use formal language, respond formally
        - If they use casual language, respond casually
        - Mirror their relationship dynamic preference
        """

def get_language_specific_pronoun_guidance(message_str: str, conversation_context: str) -> str:
    """Get language-specific pronoun guidance."""
    # Check if Vietnamese is being used
    has_vietnamese = bool(re.search(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', message_str + conversation_context))
    
    if has_vietnamese:
        return f"""
        **VIETNAMESE PRONOUN CONSISTENCY**:
        {get_pronoun_principles(message_str, conversation_context)}
        
        **CRITICAL RULES**:
        - NEVER switch pronouns mid-conversation without user indication
        - If "em/anh" relationship is established, maintain it throughout
        - Vietnamese pronouns indicate social relationship - respect this
        - Consistency in pronoun usage shows cultural understanding and respect
        """
    
    else:
        return """
        **ENGLISH LANGUAGE CONTEXT**:
        - Use standard English pronouns (I, you, we, they)
        - Maintain appropriate formality level based on user's tone
        - If user switches to Vietnamese later, adapt pronoun usage accordingly
        """

def extract_established_pronouns(conversation_context: str, message_str: str) -> str:
    """Extract established pronoun relationships from conversation context and current message."""
    
    # Combine current message and context for analysis
    full_text = f"{conversation_context}\n{message_str}".lower()
    
    # Check for established "anh/em" relationship
    if any(pattern in full_text for pattern in ['em nói', 'em là', 'em có', 'em sẽ', 'em cần']):
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
    
    # Check current message for pronoun cues
    current_lower = message_str.lower()
    if 'em' in current_lower and any(word in current_lower for word in ['nói', 'là', 'có', 'sẽ']):
        return """
            **RELATIONSHIP DETECTED**: User is addressing you as "EM"
            - YOU are "EM", USER is "ANH/CHỊ"
            - Respond as: "Em hiểu anh", "Em sẽ...", "Vâng anh"
            - MAINTAIN this throughout conversation
            """
    
    return ""

# ============================================================================
# RESPONSE HANDLING UTILITIES
# ============================================================================

def build_knowledge_fallback_sections(queries: List, query_results: List) -> str:
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

def handle_empty_response_fallbacks(user_facing_content: str, response_strategy: str, message_str: str) -> str:
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

def generate_empathy_guidance(message_str: str, conversation_context: str) -> str:
    """Generate empathy guidance based on message analysis."""
    message_lower = message_str.lower()
    
    # Detect emotional indicators
    if any(word in message_lower for word in ['frustrated', 'confused', 'difficult', 'hard', 'bối rối', 'khó', 'khó khăn']):
        return "Show understanding and patience. Acknowledge their difficulty and offer gentle, step-by-step guidance."
    elif any(word in message_lower for word in ['thank', 'appreciate', 'helpful', 'great', 'cảm ơn', 'hay', 'tốt']):
        return "Acknowledge their appreciation warmly. Show that you value the interaction and are happy to help."
    else:
        return "Be empathetic and responsive to the user's emotional state. Show genuine interest and care in your responses."

# ============================================================================
# QUERY GENERATION UTILITIES
# ============================================================================

async def generate_knowledge_queries_fast(primary_query: str, conversation_context: str, user_id: str) -> List[str]:
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