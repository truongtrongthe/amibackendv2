import json
import re
import time
import asyncio
import traceback
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Tuple
from functools import lru_cache
from langchain_openai import ChatOpenAI
from utilities import logger
from brain_singleton import get_brain, set_graph_version, is_brain_loaded, load_brain_vectors, get_current_graph_version

brain = get_brain()

def extract_structured_data_from_raw(raw_text: str) -> Dict[str, str]:
    """
    Extract structured data from raw text using regex patterns.
    
    Args:
        raw_text: The raw text from a knowledge entry
        
    Returns:
        Dictionary with extracted fields (title, description, content, takeaways, etc.)
    """
    structured_data = {}
    
    # Extract title
    title_match = re.search(r'Title:\s*(.*?)(?:\n|$)', raw_text)
    if title_match:
        structured_data["title"] = title_match.group(1).strip()
    
    # Extract description
    desc_match = re.search(r'Description:\s*(.*?)(?:\nContent:|\nTakeaways:|\nDocument Summary:|\n\n|$)', raw_text, re.DOTALL)
    if desc_match:
        structured_data["description"] = desc_match.group(1).strip()
    
    # Extract content
    content_match = re.search(r'Content:\s*(.*?)(?:\nTakeaways:|\nDocument Summary:|\n\n|$)', raw_text, re.DOTALL)
    if content_match:
        structured_data["content"] = content_match.group(1).strip()
    
    # Extract application method from takeaways
    takeaways_match = re.search(r'Takeaways:\s*(.*?)(?:\nDocument Summary:|\n\n|$)', raw_text, re.DOTALL)
    if takeaways_match:
        takeaways_text = takeaways_match.group(1).strip()
        app_method_match = re.search(r'Application Method:\s*(.*?)(?:\n\n\d|\n\n$|$)', takeaways_text, re.DOTALL)
        if app_method_match:
            structured_data["application_method"] = app_method_match.group(1).strip()
        else:
            structured_data["takeaways"] = takeaways_text
    
    # Extract document summary
    summary_match = re.search(r'Document Summary:\s*(.*?)(?:\nCross-Cluster Connections:|\n\n|$)', raw_text, re.DOTALL)
    if summary_match:
        structured_data["document_summary"] = summary_match.group(1).strip()
    
    # Extract cross-cluster connections
    connections_match = re.search(r'Cross-Cluster Connections:\s*(.*?)(?:\n\n|$)', raw_text, re.DOTALL)
    if connections_match:
        structured_data["cross_cluster_connections"] = connections_match.group(1).strip()
    
    return structured_data

def optimize_knowledge_context(knowledge_entries: List[Dict], user_query: str, max_chars: int = 2500) -> str:
    """
    Optimize knowledge context by prioritizing and formatting relevant knowledge
    entries for the prompt.
    
    Args:
        knowledge_entries: List of knowledge entry dictionaries
        user_query: Original user query for prioritization
        max_chars: Maximum characters allowed in the context
        
    Returns:
        Formatted knowledge context string
    """
    if not knowledge_entries:
        return ""
    
    try:
        # Step 1: Create a scoring function to prioritize entries
        def score_entry(entry: Dict) -> float:
            score = entry.get("similarity", 0)
            
            # Bonus for profile-matched entries
            if entry.get("profile_match"):
                score += 0.1
            
            # Bonus for entries from first phase
            if entry.get("phase") == "initial":
                score += 0.05
            
            # Bonus for entries with structured data
            if entry.get("structured") and len(entry.get("structured", {})) > 0:
                score += 0.03
            
            return score
        
        # Step 2: Sort entries by score
        sorted_entries = sorted(knowledge_entries, key=score_entry, reverse=True)
        
        # Step 3: Format entries in priority order until max_chars is reached
        formatted_context = ""
        added_ids = set()
        
        for entry in sorted_entries:
            # Skip if already added
            if entry.get("id") in added_ids:
                continue
            
            # Format this entry
            entry_text = format_knowledge_entry(entry)
            
            # Check if adding this would exceed max chars
            if len(formatted_context) + len(entry_text) <= max_chars:
                formatted_context += entry_text + "\n\n"
                added_ids.add(entry.get("id"))
            else:
                # If we can't add more full entries, we're done
                break
        
        return formatted_context.strip()
    
    except Exception as e:
        logger.error(f"Error optimizing knowledge context: {str(e)}")
        # Fallback: just concatenate raw text of first few entries
        fallback_text = ""
        for entry in knowledge_entries[:3]:
            raw = entry.get("raw", "")
            if raw:
                fallback_text += raw[:500] + "...\n\n"  # Truncate each entry
        
        return fallback_text[:max_chars]

async def ensure_brain_loaded(graph_version_id: str = "") -> bool:
    """
    Ensure the brain is loaded and ready to use.
    
    Args:
        graph_version_id: Optional graph version ID
        
    Returns:
        True if brain is loaded successfully, False otherwise
    """
    try:
        # Set graph version if provided
        if graph_version_id:
            current = get_current_graph_version()
            if current != graph_version_id:
                set_graph_version(graph_version_id)
                logger.info(f"Set graph version to {graph_version_id}")
        
        # Check if brain is already loaded
        if is_brain_loaded():
            return True
        
        # Otherwise load the brain
        await load_brain_vectors()
        return is_brain_loaded()
        
    except Exception as e:
        logger.error(f"Error ensuring brain loaded: {str(e)}")
        return False

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    Simple heuristic-based detection for Vietnamese vs. English.
    
    Args:
        text: The text to analyze
        
    Returns:
        Language code ('vi' for Vietnamese, 'en' for English)
    """
    # Vietnamese-specific characters and words
    vietnamese_chars = set("ăâêôơưđ")
    vietnamese_words = [
        "không", "của", "và", "là", "được", "có", "tôi", "cho", "một", "để",
        "trong", "được", "người", "những", "nhưng", "với", "các", "mình", "này", "đã",
        "làm", "khi", "giúp", "từ", "cách", "như", "thể", "nếu", "vì", "tại"
    ]
    
    # Convert text to lowercase for better matching
    text_lower = text.lower()
    
    # Check for Vietnamese characters
    if any(char in vietnamese_chars for char in text_lower):
        return "vietnamese"
    
    # Check for Vietnamese words
    if any(word in text_lower for word in vietnamese_words):
        return "vietnamese"
    
    # Default to English
    return "english"

def process_knowledge_context(knowledge_entries: List[Dict], user_query: str) -> str:
    """
    Process knowledge entries into a formatted context for the LLM.
    
    Args:
        knowledge_entries: List of knowledge entry dictionaries
        user_query: The user's query for relevance scoring
        
    Returns:
        Formatted knowledge context string
    """
    if not knowledge_entries:
        return ""
    
    try:
        # Group entries by type for better organization
        entries_by_type = {
            "high_relevance": [],
            "medium_relevance": [],
            "low_relevance": []
        }
        
        # Calculate median similarity for threshold
        similarities = [entry.get("similarity", 0) for entry in knowledge_entries]
        median_similarity = sorted(similarities)[len(similarities) // 2] if similarities else 0
        
        # Create threshold values
        high_threshold = median_similarity + 0.1
        low_threshold = median_similarity - 0.1
        
        # Categorize entries by relevance
        for entry in knowledge_entries:
            similarity = entry.get("similarity", 0)
            if similarity >= high_threshold:
                entries_by_type["high_relevance"].append(entry)
            elif similarity <= low_threshold:
                entries_by_type["low_relevance"].append(entry)
            else:
                entries_by_type["medium_relevance"].append(entry)
        
        # Format with most relevant first, limited to 2500 chars total
        formatted_context = "KNOWLEDGE CONTEXT:\n\n"
        char_count = len(formatted_context)
        max_chars = 2500
        
        # Process high relevance
        for entry in entries_by_type["high_relevance"]:
            entry_text = format_knowledge_entry(entry)
            if char_count + len(entry_text) + 10 <= max_chars:
                formatted_context += f"[HIGH RELEVANCE]\n{entry_text}\n\n"
                char_count += len(entry_text) + 20  # Account for added tags
        
        # Process medium relevance
        for entry in entries_by_type["medium_relevance"]:
            entry_text = format_knowledge_entry(entry)
            if char_count + len(entry_text) + 10 <= max_chars:
                formatted_context += f"[MEDIUM RELEVANCE]\n{entry_text}\n\n"
                char_count += len(entry_text) + 22  # Account for added tags
        
        # Process low relevance (only if space allows)
        for entry in entries_by_type["low_relevance"]:
            entry_text = format_knowledge_entry(entry)
            if char_count + len(entry_text) + 10 <= max_chars:
                formatted_context += f"[ADDITIONAL INFO]\n{entry_text}\n\n"
                char_count += len(entry_text) + 21  # Account for added tags
        
        return formatted_context.strip()
        
    except Exception as e:
        logger.error(f"Error processing knowledge context: {str(e)}")
        # Fallback: just concatenate the first few entries
        fallback_text = "KNOWLEDGE CONTEXT:\n\n"
        for entry in knowledge_entries[:3]:
            fallback_text += format_knowledge_entry(entry) + "\n\n"
        
        # Ensure we don't exceed the maximum length
        if len(fallback_text) > 2500:
            fallback_text = fallback_text[:2497] + "..."
        
        return fallback_text

def format_knowledge_entry(entry: Dict) -> str:
    """
    Format a knowledge entry for inclusion in a prompt.
    
    Args:
        entry: Knowledge entry dictionary
        
    Returns:
        Formatted string representation of the knowledge entry
    """
    # Extract entry components
    entry_id = entry.get("id", "unknown")
    similarity = entry.get("similarity", 0)
    raw_text = entry.get("raw", "")
    structured = entry.get("structured", {})
    
    # Format with structured data if available
    if structured:
        parts = []
        
        # Add title if available
        if "title" in structured:
            parts.append(f"Title: {structured['title']}")
        
        # Add description if available
        if "description" in structured:
            parts.append(f"Description: {structured['description']}")
        
        # Add application method or takeaways if available
        if "application_method" in structured:
            parts.append(f"Application: {structured['application_method']}")
        elif "takeaways" in structured:
            parts.append(f"Takeaways: {structured['takeaways']}")
        
        # Fall back to raw text if structured extraction failed
        if not parts and raw_text:
            # Clean up raw text and truncate if needed
            cleaned_text = raw_text.strip()
            if len(cleaned_text) > 500:  # Limit to 500 chars
                cleaned_text = cleaned_text[:497] + "..."
            return cleaned_text
        
        return "\n".join(parts)
    
    # Fall back to raw text if no structured data
    if raw_text:
        # Clean up raw text and truncate if needed
        cleaned_text = raw_text.strip()
        if len(cleaned_text) > 500:  # Limit to 500 chars
            cleaned_text = cleaned_text[:497] + "..."
        return cleaned_text
    
    # Last resort if no content available
    return f"Entry {entry_id} (Relevance: {similarity:.2f})"

def extract_key_knowledge(knowledge_context: str, conversation_context: str) -> str:
    """
    Extract the most relevant parts from knowledge context based on the conversation.
    
    Args:
        knowledge_context: The full knowledge context text
        conversation_context: The conversation context to use for relevance
        
    Returns:
        Reduced knowledge context focusing on the most relevant parts
    """
    if not knowledge_context:
        return ""
    
    try:
        # Get key terms from the conversation (last user message)
        key_terms = []
        user_msg = ""
        
        for line in conversation_context.split('\n'):
            if line.startswith("User:"):
                user_msg = line[5:].strip()
                break
        
        # Extract potential key terms from user message
        if user_msg:
            # Remove common words and punctuation
            common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "and", "but", "or", "is", "are",
                           "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "of"}
            
            # Split by common delimiters and convert to lowercase
            words = re.split(r'[ \t\n,.?!:;()\[\]{}"]', user_msg.lower())
            words = [w for w in words if w and w not in common_words and len(w) > 2]
            
            # Take most frequent terms
            from collections import Counter
            term_counter = Counter(words)
            key_terms = [term for term, _ in term_counter.most_common(5)]
        
        # Split knowledge into sections/paragraphs
        sections = re.split(r'\n\n+', knowledge_context)
        
        # Score each section based on key terms
        scored_sections = []
        for section in sections:
            score = 1  # Base score
            
            # Score by key term presence
            for term in key_terms:
                if term.lower() in section.lower():
                    score += 3
            
            # Bonus for sections with "Title:", "Description:", etc.
            if re.search(r'(Title|Description|Application|Takeaways):', section):
                score += 5
            
            # Penalize very long sections
            if len(section) > 300:
                score -= 2
            
            scored_sections.append((section, score))
        
        # Sort by score and take top sections
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        top_sections = [section for section, _ in scored_sections[:3]]
        
        # Combine and return top sections
        reduced_context = "\n\n".join(top_sections)
        
        # Ensure we're within limit
        if len(reduced_context) > 1500:
            reduced_context = reduced_context[:1497] + "..."
        
        return reduced_context
        
    except Exception as e:
        logger.error(f"Error extracting key knowledge: {str(e)}")
        # Return truncated original context as fallback
        return knowledge_context[:1000] + "..."


def build_user_profile(conversation_context: str, last_user_message: str) -> Dict:
    """
    Build a comprehensive user profile based on conversation history and the latest message.
    This profile helps tailor knowledge retrieval and response generation.
    
    Args:
        conversation_context: The full conversation history
        last_user_message: The most recent user message
        
    Returns:
        A dictionary containing the user profile with multiple dimensions
    """
    # Default profile with fallback values
    default_profile = {
        "identity": {
            "expertise_level": "intermediate",  # beginner, intermediate, advanced
            "language_preference": "en"  # en, vi, etc.
        },
        "segment": {
            "category": "general",  # general, entrepreneur, investor, etc.
            "interest_level": "moderate",  # low, moderate, high
            "customer_stage": "exploring"  # exploring, evaluating, committed
        },
        "communication": {
            "style": "casual",  # formal, casual, technical
            "tone": "neutral",  # friendly, neutral, serious
            "detail_preference": "balanced"  # minimal, balanced, detailed
        },
        "emotional_state": {
            "primary": "neutral",  # curious, frustrated, urgent, neutral
            "urgency": "normal"  # low, normal, high
        },
        "query_characteristics": {
            "type": "informational",  # informational, problem-solving, opinion
            "complexity": "moderate",  # simple, moderate, complex
            "knowledge_areas": []  # list of relevant domains/topics
        }
    }
    
    try:
        # Detect language preference (simple heuristic, would be better with language detection)
        vietnamese_indicators = ["không", "của", "và", "là", "được", "có", "tôi", "cho", "một", "để"]
        if any(word in last_user_message.lower() for word in vietnamese_indicators):
            default_profile["identity"]["language_preference"] = "vi"
        
        # Detect expertise level based on terminology and question sophistication
        technical_terms = ["roi", "ebitda", "valuation", "cash flow", "tư vấn", "đầu tư", 
                          "portfolio", "scaling", "investment", "strategy", "strategic", 
                          "acquisition", "term sheet", "venture", "due diligence"]
        
        technical_term_count = sum(1 for term in technical_terms if term.lower() in last_user_message.lower())
        
        if technical_term_count >= 3:
            default_profile["identity"]["expertise_level"] = "advanced"
        elif technical_term_count >= 1:
            default_profile["identity"]["expertise_level"] = "intermediate"
        else:
            default_profile["identity"]["expertise_level"] = "beginner"
        
        # Detect segment based on keywords in conversation
        entrepreneur_indicators = ["startup", "business", "venture", "entrepreneur", "founding", "scaling", 
                                  "kinh doanh", "startups", "khởi nghiệp", "gọi vốn"]
        investor_indicators = ["investment", "returns", "portfolio", "investing", "investor", 
                              "đầu tư", "nhà đầu tư", "lợi nhuận", "danh mục"]
        
        entrepreneur_count = sum(1 for term in entrepreneur_indicators if term.lower() in conversation_context.lower())
        investor_count = sum(1 for term in investor_indicators if term.lower() in conversation_context.lower())
        
        if entrepreneur_count > investor_count and entrepreneur_count >= 2:
            default_profile["segment"]["category"] = "entrepreneur"
        elif investor_count > entrepreneur_count and investor_count >= 2:
            default_profile["segment"]["category"] = "investor"
        
        # Detect communication style
        formal_indicators = ["would you", "please", "kindly", "appreciate", "thank you", "respectfully", 
                            "could you please", "I would like to", "lam ơn", "xin vui lòng"]
        technical_indicators = ["specifically", "details", "analysis", "compare", "data", "statistics", 
                               "thoroughly", "chi tiết", "phân tích", "số liệu"]
        
        formal_count = sum(1 for term in formal_indicators if term.lower() in last_user_message.lower())
        technical_count = sum(1 for term in technical_indicators if term.lower() in last_user_message.lower())
        
        if formal_count >= 2:
            default_profile["communication"]["style"] = "formal"
        elif technical_count >= 2:
            default_profile["communication"]["style"] = "technical"
        else:
            default_profile["communication"]["style"] = "casual"
        
        # Detect emotional state
        urgent_indicators = ["urgent", "quickly", "asap", "immediately", "need right now", "hurry", 
                            "khẩn cấp", "ngay lập tức", "càng sớm càng tốt"]
        frustrated_indicators = ["not working", "frustrated", "tried", "still", "doesn't work", "issue", "problem", 
                               "gặp vấn đề", "không hoạt động", "lỗi"]
        curious_indicators = ["curious", "wondering", "interested", "tell me more", "how does", "why is", 
                             "tò mò", "tại sao", "như thế nào"]
        
        urgent_count = sum(1 for term in urgent_indicators if term.lower() in last_user_message.lower())
        frustrated_count = sum(1 for term in frustrated_indicators if term.lower() in last_user_message.lower())
        curious_count = sum(1 for term in curious_indicators if term.lower() in last_user_message.lower())
        
        if urgent_count >= 1:
            default_profile["emotional_state"]["primary"] = "urgent"
            default_profile["emotional_state"]["urgency"] = "high"
        elif frustrated_count >= 2:
            default_profile["emotional_state"]["primary"] = "frustrated"
        elif curious_count >= 1:
            default_profile["emotional_state"]["primary"] = "curious"
        
        # Determine query type and complexity
        question_indicators = ["?", "what", "how", "why", "when", "who", "where", 
                              "tại sao", "như thế nào", "khi nào", "ai", "ở đâu"]
        problem_solving_indicators = ["help", "solve", "fix", "resolve", "solution", "advice", 
                                     "giúp", "giải quyết", "cách", "tư vấn"]
        
        has_question = any(term in last_user_message.lower() for term in question_indicators)
        has_problem = any(term in last_user_message.lower() for term in problem_solving_indicators)
        
        if has_problem:
            default_profile["query_characteristics"]["type"] = "problem-solving"
        elif has_question:
            default_profile["query_characteristics"]["type"] = "informational"
        else:
            default_profile["query_characteristics"]["type"] = "opinion"
        
        # Estimate complexity by message length and structure
        words = last_user_message.split()
        if len(words) > 30:
            default_profile["query_characteristics"]["complexity"] = "complex"
        elif len(words) > 10:
            default_profile["query_characteristics"]["complexity"] = "moderate"
        else:
            default_profile["query_characteristics"]["complexity"] = "simple"
        
        # Extract knowledge areas
        knowledge_area_keywords = {
            "funding": ["funding", "investment", "VC", "venture capital", "gọi vốn", "đầu tư"],
            "strategy": ["strategy", "growth", "scaling", "chiến lược", "tăng trưởng", "mở rộng"],
            "operations": ["operations", "management", "team", "quản lý", "vận hành", "đội ngũ"],
            "market": ["market", "industry", "competitor", "thị trường", "ngành", "đối thủ"],
            "product": ["product", "development", "features", "sản phẩm", "phát triển", "tính năng"],
            "legal": ["legal", "terms", "agreement", "pháp lý", "điều khoản", "hợp đồng"]
        }
        
        detected_areas = []
        for area, keywords in knowledge_area_keywords.items():
            if any(keyword.lower() in last_user_message.lower() for keyword in keywords):
                detected_areas.append(area)
        
        default_profile["query_characteristics"]["knowledge_areas"] = detected_areas if detected_areas else ["general"]
        
        # Further refinement based on message content
        if "compare" in last_user_message.lower() or "difference" in last_user_message.lower():
            default_profile["query_characteristics"]["complexity"] = "complex"
            default_profile["communication"]["detail_preference"] = "detailed"
        
        # Detect interest level based on question specificity and follow-up patterns
        if "more detail" in conversation_context.lower() or "tell me more" in conversation_context.lower():
            default_profile["segment"]["interest_level"] = "high"
        
        # Detect customer stage from conversation context patterns
        if "already using" in conversation_context.lower() or "đang sử dụng" in conversation_context.lower():
            default_profile["segment"]["customer_stage"] = "committed"
        elif "considering" in conversation_context.lower() or "đang cân nhắc" in conversation_context.lower():
            default_profile["segment"]["customer_stage"] = "evaluating"
        
        return default_profile
        
    except Exception as e:
        logger.error(f"Error building user profile: {str(e)}")
        return default_profile

def generate_profile_enhanced_queries(base_query: str, user_profile: Dict) -> List[str]:
    """
    Generate multiple search queries enhanced with user profile information
    to improve knowledge retrieval relevance.
    
    Args:
        base_query: The original user query
        user_profile: The user profile dictionary
        
    Returns:
        A list of profile-enhanced search queries
    """
    enhanced_queries = [base_query]  # Always include the original query
    
    try:
        # Extract key profile dimensions
        segment = user_profile.get("segment", {}).get("category", "general")
        expertise = user_profile.get("identity", {}).get("expertise_level", "intermediate")
        knowledge_areas = user_profile.get("query_characteristics", {}).get("knowledge_areas", [])
        query_type = user_profile.get("query_characteristics", {}).get("type", "informational")
        emotional_state = user_profile.get("emotional_state", {}).get("primary", "neutral")
        
        # Create segment-focused query
        if segment != "general":
            segment_query = f"{base_query} for {segment}s"
            enhanced_queries.append(segment_query)
        
        # Create knowledge area focused queries
        for area in knowledge_areas[:2]:  # Limit to top 2 knowledge areas
            if area != "general":
                knowledge_query = f"{base_query} {area}"
                enhanced_queries.append(knowledge_query)
        
        # Create expertise-level focused query
        if expertise in ["beginner", "advanced"]:
            expertise_query = f"{base_query} for {expertise}s"
            enhanced_queries.append(expertise_query)
        
        # Create query type focused query
        if query_type == "problem-solving":
            problem_query = f"solution for {base_query}"
            enhanced_queries.append(problem_query)
        
        # Create emotional state focused query for urgent or frustrated states
        if emotional_state in ["urgent", "frustrated"]:
            emotional_query = f"quick {base_query}" if emotional_state == "urgent" else f"troubleshooting {base_query}"
            enhanced_queries.append(emotional_query)
        
        # Remove duplicates while preserving order
        unique_queries = []
        for query in enhanced_queries:
            if query not in unique_queries:
                unique_queries.append(query)
        
        return unique_queries
        
    except Exception as e:
        logger.error(f"Error generating enhanced queries: {str(e)}")
        return [base_query]  # Fall back to the original query only

