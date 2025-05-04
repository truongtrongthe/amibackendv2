"""
Knowledge-enhanced user profiling system.
This module provides functions for building comprehensive user profiles based on conversation context
and knowledge retrieved from the brain.
"""

import re
import time
import asyncio
import json
from typing import List, Dict, Any, Optional
from utilities import logger
from brain_singleton import get_brain
from tool_helpers import (
    extract_structured_data_from_raw,
    detect_language,
    ensure_brain_loaded,
    prepare_knowledge
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Configuration constants
CLASSIFICATION_TERMS = [
    "nhóm chán nản", 
    "nhóm tự tin", 
    "nhóm chưa rõ tâm lý", 
    "phân loại khách hàng", 
    "phân nhóm"
]

# Initialize global state dictionary
state = {}

# Initialize the LangChain ChatOpenAI model
LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    streaming=False
)

def detect_vietnamese_language(text: str) -> bool:
    """
    More accurate detection of Vietnamese language in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Boolean indicating if text is likely Vietnamese
    """
    # Vietnamese-specific characters
    vn_chars = set("ăâêôơưđáàảãạắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ")
    
    # Common Vietnamese words
    vn_words = [
        "anh", "tôi", "bạn", "bị", "của", "và", "là", "được", "có", "cho", 
        "một", "để", "trong", "người", "những", "không", "với", "các", "mình", 
        "này", "đã", "khi", "từ", "cách", "như", "thể", "nếu", "vì", "tại"
    ]
    
    # Check for Vietnamese characters
    text_lower = text.lower()
    if any(char in vn_chars for char in text):
        return True
    
    # Check for common Vietnamese words
    words = re.findall(r'\b\w+\b', text_lower)
    vn_word_count = sum(1 for word in words if word in vn_words)
    
    if vn_word_count >= 1 or "anh bị" in text_lower or "em bị" in text_lower:
        return True
        
    return False


async def build_user_profile(conversation_context: str, last_user_message: str, graph_version_id: str = "") -> Dict:
    """
    Build a user profile as a descriptive portrait string.
    Uses a knowledge-enhanced approach that retrieves relevant knowledge about user analysis,
    then prompts an LLM to create a comprehensive description of the user.
    
    Args:
        conversation_context: The full conversation history
        last_user_message: The most recent user message
        graph_version_id: The version ID of the knowledge graph to query (optional)
        
    Returns:
        Dictionary with the user portrait as a string and metadata
    """
    try:
        # Detect language
        language = "vi" if detect_vietnamese_language(last_user_message) else "en"
        
        # Skip knowledge retrieval if no graph_version_id provided
        if not graph_version_id:
            logger.info("No graph version ID provided, proceeding with LLM-only profiling")
            return await create_user_portrait(conversation_context, last_user_message, [], "")
        
        # Prepare broader queries to find relevant analysis methods
        understanding_queries = [
            "how to analyze user needs",
            "how to understand user psychology",
            "user profiling techniques",
            "user communication patterns",
            "các phương pháp phân tích nhu cầu người dùng",
            "cách hiểu tâm lý người dùng",
            "kỹ thuật phân tích người dùng",
            "mô hình giao tiếp với người dùng"
        ]
        
        # Select language-specific queries based on detected language
        if language == "vi":
            queries_to_use = understanding_queries[4:8]  # Vietnamese queries
            # Add a couple English queries as backup
            queries_to_use.extend(understanding_queries[0:2])
        else:
            queries_to_use = understanding_queries[0:4]  # English queries
            # Add a couple Vietnamese queries as backup
            queries_to_use.extend(understanding_queries[4:6])
        
        # Ensure brain is loaded
        brain_loaded = await ensure_brain_loaded(graph_version_id)
        if not brain_loaded:
            logger.warning("Brain loading failed, proceeding with LLM-only profiling")
            return await create_user_portrait(conversation_context, last_user_message, [], "")
        
        # Get brain instance
        brain = get_brain()
        knowledge_entries = []
        
        # Retrieve knowledge using multiple queries for better coverage
        logger.info("Fetching knowledge on user analysis techniques")
        
        # Prioritize classification knowledge
        classification_queries = [
            "phân loại khách hàng",
            "nhóm tâm lý khách hàng",
            "phân nhóm khách hàng dựa trên tâm lý",
            "classification of customers"
        ]
        
        # Add these queries to the front of the list to prioritize classification knowledge
        combined_queries = classification_queries + queries_to_use
        
        for query in combined_queries:
            try:
                # Increase top_k to get more diverse results
                results = await brain.get_similar_vectors_by_text(query, top_k=2)
                
                for vector_id, vector, metadata, similarity in results:
                    # Use a slightly lower threshold for more coverage
                    if similarity < 0.28:  # Reduced threshold
                        continue
                    
                    # Skip duplicates
                    if any(entry.get("id") == vector_id for entry in knowledge_entries):
                        logger.info(f"Skipping duplicate knowledge entry: {vector_id}")
                        continue
                    
                    # Get the raw text from the results
                    raw_text = metadata.get("raw", "")
                    
                    # Prioritize entries containing classification terms
                    boost = 0.0
                    if any(term in raw_text.lower() for term in CLASSIFICATION_TERMS):
                        boost = 0.4
                        logger.info(f"Boosting entry {vector_id} for containing classification terms")
                    
                    # Create a simplified knowledge entry with raw data
                    knowledge_entry = {
                        "id": vector_id,
                        "query": query,
                        "raw": raw_text,
                        "similarity": float(similarity) + boost,
                        "priority": 1 if boost > 0 else 0  # Priority flag for classification entries
                    }
                    
                    knowledge_entries.append(knowledge_entry)
            except Exception as e:
                logger.warning(f"Error retrieving knowledge for query '{query}': {str(e)}")
        
        # Process the knowledge entries using prepare_knowledge
        #logger.info(f"Retrieved {knowledge_entries} knowledge entries to inform user profiling")
        
        if knowledge_entries:
            # Sort entries by priority (classification entries first) and then by similarity
            sorted_entries = sorted(knowledge_entries, key=lambda x: (-x.get("priority", 0), -x.get("similarity", 0)))
            
            # Take top entries prioritizing classification entries
            classification_entries = [entry for entry in sorted_entries if entry.get("priority", 0) > 0]
            other_entries = [entry for entry in sorted_entries if entry.get("priority", 0) == 0]
            
            # Ensure we have at least some classification entries
            selected_entries = classification_entries[:2]  # Take up to 2 classification entries
            # Fill the rest with other entries if needed
            if len(selected_entries) < 3:
                selected_entries.extend(other_entries[:3-len(selected_entries)])
                
            logger.info(f"Selected {len(selected_entries)} entries for user profiling, including {len(classification_entries[:2])} classification entries")
            
            # Save these entries to state so CoT analysis can use the same ones
            # This ensures consistency between user profiling and later analysis
            state["profiling_knowledge_entries"] = selected_entries
            logger.info("Saved profiling knowledge entries to state for consistent CoT analysis")
            
            # Use prepare_knowledge to create a cohesive knowledge context with exact terminology preservation
            knowledge_context = prepare_knowledge(
                selected_entries,
                last_user_message,
                max_chars=10000,  # Limit for user profiling
                target_classification=None,  # No specific classification for general profiling
                preserve_exact_terminology=True  # Preserve original classification names exactly
            )
            logger.info(f"Knowledge prepared for user profiling: {knowledge_context}")
        else:
            knowledge_context = ""
            logger.info("No knowledge entries found for user profiling")
        
        # Generate the user portrait using the knowledge context
        # Pass both the raw entries and the processed knowledge context
        return await create_user_portrait(conversation_context, last_user_message, knowledge_entries, knowledge_context)
        
    except Exception as e:
        logger.error(f"Error in user portrait creation: {str(e)}")
        
        # Return a very simple fallback
        return {
            "portrait": f"User communicating in {'Vietnamese' if detect_vietnamese_language(last_user_message) else 'English'}, asking about: {last_user_message[:100]}...",
            "method": "error_fallback",
            "error": str(e)
        }


async def create_user_portrait(conversation_context: str, last_user_message: str, knowledge: List[Dict], knowledge_context: str) -> Dict:
    """
    Creates a rich, descriptive textual portrait of the user based on conversation and knowledge.
    
    Args:
        conversation_context: The full conversation history
        last_user_message: The most recent user message
        knowledge: List of knowledge entries to inform the portrait (raw entries)
        knowledge_context: The processed knowledge context from prepare_knowledge
        
    Returns:
        Dictionary with the user portrait and metadata
    """
    try:
        # Determine language for output
        is_vietnamese = detect_vietnamese_language(last_user_message)
        
        # Create a knowledge-informed portrait prompt with focus on customer classification
        prompt = f"""
        Create a rich, descriptive psychological portrait of the user based on their conversation and our knowledge base.
        
        {f'KNOWLEDGE FOR YOUR ANALYSIS (CONTAINS CUSTOMER CLASSIFICATION FRAMEWORKS):\\n{knowledge_context}' if knowledge_context else 'Use your psychological expertise to analyze this conversation.'}
        
        CONVERSATION:
        {conversation_context}
        
        CLASSIFICATION INSTRUCTIONS:
        1. FIRST: Carefully extract the EXACT classification framework from the KNOWLEDGE section
           - Identify all classification categories/segments presented in the knowledge
           - Note the exact terminology used for each classification
        2. SECOND: Extract the SPECIFIC CRITERIA that define each classification
           - Look for time thresholds, frequency metrics, behavioral indicators, qualifying conditions
           - Pay attention to examples that illustrate how to apply each classification
        3. THIRD: Apply these criteria STRICTLY to classify the user
           - Match the user's statements and behavior patterns to the classification criteria
           - Look for explicit signals in the conversation that align with specific classifications
           - Only use the "unclear" or "undetermined" classification if the user truly doesn't fit other categories
        4. Create a SINGLE COHERENT PARAGRAPH that properly categorizes the user according to our framework
        5. Respond in the SAME LANGUAGE as the user ({('Vietnamese' if is_vietnamese else 'English')})
        
        Your portrait MUST include:
        - User's psychological state and communication patterns
        - Their EXACT CLASSIFICATION from our framework in BOLD with ** markers (e.g., **Nhóm Chán Nản**, **Nhóm Tự Tin**)
        - Process state if appropriate (e.g., **Đang Tìm Hiểu** vs **Đã Quyết Định**)
        - 2-3 key needs or pain points specific to their classification
        - Recommended approach strategy based on our framework's guidance for this classification
        
        IMPORTANT:
        - Apply the classification framework EXACTLY as described in the knowledge
        - Use the classification criteria rigorously - don't make assumptions beyond the provided criteria
        - Only use the "unclear" classification type when clearly appropriate based on knowledge criteria
        - ALWAYS mark the chosen classification with ** for visibility (e.g., **[Classification]**)
        - Return ONLY a single descriptive paragraph (200-250 words)
        """
        
        # Use the LangChain ChatOpenAI interface
        messages = [
            SystemMessage(content="You are an expert psychologist with deep empathy, specializing in creating insightful portraits of people based on their communication patterns. Your strength is recognizing patterns, applying appropriate classification frameworks, and formatting JSON responses exactly as requested."),
            HumanMessage(content=prompt)
        ]
        
        # Call the model with global LLM instance
        response = await LLM.ainvoke(messages, temperature=0.1)
        
        # Extract the portrait text
        portrait_text = response.content.strip()
        
        # Basic validation - ensure we got a substantial response
        if len(portrait_text) < 50:
            logger.warning("Portrait text too short, using fallback")
            portrait_text = f"User communicating in {'Vietnamese' if detect_vietnamese_language(last_user_message) else 'English'}, asking about: {last_user_message[:100]}..."
            
        # Return the portrait in a simplified structure
        return {
            "portrait": portrait_text,
            "method": "knowledge_enhanced" if knowledge_context else "llm_only",
            "knowledge_sources": len(knowledge)
        }
            
    except Exception as e:
        logger.error(f"Error in portrait creation: {str(e)}")
        return {
            "portrait": f"User communicating in {'Vietnamese' if detect_vietnamese_language(last_user_message) else 'English'}, asking about: {last_user_message[:100]}...",
            "method": "error_fallback",
            "error": str(e)
        }


def format_user_profile_for_prompt(user_profile: Dict) -> str:
    """
    Format the user portrait for inclusion in Chain of Thought (CoT) prompts.
    Creates a well-structured section that provides valuable user insights to the model.
    
    Args:
        user_profile: The user profile dictionary containing the portrait
        
    Returns:
        Formatted user understanding section for CoT prompts
    """
    # Check if we have a portrait
    if "portrait" in user_profile and user_profile["portrait"]:
        portrait = user_profile["portrait"]
        
        # Get method metadata for context
        method = user_profile.get("method", "unknown")
        knowledge_sources = user_profile.get("knowledge_sources", 0)
        
        # Create a structured, informative section for CoT
        cot_text = "USER UNDERSTANDING:\n"
        
        # Add portrait with proper formatting
        cot_text += f"{portrait}\n"
        
        # Add a subtle separator for visual clarity
        cot_text += "\n---\n"
        
        # Add a brief metadata section for the model
        if method == "knowledge_enhanced":
            cot_text += f"Note: This user understanding was generated with {knowledge_sources} knowledge sources.\n"
        
        return cot_text
    
    # Provide a basic fallback if no portrait is available
    return "USER UNDERSTANDING: Limited information available about the user."

