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
    ensure_brain_loaded
)
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

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
            return await create_user_portrait(conversation_context, last_user_message, [])
        
        # Prepare the exact queries specified for knowledge retrieval
        understanding_queries = [
            "how to analyze user",
            "how to classify user",
            "how to understand emotional and psychological status of user",
            "làm thế nào để nhận diện chân dung người dùng",
            "làm thế nào để phân loại người dùng",
            "làm thế nào để hiểu tâm lý và tình trạng tâm lý của người dùng"
        ]
        
        # Ensure brain is loaded
        brain_loaded = await ensure_brain_loaded(graph_version_id)
        if not brain_loaded:
            logger.warning("Brain loading failed, proceeding with LLM-only profiling")
            return await create_user_portrait(conversation_context, last_user_message, [])
        
        # Get brain instance
        brain = get_brain()
        structured_knowledge = []
        
        # Retrieve knowledge using the exact 3 queries
        logger.info("Fetching knowledge on user analysis techniques")
        
        for query in understanding_queries:
            try:
                # Fetch knowledge for this query
                results = await brain.get_similar_vectors_by_text(query, top_k=1)
                
                for vector_id, vector, metadata, similarity in results:
                    if similarity < 0.303:  # Relevance threshold
                        continue
                    
                    # Get the raw text from the results
                    raw_text = metadata.get("raw", "")
                    
                    # Extract structured data using the helper function
                    structured_data = extract_structured_data_from_raw(raw_text)
                    
                    # Create an enriched knowledge entry with both raw and structured data
                    knowledge_entry = {
                        "id": vector_id,
                        "query": query,
                        "structured": structured_data,
                        "raw": raw_text,
                        "similarity": float(similarity)
                    }
                    
                    structured_knowledge.append(knowledge_entry)
            except Exception as e:
                logger.warning(f"Error retrieving knowledge for query '{query}': {str(e)}")
        
        # Generate the user portrait using the structured knowledge
        logger.info(f"Retrieved {len(structured_knowledge)} structured knowledge entries to inform user profiling")
        return await create_user_portrait(conversation_context, last_user_message, structured_knowledge)
        
    except Exception as e:
        logger.error(f"Error in user portrait creation: {str(e)}")
        
        # Return a very simple fallback
        return {
            "portrait": f"User communicating in {'Vietnamese' if detect_vietnamese_language(last_user_message) else 'English'}, asking about: {last_user_message[:100]}...",
            "method": "error_fallback",
            "error": str(e)
        }


async def create_user_portrait(conversation_context: str, last_user_message: str, knowledge: List[Dict]) -> Dict:
    """
    Creates a rich, descriptive textual portrait of the user based on conversation and knowledge.
    
    Args:
        conversation_context: The full conversation history
        last_user_message: The most recent user message
        knowledge: List of knowledge entries to inform the portrait
        
    Returns:
        Dictionary with the user portrait and metadata
    """
    try:
        # Format knowledge for the prompt
        knowledge_context = ""
        knowledge_classification_hints = []
        
        if knowledge and len(knowledge) > 0:
            knowledge_context = "Use the following knowledge to guide your analysis:\n\n"
            
            for entry in knowledge:
                # Get the structured data if available
                structured = entry.get("structured", {})
                title = structured.get("title", "")
                content = structured.get("content", "")
                description = structured.get("description", "")
                takeaways = structured.get("takeaways", "")
                application_method = structured.get("application_method", "")
                
                # Add debug logging for application_method structure
                if application_method:
                    logger.info(f"Application method type: {type(application_method).__name__}")
                    if isinstance(application_method, dict):
                        logger.info(f"Application method keys: {list(application_method.keys())}")
                        if "title" in application_method:
                            logger.info(f"Application method title: {application_method['title']}")
                            
                # If we have an application method, capture it
                if application_method:
                    # Check if application_method is a dictionary and extract title
                    if isinstance(application_method, dict) and "title" in application_method:
                        knowledge_classification_hints.append(application_method["title"])
                    # If it's a string, use it directly
                    elif isinstance(application_method, str):
                        knowledge_classification_hints.append(application_method)
                    # If it's some other structure, convert to string safely
                    else:
                        try:
                            knowledge_classification_hints.append(str(application_method))
                        except:
                            # If conversion fails, skip it
                            logger.warning(f"Could not process application method for classification hints: {type(application_method)}")
                
                # Format the structured data for the prompt
                knowledge_context += f"--- USER ANALYSIS GUIDANCE ---\n"
                if title:
                    knowledge_context += f"Topic: {title}\n"
                if description:
                    knowledge_context += f"Description: {description}\n"
                if content:
                    knowledge_context += f"Content: {content[:1000]}...\n" if len(content) > 1000 else f"Content: {content}\n"
                if takeaways:
                    knowledge_context += f"Key Takeaways: {takeaways}\n"
                if application_method:
                    # Handle application_method based on its type
                    if isinstance(application_method, dict):
                        # Format dictionary application method
                        app_title = application_method.get("title", "Application Method")
                        knowledge_context += f"Application Method: {app_title}\n"
                        
                        # Include steps if available
                        steps = application_method.get("steps", [])
                        if steps:
                            knowledge_context += "Steps:\n"
                            for i, step in enumerate(steps):
                                step_title = step.get("title", f"Step {i+1}")
                                knowledge_context += f"- {step_title}\n"
                    else:
                        # Simple string case
                        knowledge_context += f"Application Method: {application_method}\n"
                
                # If no structured data, fall back to raw text
                if not (title or content or description or takeaways or application_method):
                    raw_text = entry.get("raw", "")
                    clean_text = raw_text.replace("\n\n", " ").replace("\n", " ")[:2000]
                    knowledge_context += f"Guidance: {clean_text}\n"
                
                knowledge_context += "\n"
        
        # Add special instruction for respecting the knowledge context
        knowledge_instruction = ""
        if knowledge:
            knowledge_instruction = (
                "\n\nIMPORTANT: The knowledge provided contains specific classification methods "
                "and approaches to categorize and understand users. Pay close attention to these "
                "frameworks and explicitly apply them in your analysis. For example, if the knowledge "
                "mentions specific user categories (like 'Confident Group' or 'Discouraged Group'), "
                "classification criteria, behavioral patterns, or communication strategies, use these "
                "exact frameworks to classify this user."
            )
            
            if knowledge_classification_hints:
                knowledge_instruction += f"\n\nSpecifically, consider these classification methods mentioned in the knowledge: {', '.join(knowledge_classification_hints)}"
            
        logger.info(f"Knowledge context before building PORTRAIT: {knowledge_context}")
        
        # Create a prompt focused on generating a descriptive portrait
        prompt = f"""
        Analyze this conversation to create a rich, descriptive portrait of the user using language of the user.
        
        {knowledge_context}{knowledge_instruction}
        
        CONVERSATION:
        {conversation_context}
        
        Create a comprehensive psychological portrait of this user that describes:
        
        1. Personal Information: Extract any personal details like name, age, location, occupation, gender, relationship status, family status, etc. Infer these details from context if not explicitly stated.
        
        2. Identity: Who they are (language, expertise level, self-perception, self-image, social identity)
        
        3. Psychological State: Their current emotions, concerns, confidence level, anxiety level, stress factors
        
        4. Desires and Motivations:
           - Explicit needs: What they directly state they want
           - Implicit/hidden needs: What they might want but aren't directly expressing
           - Core motivations driving their behavior
           - Barriers preventing them from achieving their goals
           - What they might be avoiding or afraid to address directly
        
        5. Communication Style: How they express themselves (style, directness, formality, vocabulary level, precision)
        
        6. Approach Strategy: How we should engage with them (tone, communication strategy, level of directness, emotional support needed)
        
        7. Classification: EXPLICITLY classify this user according to the frameworks mentioned in the knowledge. If the knowledge mentions specific user categories, classification criteria, or behavioral patterns, directly state which category this user belongs to and why. Use the exact terminology from the knowledge. PUT THIS CLASSIFICATION IN BOLD by surrounding it with ** symbols in markdown format (e.g., **Discouraged Group**).
        
        IMPORTANT: Respond in the SAME LANGUAGE that the user is using in the conversation. If they write in Vietnamese, your portrait must be in Vietnamese. If they write in English, respond in English.
        
        Format your response as a cohesive, flowing paragraph that paints a complete picture of this person, 
        not as a list of attributes. Make your portrait nuanced, insightful, and specific to this person.
        Pay particular attention to uncovering their hidden desires and unstated needs.
        
        USER PORTRAIT:
        """
        
        # Call the LLM to generate the portrait
        try:
            # Use the LangChain ChatOpenAI interface
            messages = [
                SystemMessage(content="You are an expert psychologist with deep empathy, specializing in creating insightful portraits of people based on their communication. Your strength is recognizing patterns and applying appropriate classification frameworks from provided knowledge."),
                HumanMessage(content=prompt)
            ]
            
            # Call the model
            response = await LLM.ainvoke(messages)
            
            # Extract the portrait text
            portrait_text = response.content.strip()
            
            # Basic validation - ensure we got a substantial response
            if len(portrait_text) < 50:
                logger.warning("Portrait text too short, using fallback")
                portrait_text = f"User communicating in {'Vietnamese' if detect_vietnamese_language(last_user_message) else 'English'}, asking about: {last_user_message[:100]}..."
                
            # Return the portrait in a simplified structure
            return {
                "portrait": portrait_text,
                "method": "knowledge_enhanced" if knowledge else "llm_only",
                "knowledge_sources": len(knowledge)
            }
                
        except Exception as llm_error:
            logger.error(f"Error generating portrait with LLM: {str(llm_error)}")
            return {
                "portrait": f"User communicating in {'Vietnamese' if detect_vietnamese_language(last_user_message) else 'English'}, asking about: {last_user_message[:100]}...",
                "method": "error_fallback",
                "error": str(llm_error)
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

