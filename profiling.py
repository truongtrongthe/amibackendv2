import re
import time
import asyncio
import json
import hashlib
import traceback
from typing import List, Dict, Any, Optional
from cachetools import TTLCache
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
from profile_cache import store_profile_knowledge
import os
from profile_helper import detect_vietnamese_language


from utilities import logger
# Additional configuration to handle Unicode in JSON serialization if logs are processed further
import json
# Custom JSON encoder to prevent escaping of non-ASCII characters
class UnicodeJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        kwargs['ensure_ascii'] = False
        super().__init__(*args, **kwargs)

# If logs are serialized to JSON, use the custom encoder
def custom_json_renderer(event_dict):
    return json.dumps(event_dict, cls=UnicodeJSONEncoder)

# Configuration constants
CLASSIFICATION_TERMS = [
    "nhóm chán nản", 
    "nhóm tự tin", 
    "nhóm chưa rõ tâm lý", 
    "phân loại khách hàng", 
    "phân nhóm"
]
SIMILARITY_THRESHOLD = float(os.getenv("PROFILE_SIMILARITY_THRESHOLD", 0.28))
CLASSIFICATION_BOOST = float(os.getenv("PROFILE_CLASSIFICATION_BOOST", 0.4))
KNOWLEDGE_CACHE_TTL = int(os.getenv("KNOWLEDGE_CACHE_TTL", 3600))
PORTRAIT_CACHE_TTL = int(os.getenv("PORTRAIT_CACHE_TTL", 3600))

# Initialize caches
knowledge_cache = TTLCache(maxsize=1000, ttl=KNOWLEDGE_CACHE_TTL)
portrait_cache = TTLCache(maxsize=500, ttl=PORTRAIT_CACHE_TTL)

# Initialize the LangChain ChatOpenAI model
LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    streaming=False
)

# Prompt templates
PROMPT_TEMPLATES = {
    "user_portrait": """
Create a rich, descriptive psychological portrait of the user based on their conversation and our knowledge base.

{knowledge_section}

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
5. Respond ENTIRELY and EXCLUSIVELY in the SAME LANGUAGE as the user ({language}). This is ABSOLUTELY CRITICAL - if the user language is Vietnamese, EVERY SINGLE WORD and SENTENCE in your response MUST be in Vietnamese. Do NOT include ANY English text or explanations in your response.

Your portrait MUST include:
- User's psychological state and communication patterns
- Their EXACT CLASSIFICATION from our framework in BOLD with ** markers (e.g., **Nhóm Chán Nản**, **Nhóm Tự Tin**)
- Process state if appropriate (e.g., **Đang Tìm Hiểu** vs **Đã Quyết Định**)
- 2-3 key needs or pain points specific to their classification
- Recommended approach strategy based on our framework's guidance for this classification, explicitly referencing specific application methods (e.g., 'Giao tiếp với nhóm khách Hàng thoải mái, cởi mở') and implementation steps from the knowledge context
- Identification of missing information in the user portrait that could enhance understanding (e.g., demographic details like age, specific triggers for their condition), explicitly stating what next steps should focus on gathering

IMPORTANT:
- Apply the classification framework EXACTLY as described in the knowledge
- Use the classification criteria rigorously - don't make assumptions beyond the provided criteria
- Only use the "unclear" classification type when clearly appropriate based on knowledge criteria
- ALWAYS mark the chosen classification with ** for visibility (e.g., **[Classification]**)
- Return ONLY a single descriptive paragraph (200-250 words)
- When crafting the recommended approach, directly reference and incorporate details from the 'What' and 'How' sections of application methods and the step-by-step implementation guide provided in the knowledge context
- Proactively suggest areas of missing information to guide next steps in building a more complete user profile
- STRICTLY adhere to the user's language ({language}) for ALL text in the response. ABSOLUTELY NO text should be in any other language, including English explanations or translations.
"""
}

async def build_user_profile(conversation_context: str, last_user_message: str, graph_version_id: str = "", state: Optional[Dict] = None) -> Dict:
    """
    Build a user profile as a descriptive portrait string.
    Uses a knowledge-enhanced approach that retrieves relevant knowledge about user analysis,
    then prompts an LLM to create a comprehensive description of the user.
    
    Args:
        conversation_context: The full conversation history
        last_user_message: The most recent user message
        graph_version_id: The version ID of the knowledge graph to query (optional)
        state: Optional state dictionary to store profiling knowledge entries
        
    Returns:
        Dictionary with the user portrait as a string and metadata
    """
    try:
        # Use provided state or create a temporary one
        local_state = state if state is not None else {}
        
        # Detect language using detect_vietnamese_language for better Vietnamese detection
        is_vietnamese = detect_vietnamese_language(last_user_message)
        language_name = "Vietnamese" if is_vietnamese else "English"
        
        # Skip knowledge retrieval if no graph_version_id provided
        if not graph_version_id:
            logger.info(f"No graph version ID provided, proceeding with LLM-only profiling: {language_name}")
            return await create_user_portrait(conversation_context, last_user_message, [], "", language_name)
        
        # Cache key for knowledge
        knowledge_cache_key = f"{graph_version_id}:{hashlib.md5(last_user_message.encode()).hexdigest()}"
        if knowledge_cache_key in knowledge_cache:
            knowledge_entries = knowledge_cache[knowledge_cache_key]
            logger.info(f"Using cached knowledge entries: {knowledge_cache_key}")
        else:
            # Prepare broader queries to find relevant analysis methods
            understanding_queries = [
                "how to analyze user needs",
                "user profiling techniques",
                "how to understand user psychology",
                "user communication patterns",
                "các phương pháp phân tích nhu cầu người dùng",
                "cách hiểu tâm lý người dùng",
                "kỹ thuật phân tích người dùng",
                "cách xây dựng chân dung người dùng"
            ]
            
            # Select language-specific queries
            queries_to_use = understanding_queries[4:8] if is_vietnamese else understanding_queries[0:4]
            queries_to_use.extend(understanding_queries[0:2] if is_vietnamese else understanding_queries[4:6])
            
            # Prioritize classification knowledge
            classification_queries = [
                "phân loại khách hàng",
                "nhóm tâm lý khách hàng",
                "classification of customers"
            ]
            
            combined_queries = classification_queries + queries_to_use
            
            # Ensure brain is loaded
            brain_loaded = await ensure_brain_loaded(graph_version_id)
            if not brain_loaded:
                logger.warning("Brain loading failed, proceeding with LLM-only profiling")
                return await create_user_portrait(conversation_context, last_user_message, [], "", language_name)
            
            # Get brain instance
            brain = get_brain()
            knowledge_entries = []
            
            logger.info("Fetching knowledge on user analysis techniques")
            for query in combined_queries:
                try:
                    results = await brain.get_similar_vectors_by_text(query, top_k=5)
                    for vector_id, _, metadata, similarity in results:
                        if similarity < SIMILARITY_THRESHOLD:
                            continue
                        
                        if any(entry.get("id") == vector_id for entry in knowledge_entries):
                            logger.info(f"Skipping duplicate knowledge entry: {vector_id}")
                            continue
                        
                        raw_text = metadata.get("raw", "")
                        boost = CLASSIFICATION_BOOST if any(term in raw_text.lower() for term in CLASSIFICATION_TERMS) else 0.0
                        if boost:
                            logger.info(f"Boosting entry for classification terms: {vector_id}")
                        
                        knowledge_entries.append({
                            "id": vector_id,
                            "query": query,
                            "raw": raw_text,
                            "similarity": float(similarity) + boost,
                            "priority": 1 if boost > 0 else 0
                        })
                except Exception as e:
                    logger.warning(f"Error retrieving knowledge: {query}, {str(e)}")
            
            knowledge_cache[knowledge_cache_key] = knowledge_entries
            logger.info(f"Cached knowledge entries: {knowledge_cache_key}, {len(knowledge_entries)}")
        
        # Process knowledge entries
        if knowledge_entries:
            sorted_entries = sorted(knowledge_entries, key=lambda x: (-x.get("priority", 0), -x.get("similarity", 0)))
            selected_entries = sorted_entries[:5]  # Take top 3 entries
            logger.info(f"Selected knowledge entries for profiling: {len(selected_entries)}")
            
            # Store in state and profile_cache
            local_state["profiling_knowledge_entries"] = selected_entries
            store_profile_knowledge(selected_entries)
            logger.info(f"Stored profiling knowledge entries: {state is not None}, {True}")
            
            knowledge_context = prepare_knowledge(
                selected_entries,
                last_user_message,
                max_chars=10000,
                target_classification=None,
                preserve_exact_terminology=True
            )
            logger.info(f"Knowledge For Profiling: {knowledge_context}")
        else:
            knowledge_context = ""
            logger.info(f"No knowledge entries found for user profiling: {language_name}")
        
        return await create_user_portrait(conversation_context, last_user_message, knowledge_entries, knowledge_context, language_name)
        
    except Exception as e:
        logger.error(f"Error in user portrait creation: {str(e)}", traceback=traceback.format_exc())
        return {
            "portrait": f"User communicating in {language_name}, asking about: {last_user_message[:100]}...",
            "method": "error_fallback",
            "error": str(e)
        }

async def create_user_portrait(conversation_context: str, last_user_message: str, knowledge: List[Dict], knowledge_context: str, language: str) -> Dict:
    """
    Creates a rich, descriptive textual portrait of the user based on conversation and knowledge.
    
    Args:
        conversation_context: The full conversation history
        last_user_message: The most recent user message
        knowledge: List of knowledge entries to inform the portrait
        knowledge_context: The processed knowledge context
        language: The detected language name (e.g., "Vietnamese", "English")
        
    Returns:
        Dictionary with the user portrait and metadata
    """
    try:
        # Cache key for portrait
        cache_key = hashlib.md5((conversation_context + knowledge_context).encode()).hexdigest()
        if cache_key in portrait_cache:
            logger.info(f"Using cached user portrait: {cache_key}")
            return portrait_cache[cache_key]
        
        # Create knowledge section for prompt
        knowledge_section = f"KNOWLEDGE FOR YOUR ANALYSIS (CONTAINS CUSTOMER CLASSIFICATION FRAMEWORKS):\n{knowledge_context}" if knowledge_context else "Use your psychological expertise to analyze this conversation."
        
        # Create prompt
        try:
            prompt = PROMPT_TEMPLATES["user_portrait"].format(
                knowledge_section=knowledge_section,
                conversation_context=conversation_context,
                language=language
            )
        except KeyError as ke:
            logger.error(f"KeyError in prompt formatting: {str(ke)}, template: user_portrait")
            raise
        
        # Use LangChain interface
        messages = [
            SystemMessage(content="You are an expert psychologist specializing in creating insightful user portraits based on communication patterns. Apply classification frameworks rigorously and format responses as plain text paragraphs."),
            HumanMessage(content=prompt)
        ]
        
        # Call LLM
        response = await LLM.ainvoke(messages, temperature=0.1)
        portrait_text = response.content.strip()
        
        # Remove any unwanted JSON or code block formatting if present
        portrait_text = portrait_text.replace('```json', '').replace('```', '').strip()
        if portrait_text.startswith('{') and portrait_text.endswith('}'):
            # Attempt to extract the main content if it's JSON formatted
            try:
                import json
                parsed = json.loads(portrait_text)
                if 'psychological_portrait' in parsed:
                    portrait_text = parsed['psychological_portrait']
            except:
                pass  # If parsing fails, keep the original text
        
        # Validate response
        if len(portrait_text) < 50:
            logger.warning("Portrait text too short, using fallback")
            portrait_text = f"User communicating in {language}, asking about: {last_user_message[:100]}..."
        
        # Create result
        result = {
            "portrait": portrait_text,
            "method": "knowledge_enhanced" if knowledge_context else "llm_only",
            "knowledge_sources": len(knowledge)
        }
        
        # Cache result
        portrait_cache[cache_key] = result
        logger.info(f"Cached user portrait: {cache_key}")
        logger.info(f"Generated user portrait: {portrait_text[:200]}...")
        
        return result
            
    except Exception as e:
        logger.error(f"Error in portrait creation: {str(e)}", traceback=traceback.format_exc())
        return {
            "portrait": f"User communicating in {language}, asking about: {last_user_message[:100]}...",
            "method": "error_fallback",
            "error": str(e)
        }

def format_user_profile_for_prompt(user_profile: Dict) -> str:
    """
    Format the user portrait for inclusion in Chain of Thought (CoT) prompts.
    
    Args:
        user_profile: The user profile dictionary containing the portrait
        
    Returns:
        Formatted user understanding section for CoT prompts
    """
    if "portrait" in user_profile and user_profile["portrait"]:
        portrait = user_profile["portrait"]
        method = user_profile.get("method", "unknown")
        knowledge_sources = user_profile.get("knowledge_sources", 0)
        
        cot_text = "USER UNDERSTANDING:\n"
        cot_text += f"{portrait}\n"
        cot_text += "\n---\n"
        if method == "knowledge_enhanced":
            cot_text += f"Note: This user understanding was generated with {knowledge_sources} knowledge sources.\n"
        
        return cot_text
    
    return "USER UNDERSTANDING: Limited information available about the user."