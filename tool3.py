import json
import re
import time
import asyncio
import traceback
import hashlib
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from pydantic import BaseModel
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from cachetools import TTLCache
import os

from utilities import logger as default_logger
from brain_singleton import get_brain
from tool_helpers import (
    detect_language,
    ensure_brain_loaded
)

# Initialize structured logging
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger()

# Configuration
RESPONSE_TIMEOUT = float(os.getenv("RESPONSE_TIMEOUT", 7))
MAX_SOCKET_SIZE = int(os.getenv("MAX_SOCKET_SIZE", 15000))
KNOWLEDGE_CACHE_TTL = int(os.getenv("KNOWLEDGE_CACHE_TTL", 3600))
RESPONSE_CACHE_TTL = int(os.getenv("RESPONSE_CACHE_TTL", 3600))
PROFILE_CACHE_TTL = int(os.getenv("PROFILE_CACHE_TTL", 3600))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 30))

# LLM instances
LLM = ChatOpenAI(model="gpt-4o", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

# Caches
knowledge_cache = TTLCache(maxsize=1000, ttl=KNOWLEDGE_CACHE_TTL)
response_cache = TTLCache(maxsize=500, ttl=RESPONSE_CACHE_TTL)
portrait_cache = TTLCache(maxsize=500, ttl=PROFILE_CACHE_TTL)
profile_knowledge_cache = TTLCache(maxsize=1000, ttl=PROFILE_CACHE_TTL)

# Performance tracking
PERF_METRICS = {}

# Access brain singleton
brain = get_brain()

# Configuration constants
CLASSIFICATION_TERMS = [
    "nhóm chán nản", 
    "nhóm tự tin", 
    "nhóm chưa rõ tâm lý", 
    "phân loại khách hàng", 
    "phân nhóm"
]

class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools = {}
        self.tool_schemas = {}
    
    def register_tool(self, name: str, handler, schema: Dict):
        """Register a tool with its handler and schema."""
        self.tools[name] = handler
        self.tool_schemas[name] = schema
        logger.info("Registered tool", tool_name=name)
    
    def get_tool_schemas(self) -> List[Dict]:
        """Get all tool schemas for LLM tool calling."""
        return list(self.tool_schemas.values())
    
    def get_tool_handler(self, name: str):
        """Get a tool handler by name."""
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        return self.tools[name]
    
    def get_tool_names(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self.tools.keys())

# Tool schemas
class CotKnowledgeAnalysisParams(BaseModel):
    conversation_context: str
    graph_version_id: str
    thread_id: Optional[str] = None
    state: Optional[Dict] = {}

COT_KNOWLEDGE_ANALYSIS_SCHEMA = {
    "name": "cot_knowledge_analysis_tool",
    "description": "Combine knowledge retrieval, user profiling, analysis, and response generation using CoT",
    "parameters": CotKnowledgeAnalysisParams.model_json_schema()
}

# Prompt templates
PROMPT_TEMPLATES = {
    "classification": """
Analyze the conversation to classify the user based on the provided knowledge framework.

KNOWLEDGE:
{knowledge_context}

CONVERSATION:
{conversation_context}

TASK:
1. Identify the EXACT classification categories from the knowledge (e.g., Nhóm Chán Nản, Nhóm Tự Tin).
2. Extract specific criteria for each category (e.g., behavioral indicators, frequency metrics).
3. Classify the user into ONE category based on the conversation, using strict criteria.
4. Respond in {language} with a JSON object:
   ```json
   {{
     "classification": "<category>",
     "criteria_matched": ["<criterion1>", "<criterion2>", ...]
   }}
   ```
""",
    "profile_analysis": """
Analyze the user profile and conversation to determine next actions.

USER PROFILE:
{profile_summary}

CONVERSATION:
{conversation_context}

KNOWLEDGE:
{knowledge_context}

TASK:
1. Interpret the profile using the knowledge to understand user needs and psychological state.
2. Analyze the conversation to identify immediate needs or intents.
3. Determine 2-3 specific next actions (e.g., reassure, persuade, provide information).
4. Respond in {language} with a JSON object:
   ```json
   {{
     "needs": ["<need1>", "<need2>", ...],
     "next_actions": ["<action1>", "<action2>", ...]
   }}
   ```
""",
    "response_generation": """
Generate a persuasive response based on the user profile, analysis, and knowledge.

USER PROFILE:
{profile_summary}

CONVERSATION:
{conversation_context}

ANALYSIS:
{analysis}

NEXT ACTIONS:
{next_actions}

KNOWLEDGE:
{knowledge_context}

PERSONALITY:
{personality_instructions}

TASK:
Create a concise, persuasive response in {language} that:
1. Addresses the user's needs compellingly
2. Sounds like a friendly human
3. Uses cultural terms (e.g., "Chị ơi" in Vietnamese, "Hi there" in English)
4. Keeps content brief (2-3 paragraphs)
5. Includes a natural call-to-action

CONVERSATIONAL STYLE GUIDE:
- In Vietnamese: Use "Chị ơi/Anh ơi" and particles like "nhé", "ạ", "nha"
- In English: Use friendly openings and closings
- Keep sentences short, use contractions, avoid repetition
- Adapt the persuasive script to be conversational while maintaining key points
""",
    "knowledge_sniff": """
Extract relevant knowledge segments from raw data based on the context.

RAW KNOWLEDGE:
{raw_knowledge}

CONTEXT:
{context}

TASK:
1. Identify segments relevant to the context (e.g., classification criteria, application methods).
2. Extract specific sections (e.g., titled sections, numbered steps) that match the context.
3. Preserve exact terminology and structure.
4. Respond in {language} with a JSON object:
   ```json
   {{
     "relevant_segments": [
       {{"title": "<title>", "content": "<content>"}},
       ...
     ]
   }}
   ```
"""
}

# Initialize tool registry
tool_registry = ToolRegistry()

def store_profile_knowledge(knowledge_entries: List[Dict], cache_key: Optional[str] = None) -> None:
    """
    Store profile-related knowledge entries in the cache.
    
    Args:
        knowledge_entries: List of knowledge entry dictionaries
        cache_key: Optional cache key; if None, a default key is generated
    """
    try:
        if not knowledge_entries:
            logger.info("No knowledge entries to store")
            return
        
        if cache_key is None:
            cache_key = hashlib.md5(json.dumps(knowledge_entries, sort_keys=True).encode()).hexdigest()
        
        profile_knowledge_cache[cache_key] = knowledge_entries
        logger.info("Stored profile knowledge", cache_key=cache_key, entries=len(knowledge_entries))
    except Exception as e:
        logger.error("Error storing profile knowledge", error=str(e), traceback=traceback.format_exc())

def get_profile_knowledge(cache_key: str) -> List[Dict]:
    """
    Retrieve profile-related knowledge entries from the cache.
    
    Args:
        cache_key: The cache key to retrieve entries
        
    Returns:
        List of knowledge entry dictionaries, or empty list if not found
    """
    try:
        knowledge_entries = profile_knowledge_cache.get(cache_key, [])
        logger.info("Retrieved profile knowledge", cache_key=cache_key, entries=len(knowledge_entries))
        return knowledge_entries
    except Exception as e:
        logger.error("Error retrieving profile knowledge", error=str(e), traceback=traceback.format_exc())
        return []

def preprocess_knowledge(raw_text: str, context: str) -> List[Dict]:
    """
    Preprocess raw knowledge to extract relevant segments using regex and keyword matching.
    
    Args:
        raw_text: The raw knowledge text
        context: The context (e.g., user query, classification) to guide extraction
        
    Returns:
        List of dictionaries with titles and content segments
    """
    segments = []
    context_terms = [term.lower() for term in context.split() if len(term) > 3]
    
    # Extract titled sections (e.g., Application Method, Steps)
    section_matches = re.finditer(r'(Application Method|Title|Steps):[\s\n]*(.*?)(?=(?:Application Method|Title|Steps):|$)', raw_text, re.DOTALL | re.IGNORECASE)
    for match in section_matches:
        title = match.group(1).strip()
        content = match.group(2).strip()
        relevance = sum(1 for term in context_terms if term in content.lower())
        
        if relevance > 0 or any(term in content.lower() for term in CLASSIFICATION_TERMS):
            segments.append({"title": title, "content": content, "relevance": relevance})
    
    # Extract numbered steps
    step_matches = re.finditer(r'(\d+\.\s*\*\*[^\*]+\*\*.*?)(?=\d+\.\s*\*\*|$)', raw_text, re.DOTALL)
    for match in step_matches:
        content = match.group(1).strip()
        relevance = sum(1 for term in context_terms if term in content.lower())
        if relevance > 0:
            segments.append({"title": "Step", "content": content, "relevance": relevance})
    
    # Sort by relevance
    return sorted(segments, key=lambda x: x["relevance"], reverse=True)[:3]

async def sniff_knowledge(raw_knowledge: str, context: str, language: str) -> List[Dict]:
    """
    Use LLM to extract relevant knowledge segments dynamically.
    
    Args:
        raw_knowledge: The raw knowledge text
        context: The context to guide extraction
        language: The language for output
        
    Returns:
        List of dictionaries with relevant segments
    """
    cache_key = hashlib.md5((raw_knowledge + context).encode()).hexdigest()
    if cache_key in knowledge_cache:
        logger.info("Using cached sniffed knowledge", cache_key=cache_key)
        return knowledge_cache[cache_key]
    
    prompt = PROMPT_TEMPLATES["knowledge_sniff"].format(
        raw_knowledge=raw_knowledge[:5000],  # Limit input size
        context=context,
        language=language
    )
    
    try:
        response = await LLM.ainvoke(prompt, temperature=0.1)
        segments = json.loads(response.content)["relevant_segments"]
        knowledge_cache[cache_key] = segments
        logger.info("Sniffed and cached knowledge segments", cache_key=cache_key, segments=len(segments))
        return segments
    except Exception as e:
        logger.error("Knowledge sniffing error", error=str(e))
        return []

async def prepare_knowledge(knowledge_entries: List[Dict], user_query: str, max_chars: int = 10000, target_classification: str = None, preserve_exact_terminology: bool = False, language: str = "Vietnamese") -> str:
    """
    Prepare knowledge context by processing raw knowledge text and creating a unified instruction set.
    
    Args:
        knowledge_entries: List of knowledge entry dictionaries
        user_query: Original user query for prioritization
        max_chars: Maximum characters allowed in the context
        target_classification: Optional classification to filter segments
        preserve_exact_terminology: If True, preserves exact classification terms
        language: The language for formatting (default: "Vietnamese")
    
    Returns:
        Formatted knowledge context string
    """
    if not knowledge_entries:
        return ""
    
    try:
        # Preprocess entries to extract relevant segments
        all_segments = []
        for entry in knowledge_entries[:5]:  # Limit to top 5 entries
            raw_text = entry.get("raw", "")
            entry_id = entry.get("id", "unknown")
            if not raw_text:
                continue
            
            segments = preprocess_knowledge(raw_text, user_query)
            if not segments:  # If preprocessing finds nothing, try LLM sniffing
                segments = await sniff_knowledge(raw_text, user_query, language)
            
            all_segments.extend(segments)
        
        # Sort and deduplicate segments
        unique_segments = []
        seen = set()
        for segment in sorted(all_segments, key=lambda x: x.get("relevance", 0), reverse=True):
            segment_key = hashlib.md5(segment["content"].encode()).hexdigest()
            if segment_key not in seen:
                unique_segments.append(segment)
                seen.add(segment_key)
        
        # Format segments
        knowledge_input = ""
        for i, segment in enumerate(unique_segments[:3]):  # Limit to top 3
            title = segment["title"]
            content = segment["content"]
            if language == "Vietnamese":
                knowledge_input += f"MỤC KIẾN THỨC {i+1}:\nTiêu đề: {title}\nNội dung:\n{content}\n\n----\n\n"
            else:
                knowledge_input += f"KNOWLEDGE ENTRY {i+1}:\nTitle: {title}\nContent:\n{content}\n\n----\n\n"
        
        # Synthesize with LLM if needed
        if len(unique_segments) > 1:  # Only synthesize if multiple segments
            llm = ChatOpenAI(model="gpt-4o", temperature=0.05)
            lang_instruction = "Respond ENTIRELY in Vietnamese." if language == "Vietnamese" else "Respond ENTIRELY in English."
            preservation_instruction = """
            CRITICAL INSTRUCTION: Preserve EXACT terminology of classification categories.
            - Copy classification names verbatim (e.g., "Nhóm Chán Nản").
            - Do not translate or reinterpret terms.
            """ if preserve_exact_terminology else ""
            
            heading = "## Kiến Thức Được Tìm Thấy" if language == "Vietnamese" else "## Knowledge Found"
            synthesis_prompt = f"""
            Synthesize these knowledge segments into a concise narrative.

            LANGUAGE INSTRUCTION: {lang_instruction}
            USER QUERY: {user_query}
            {f"USER CLASSIFICATION: {target_classification}" if target_classification else ""}
            {preservation_instruction}

            {knowledge_input}

            TASK:
            Create a brief narrative (50-75 words) summarizing the segments, followed by the segments themselves.
            FORMAT:
            {heading}
            [Narrative]

            {knowledge_input}
            """
            
            response = await llm.ainvoke(synthesis_prompt, temperature=0.05)
            synthesized_knowledge = response.content if hasattr(response, 'content') else str(response)
            
            if len(synthesized_knowledge) > max_chars:
                synthesized_knowledge = synthesized_knowledge[:max_chars-3] + "..."
            
            return synthesized_knowledge
        
        return knowledge_input[:max_chars] if knowledge_input else ""
    
    except Exception as e:
        logger.error("Error in prepare_knowledge", error=str(e), traceback=traceback.format_exc())
        fallback_content = ""
        for entry in knowledge_entries[:2]:
            raw_text = entry.get("raw", "")
            if raw_text:
                fallback_content += f"--- KNOWLEDGE ---\n{raw_text[:1000]}\n\n"
        
        if len(fallback_content) > max_chars:
            fallback_content = fallback_content[:max_chars-3] + "..."
        return fallback_content

async def build_user_profile(conversation_context: str, last_user_message: str, graph_version_id: str = "", state: Optional[Dict] = None) -> Dict:
    """
    Build a user profile as a descriptive portrait string.
    
    Args:
        conversation_context: The full conversation history
        last_user_message: The most recent user message
        graph_version_id: The version ID of the knowledge graph to query
        state: Optional state dictionary to store profiling knowledge entries
        
    Returns:
        Dictionary with the user portrait and metadata
    """
    local_state = state if state is not None else {}
    try:
        language = detect_language(last_user_message)
        language_name = "Vietnamese" if language == "vietnamese" else "English"
        
        if not graph_version_id:
            logger.info("No graph version ID, using LLM-only profiling")
            return await create_user_portrait(conversation_context, last_user_message, [], "", language_name)
        
        # Iterative knowledge sniffing
        classification_queries = [
            "phân loại khách hàng",
            "nhóm tâm lý khách hàng",
            "classification of customers"
        ]
        knowledge_cache_key = f"{graph_version_id}:profile:{hashlib.md5(last_user_message.encode()).hexdigest()}"
        
        # First, try classification knowledge
        classification_entries = await fetch_knowledge(classification_queries, graph_version_id, f"{knowledge_cache_key}:classification")
        classification_context = await prepare_knowledge(
            classification_entries,
            last_user_message,
            max_chars=5000,
            language=language_name,
            preserve_exact_terminology=True
        )
        
        # Classify user to guide further knowledge retrieval
        classification_prompt = PROMPT_TEMPLATES["classification"].format(
            knowledge_context=classification_context or "No classification knowledge available.",
            conversation_context=conversation_context,
            language=language_name
        )
        try:
            response = await LLM.ainvoke(classification_prompt)
            classification_data = json.loads(response.content)
            logger.info("User classified", classification=classification_data["classification"])
        except Exception as e:
            logger.error("Classification error", error=str(e))
            classification_data = {"classification": "Nhóm Chưa Rõ Tâm Lý", "criteria_matched": []}
        
        # Fetch profiling knowledge based on classification
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
        queries_to_use = understanding_queries[4:8] if language == "vietnamese" else understanding_queries[0:4]
        queries_to_use.extend(understanding_queries[0:2] if language == "vietnamese" else understanding_queries[4:6])
        combined_queries = queries_to_use + [f"phân tích {classification_data['classification']}"]
        
        knowledge_entries = await fetch_knowledge(combined_queries, graph_version_id, knowledge_cache_key)
        if knowledge_entries:
            sorted_entries = sorted(knowledge_entries, key=lambda x: (-x.get("priority", 0), -x.get("similarity", 0)))
            selected_entries = sorted_entries[:3]
            local_state["profiling_knowledge_entries"] = selected_entries
            store_profile_knowledge(selected_entries, cache_key=knowledge_cache_key)
            logger.info("Stored profiling knowledge", in_state=state is not None, in_cache=True)
            
            knowledge_context = await prepare_knowledge(
                selected_entries,
                last_user_message,
                max_chars=10000,
                target_classification=classification_data["classification"],
                preserve_exact_terminology=True,
                language=language_name
            )
        else:
            knowledge_context = classification_context
            logger.info("No additional profiling knowledge found")
        
        profile = await create_user_portrait(conversation_context, last_user_message, knowledge_entries, knowledge_context, language_name)
        profile["classification"] = classification_data["classification"]
        profile["criteria_matched"] = classification_data["criteria_matched"]
        return profile
        
    except Exception as e:
        logger.error("Error in user portrait creation", error=str(e), traceback=traceback.format_exc())
        return {
            "portrait": f"User communicating in {language_name}, asking about: {last_user_message[:100]}...",
            "method": "error_fallback",
            "error": str(e)
        }

async def create_user_portrait(conversation_context: str, last_user_message: str, knowledge: List[Dict], knowledge_context: str, language: str) -> Dict:
    """
    Creates a rich, descriptive textual portrait of the user.
    
    Args:
        conversation_context: The full conversation history
        last_user_message: The most recent user message
        knowledge: List of knowledge entries
        knowledge_context: Processed knowledge context
        language: The language for output (e.g., "Vietnamese", "English")
        
    Returns:
        Dictionary with the user portrait and metadata
    """
    try:
        cache_key = hashlib.md5((conversation_context + knowledge_context).encode()).hexdigest()
        if cache_key in portrait_cache:
            logger.info("Using cached user portrait", cache_key=cache_key)
            return portrait_cache[cache_key]
        
        prompt = f"""
        Create a rich, descriptive psychological portrait of the user based on their conversation and our knowledge base.
        
        {f'KNOWLEDGE FOR YOUR ANALYSIS (CONTAINS CUSTOMER CLASSIFICATION FRAMEWORKS):\n{knowledge_context}' if knowledge_context else 'Use your psychological expertise to analyze this conversation.'}
        
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
        5. Respond in the SAME LANGUAGE as specified ({language})
        
        Your portrait MUST include:
        - User's psychological state and communication patterns
        - Their EXACT CLASSIFICATION from our framework in BOLD with ** markers (e.g., **Nhóm Chán Nản**)
        - Process state if appropriate (e.g., **Đang Tìm Hiểu** vs **Đã Quyết Định**)
        - 2-3 key needs or pain points specific to their classification
        - Recommended approach strategy referencing specific application methods and implementation steps
        - Identification of missing information (e.g., age, specific triggers)
        
        IMPORTANT:
        - Apply the classification framework EXACTLY as described
        - Use rigorous criteria, no assumptions
        - Mark classification with **
        - Return a single paragraph (200-250 words)
        """
        
        messages = [
            SystemMessage(content="You are an expert psychologist specializing in creating insightful user portraits."),
            HumanMessage(content=prompt)
        ]
        
        response = await LLM.ainvoke(messages, temperature=0.1)
        portrait_text = response.content.strip()
        
        if len(portrait_text) < 50:
            logger.warning("Portrait text too short, using fallback")
            portrait_text = f"User communicating in {language}, asking about: {last_user_message[:100]}..."
        
        result = {
            "portrait": portrait_text,
            "method": "knowledge_enhanced" if knowledge_context else "llm_only",
            "knowledge_sources": len(knowledge)
        }
        
        portrait_cache[cache_key] = result
        logger.info("Cached user portrait", cache_key=cache_key)
        
        return result
            
    except Exception as e:
        logger.error("Error in portrait creation", error=str(e), traceback=traceback.format_exc())
        return {
            "portrait": f"User communicating in {language}, asking about: {last_user_message[:100]}...",
            "method": "error_fallback",
            "error": str(e)
        }

def format_user_profile_for_prompt(user_profile: Dict) -> str:
    """
    Format the user portrait for inclusion in CoT prompts.
    
    Args:
        user_profile: The user profile dictionary containing the portrait
        
    Returns:
        Formatted user understanding section
    """
    if "portrait" in user_profile and user_profile["portrait"]:
        portrait = user_profile["portrait"]
        method = user_profile.get("method", "unknown")
        knowledge_sources = user_profile.get("knowledge_sources", 0)
        
        cot_text = "USER UNDERSTANDING:\n"
        cot_text += f"{portrait}\n"
        cot_text += "\n---\n"
        if method == "knowledge_enhanced":
            cot_text += f"Note: Generated with {knowledge_sources} knowledge sources.\n"
        
        return cot_text
    
    return "USER UNDERSTANDING: Limited information available about the user."

async def fetch_knowledge(queries: List[str], graph_version_id: str, cache_key: str) -> List[Dict]:
    """Fetch knowledge from the brain for given queries."""
    if cache_key in knowledge_cache:
        logger.info("Using cached knowledge", cache_key=cache_key)
        return knowledge_cache[cache_key]
    
    knowledge_entries = []
    try:
        if await ensure_brain_loaded(graph_version_id):
            for query in queries:
                results = await brain.get_similar_vectors_by_text(query, top_k=2)
                for vector_id, _, metadata, similarity in results:
                    if any(entry.get("id") == vector_id for entry in knowledge_entries):
                        continue
                    knowledge_entries.append({
                        "id": vector_id,
                        "similarity": float(similarity),
                        "raw": metadata.get("raw", ""),
                        "query": query
                    })
        knowledge_cache[cache_key] = knowledge_entries
        logger.info("Fetched and cached knowledge", cache_key=cache_key, entries=len(knowledge_entries))
    except Exception as e:
        logger.error("Knowledge retrieval error", error=str(e))
    
    return knowledge_entries

async def understand_user(conversation_context: str, last_user_message: str, graph_version_id: str, state: Dict) -> Dict:
    """Step 1: Understand the user through profiling and classification."""
    # 1.1: Knowledge for user classification
    classification_cache_key = f"{graph_version_id}:classification:{hashlib.md5(last_user_message.encode()).hexdigest()}"
    classification_knowledge = get_profile_knowledge(classification_cache_key)
    if not classification_knowledge:
        classification_queries = [
            "phân loại khách hàng",
            "nhóm tâm lý khách hàng",
            "classification of customers"
        ]
        classification_knowledge = await fetch_knowledge(classification_queries, graph_version_id, classification_cache_key)
        store_profile_knowledge(classification_knowledge, classification_cache_key)
    
    classification_context = await prepare_knowledge(
        classification_knowledge,
        last_user_message,
        max_chars=5000,
        language="Vietnamese" if detect_language(last_user_message) == "vietnamese" else "English"
    )
    classification_prompt = PROMPT_TEMPLATES["classification"].format(
        knowledge_context=classification_context or "No classification knowledge available.",
        conversation_context=conversation_context,
        language="Vietnamese" if detect_language(last_user_message) == "vietnamese" else "English"
    )
    
    try:
        response = await LLM.ainvoke(classification_prompt)
        classification_data = json.loads(response.content)
        logger.info("User classified", classification=classification_data["classification"])
    except Exception as e:
        logger.error("Classification error", error=str(e))
        classification_data = {"classification": "Nhóm Chưa Rõ Tâm Lý", "criteria_matched": []}
    
    # 1.2: Build user profile
    profile = await build_user_profile(conversation_context, last_user_message, graph_version_id, state)
    profile["classification"] = classification_data["classification"]
    profile["criteria_matched"] = classification_data["criteria_matched"]
    state["analysis"] = state.get("analysis", {})
    state["analysis"]["user_profile"] = profile
    
    return profile

async def analyze_profile(conversation_context: str, user_profile: Dict, graph_version_id: str) -> Dict:
    """Step 2: Analyze the user profile and messages."""
    # 2.1: Knowledge for profile understanding
    analysis_cache_key = f"{graph_version_id}:analysis:{hashlib.md5(str(user_profile).encode()).hexdigest()}"
    analysis_knowledge = get_profile_knowledge(analysis_cache_key)
    if not analysis_knowledge:
        analysis_queries = [
            "hiểu tâm lý khách hàng",
            "phân tích nhu cầu người dùng",
            "understand customer psychology",
            "analyze user needs"
        ]
        analysis_knowledge = await fetch_knowledge(analysis_queries, graph_version_id, analysis_cache_key)
        store_profile_knowledge(analysis_knowledge, analysis_cache_key)
    
    analysis_context = await prepare_knowledge(
        analysis_knowledge,
        conversation_context,
        max_chars=5000,
        target_classification=user_profile.get("classification"),
        language="Vietnamese" if detect_language(conversation_context) == "vietnamese" else "English"
    )
    analysis_prompt = PROMPT_TEMPLATES["profile_analysis"].format(
        profile_summary=format_user_profile_for_prompt(user_profile),
        conversation_context=conversation_context,
        knowledge_context=analysis_context or "No analysis knowledge available.",
        language="Vietnamese" if detect_language(conversation_context) == "vietnamese" else "English"
    )
    
    try:
        response = await LLM.ainvoke(analysis_prompt)
        analysis_data = json.loads(response.content)
        logger.info("Profile analyzed", needs=analysis_data["needs"], actions=analysis_data["next_actions"])
    except Exception as e:
        logger.error("Profile analysis error", error=str(e))
        analysis_data = {"needs": ["Unknown"], "next_actions": ["Provide general information"]}
    
    return analysis_data

async def plan_and_respond(conversation_context: str, user_profile: Dict, analysis_data: Dict, graph_version_id: str, thread_id: Optional[str]) -> AsyncGenerator[str, None]:
    """Step 3: Plan next actions and generate response."""
    # 3.1: Knowledge for next actions
    action_cache_key = f"{graph_version_id}:actions:{hashlib.md5(str(analysis_data['next_actions']).encode()).hexdigest()}"
    action_knowledge = get_profile_knowledge(action_cache_key)
    if not action_knowledge:
        action_queries = [f"Cách thực hiện {action}" for action in analysis_data["next_actions"]]
        action_knowledge = await fetch_knowledge(action_queries, graph_version_id, action_cache_key)
        store_profile_knowledge(action_knowledge, action_cache_key)
    
    action_context = await prepare_knowledge(
        action_knowledge,
        conversation_context,
        max_chars=5000,
        target_classification=user_profile.get("classification"),
        language="Vietnamese" if detect_language(conversation_context) == "vietnamese" else "English"
    )
    
    # 3.2: Generate response
    last_message = conversation_context.split('\n')[-1].replace("User:", "").strip()
    language_preference = detect_language(last_message)
    prompt = PROMPT_TEMPLATES["response_generation"].format(
        profile_summary=format_user_profile_for_prompt(user_profile),
        conversation_context=conversation_context,
        analysis=json.dumps(analysis_data, ensure_ascii=False),
        next_actions="\n".join(analysis_data["next_actions"]),
        knowledge_context=action_context or "No action knowledge available.",
        personality_instructions="""
        You are a sophisticated, energetic AI:
        - Mirror user's style and formality
        - Use clear, rich vocabulary
        - Show enthusiasm and interest
        - Be persuasive and compelling
        - Adapt to cultural nuances
        """,
        language=language_preference
    )
    
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    if cache_key in response_cache:
        logger.info("Using cached response", cache_key=cache_key)
        for chunk in response_cache[cache_key]:
            yield chunk
        return
    
    full_response = []
    buffer = ""
    
    try:
        start_time = time.time()
        async for chunk in StreamLLM.astream(prompt, temperature=0.05):
            if time.time() - start_time > RESPONSE_TIMEOUT:
                logger.warning("Response generation timed out", timeout=RESPONSE_TIMEOUT)
                break
            
            content = chunk.content
            buffer += content
            full_response.append(content)
            if len(buffer) >= BUFFER_SIZE or content.endswith((".", "!", "?", "\n", " ")):
                yield buffer
                buffer = ""
        
        if buffer:
            yield buffer
            full_response.append(buffer)
        
        response_cache[cache_key] = full_response
        logger.info("Cached response", cache_key=cache_key)
        
    except Exception as e:
        logger.error("Response generation error", error=str(e), traceback=traceback.format_exc())
        if not full_response:
            full_response = [
                "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại nha." if language_preference == "vietnamese"
                else "Sorry, an error occurred. Please try again."
            ]
        for chunk in full_response:
            yield chunk

async def cot_knowledge_analysis_actions_handler(params: CotKnowledgeAnalysisParams) -> AsyncGenerator[Dict, None]:
    """Chain-of-Thought handler for user understanding, analysis, and response."""
    start_time = time.time()
    state = params.state or {}
    state.setdefault("cot_results", {})
    
    # Step 1: Understand the user
    user_profile = await understand_user(
        params.conversation_context,
        params.conversation_context.split('\n')[-1].replace("User:", "").strip(),
        params.graph_version_id,
        state
    )
    yield {
        "type": "analysis",
        "content": f"Classified user as {user_profile.get('classification', 'Unknown')}",
        "complete": False,
        "thread_id": params.thread_id,
        "status": "profiling"
    }
    
    # Step 2: Analyze profile and messages
    analysis_data = await analyze_profile(params.conversation_context, user_profile, params.graph_version_id)
    state["cot_results"]["analysis"] = analysis_data
    yield {
        "type": "analysis",
        "content": f"Identified needs: {', '.join(analysis_data['needs'])}",
        "complete": True,
        "thread_id": params.thread_id,
        "status": "analysis_complete",
        "user_profile": user_profile,
        "needs": analysis_data["needs"],
        "next_actions": analysis_data["next_actions"]
    }
    
    # Step 3: Plan and respond
    async for response_chunk in plan_and_respond(
        params.conversation_context,
        user_profile,
        analysis_data,
        params.graph_version_id,
        params.thread_id
    ):
        yield {
            "type": "response",
            "content": response_chunk,
            "complete": False,
            "thread_id": params.thread_id,
            "status": "responding"
        }
    
    # Complete processing
    yield {
        "type": "status",
        "content": "Processing complete",
        "complete": True,
        "thread_id": params.thread_id,
        "status": "complete"
    }
    
    logger.info("CoT handler completed", duration=time.time() - start_time)

# Register tools
tool_registry.register_tool(
    "cot_knowledge_analysis_tool",
    cot_knowledge_analysis_actions_handler,
    COT_KNOWLEDGE_ANALYSIS_SCHEMA
)

async def process_llm_with_tools(
    user_message: str,
    conversation_history: List[Dict],
    state: Dict,
    graph_version_id: str,
    thread_id: Optional[str] = None
) -> AsyncGenerator[Union[str, Dict], None]:
    """Process user message with LLM and tools."""
    overall_start_time = time.time()
    PERF_METRICS.clear()
    PERF_METRICS["overall_start"] = overall_start_time
    
    # Build conversation context
    conversation_context = f"User: {user_message}\n"
    if conversation_history:
        recent_messages = [
            f"{'AI' if msg['role'] == 'assistant' else 'User'}: {msg['content']}"
            for msg in conversation_history[-15:-1]
            if msg.get("role") and msg.get("content")
        ]
        if recent_messages:
            conversation_context = "\n".join(recent_messages) + "\n" + conversation_context
    
    logger.info("Conversation history processed", messages=len(conversation_history))
    
    # Process with CoT handler
    cot_params = CotKnowledgeAnalysisParams(
        conversation_context=conversation_context,
        graph_version_id=graph_version_id,
        thread_id=thread_id,
        state=state
    )
    
    response_buffer = ""
    try:
        async for result in cot_knowledge_analysis_actions_handler(cot_params):
            if thread_id and result.get("type") in ["analysis", "response", "status"]:
                try:
                    from socketio_manager import (
                        emit_analysis_event,
                        emit_knowledge_event,
                        emit_next_action_event
                    )
                    emit_func = {
                        "analysis": emit_analysis_event,
                        "response": emit_next_action_event,
                        "status": emit_analysis_event
                    }.get(result["type"])
                    if emit_func:
                        emit_func(thread_id, result)
                        logger.debug(f"Emitted {result['type']} event via socketio", thread_id=thread_id)
                    else:
                        yield result
                        logger.debug(f"Yielded {result['type']} event directly", thread_id=thread_id)
                except Exception as e:
                    logger.error(f"Error emitting {result.get('type')} event", error=str(e))
                    yield result
                    logger.debug(f"Yielded {result['type']} event after emission failure", thread_id=thread_id)
            else:
                yield result
                logger.debug(f"Yielded {result.get('type', 'unknown')} event", thread_id=thread_id)
            
            if result.get("type") == "response":
                response_buffer += result["content"]
    
    except Exception as e:
        logger.error("CoT processing error", error=str(e), traceback=traceback.format_exc())
        yield {"type": "error", "component": "cot_handler", "error": str(e)}
    
    # Update state
    if response_buffer:
        state.setdefault("messages", []).append({"role": "assistant", "content": response_buffer})
        state["prompt_str"] = response_buffer
    
    yield {"state": state}
    logger.info("Processing completed", duration=time.time() - overall_start_time)

async def execute_tool_call(
    tool_call: Dict,
    params: Dict,
    thread_id: Optional[str] = None
) -> AsyncGenerator[Dict, None]:
    """Execute a single tool call."""
    try:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})
        
        handler = tool_registry.get_tool_handler(tool_name)
        if not handler:
            logger.error("Tool not found", tool_name=tool_name)
            yield {"error": f"Tool not found: {tool_name}"}
            return
        
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except:
                tool_args = {"query": tool_args}
        
        call_params = {**tool_args, **params}
        if thread_id:
            call_params["thread_id"] = thread_id
        
        logger.info("Executing tool call", tool_name=tool_name, params=call_params)
        
        start_time = time.time()
        async for result in handler(call_params):
            yield result
        
        logger.info("Tool call completed", tool_name=tool_name, duration=time.time() - start_time)
    except Exception as e:
        logger.error("Tool execution error", error=str(e), traceback=traceback.format_exc())
        yield {"error": f"Tool execution error: {str(e)}"}