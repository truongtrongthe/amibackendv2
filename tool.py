import json
import re
import time
import asyncio
import traceback
import hashlib
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from functools import lru_cache
from pydantic import BaseModel
import structlog
from langchain_openai import ChatOpenAI
from cachetools import TTLCache
import os

from utilities import logger as default_logger
from brain_singleton import get_brain
from tool_helpers import (
    detect_language,
    ensure_brain_loaded,
    prepare_knowledge,
    build_analyse_profile_query
)
from profiling import (
    build_user_profile,
    format_user_profile_for_prompt
)
from profile_cache import get_profile_knowledge

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
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 30))

# LLM instances
LLM = ChatOpenAI(model="gpt-4o", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

# Caches
knowledge_cache = TTLCache(maxsize=1000, ttl=KNOWLEDGE_CACHE_TTL)
response_cache = TTLCache(maxsize=500, ttl=RESPONSE_CACHE_TTL)

# Performance tracking
PERF_METRICS = {}

# Access brain singleton
brain = get_brain()

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
class CombinedAnalysisActionsParams(BaseModel):
    conversation_context: str
    graph_version_id: str
    knowledge_context: Optional[str] = ""
    additional_instructions: Optional[str] = ""

class KnowledgeQueryParams(BaseModel):
    queries: List[str]
    graph_version_id: str
    top_k: Optional[int] = 3

class ResponseGenerationParams(BaseModel):
    conversation_context: str
    analysis: str
    next_actions: str
    knowledge_context: str
    personality_instructions: str
    knowledge_found: bool = False
    user_profile: Optional[Dict] = {}

class CotKnowledgeAnalysisParams(BaseModel):
    conversation_context: str
    graph_version_id: str
    thread_id: Optional[str] = None
    state: Optional[Dict] = {}

COMBINED_ANALYSIS_ACTIONS_SCHEMA = {
    "name": "combined_analysis_actions_tool",
    "description": "Analyze conversation context and determine next actions",
    "parameters": CombinedAnalysisActionsParams.model_json_schema()
}

KNOWLEDGE_QUERY_SCHEMA = {
    "name": "knowledge_query_tool",
    "description": "Query the knowledge base for relevant information",
    "parameters": KnowledgeQueryParams.model_json_schema()
}

RESPONSE_GENERATION_SCHEMA = {
    "name": "response_generation_tool",
    "description": "Generate a user response based on context and analysis",
    "parameters": ResponseGenerationParams.model_json_schema()
}

COT_KNOWLEDGE_ANALYSIS_SCHEMA = {
    "name": "cot_knowledge_analysis_tool",
    "description": "Combine knowledge retrieval, analysis, and actions using CoT",
    "parameters": CotKnowledgeAnalysisParams.model_json_schema()
}

# Prompt templates
PROMPT_TEMPLATES = {
    "response_generation": """
# Context
Latest message: {last_message}

{user_understanding}

# Analysis Summary
{analysis}

# Actions Needed
{next_actions}

{knowledge_section}

{persuasive_script_section}

# Personality
{personality_instructions}

# Task
Create a concise, persuasive response in {language} that:
1. Directly addresses the user's needs in a compelling way
2. Sounds like a friendly human, not an AI
3. Uses appropriate cultural terms of address and conversational markers
4. Keeps the content brief but complete (aim for 2-3 short paragraphs)
5. Includes a natural, friendly call-to-action

CONVERSATIONAL STYLE GUIDE:
- In Vietnamese: Use terms like "Chị ơi/Anh ơi" at the beginning, and particles like "nhé", "ạ", "nha"
- In English: Use friendly openings like "Hi there" and closings like "Let me know if you need anything else"
- Keep sentences short and simple
- Use contractions and everyday phrases
- Sound conversational, not formal
- AVOID REPEATING the same prefixes or greetings
- VARY expressions to sound natural

HIGHEST PRIORITY - PERSUASIVE SCRIPT ADAPTATION:
1. Use the persuasive sales script as the PRIMARY SOURCE
2. Adapt to be conversational and human-sounding
3. Maintain key selling points and value propositions
4. Shorten lengthy explanations
5. Add appropriate conversational markers
""",
    "analysis": """
Based on the user's message, profile, and knowledge:

USER MESSAGE: {last_user_message}
USER PROFILE: {profile_summary}
KNOWLEDGE: {knowledge_context}

Provide:
ANALYSIS: [Incorporate communication style, classification, decision state, needs, strategy]
RECOMMENDED TECHNIQUES: [Extracted from knowledge]
SPECIFIC INSTRUCTIONS: [Copied verbatim]
KEY QUERIES: [Formatted as "Cách thực hiện {technique}"]
""",
    "actions": """
# Sales Automation Next Steps and Persuasive Script

**User Message**: {last_user_message}
**User Classification**: {user_classification}
**User Profile**: {profile_summary}
**Analysis**: {analysis}
**Recommended Techniques**: {techniques}
**Specific Instructions**: {instructions}
**Knowledge**: {knowledge}

## Task
### Part 1: Sales Automation Next Steps
For each technique:
1. Identify by name in **bold**.
2. Extract steps from instructions.
3. Adapt steps based on profile.
4. Preserve original wording.

### Part 2: Persuasive Script
PERSUASIVE_SCRIPT:
[Script implementing steps, using knowledge phrases]
END_SCRIPT
"""
}

# Initialize tool registry
tool_registry = ToolRegistry()

async def response_generation_handler(params: ResponseGenerationParams) -> AsyncGenerator[str, None]:
    """Generate a response with user profile awareness and persuasive script."""
    start_time = time.time()
    PERF_METRICS["response_generation_start"] = start_time
    
    # Extract parameters
    conversation_context = params.conversation_context
    analysis = params.analysis
    next_actions = params.next_actions
    knowledge_context = params.knowledge_context
    personality_instructions = params.personality_instructions
    user_profile = params.user_profile or {}
    
    # Determine language
    language_preference = detect_language(conversation_context)
    if user_profile.get("portrait", "").lower().find("vietnamese") != -1:
        language_preference = "vietnamese"
    elif user_profile.get("portrait", "").lower().find("english") != -1:
        language_preference = "english"
    
    # Format user profile
    user_understanding = format_user_profile_for_prompt(user_profile) if user_profile else ""
    
    # Simplify personality instructions if too long
    if len(personality_instructions) > 500:
        personality_instructions = """
        You are a warm, conversational assistant:
        - Use casual, everyday language
        - Include cultural terms (e.g., "Chị ơi" in Vietnamese)
        - Keep responses concise (2-3 paragraphs)
        - Mirror user's style
        - Use appropriate emotional markers
        """
    
    # Get last user message
    last_message = ""
    for line in conversation_context.split('\n'):
        if line.startswith("User:"):
            last_message = line[5:].strip()
    
    # Build prompt
    persuasive_script = user_profile.get('persuasive_script', '')
    prompt = PROMPT_TEMPLATES["response_generation"].format(
        last_message=last_message,
        user_understanding=user_understanding,
        analysis=analysis,
        next_actions=next_actions,
        knowledge_section=f"# Knowledge\n{knowledge_context}" if knowledge_context else "",
        persuasive_script_section=f"# Persuasive Script\n{persuasive_script}" if persuasive_script else "",
        personality_instructions=personality_instructions,
        language=language_preference
    )
    
    # Check cache
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    if cache_key in response_cache:
        logger.info("Using cached response", cache_key=cache_key)
        for chunk in response_cache[cache_key]:
            yield chunk
        return
    
    # Stream response
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
        
        # Cache the response
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
    
    PERF_METRICS["response_generation_end"] = time.time()
    logger.info("Response generation completed", duration=PERF_METRICS["response_generation_end"] - start_time)

async def build_user_profile_step(context: str, graph_version_id: str, state: Dict) -> Dict:
    """STEP 1: Build or retrieve user profile."""
    if state.get("analysis", {}).get("user_profile"):
        logger.info("Using cached user profile")
        return state["analysis"]["user_profile"]
    
    try:
        last_message = context.split('\n')[-1].replace("User:", "").strip()
        profile = await build_user_profile(context, last_message, graph_version_id, state=state)
        state["analysis"] = state.get("analysis", {})
        state["analysis"]["user_profile"] = profile
        state["profiling_knowledge_entries"] = state.get("profiling_knowledge_entries", [])
        logger.info("Built user profile", profile=profile)
        return profile
    except Exception as e:
        logger.error("Error building user profile", error=str(e))
        return {"portrait": f"User query: {context.split('\n')[-1]}"}

async def fetch_knowledge_step(user_profile: Dict, graph_version_id: str, state: Dict, last_user_message: str) -> Dict[str, List[Dict]]:
    """STEP 2: Fetch categorized knowledge from cache or brain."""
    cache_key = f"{graph_version_id}:{user_profile.get('segment', {}).get('category', '')}"
    if cache_key in knowledge_cache:
        logger.info("Using cached knowledge", cache_key=cache_key)
        return knowledge_cache[cache_key]
    
    if state.get("profiling_knowledge_entries"):
        logger.info("Using state knowledge entries")
        return {
            "user_analysis": state["profiling_knowledge_entries"],
            "technique_implementation": [],
            "additional": []
        }
    
    knowledge = {
        "user_analysis": [],
        "technique_implementation": [],
        "additional": []
    }
    
    try:
        if await ensure_brain_loaded(graph_version_id):
            # Fetch user analysis knowledge
            queries = build_analyse_profile_query(user_profile)[:3]
            for query in queries:
                results = await brain.get_similar_vectors_by_text(query, top_k=2)
                for vector_id, _, metadata, similarity in results:
                    if any(entry.get("id") == vector_id for entry in knowledge["user_analysis"]):
                        continue
                    knowledge["user_analysis"].append({
                        "id": vector_id,
                        "similarity": float(similarity),
                        "raw": metadata.get("raw", ""),
                        "query": query,
                        "phase": "user_analysis"
                    })
            
            state["profiling_knowledge_entries"] = knowledge["user_analysis"]
            knowledge_cache[cache_key] = knowledge
            logger.info("Fetched knowledge", user_analysis_entries=len(knowledge["user_analysis"]))
    except Exception as e:
        logger.error("Knowledge retrieval error", error=str(e))
    
    return knowledge

async def generate_analysis_step(user_profile: Dict, knowledge: Dict[str, List[Dict]], context: str) -> tuple[str, List[str], str, List[str]]:
    """STEP 3: Generate analysis, techniques, instructions, and key queries."""
    last_user_message = context.split('\n')[-1].replace("User:", "").strip()
    user_classification = user_profile.get("segment", {}).get("category", "general")
    knowledge_context = prepare_knowledge(
        knowledge["user_analysis"], 
        last_user_message, 
        target_classification=user_classification
    )
    profile_summary = format_user_profile_for_prompt(user_profile) if "portrait" in user_profile else ""
    
    try:
        # Safely format the prompt to avoid KeyError
        prompt = PROMPT_TEMPLATES.get("analysis", "Default analysis prompt: {last_user_message}").format(
            last_user_message=last_user_message,
            profile_summary=profile_summary,
            knowledge_context=knowledge_context or "No specific knowledge found."
        )
    except KeyError as e:
        logger.error("KeyError in prompt formatting", error=str(e))
        prompt = f"Analyze the user message: {last_user_message}"
    
    try:
        response = await LLM.ainvoke(prompt, temperature=0.1)
        full_analysis = response.content
        
        # Parse response
        analysis_part = full_analysis.split("ANALYSIS:")[1].split("RECOMMENDED TECHNIQUES:")[0].strip() if "ANALYSIS:" in full_analysis else full_analysis
        techniques_part = full_analysis.split("RECOMMENDED TECHNIQUES:")[1].split("SPECIFIC INSTRUCTIONS:")[0].strip() if "RECOMMENDED TECHNIQUES:" in full_analysis else ""
        instructions_part = full_analysis.split("SPECIFIC INSTRUCTIONS:")[1].split("KEY QUERIES:")[0].strip() if "SPECIFIC INSTRUCTIONS:" in full_analysis else ""
        queries_part = full_analysis.split("KEY QUERIES:")[1].strip() if "KEY QUERIES:" in full_analysis else ""
        
        techniques = [re.sub(r'^\s*[\d\.\-\*]+\s*', '', line.strip()) for line in techniques_part.split('\n') if line.strip() and len(line) > 3]
        key_queries = [re.sub(r'^\s*[\d\.\-\*]+\s*', '', line.strip()) for line in queries_part.split('\n') if line.strip() and len(line) > 3]
        
        if not techniques:
            techniques = [f"Phương pháp tiếp cận nhóm {user_classification}"]
        if not key_queries:
            key_queries = [f"Cách thực hiện {techniques[0]}" if techniques else f"Phương pháp điều trị cho nhóm {user_classification}"]
        
        logger.info("Generated analysis", analysis_len=len(analysis_part), techniques=techniques, queries=key_queries)
        return analysis_part, techniques, instructions_part, key_queries
    except Exception as e:
        logger.error("Analysis generation error", error=str(e))
        return (
            f"Analyzing query: {last_user_message}",
            [f"Phương pháp tiếp cận nhóm {user_classification}"],
            "",
            [f"Phương pháp điều trị cho nhóm {user_classification}"]
        )

async def fetch_technique_knowledge_step(techniques: List[str], key_queries: List[str], user_profile: Dict, graph_version_id: str) -> Dict[str, List[Dict]]:
    """STEP 4: Fetch technique implementation and additional knowledge."""
    knowledge = {
        "technique_implementation": [],
        "additional": []
    }
    
    try:
        if await ensure_brain_loaded(graph_version_id):
            # Fetch technique implementation knowledge
            for technique in techniques[:2]:
                query = f"Cách thực hiện {technique}"
                results = await brain.get_similar_vectors_by_text(query, top_k=2)
                for vector_id, _, metadata, similarity in results:
                    if any(entry.get("id") == vector_id for entry in knowledge["technique_implementation"]):
                        continue
                    knowledge["technique_implementation"].append({
                        "id": vector_id,
                        "similarity": float(similarity),
                        "raw": metadata.get("raw", ""),
                        "query": query,
                        "phase": "technique_implementation"
                    })
            
            # Fetch additional knowledge
            for query in key_queries:
                results = await brain.get_similar_vectors_by_text(query, top_k=2)
                for vector_id, _, metadata, similarity in results:
                    if any(entry.get("id") == vector_id for entry in knowledge["additional"]) or \
                       any(entry.get("id") == vector_id for entry in knowledge["technique_implementation"]):
                        continue
                    knowledge["additional"].append({
                        "id": vector_id,
                        "similarity": float(similarity),
                        "raw": metadata.get("raw", ""),
                        "query": query,
                        "phase": "additional_knowledge"
                    })
        
        logger.info(
            "Fetched technique knowledge",
            technique_entries=len(knowledge["technique_implementation"]),
            additional_entries=len(knowledge["additional"])
        )
        return knowledge
    except Exception as e:
        logger.error("Technique knowledge retrieval error", error=str(e))
        return knowledge

async def plan_actions_step(techniques: List[str], instructions: str, user_profile: Dict, knowledge: Dict[str, List[Dict]], last_user_message: str) -> tuple[str, str]:
    """STEP 5: Plan next actions and generate persuasive script."""
    user_classification = user_profile.get("segment", {}).get("category", "general")
    
    # Combine knowledge
    user_analysis_context = prepare_knowledge(
        knowledge.get("user_analysis", []),
        last_user_message,
        target_classification=user_classification
    )
    technique_context = prepare_knowledge(
        knowledge.get("technique_implementation", []),
        last_user_message,
        target_classification=user_classification
    )
    additional_context = prepare_knowledge(
        knowledge.get("additional", []),
        last_user_message,
        target_classification=user_classification
    )
    
    combined_knowledge = ""
    if user_analysis_context:
        combined_knowledge += f"## USER ANALYSIS KNOWLEDGE:\n{user_analysis_context}\n\n"
    if technique_context:
        combined_knowledge += f"## TECHNIQUE IMPLEMENTATION KNOWLEDGE:\n{technique_context}\n\n"
    if additional_context:
        combined_knowledge += f"## ADDITIONAL KNOWLEDGE:\n{additional_context}\n\n"
    
    prompt = PROMPT_TEMPLATES["actions"].format(
        last_user_message=last_user_message,
        user_classification=user_classification,
        profile_summary=format_user_profile_for_prompt(user_profile) if user_profile else "",
        analysis="",
        techniques="\n".join(f"- {tech}" for tech in techniques),
        instructions=instructions,
        knowledge=combined_knowledge
    )
    
    try:
        next_actions = ""
        persuasive_script = ""
        async for chunk in StreamLLM.astream(prompt):
            content = chunk.content
            next_actions += content
            if "PERSUASIVE_SCRIPT:" in next_actions and "END_SCRIPT" in next_actions:
                script_start = next_actions.find("PERSUASIVE_SCRIPT:")
                script_end = next_actions.find("END_SCRIPT", script_start)
                if script_start != -1 and script_end != -1:
                    persuasive_script = next_actions[script_start + len("PERSUASIVE_SCRIPT:"):script_end].strip()
        
        clean_actions = next_actions.replace("END_SCRIPT", "END_SCRIPT\n") if persuasive_script else next_actions
        logger.info("Planned actions", actions_len=len(clean_actions), script_len=len(persuasive_script))
        return clean_actions, persuasive_script
    except Exception as e:
        logger.error("Actions planning error", error=str(e))
        return (
            "1. Research more\n2. Consult resources\n3. Try specific query",
            "Xin lỗi, tôi cần thêm thông tin để hỗ trợ bạn." if detect_language(last_user_message) == "vietnamese" else "Sorry, I need more information to assist you."
        )

async def cot_knowledge_analysis_actions_handler(params: CotKnowledgeAnalysisParams) -> AsyncGenerator[Dict, None]:
    """Chain-of-Thought handler for knowledge, analysis, and actions."""
    start_time = time.time()
    state = params.state or {}
    state.setdefault("cot_results", {})
    
    # STEP 1: Build user profile
    user_profile = await build_user_profile_step(params.conversation_context, params.graph_version_id, state)
    yield {
        "type": "analysis",
        "content": "Analyzing your message...",
        "complete": False,
        "thread_id": params.thread_id,
        "status": "analyzing"
    }
    
    # STEP 2: Fetch user analysis knowledge
    last_user_message = params.conversation_context.split('\n')[-1].replace("User:", "").strip()
    knowledge = await fetch_knowledge_step(user_profile, params.graph_version_id, state, last_user_message)
    yield {
        "type": "knowledge",
        "content": knowledge["user_analysis"],
        "complete": False,
        "thread_id": params.thread_id,
        "status": "searching"
    }
    
    # STEP 3: Generate analysis
    analysis, techniques, instructions, key_queries = await generate_analysis_step(user_profile, knowledge, params.conversation_context)
    state["cot_results"]["analysis"] = analysis
    state["cot_results"]["techniques"] = techniques
    state["cot_results"]["instructions"] = instructions
    state["cot_results"]["key_queries"] = key_queries
    yield {
        "type": "analysis",
        "content": analysis,
        "complete": True,
        "thread_id": params.thread_id,
        "status": "complete",
        "user_profile": user_profile,
        "techniques": techniques,
        "instructions": instructions
    }
    
    # STEP 4: Fetch technique and additional knowledge
    technique_knowledge = await fetch_technique_knowledge_step(techniques, key_queries, user_profile, params.graph_version_id)
    knowledge.update(technique_knowledge)
    if technique_knowledge["technique_implementation"] or technique_knowledge["additional"]:
        yield {
            "type": "knowledge",
            "content": technique_knowledge["technique_implementation"] + technique_knowledge["additional"],
            "complete": False,
            "thread_id": params.thread_id,
            "status": "searching"
        }
    
    # STEP 5: Plan actions and generate persuasive script
    next_actions, persuasive_script = await plan_actions_step(techniques, instructions, user_profile, knowledge, last_user_message)
    state["cot_results"]["next_actions"] = next_actions
    state["cot_results"]["persuasive_script"] = persuasive_script
    user_profile["persuasive_script"] = persuasive_script
    
    # Chunk large actions if necessary
    if len(next_actions) > MAX_SOCKET_SIZE:
        chunks = [next_actions[i:i + MAX_SOCKET_SIZE] for i in range(0, len(next_actions), MAX_SOCKET_SIZE)]
        for i, chunk in enumerate(chunks):
            event = {
                "type": "next_actions",
                "content": chunk,
                "complete": i == len(chunks) - 1,
                "thread_id": params.thread_id,
                "status": "chunked" if i < len(chunks) - 1 else "complete",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "user_profile": user_profile if i == 0 else None
            }
            yield event
            logger.debug("Yielded next_actions chunk", chunk_index=i, thread_id=params.thread_id)
            await asyncio.sleep(0.1)
    else:
        event = {
            "type": "next_actions",
            "content": next_actions,
            "complete": True,
            "thread_id": params.thread_id,
            "status": "complete",
            "user_profile": user_profile
        }
        yield event
        logger.debug("Yielded next_actions event", thread_id=params.thread_id)
    
    # Complete knowledge event
    all_knowledge = (
        knowledge["user_analysis"] +
        knowledge["technique_implementation"] +
        knowledge["additional"]
    )
    event = {
        "type": "knowledge",
        "content": all_knowledge,
        "complete": True,
        "thread_id": params.thread_id,
        "status": "complete",
        "stats": {"total_results": len(all_knowledge)}
    }
    yield event
    logger.debug("Yielded knowledge complete event", thread_id=params.thread_id)
    
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
    
    # Initial status
    event = {"type": "status", "status": "processing", "message": "Processing your request..."}
    yield event
    logger.debug("Yielded initial status event", thread_id=thread_id)
    
    # Process with CoT handler
    cot_params = CotKnowledgeAnalysisParams(
        conversation_context=conversation_context,
        graph_version_id=graph_version_id,
        thread_id=thread_id,
        state=state
    )
    
    analysis_results = ""
    next_actions_results = ""
    knowledge_entries = []
    knowledge_context = ""
    knowledge_found = False
    
    try:
        async for result in cot_knowledge_analysis_actions_handler(cot_params):
            if thread_id and result.get("type") in ["analysis", "knowledge", "next_actions"]:
                try:
                    from socketio_manager import (
                        emit_analysis_event,
                        emit_knowledge_event,
                        emit_next_action_event
                    )
                    emit_func = {
                        "analysis": emit_analysis_event,
                        "knowledge": emit_knowledge_event,
                        "next_actions": emit_next_action_event
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
            
            if result.get("type") == "analysis" and result.get("complete"):
                analysis_results = result.get("content", "")
            elif result.get("type") == "next_actions" and result.get("complete"):
                next_actions_results = result.get("content", "")
            elif result.get("type") == "knowledge" and not result.get("complete"):
                knowledge_entries.extend(result.get("content", []))
    except Exception as e:
        logger.error("CoT processing error", error=str(e), traceback=traceback.format_exc())
        yield {"type": "error", "component": "cot_handler", "error": str(e)}
    
    # Format knowledge context
    if knowledge_entries:
        user_classification = state.get("cot_results", {}).get("user_profile", {}).get("segment", {}).get("category")
        knowledge_context = prepare_knowledge(knowledge_entries, user_message, target_classification=user_classification)
        knowledge_found = True
    
    # Generate response
    yield {"type": "status", "status": "generating", "message": "Generating response..."}
    
    response_params = ResponseGenerationParams(
        conversation_context=conversation_context,
        analysis=analysis_results,
        next_actions=next_actions_results,
        knowledge_context=knowledge_context,
        personality_instructions="""
        You are a sophisticated, energetic AI:
        - Mirror user's style and formality
        - Use clear, rich vocabulary
        - Show enthusiasm and interest
        - Be persuasive and compelling
        - Adapt to cultural nuances
        """,
        knowledge_found=knowledge_found,
        user_profile=state.get("cot_results", {}).get("user_profile", {})
    )
    
    response_buffer = ""
    try:
        async for chunk in response_generation_handler(response_params):
            response_buffer += chunk
            yield chunk
    except Exception as e:
        logger.error("Response generation error", error=str(e))
        fallback = (
            "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại nha." if detect_language(user_message) == "vietnamese"
            else "Sorry, an error occurred. Please try again."
        )
        if "Discouraged Group" in response_params.user_profile.get("portrait", ""):
            fallback = (
                "Xuất tinh sớm là tình trạng phổ biến, và rất nhiều nam giới gặp phải. "
                "Có nhiều phương pháp hữu ích như bài tập Kegel, kỹ thuật thở sâu, và phương pháp start-stop. "
                "Bạn có thể thử hoặc tham khảo ý kiến chuyên gia y tế nhé."
            ) if detect_language(user_message) == "vietnamese" else (
                "Premature ejaculation is common and nothing to be ashamed of. "
                "There are helpful methods like Kegel exercises, deep breathing, and the start-stop technique. "
                "You can try these or consult a healthcare professional."
            )
        yield fallback
        response_buffer = fallback
    
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