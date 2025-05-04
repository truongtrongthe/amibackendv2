"""
Tool calling implementation for the MC system with optimized performance.
This file defines the tool schemas and handlers for LLM tool calling with combined operations,
parallel processing, and optimized prompts.
"""

import json
import re
import time
import asyncio
import traceback
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Tuple
from functools import lru_cache
from langchain_openai import ChatOpenAI
from utilities import logger

from brain_singleton import get_brain
from tool_helpers import (
    detect_language,
    ensure_brain_loaded,
    prepare_knowledge,
    build_analyse_profile_query
)
from profile_helper import (
    build_user_profile,
    format_user_profile_for_prompt
)
from response_optimization import ResponseProcessor, ResponseFilter

# Use the same LLM instances as mc.py
LLM = ChatOpenAI(model="gpt-4o", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

# Performance tracking
PERF_METRICS = {}

# Access the brain singleton once
brain = get_brain()

# Add a module-level variable to store results from the CoT handler
_last_cot_results = {
    "analysis_content": "",
    "next_actions_content": "",
    "clean_next_actions": "",
    "knowledge_entries": [],
    "knowledge_context": "",
    "persuasive_script": ""
}

class ToolRegistry:
    """Registry for all available tools"""
    
    def __init__(self):
        self.tools = {}
        self.tool_schemas = {}
    
    def register_tool(self, name: str, handler, schema: Dict):
        """Register a tool with its handler and schema"""
        self.tools[name] = handler
        self.tool_schemas[name] = schema
        logger.info(f"Registered tool: {name}")
    
    def get_tool_schemas(self) -> List[Dict]:
        """Get all tool schemas in a format suitable for LLM tool calling"""
        return list(self.tool_schemas.values())
    
    def get_tool_handler(self, name: str):
        """Get a tool handler by name"""
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        return self.tools[name]
    
    def get_tool_names(self) -> List[str]:
        """Get list of all registered tool names"""
        return list(self.tools.keys())


# Initialize the tool registry
tool_registry = ToolRegistry()


# OPTIMIZED TOOL SCHEMAS
# Combined analysis and next actions tool schema
COMBINED_ANALYSIS_ACTIONS_SCHEMA = {
    "name": "combined_analysis_actions_tool",
    "description": "Analyze the conversation context and determine next actions in a single operation",
    "parameters": {
        "type": "object",
        "properties": {
            "conversation_context": {
                "type": "string",
                "description": "The full conversation history to analyze"
            },
            "graph_version_id": {
                "type": "string",
                "description": "The version ID of the knowledge graph to query for profile information"
            },
            "knowledge_context": {
                "type": "string",
                "description": "The retrieved knowledge relevant to the conversation"
            },
            "additional_instructions": {
                "type": "string",
                "description": "Optional additional instructions for the analysis",
                "default": ""
            }
        },
        "required": ["conversation_context", "graph_version_id"]
    }
}

KNOWLEDGE_QUERY_SCHEMA = {
    "name": "knowledge_query_tool",
    "description": "Query the knowledge base for information relevant to the user's request",
    "parameters": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search queries to run against the knowledge base"
            },
            "graph_version_id": {
                "type": "string",
                "description": "The version ID of the knowledge graph to query"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return per query",
                "default": 3
            }
        },
        "required": ["queries", "graph_version_id"]
    }
}

RESPONSE_GENERATION_SCHEMA = {
    "name": "response_generation_tool",
    "description": "Generate a response to the user based on conversation context, analysis, and knowledge",
    "parameters": {
        "type": "object",
        "properties": {
            "conversation_context": {
                "type": "string",
                "description": "The conversation history"
            },
            "analysis": {
                "type": "string",
                "description": "The analysis of the conversation"
            },
            "next_actions": {
                "type": "string",
                "description": "The planned next actions"
            },
            "knowledge_context": {
                "type": "string",
                "description": "The retrieved knowledge"
            },
            "personality_instructions": {
                "type": "string",
                "description": "Instructions for the AI's personality"
            },
            "knowledge_found": {
                "type": "boolean",
                "description": "Whether relevant knowledge was found",
                "default": False
            }
        },
        "required": ["conversation_context", "analysis", "next_actions", "personality_instructions", "knowledge_context"]
    }
}

# Register the CoT handler schema
COT_KNOWLEDGE_ANALYSIS_SCHEMA = {
    "name": "cot_knowledge_analysis_tool",
    "description": "Combine knowledge retrieval with analysis and next actions using Chain-of-Thought reasoning",
    "parameters": {
        "type": "object",
        "properties": {
            "conversation_context": {
                "type": "string",
                "description": "The full conversation history to analyze"
            },
            "graph_version_id": {
                "type": "string",
                "description": "The version ID of the knowledge graph to query"
            }
        },
        "required": ["conversation_context", "graph_version_id"]
    }
}

# Optimized response generation handler
async def response_generation_handler(params: Dict) -> AsyncGenerator[str, None]:
    """
    Optimized handler for response generation with user profile awareness.
    
    Args:
        params: Dictionary containing conversation_context, analysis, next_actions, 
                knowledge_context, personality_instructions, knowledge_found,
                and user_profile (optional)
                
    Yields:
        Streamed response chunks as strings
    """
    start_time = time.time()
    PERF_METRICS["response_generation_start"] = start_time
    
    conversation_context = params.get("conversation_context", "")
    analysis = params.get("analysis", "")
    next_actions = params.get("next_actions", "")
    knowledge_context = params.get("knowledge_context", "")
    personality_instructions = params.get("personality_instructions", "")
    user_profile = params.get("user_profile", _last_cot_results.get("user_profile", {}))
    
    # Get user profile information or use defaults
    language_preference = "vietnamese"
    
    # Extract basic language information from the portrait or fallback to detection
    if user_profile and "portrait" in user_profile:
        # Try to extract language from portrait
        portrait = user_profile.get("portrait", "")
        if "communicating in Vietnamese" in portrait:
            language_preference = "vietnamese"
        elif "communicating in English" in portrait:
            language_preference = "english"
        else:
            # Fall back to simple language detection
            language_preference = detect_language(conversation_context)
    else:
        # Fall back to simple language detection if no profile
        language_preference = detect_language(conversation_context)
    
    detected_language = language_preference
    
    # Get the user portrait to include in the prompt
    user_understanding = format_user_profile_for_prompt(user_profile) if user_profile else ""
    
    # OPTIMIZATION: Simplify personality instructions if too long
    if personality_instructions and len(personality_instructions) > 500:
        personality_instructions = """
        You are a warm, conversational assistant who communicates naturally like a friendly human would:
        - Use casual, everyday language rather than formal or technical phrasing
        - Include culturally appropriate terms of address (like "Chị ơi", "Anh ơi" in Vietnamese)
        - Keep responses concise and to the point (prefer 2-3 short paragraphs)
        - Mirror the user's communication style while maintaining clarity
        - Use appropriate emotional markers and conversational particles for the language
        """
    
    
    # Get the last user message for better context
    last_message = ""
    for line in conversation_context.split('\n'):
        if line.startswith("User:"):
            last_message = line
    
    # OPTIMIZATION: Simplify the prompt structure but incorporate user understanding
    prompt = f"""
    # Context
    Latest message: {last_message}
    
    {user_understanding}
    
    # Analysis Summary
    {analysis}  
    
    # Actions Needed
    {next_actions}
    
    {f"# Knowledge\\n{knowledge_context}" if knowledge_context else ""}
    
    {f"# Persuasive Sales Script\\n{_last_cot_results.get('persuasive_script', '')}" if _last_cot_results.get('persuasive_script') else ""}
    
    # Personality
    {personality_instructions}
    
    # Task
    Create a concise, persuasive response in {detected_language} that:
    1. Directly addresses the user's needs in a compelling way
    2. Sounds like a friendly human, not an AI
    3. Uses appropriate cultural terms of address and conversational markers
    4. Keeps the content brief but complete (aim for 2-3 short paragraphs)
    5. Includes a natural, friendly call-to-action
    
    CONVERSATIONAL STYLE GUIDE:
    - In Vietnamese: Use appropriate terms like "Chị ơi/Anh ơi" at the beginning, and conversational particles like "nhé", "ạ", "nha" where natural
    - In English: Use friendly openings like "Hi there" and conversational closings like "Let me know if you need anything else"
    - Keep sentences short and simple
    - Use contractions and everyday phrases
    - Sound like you're chatting, not writing a formal document
    - AVOID REPEATING the same prefixes or greeting patterns across consecutive messages
    - VARY your expressions and phrasing to sound more natural
    
    HIGHEST PRIORITY - PERSUASIVE SCRIPT ADAPTATION:
    1. Use the persuasive sales script as your PRIMARY SOURCE for content
    2. Adapt the script to be more conversational and human-sounding
    3. Maintain all key selling points and value propositions
    4. Shorten lengthy explanations while preserving the main message
    5. Add appropriate conversational markers and cultural terms
    
    RESPONSE GUIDANCE:
    - Begin with the appropriate conversational greeting for the language
    - Present the main value proposition concisely
    - Keep technical details minimal but sufficient
    - End with a warm, friendly close and simple call-to-action
    - If this is a follow-up message, don't repeat the same greeting from previous messages
    
    IMPORTANT: Your response should feel like a message from a helpful friend rather than a formal business communication. Make it conversational, concise, and culturally appropriate. Vary your language patterns between messages to sound more human.
    """
    
    # OPTIMIZATION: Set a timeout for response generation
    RESPONSE_TIMEOUT = 7  # 7 seconds timeout
    
    # Response optimization components
    # OPTIMIZATION: Collect the full response with larger buffer size
    full_response = ""
    buffer = ""
    buffer_size = 30  # Increased buffer size for efficiency
    
    try:
        logger.info(f"Starting prompt response generation stream")
        PERF_METRICS["response_stream_start"] = time.time()
        
        # Set up streaming with timeout
        response_task = None  # Initialize to None to avoid reference errors
        
        try:
            # Process with timeout
            logger.info(f"Sending response generation request to OpenAI")
            async with asyncio.timeout(RESPONSE_TIMEOUT):
                async for chunk in StreamLLM.astream(prompt,temperature=0.05):
                    content = chunk.content
                    buffer += content
                    full_response += content
                    
                    # OPTIMIZATION: Stream more frequently for better UX
                    if len(buffer) >= buffer_size or content.endswith((".", "!", "?", "\n", " ")):
                        yield buffer
                        buffer = ""

        except asyncio.TimeoutError:
            logger.warning(f"Response generation timed out after {RESPONSE_TIMEOUT}s, using partial response")
            # Don't try to cancel the streaming task since it's not stored as a task
            # Just yield what we've got so far
            if not full_response:
                # If we have nothing, generate a simple response based on language
                if detected_language == "vietnamese":
                    full_response = "Tôi đang xử lý thông tin. Vui lòng cho tôi thêm chi tiết để hỗ trợ bạn tốt hơn."
                else:
                    full_response = "I'm processing your request. Could you provide more details so I can help you better?"
                yield full_response
        except Exception as e:
            logger.error(f"Error in response generation streaming: {str(e)}")
            logger.error(traceback.format_exc())
            # Fall back to language-specific error messages
            if not full_response:
                if detected_language == "vietnamese":
                    full_response = "Xin lỗi, tôi đang gặp sự cố khi xử lý yêu cầu của bạn. Vui lòng thử lại."
                else:
                    full_response = "I encountered an issue while generating a response. Let me try again."
                yield full_response
        
        # Yield any remaining buffer
        if buffer:
            yield buffer
        
        PERF_METRICS["response_stream_end"] = time.time()
        logger.info(f"Response streaming completed in {PERF_METRICS['response_stream_end'] - PERF_METRICS['response_stream_start']:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in response generation: {str(e)}")
        # Fall back to language-specific error messages
        if detected_language == "vietnamese":
            yield "Xin lỗi, tôi đang gặp sự cố khi xử lý yêu cầu của bạn. Vui lòng thử lại."
        else:
            yield "I encountered an issue while generating a response. Let me try again."
    
    PERF_METRICS["response_generation_end"] = time.time()
    logger.info(f"Response generation completed in {time.time() - start_time:.2f}s")

# Add the Chain of Thought handler
async def cot_knowledge_analysis_actions_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Simplified Chain-of-Thought handler with a linear approach:
    1) Generate profile-enhanced queries
    2) Fetch knowledge for user analysis
    3) Generate action-oriented queries 
    4) Fetch solution knowledge
    5) Implement next actions
    
    Args:
        params: Dictionary containing conversation_context, graph_version_id
                
    Yields:
        Dict events with streaming results for knowledge, analysis, and next actions
    """
    start_time = time.time()
    
    # Access the global variable to store results
    global _last_cot_results
    _last_cot_results = {
        "analysis_content": "",
        "next_actions_content": "",
        "knowledge_entries": [],
        "knowledge_context": "",
        "user_profile": {},
        "persuasive_script": ""
    }
    
    conversation_context = params.get("conversation_context", "")
    graph_version_id = params.get("graph_version_id", "")
    thread_id = params.get("_thread_id")
    
    # Extract user message for knowledge retrieval
    recent_messages = []
    for line in conversation_context.strip().split('\n'):
        if line.startswith("User:") or line.startswith("AI:"):
            recent_messages.append(line)
    
    # Keep last 30 messages for richer context
    recent_messages = recent_messages[-30:] if len(recent_messages) > 30 else recent_messages
    
    # Get most recent user message for initial knowledge search
    last_user_message = ""
    for line in reversed(recent_messages):
        if line.startswith("User:"):
            last_user_message = line[5:].strip()
            break
    
    # Create a comprehensive context from recent messages
    context_window = "\n".join(recent_messages)
    
    logger.info(f"Starting simplified CoT with context of {len(recent_messages)} messages")
    logger.info(f"User query: '{last_user_message}'")
    
    # STEP 1: Begin analysis and build user profile
    # Send analysis starting event
    yield {
        "type": "analysis",
        "content": "Analyzing your message...",
        "complete": False,
        "thread_id": thread_id,
        "status": "analyzing"
    }
    
    # OPTIMIZATION: Simplified user profile generation
    try:
        user_profile = await build_user_profile(context_window, last_user_message, graph_version_id)
        _last_cot_results["user_profile"] = user_profile
        logger.info(f"User profile built successfully: {user_profile}")
    except Exception as e:
        logger.error(f"Error in fast user profile creation: {e}")
        user_profile = {"portrait": f"User query: {last_user_message}"}
    
    # STEP 2: Generate profile-enhanced queries and fetch knowledge
    yield {
        "type": "knowledge",
        "content": [],
        "complete": False,
        "thread_id": thread_id,
        "status": "searching"
    }
    
    user_analysis_knowledge = []
    try:
        # Generate profile-enhanced queries
        enhanced_queries = build_analyse_profile_query(user_profile)
        logger.info(f"Generated {len(enhanced_queries)} profile queries: {enhanced_queries}")
        
        # Load the brain for knowledge retrieval
        brain_loaded = await ensure_brain_loaded(graph_version_id)
        
        if brain_loaded:
            global brain
            brain = get_brain()
            
            # Fetch knowledge using enhanced queries (limit to first 3 queries)
            for query in enhanced_queries[:3]:
                results = await brain.get_similar_vectors_by_text(query, top_k=2)
                
                # Process knowledge results
                for vector_id, vector, metadata, similarity in results:
                    # Skip duplicates
                    if any(entry.get("id") == vector_id for entry in user_analysis_knowledge):
                        continue
                    
                    raw_text = metadata.get("raw", "")
                    
                    entry = {
                        "id": vector_id,
                        "similarity": float(similarity),
                        "raw": raw_text,
                        "query": query,
                        "phase": "user_analysis"
                    }
                    user_analysis_knowledge.append(entry)
            
            # Send knowledge results event
            if user_analysis_knowledge:
                yield {
                    "type": "knowledge",
                    "content": user_analysis_knowledge,
                    "complete": False,
                    "thread_id": thread_id,
                    "status": "searching"
                }
                #logger.info(f"ENHANCED PORTRAIT KNOWLEDGE:{user_analysis_knowledge} ")
    except Exception as e:
        logger.error(f"Error in knowledge retrieval: {e}")
    
    # STEP 3: Generate user analysis with actionable insights and extract techniques
    analysis_content = ""
    approach_techniques = []
    try:
        # Extract user classification from profile if available
        user_classification = None
        if user_profile and "segment" in user_profile and "category" in user_profile["segment"]:
            user_classification = user_profile["segment"]["category"]
        elif user_profile and "portrait" in user_profile:
            # Try to extract classification from portrait using regex
            classification_match = re.search(r'nhóm\s+(\w+\s+\w+|\w+)', user_profile["portrait"], re.IGNORECASE)
            if classification_match:
                user_classification = classification_match.group(1)
            else:
                # Try looking for classification in **bold** text
                bold_match = re.search(r'\*\*([^*]+)\*\*', user_profile["portrait"])
                if bold_match:
                    user_classification = bold_match.group(1)
        
        logger.info(f"Extracted user classification for knowledge filtering: {user_classification}")
        
        # Format knowledge for the LLM, passing the user classification
        knowledge_context = prepare_knowledge(
            user_analysis_knowledge, 
            last_user_message, 
            target_classification=user_classification
        ) if user_analysis_knowledge else ""
        
        # Store knowledge context for later use
        _last_cot_results["knowledge_context"] = knowledge_context
        _last_cot_results["knowledge_entries"] = user_analysis_knowledge
        
        logger.info(f"PREPARED KNOWLEDGE FOR USER ANALYSIS: {knowledge_context}")
        
        # Format user profile for the LLM
        profile_summary = format_user_profile_for_prompt(user_profile) if "portrait" in user_profile else ""
        
        # Create a comprehensive prompt for user analysis and technique extraction
        analysis_prompt = f"""
        Based on the user's message, profile, and the knowledge found, analyze their needs and EXTRACT specific techniques and instructions from the KNOWLEDGE provided.
        
        USER MESSAGE: {last_user_message}
        
        USER PROFILE:
        {profile_summary}
        
        KNOWLEDGE (critically important - extract techniques and instructions from here):
        {knowledge_context if knowledge_context else "No specific knowledge found."}
        
        Provide your response in this exact format:
        
        ANALYSIS: [Your detailed analysis of the user's needs and situation]
        
        RECOMMENDED TECHNIQUES: [EXTRACT the most appropriate techniques or approaches mentioned in the KNOWLEDGE for this user type. DO NOT invent techniques - only include ones found in the knowledge provided]
        
        SPECIFIC INSTRUCTIONS: [COPY-PASTE the EXACT numbered steps from the "Implementation Guide" section of the KNOWLEDGE. Do not change a single word - use the precise wording including all examples and quotation marks. If there's no Implementation Guide section, copy the most detailed steps from the Application Methods sections]
        
        KEY QUERIES: [List 2 specific queries that begin with "Cách thực hiện" (Vietnamese) or "How to implement" (English) followed by the EXACT name of the recommended techniques you identified above. For example: "Cách thực hiện Trấn An Khách Hàng Nhóm Chán Nản" or "How to implement Customer Reassurance Technique"]
        
        IMPORTANT: 
        1. Respond in the SAME LANGUAGE that the user is using
        2. For RECOMMENDED TECHNIQUES, use the EXACT headings from the "Application Methods for [Classification]" section
        3. For SPECIFIC INSTRUCTIONS, COPY-PASTE the full Implementation Guide steps verbatim with quotation marks intact
        4. For KEY QUERIES, format exactly as shown using "Cách thực hiện" for Vietnamese or "How to implement" for English
        5. Do NOT add any explanations or additional text beyond what's specifically requested in each section
        """
        
        # Generate the analysis, techniques, and key queries
        response = await LLM.ainvoke(analysis_prompt,temperature=0.1)
        
        
        full_analysis = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"FULL ANALYSIS RESPONSE: {full_analysis}")
        
        # Parse out the analysis, techniques, instructions and key queries
        analysis_part = ""
        techniques_part = ""
        instructions_part = ""
        key_queries = []
        approach_techniques = []
        
        # Extract analysis
        if "ANALYSIS:" in full_analysis:
            parts = full_analysis.split("ANALYSIS:")[1].split("RECOMMENDED TECHNIQUES:")[0] if "RECOMMENDED TECHNIQUES:" in full_analysis else full_analysis.split("ANALYSIS:")[1]
            analysis_part = parts.strip()
        
        # Extract techniques
        if "RECOMMENDED TECHNIQUES:" in full_analysis:
            parts = full_analysis.split("RECOMMENDED TECHNIQUES:")[1].split("SPECIFIC INSTRUCTIONS:")[0] if "SPECIFIC INSTRUCTIONS:" in full_analysis else full_analysis.split("RECOMMENDED TECHNIQUES:")[1]
            techniques_part = parts.strip()
            
            # Process techniques into a list
            for line in techniques_part.split('\n'):
                line = line.strip()
                if line and len(line) > 3 and "No specific techniques found" not in line:
                    # Clean up the line (remove numbers, bullet points)
                    technique = re.sub(r'^\s*[\d\.\-\*]+\s*', '', line)
                    approach_techniques.append(technique)
            
            logger.info(f"Extracted {len(approach_techniques)} techniques from knowledge: {approach_techniques}")
        
        # Extract specific instructions
        if "SPECIFIC INSTRUCTIONS:" in full_analysis:
            parts = full_analysis.split("SPECIFIC INSTRUCTIONS:")[1].split("KEY QUERIES:")[0] if "KEY QUERIES:" in full_analysis else full_analysis.split("SPECIFIC INSTRUCTIONS:")[1]
            instructions_part = parts.strip()
            logger.info(f"Extracted instructions from knowledge: {instructions_part[:100]}...")
        
        # Extract key queries
        if "KEY QUERIES:" in full_analysis:
            queries_text = full_analysis.split("KEY QUERIES:")[1].strip()
            for line in queries_text.split('\n'):
                if line.strip() and "No specific queries" not in line:
                    # Clean up the line (remove numbers, bullet points)
                    query = re.sub(r'^\s*[\d\.\-\*]+\s*', '', line.strip())
                    if query and len(query) > 3:
                        key_queries.append(query)
        logger.info(f"EXTRACTED KEY QUERIES: {key_queries}")
        # Fallback if format isn't followed
        if not analysis_part:
            analysis_part = full_analysis
        
        if not approach_techniques:
            # Generate basic techniques based on user classification but mark as fallback
            user_classification = user_profile.get("segment", {}).get("category", "general")
            approach_techniques = [
                f"Phương pháp tiếp cận nhóm {user_classification}"
            ]
            logger.info("No techniques found in knowledge - using fallback")
        
        if not key_queries:
            # Generate basic key queries based on techniques or classification
            if approach_techniques:
                key_queries = [f"Cách thực hiện {approach_techniques[0]}"]
            else:
                user_classification = user_profile.get("segment", {}).get("category", "general")
                key_queries = [f"Phương pháp điều trị cho nhóm {user_classification}"]
            logger.info("No key queries extracted - using fallback")
        
        # Store analysis content, techniques and instructions
        analysis_content = analysis_part
        _last_cot_results["analysis_content"] = analysis_content
        _last_cot_results["approach_techniques"] = approach_techniques
        _last_cot_results["specific_instructions"] = instructions_part
        
        # Stream the analysis
        yield {
            "type": "analysis",
            "content": analysis_content,
            "complete": True,
            "thread_id": thread_id,
            "status": "complete",
            "user_profile": user_profile,
            "techniques": approach_techniques,
            "instructions": instructions_part
        }
        logger.info("Analysis and knowledge extraction completed and streamed")
        
        
        # STEP 4: Generate technique-specific queries and fetch implementation knowledge
        technique_knowledge = []
        if brain_loaded and approach_techniques:
            for technique in approach_techniques[:2]:  # Limit to 2 techniques for efficiency
                # Create a query about how to implement this technique
                technique_query = f"Cách thực hiện {technique}"
                logger.info(f"Searching for implementation knowledge with query: {technique_query}")
                
                # Search for knowledge about implementing this technique
                results = await brain.get_similar_vectors_by_text(technique_query, top_k=2)
                
                # Process knowledge results
                for vector_id, vector, metadata, similarity in results:
                    # Skip duplicates
                    if any(entry.get("id") == vector_id for entry in technique_knowledge) or \
                       any(entry.get("id") == vector_id for entry in user_analysis_knowledge):
                        continue
                    
                    raw_text = metadata.get("raw", "")
                    
                    entry = {
                        "id": vector_id,
                        "similarity": float(similarity),
                        "raw": raw_text,
                        "query": technique_query,
                        "phase": "technique_implementation"
                    }
                    technique_knowledge.append(entry)
            
            if technique_knowledge:
                logger.info(f"Found {len(technique_knowledge)} technique implementation knowledge entries")
        
        # STEP 4.5: Get additional knowledge from key queries
        additional_knowledge = []
        if brain_loaded and key_queries:
            # Search for knowledge related to each key query
            for query in key_queries:
                results = await brain.get_similar_vectors_by_text(query, top_k=2)
                
                # Process knowledge results
                for vector_id, vector, metadata, similarity in results:
                    # Skip duplicates
                    if any(entry.get("id") == vector_id for entry in additional_knowledge) or \
                       any(entry.get("id") == vector_id for entry in technique_knowledge) or \
                       any(entry.get("id") == vector_id for entry in user_analysis_knowledge):
                        continue
                    
                    raw_text = metadata.get("raw", "")
                    
                    entry = {
                        "id": vector_id,
                        "similarity": float(similarity),
                        "raw": raw_text,
                        "query": query,
                        "phase": "additional_knowledge"
                    }
                    additional_knowledge.append(entry)
            
            if additional_knowledge:
                logger.info(f"Found {len(additional_knowledge)} additional knowledge entries from key queries")
        
        # STEP 5: Generate next actions using all collected knowledge
        # Format the combined knowledge for the LLM
        technique_context = prepare_knowledge(
            technique_knowledge, 
            last_user_message, 
            target_classification=user_classification
        ) if technique_knowledge else ""
        
        solution_context = prepare_knowledge(
            additional_knowledge, 
            last_user_message,
            target_classification=user_classification
        ) if additional_knowledge else ""
        
        # Store all knowledge for later use
        if technique_knowledge:
            _last_cot_results["technique_knowledge"] = technique_knowledge
            _last_cot_results["technique_context"] = technique_context
            
        if additional_knowledge:
            _last_cot_results["additional_knowledge"] = additional_knowledge
            _last_cot_results["solution_context"] = solution_context
            
        # Combine all knowledge entries for complete context
        _last_cot_results["knowledge_entries"] = user_analysis_knowledge + technique_knowledge + additional_knowledge
        
        # Create a comprehensive knowledge context with all knowledge entries
        all_knowledge = user_analysis_knowledge + technique_knowledge + additional_knowledge
        
        # Trust the LLM to extract relevant information by providing clear instructions
        combined_knowledge = ""
        
        # Add each knowledge source with clear section headers for context
        if knowledge_context:
            combined_knowledge += f"## USER ANALYSIS KNOWLEDGE:\n{knowledge_context}\n\n"
        if technique_context:
            combined_knowledge += f"## TECHNIQUE IMPLEMENTATION KNOWLEDGE:\n{technique_context}\n\n"
        if solution_context:
            combined_knowledge += f"## SOLUTION KNOWLEDGE:\n{solution_context}\n\n"
        
        #### BRIAN MOVED HERE BY 04-05-2025 - JUST BEFORE THE ACTIONS PROMPT ####
        
        {analysis_content}
        
        logger.info(f"RECOMMENDED TECHNIQUES (ALREADY IDENTIFIED): {_last_cot_results.get("approach_techniques", [])}.")
        logger.info(f"SPECIFIC INSTRUCTIONS (ALREADY IDENTIFIED):{_last_cot_results.get("specific_instructions", "")}")

        # Store the combined knowledge for later use
        _last_cot_results["combined_knowledge_context"] = combined_knowledge
        
        # Create a prompt for generating actionable next steps that instructs the LLM
        # to identify important information itself
        actions_prompt = f"""
        Based on the user's message, our analysis, and the knowledge we've found, provide specific, actionable next steps and a persuasive sales script.
        
        USER MESSAGE: {last_user_message}
        
        USER CLASSIFICATION: {user_profile.get("segment", {}).get("category", "general")}
        
        USER PROFILE INFORMATION:
        {format_user_profile_for_prompt(user_profile) if user_profile else ""}
        
        OUR ANALYSIS:
        {analysis_content}
        
        RECOMMENDED TECHNIQUES (ALREADY IDENTIFIED):
        {_last_cot_results.get("approach_techniques", [])}
        
        SPECIFIC INSTRUCTIONS (ALREADY IDENTIFIED):
        {_last_cot_results.get("specific_instructions", "")}
        
        {f"TECHNIQUE IMPLEMENTATION KNOWLEDGE:\n{technique_context}" if technique_context else ""}
        
        {f"SPECIFIC SOLUTIONS:\n{solution_context}" if solution_context else ""}
        
        {f"COMBINED KNOWLEDGE:\n{combined_knowledge}" if combined_knowledge else ""}
        
        ## PART 1: SALES AUTOMATION NEXT STEPS
        NAME and BOLD the specific techniques from the RECOMMENDED TECHNIQUES list.
        Extract the exact actions associated with each technique from the SPECIFIC INSTRUCTIONS.
        THEN ADAPT each step based on information already available in the USER PROFILE and prior conversation context.
        
        IMPORTANT INSTRUCTIONS:
        1. IDENTIFY each recommended technique by name (e.g., "**Trấn An Khách Hàng Nhóm Chán Nản**")
        2. BOLD each technique name using markdown format (with ** before and after)
        3. EXTRACT the exact implementation steps for each technique from the SPECIFIC INSTRUCTIONS
        4. ADAPT implementation steps based on context - SKIP steps requesting information that is ALREADY KNOWN from the USER PROFILE
        5. ORGANIZE these steps in the sequence they appear in the SPECIFIC INSTRUCTIONS, after adaptation
        6. PRESERVE all quotes, examples, and language exactly as written except for the adapted steps
        7. For information that we ALREADY KNOW (like profession, user classification, etc.), DO NOT include steps to gather this information again
        
        FORMATTING REQUIREMENTS:
        1. Start each technique section with its BOLDED name (e.g., "**Technique Name:**")
        2. List the implementation steps under each technique
        3. Use exact wording from the SPECIFIC INSTRUCTIONS for non-adapted steps
        4. Include all quotes exactly as they appear (e.g., "Em hiểu rằng việc này...")
        5. Format as a clean, structured list with clear technique headings
        
        ## PART 2: PERSUASIVE SALES SCRIPT
        SEARCH the COMBINED KNOWLEDGE for detailed instructions on HOW to perform each technique, then create a script that implements these techniques.
        IMPORTANT: The script must be consistent with the ADAPTED steps from Part 1 - do not include script elements for steps that were skipped.
        
        IMPORTANT SCRIPT REQUIREMENTS:
        1. For EACH technique in Part 1, FIND specific implementation details in the COMBINED KNOWLEDGE
        2. LOOK FOR sections labeled "What", "How", "Implementation Guide", or "Bước" (steps)
        3. EXTRACT exact phrases, conversation examples, and quoted text from these sections
        4. STRUCTURE your script to follow the exact sequence of techniques identified in Part 1
        5. COPY the exact wording of quotes, examples, and key phrases from the knowledge
        6. SKIP any dialogue requesting information that's ALREADY KNOWN from the user profile
        
        The script should:
        1. Start with the EXACT empathetic opening phrases found in the knowledge
        2. USE the PRECISE reassurance statements found in the Implementation Guide
        3. INCLUDE the EXACT refund guarantee language as written in the knowledge
        4. INCORPORATE the SPECIFIC expert support phrasing from the knowledge
        5. MAINTAIN the EXACT conversational flow described in the techniques
        6. SKIP any steps asking for information we already have (like profession, classification)
        
        FORMAT THE SALES SCRIPT AS:
        
        PERSUASIVE_SCRIPT:
        [Your script here that EXACTLY follows the ADAPTED SPECIFIC INSTRUCTIONS using the precise language and quotes]
        END_SCRIPT
        
        IMPORTANT:
        - Respond in the SAME LANGUAGE as the user
        - DO NOT CREATE NEW CONTENT - use only what's provided in the instructions and knowledge
        - PRESERVE all quoted text exactly as written
        - EXECUTE the specified techniques precisely as described
        - RECOGNIZE INFORMATION ALREADY IN THE USER PROFILE and avoid steps asking for it again
        """
        
        # Define the system prompt for actions
        system_prompt_actions = "You are a sales specialist that creates persuasive, actionable recommendations based on accurate knowledge."
        
        # Process actions through the LLM
        next_actions_content = ""
        
        # Process chunk by chunk to extract the script
        async for chunk in StreamLLM.astream(actions_prompt):
            content = chunk.content if hasattr(chunk, 'content') else chunk
            if content:
                next_actions_content += content
                
                # Check if we have a complete script section
                if "PERSUASIVE_SCRIPT:" in next_actions_content and "END_SCRIPT" in next_actions_content:
                    script_start = next_actions_content.find("PERSUASIVE_SCRIPT:")
                    script_end = next_actions_content.find("END_SCRIPT", script_start)
                    if script_start != -1 and script_end != -1:
                        persuasive_script = next_actions_content[script_start + len("PERSUASIVE_SCRIPT:"):script_end].strip()
                        _last_cot_results["persuasive_script"] = persuasive_script
                        logger.info(f"Extracted persuasive script: {persuasive_script[:100]}...")
                
                # Stream the content
                yield {"content": content}
        
        # Store and format next actions
        _last_cot_results["next_actions_content"] = next_actions_content
        
        # Clean up the next_actions_content to remove the script section if needed
        if "PERSUASIVE_SCRIPT:" in next_actions_content and "END_SCRIPT" in next_actions_content:
            # Don't split - keep both parts together for next_actions
            # The script section is important and should not be removed
            clean_actions = next_actions_content.replace("END_SCRIPT", "END_SCRIPT\n")
            _last_cot_results["clean_next_actions"] = clean_actions
            
            # Extract the script separately for response generation
            script_start = next_actions_content.find("PERSUASIVE_SCRIPT:")
            script_end = next_actions_content.find("END_SCRIPT", script_start)
            if script_start != -1 and script_end != -1:
                persuasive_script = next_actions_content[script_start + len("PERSUASIVE_SCRIPT:"):script_end].strip()
                _last_cot_results["persuasive_script"] = persuasive_script
                logger.info(f"Extracted persuasive script: {persuasive_script[:100]}...")
        else:
            _last_cot_results["clean_next_actions"] = next_actions_content
        
        logger.info(f"Next actions content length: {len(next_actions_content)} chars")
        
        # Stream next actions to the client
        next_actions_event = {
            "type": "next_actions",
            "content": _last_cot_results["clean_next_actions"],
            "complete": True,
            "thread_id": thread_id,
            "status": "complete",
            "user_profile": user_profile
        }
        
        # Check if content is truncated in the event
        if len(next_actions_event["content"]) < len(next_actions_content):
            logger.warning(f"Next actions content appears to be truncated! Event has {len(next_actions_event['content'])} chars vs original {len(next_actions_content)} chars")
        else:
            logger.info(f"Next actions content correctly sized: {len(next_actions_event['content'])} chars")
        
        # Check if content might be too large for socket (>16KB is a common limit)
        MAX_SOCKET_SIZE = 15000  # ~15KB to be safe
        if len(next_actions_event["content"]) > MAX_SOCKET_SIZE:
            logger.warning(f"Next actions content too large for single socket message: {len(next_actions_event['content'])} chars")
            
            # Split content into chunks
            content = next_actions_event["content"]
            total_chunks = (len(content) + MAX_SOCKET_SIZE - 1) // MAX_SOCKET_SIZE
            
            # Send chunks
            for i in range(total_chunks):
                chunk_start = i * MAX_SOCKET_SIZE
                chunk_end = min((i + 1) * MAX_SOCKET_SIZE, len(content))
                chunk = content[chunk_start:chunk_end]
                
                # Prepare chunk event
                chunk_event = {
                    "type": "next_actions",
                    "content": chunk,
                    "complete": i == total_chunks - 1,  # Only mark the last chunk as complete
                    "thread_id": thread_id,
                    "status": "chunked" if i < total_chunks - 1 else "complete",
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "user_profile": user_profile if i == 0 else None  # Only include profile in first chunk
                }
                
                # Emit the chunk
                if thread_id:
                    try:
                        from socketio_manager import emit_next_action_event
                        logger.info(f"Emitting next_actions chunk {i+1}/{total_chunks} to thread {thread_id}")
                        was_delivered = emit_next_action_event(thread_id, chunk_event)
                        logger.info(f"Emitted next_actions chunk {i+1}/{total_chunks} via socketio_manager")
                    except Exception as e:
                        logger.error(f"Error emitting next_actions chunk {i+1}/{total_chunks}: {str(e)}")
                        yield chunk_event
                else:
                    # Not using WebSockets, just yield the chunk
                    yield chunk_event
                
                # Small delay between chunks to prevent overwhelming the socket
                await asyncio.sleep(0.1)
            
            logger.info(f"Sent next_actions in {total_chunks} chunks")
        else:
            # Emit the next actions event over socket
            if thread_id:
                try:
                    from socketio_manager import emit_next_action_event
                    logger.info(f"Emitting next_actions event to thread {thread_id}")
                    was_delivered = emit_next_action_event(thread_id, next_actions_event)
                    logger.info(f"Emitted next_actions event via socketio_manager for thread {thread_id}")
                except Exception as e:
                    logger.error(f"Error emitting next_actions: {str(e)}")
                    logger.error(traceback.format_exc())
                    yield next_actions_event
        
        logger.info("Next actions completed and streamed")
        
        # Wrap up and return
        yield {"content": "\n\nAnalysis Complete!"}
        logger.info("CoT analysis stream complete")
        return
        
    except Exception as e:
        logger.error(f"Error generating analysis or next actions: {e}")
        
        # Send a minimal analysis if error occurs
        if not analysis_content:
            yield {
                "type": "analysis",
                "content": f"Analyzing query: {last_user_message}",
                "complete": True,
                "thread_id": thread_id,
                "status": "complete",
                "user_profile": user_profile
            }
        yield {
            "type": "next_actions",
            "content": "1. Consider researching more about this topic\n2. Consult relevant resources\n3. Try a more specific query",
            "complete": True,
            "thread_id": thread_id,
            "status": "complete",
            "user_profile": user_profile
        }
    
    # Complete the knowledge event
    all_knowledge = user_analysis_knowledge + (technique_knowledge if 'technique_knowledge' in locals() else []) + (additional_knowledge if 'additional_knowledge' in locals() else [])
    yield {
        "type": "knowledge",
        "content": all_knowledge,
        "complete": True,
        "thread_id": thread_id,
        "status": "complete",
        "stats": {"total_results": len(all_knowledge)}
    }
    
    # Log completion time
    total_time = time.time() - start_time
    logger.info(f"Simplified CoT handler completed in {total_time:.2f}s")

# Register the new CoT tool
tool_registry.register_tool(
    "cot_knowledge_analysis_tool",
    cot_knowledge_analysis_actions_handler,
    COT_KNOWLEDGE_ANALYSIS_SCHEMA
)

# Now modify the process_llm_with_tools function to use the new CoT handler
async def process_llm_with_tools(
    user_message: str,
    conversation_history: List[Dict],
    state: Dict,
    graph_version_id: str,
    thread_id: Optional[str] = None
) -> AsyncGenerator[Union[str, Dict], None]:
    """
    Process a user message using LLM with tool calling.
    Optimized implementation with more parallel processing for better performance.
    
    Args:
        user_message: The user's message
        conversation_history: List of previous messages
        state: Current conversation state
        graph_version_id: Version ID for the knowledge graph
        thread_id: Optional thread ID for WebSocket streaming
        
    Yields:
        Tool results and final response as they become available
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    import traceback
    
    overall_start_time = time.time()
    PERF_METRICS.clear()  # Reset metrics for this request
    PERF_METRICS["overall_start"] = overall_start_time
    
    # OPTIMIZATION: Include more context - use up to 15 messages for richer conversation history
    conversation_context = f"User: {user_message}\n"
    
    # Include up to 15 previous messages for context (increased from just 1-2 turns)
    if conversation_history:
        # Extract the last 15 messages excluding the current
        recent_messages = []
        message_count = 0
        max_messages = 15  # Increased from previous 1-2
        
        for msg in reversed(conversation_history[:-1]):  # Skip the current message
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role and content:
                if role == "assistant":
                    recent_messages.append(f"AI: {content}")
                elif role == "user":
                    recent_messages.append(f"User: {content}")
                
                message_count += 1
                if message_count >= max_messages:
                    break
        
        # Add messages in chronological order
        if recent_messages:
            conversation_context = "\n".join(reversed(recent_messages)) + "\n" + conversation_context
    
    # Track history processing time
    PERF_METRICS["history_processed"] = time.time()
    logger.info(f"Minimal conversation history processed in {time.time() - overall_start_time:.2f}s")
    
    use_websocket = thread_id is not None
    
    # Track events for debugging
    analysis_events_sent = 0
    knowledge_events_sent = 0
    next_actions_events_sent = 0
    
    # Reset the last CoT results
    global _last_cot_results
    _last_cot_results = {
        "analysis_content": "",
        "next_actions_content": "",
        "clean_next_actions": "",
        "knowledge_entries": [],
        "knowledge_context": "",
        "persuasive_script": ""
    }
    
    try:
        # Enhanced personality instructions that adapt to the user's style while remaining sophisticated
        personality_instructions = """
        You are a sophisticated, energetic, and insightful AI assistant with these key traits:
        - Perceptive: You mirror the formality level, energy, and communication style of the user
        - Articulate: You express ideas clearly with rich vocabulary and natural flow
        - Adaptable: You elevate your responses based on the user's expertise level
        - Engaging: You show enthusiasm and genuine interest in the user's needs
        - Persuasive: You present ideas with confidence and compelling reasoning
        """
        
        # CRITICAL FIX: Send an initial event to the frontend
        status_event = {
            "type": "status", 
            "status": "processing", 
            "message": "Processing your request..."
        }
        logger.info(f"Sending initial status event: {status_event}")
        yield status_event
        
        # Use the new CoT handler for combined knowledge retrieval, analysis, and next actions
        logger.info("Using Chain-of-Thought approach for combined knowledge and analysis")
        
        # Setup CoT parameters
        cot_params = {
            "conversation_context": conversation_context,
            "graph_version_id": graph_version_id
        }
        if thread_id:
            cot_params["_thread_id"] = thread_id
        
        # Process all events from the CoT handler
        analysis_results = ""
        next_actions_results = ""
        knowledge_context = ""
        knowledge_entries = []
        knowledge_found = False
        
        # Forward all events from the CoT handler
        try:
            async for result in cot_knowledge_analysis_actions_handler(cot_params):
                # Track event types
                if result.get("type") == "analysis":
                    analysis_events_sent += 1
                elif result.get("type") == "knowledge":
                    knowledge_events_sent += 1
                elif result.get("type") == "next_actions":
                    next_actions_events_sent += 1
                
                # CRITICAL FIX: Ensure each event has the thread_id for socket.io routing
                if thread_id and "thread_id" not in result:
                    result["thread_id"] = thread_id
                
                # Log the event being forwarded
                event_type = result.get("type", "unknown")
                is_complete = result.get("complete", False)
                event_content_preview = str(result.get("content", ""))[:30]
                #logger.info(f"Forwarding CoT {event_type} event (complete: {is_complete}): {event_content_preview}...")
                
                # CRITICAL FIX: Use socketio_manager directly for WebSocket events
                if use_websocket:
                    try:
                        # Import the correct emit function based on event type
                        if event_type == "analysis":
                            from socketio_manager import emit_analysis_event
                            logger.info(f"Emitting analysis event to thread {thread_id} (1 active sessions)")
                            was_delivered = emit_analysis_event(thread_id, result)
                            logger.info(f"Emitted analysis event via socketio_manager for thread {thread_id}")
                        elif event_type == "knowledge":
                            from socketio_manager import emit_knowledge_event
                            logger.info(f"Emitting knowledge event to thread {thread_id} (1 active sessions)")
                            was_delivered = emit_knowledge_event(thread_id, result)
                            logger.info(f"Emitted knowledge event via socketio_manager for thread {thread_id}")
                        elif event_type == "next_actions":
                            from socketio_manager import emit_next_action_event
                            logger.info(f"Emitting next_actions event to thread {thread_id}")
                            was_delivered = emit_next_action_event(thread_id, result)
                            logger.info(f"Emitted next_actions event via socketio_manager for thread {thread_id}")
                        else:
                            # For other event types, just yield them
                            yield result
                    except Exception as e:
                        logger.error(f"Error emitting {event_type} event via socketio_manager: {str(e)}")
                        logger.error(traceback.format_exc())
                        # Fall back to yielding the event
                        yield result
                else:
                    # Not using WebSockets, just yield the event
                    yield result
                
                # Collect results as they come (for backward compatibility)
                if isinstance(result, dict):
                    # Collect analysis results
                    if result.get("type") == "analysis" and result.get("complete", True):
                        analysis_results = result.get("content", "")
                        # Store user profile if available in the analysis result
                        if "user_profile" in result:
                            _last_cot_results["user_profile"] = result["user_profile"]
                    
                    # Collect next_actions results
                    elif result.get("type") == "next_actions" and result.get("complete", True):
                        next_actions_results = result.get("content", "")
                        # Store user profile if available in the next_actions result
                        if "user_profile" in result:
                            _last_cot_results["user_profile"] = result["user_profile"]
                    
                    # Collect knowledge entries
                    elif result.get("type") == "knowledge" and not result.get("complete", False):
                        entries = result.get("content", [])
                        if isinstance(entries, list):
                            knowledge_entries.extend(entries)
        
        except Exception as cot_error:
            # Just log the exception for debugging, without any fallbacks
            logger.error(f"Error in CoT processing: {cot_error}")
            logger.error(traceback.format_exc())
            logger.error(f"CoT processing failed for user message: '{user_message[:100]}...'")
            
            # Emit an error event for tracking purposes
            error_info = {
                "type": "error",
                "component": "cot_handler",
                "error": str(cot_error),
                "thread_id": thread_id,
                "message": "Chain of Thought processing error - no fallback provided"
            }
            yield error_info
        
        # Store the analysis in the state as a dictionary
        if analysis_results:
            # Create a structured analysis object for the state
            analysis_dict = {
                "content": analysis_results,
                "user_profile": _last_cot_results.get("user_profile", {})
            }
            state["analysis"] = analysis_dict
            logger.info("Final analysis stored in state as a dictionary")
        
        # Now check the module-level variable for results if needed
        if not analysis_results and _last_cot_results.get("analysis_content"):
            analysis_results = _last_cot_results["analysis_content"]
            logger.info(f"Using analysis from _last_cot_results: {analysis_results[:50]}...")
        
        if not next_actions_results and _last_cot_results.get("next_actions_content"):
            # Prefer clean next actions if available
            if _last_cot_results.get("clean_next_actions"):
                next_actions_results = _last_cot_results["clean_next_actions"]
                logger.info(f"Using clean next actions from _last_cot_results")
            else:
                next_actions_results = _last_cot_results["next_actions_content"]
                logger.info(f"Using next actions from _last_cot_results")
        
        # For knowledge, prefer what we've collected during streaming, but fall back to module var
        if not knowledge_entries and _last_cot_results["knowledge_entries"]:
            knowledge_entries = _last_cot_results["knowledge_entries"]
            logger.info(f"Using {len(knowledge_entries)} knowledge entries from _last_cot_results")
        
        # Format knowledge context from entries
        if knowledge_entries:
            # First check if we have a pre-formatted combined knowledge context
            if _last_cot_results.get("combined_knowledge_context"):
                knowledge_context = _last_cot_results["combined_knowledge_context"]
                logger.info(f"Using comprehensive combined knowledge context: {len(knowledge_context)} chars")
            # Next check for the regular pre-formatted context
            elif _last_cot_results.get("knowledge_context"):
                knowledge_context = _last_cot_results["knowledge_context"]
                logger.info(f"Using pre-formatted structured knowledge context: {len(knowledge_context)} chars")
            # Technique and solution context can be especially valuable for responses
            elif _last_cot_results.get("technique_context") or _last_cot_results.get("solution_context"):
                # Prioritize knowledge with classification and specific recommendations
                knowledge_context = ""
                if _last_cot_results.get("technique_context"):
                    knowledge_context += f"## RECOMMENDED TECHNIQUES\n{_last_cot_results['technique_context']}\n\n"
                if _last_cot_results.get("solution_context"):
                    knowledge_context += f"## SPECIFIC SOLUTIONS\n{_last_cot_results['solution_context']}\n\n"
                logger.info(f"Using technique and solution context for response: {len(knowledge_context)} chars")
            else:
                # Get user classification from profile if available
                user_classification = None
                if _last_cot_results.get("user_profile") and "segment" in _last_cot_results["user_profile"]:
                    user_classification = _last_cot_results["user_profile"]["segment"].get("category")
                    logger.info(f"Using user classification from profile: {user_classification}")
                
                # First check if we have cached profiling knowledge entries for consistency
                # This ensures we use the same knowledge entries that were used for user profiling
                if "profiling_knowledge_entries" in state and state["profiling_knowledge_entries"]:
                    logger.info("Using cached profiling knowledge entries for consistency")
                    knowledge_entries = state["profiling_knowledge_entries"]
                
                # Let prepare_knowledge do all the work of structuring the knowledge with the user classification
                knowledge_context = prepare_knowledge(
                    knowledge_entries, 
                    user_message, 
                    target_classification=user_classification,
                    max_chars=10000,  # Increased limit for better quality
                    preserve_exact_terminology=True  # Ensure consistent classification terminology
                )
                
                logger.info(f"Created knowledge context using prepare_knowledge: {len(knowledge_context)} chars")
                # Store the knowledge context for future use
                _last_cot_results["knowledge_context"] = knowledge_context
                
            knowledge_found = True
            logger.info(f"KNOWLEDGE FOR RESPONSE: {knowledge_context} chars")
        
        # Log completion of CoT processing
        PERF_METRICS["cot_processing_end"] = time.time()
        cot_time = PERF_METRICS["cot_processing_end"] - PERF_METRICS["history_processed"]
        logger.info(f"CoT processing completed in {cot_time:.2f}s")
        
        # Generate response
        # Let the user know we're generating a response
        response_status = {
            "type": "status", 
            "status": "generating", 
            "message": "Generating response..."
        }
        logger.info(f"Sending response generation status: {response_status}")
        yield response_status
        
        response_params = {
            "conversation_context": conversation_context,
            "analysis": analysis_results,
            "next_actions": next_actions_results,
            "knowledge_context": knowledge_context,
            "personality_instructions": personality_instructions,
            "knowledge_found": knowledge_found,
            "user_profile": _last_cot_results.get("user_profile", {})  # Add user profile to response params
        }
        
        # Add debug logs
        logger.info(f"Starting response generation with params: analysis={len(analysis_results)} chars, next_actions={len(next_actions_results)} chars, knowledge={len(knowledge_context)} chars")
        if _last_cot_results.get("user_profile"):
            # Try to extract key information from the user_profile for logging
            if "portrait" in _last_cot_results["user_profile"]:
                portrait = _last_cot_results["user_profile"]["portrait"]
                # Check for classification in the portrait
                if "Discouraged Group" in portrait:
                    user_classification = "Discouraged Group"
                elif "Confident Group" in portrait:
                    user_classification = "Confident Group"
                else:
                    user_classification = "Unclassified"
                logger.info(f"Including user profile in response generation with classification: {user_classification}")
            else:
                logger.info(f"Including user profile in response generation")
        
        # Stream the response
        response_buffer = ""
        try:
            async for chunk in response_generation_handler(response_params):
                # Response chunks are streamed as strings
                if isinstance(chunk, str):
                    response_buffer += chunk
                    yield chunk
                    # Log chunks as they come in
                    logger.info(f"Response chunk received: {chunk[:30]}...")
        except Exception as response_error:
            logger.error(f"Error in response generation: {response_error}")
            # If we haven't yielded anything yet, provide a fallback response
            if not response_buffer:
                if "portrait" in _last_cot_results.get("user_profile", {}) and "Discouraged Group" in _last_cot_results["user_profile"]["portrait"]:
                    # Tailored fallback for Discouraged Group
                    fallback = "Xuất tinh sớm là tình trạng phổ biến, và rất nhiều nam giới gặp phải. Đây không phải là điều gì đáng xấu hổ. Có nhiều phương pháp hữu ích để cải thiện tình trạng này như các bài tập Kegel, kỹ thuật thở sâu, và phương pháp start-stop. Bạn có thể thử những phương pháp này hoặc tham khảo ý kiến chuyên gia y tế."
                else:
                    # Generic fallback
                    fallback = "Tôi hiểu vấn đề bạn đang gặp phải. Có nhiều phương pháp và kỹ thuật có thể giúp cải thiện tình trạng này. Bạn có thể thử một số bài tập thư giãn và kỹ thuật kiểm soát. Nếu cần thiết, tham khảo ý kiến bác sĩ sẽ giúp bạn có phương pháp điều trị phù hợp."
                
                yield fallback
                response_buffer = fallback
        
        # Log the final response buffer
        logger.info(f"Final response buffer: {response_buffer[:50]}...")
        
        # After all processing, update the state
        if response_buffer:
            state["messages"].append({"role": "assistant", "content": response_buffer})
            state["prompt_str"] = response_buffer
            logger.info(f"Updated state with response of {len(response_buffer)} chars")
        else:
            logger.warning("Response buffer is empty, state not updated")
        
        # Log event counts
        logger.info(f"Total analysis events sent from trigger: {analysis_events_sent}")
        logger.info(f"Total knowledge events sent from trigger: {knowledge_events_sent}")
        logger.info(f"Total next actions events sent from trigger: {next_actions_events_sent}")
        
        # Yield the final state
        yield {"state": state}
        
        # Log overall performance metrics
        PERF_METRICS["overall_end"] = time.time()
        total_time = PERF_METRICS["overall_end"] - PERF_METRICS["overall_start"]
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        # Log detailed performance metrics
        logger.info("Performance metrics:")
        for key, value in PERF_METRICS.items():
            if key.endswith("_start"):
                end_key = key.replace("_start", "_end")
                if end_key in PERF_METRICS:
                    duration = PERF_METRICS[end_key] - value
                    logger.info(f"  {key[:-6]}: {duration:.2f}s")
            
    except Exception as e:
        logger.error(f"Error in LLM tool calling: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log additional debugging information
        logger.error(f"Error occurred during processing of message: '{user_message[:100]}...'")
        logger.error(f"Performance metrics at time of error: {PERF_METRICS}")
        
        # Return a simple error message to the frontend
        error_event = {
            "type": "error",
            "content": "An error occurred during processing.",
            "error": str(e),
            "thread_id": thread_id
        }
        yield error_event
        
        # Also yield a simple text response for compatibility with clients expecting text
        yield "DEBUG: Processing error occurred. Check logs for details."

async def execute_tool_call(tool_call: Dict, params: Dict, thread_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
    """
    Execute a single tool call and handle the results.
    
    Args:
        tool_call: The tool call dictionary containing name and parameters
        params: Common parameters to include with the tool call
        thread_id: Optional thread ID for WebSocket events
        
    Yields:
        Results from the tool handler
    """
    try:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})
        
        # Get handler for this tool
        handler = tool_registry.get_tool_handler(tool_name)
        if not handler:
            logger.error(f"No handler found for tool: {tool_name}")
            yield {"error": f"Tool not found: {tool_name}"}
            return
        
        # Merge args with common params
        if isinstance(tool_args, str):
            try:
                # Parse if it's a JSON string
                tool_args = json.loads(tool_args)
            except:
                # Fallback: create a "query" parameter if parsing fails
                tool_args = {"query": tool_args}
        
        # Ensure tool_args is a dict
        if not isinstance(tool_args, dict):
            tool_args = {"input": str(tool_args)}
        
        # Add common params
        call_params = {**tool_args, **params}
        
        # Add thread ID for WebSocket routing if provided
        if thread_id:
            call_params["_thread_id"] = thread_id
            
        logger.info(f"Executing tool call: {tool_name} with params: {str(call_params)[:100]}...")
        
        # Track tool call start time
        start_time = time.time()
        
        # Execute the tool handler
        try:
            async for result in handler(call_params):
                # Log and yield each result
                if isinstance(result, dict) and "type" in result:
                    event_type = result.get("type")
                    is_complete = result.get("complete", False)
                    logger.info(f"Tool {tool_name} returned {event_type} event (complete: {is_complete})")
                yield result
        except Exception as handler_error:
            logger.error(f"Error in tool handler {tool_name}: {str(handler_error)}")
            yield {"error": f"Tool execution error: {str(handler_error)}"}
        
        # Log completion
        execution_time = time.time() - start_time
        logger.info(f"Tool call {tool_name} completed in {execution_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error executing tool call: {str(e)}")
        logger.error(traceback.format_exc())
        yield {"error": f"Tool execution error: {str(e)}"}
