"""
Tool calling implementation for the MC system.
This file defines the tool schemas and handlers for LLM tool calling.
"""

import json
import re
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import asyncio
from langchain_openai import ChatOpenAI
from utilities import logger
from analysis import (
    stream_analysis, 
    stream_next_action,
    build_context_analysis_prompt,
    build_next_actions_prompt
)
from brain_singleton import get_brain, set_graph_version, is_brain_loaded, load_brain_vectors, get_current_graph_version

from response_optimization import ResponseProcessor, ResponseFilter

# Use the same LLM instances as mc.py
LLM = ChatOpenAI(model="gpt-4o", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

# Access the brain singleton once
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


# Tool Schemas - These define the interface for each tool
CONTEXT_ANALYSIS_SCHEMA = {
    "name": "context_analysis_tool",
    "description": "Analyze the conversation context to understand user needs, identify intents, and determine key information",
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
            "additional_instructions": {
                "type": "string",
                "description": "Optional additional instructions for the analysis",
                "default": ""
            }
        },
        "required": ["conversation_context", "graph_version_id"]
    }
}

NEXT_ACTIONS_SCHEMA = {
    "name": "next_actions_tool",
    "description": "Determine the next actions to take based on conversation context, analysis, and available knowledge",
    "parameters": {
        "type": "object",
        "properties": {
            "conversation_context": {
                "type": "string",
                "description": "The conversation history"
            },
            "context_analysis": {
                "type": "string",
                "description": "The analysis of the conversation context"
            },
            "knowledge_context": {
                "type": "string",
                "description": "The retrieved knowledge relevant to the conversation"
            }
        },
        "required": ["conversation_context", "context_analysis", "knowledge_context"]
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
        "required": ["conversation_context", "analysis", "next_actions", "personality_instructions","knowledge_context"]
    }
}

# Tool Handlers - These implement the actual functionality
async def context_analysis_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Handle context analysis requests with streaming
    
    Args:
        params: Dictionary containing conversation_context, graph_version_id, and optional additional_instructions
        
    Yields:
        Dict events with streaming analysis results
    """
    conversation_context = params.get("conversation_context", "")
    graph_version_id = params.get("graph_version_id", "")
    additional_instructions = params.get("additional_instructions", "")
    
    process_instructions = additional_instructions
    profiling_instructions = {}  # New structured instructions dictionary
    
    try:
        if graph_version_id:
            # Ensure the brain is loaded with the right version
            if graph_version_id != get_current_graph_version():
                logger.info(f"Setting graph version to: {graph_version_id}")
                version_changed = set_graph_version(graph_version_id)
                if version_changed and not is_brain_loaded():
                    logger.info(f"Loading vectors for graph version: {graph_version_id}")
                    await load_brain_vectors()
            
            # Make sure brain is loaded
            if not is_brain_loaded():
                logger.info("Brain not loaded. Loading vectors...")
                await load_brain_vectors()
            
            # Get brain instance
            global brain
            brain = get_brain()
                
            # Enhanced profiling instructions - Expanded query categories
            # Bilingual profile query (English and Vietnamese)
            profile_queries = [
                "How to build customer portraits?",
                "Làm sao để xây dựng chân dung khách hàng?",
                "Customer profiling techniques",
                "Các kỹ thuật xây dựng hồ sơ khách hàng"
            ]
            
            # Improved classification query with abstract terms
            classification_queries = [
                "customer classification frameworks",
                "Kỹ thuật phân nhóm, phân loại khách hàng",
                "How to know what category of customer is contacting me?",
                "Làm thế nào để biết người liên hệ với tôi thuộc loại nào?",
                "customer segmentation types",
                "Các loại phân khúc khách hàng"
            ]
            
            # Demographic profiling queries
            demographic_queries = [
                "techniques for identifying customer demographic information",
                "cách xác định thông tin nhân khẩu học của khách hàng",
                "how to determine customer age groups and professional background",
                "phương pháp phân loại khách hàng theo độ tuổi và nghề nghiệp"
            ]
            
            # Psychographic profiling queries
            psychographic_queries = [
                "methods to identify customer values and interests",
                "cách xác định giá trị và sở thích của khách hàng",
                "identifying customer lifestyle and personality traits",
                "kỹ thuật nhận biết phong cách sống và tính cách khách hàng"
            ]
            
            # Behavioral profiling queries
            behavioral_queries = [
                "techniques to analyze customer purchase behavior",
                "phương pháp phân tích hành vi mua hàng của khách hàng",
                "how to identify customer buying patterns and preferences",
                "cách nhận diện kiểu mẫu và sở thích mua hàng của khách hàng"
            ]
            
            # Industry-specific customer types
            industry_specific_queries = [
                "common customer types in retail industry",
                "các loại khách hàng phổ biến trong ngành bán lẻ",
                "financial services customer personas",
                "chân dung khách hàng trong lĩnh vực dịch vụ tài chính",
                "technology product customer profiles",
                "mẫu khách hàng điển hình cho sản phẩm công nghệ"
            ]
            
            # Context-aware profiling techniques
            context_aware_queries = [
                "profiling techniques for first contact with customer",
                "kỹ thuật xây dựng hồ sơ trong lần tiếp xúc đầu tiên với khách hàng",
                "how to update customer profiles in ongoing relationships",
                "cách cập nhật thông tin khách hàng trong mối quan hệ dài hạn",
                "customer profiling based on communication channels",
                "phân loại khách hàng dựa trên kênh giao tiếp"
            ]
            
            # Emotional and cultural assessment
            emotional_cultural_queries = [
                "techniques to assess customer emotional state",
                "phương pháp đánh giá trạng thái cảm xúc của khách hàng",
                "cultural nuances in Vietnamese customer communication",
                "đặc điểm văn hóa trong giao tiếp với khách hàng Việt Nam",
                "identifying customer satisfaction signals",
                "nhận biết dấu hiệu hài lòng của khách hàng"
            ]
            
            # Combine all queries for batch processing - prioritizing the most relevant categories first
            all_queries = (
                profile_queries + 
                classification_queries + 
                psychographic_queries + 
                behavioral_queries + 
                emotional_cultural_queries + 
                context_aware_queries + 
                demographic_queries + 
                industry_specific_queries
            )
            
            # Execute batch search with simplified approach
            try:
                # Use batch similarity search for all queries at once
                batch_results = await brain.batch_similarity_search(all_queries, top_k=10)
                
                # Process results
                combined_entries = []
                seen_ids = set()
                
                # Process each query's results
                for query in all_queries:
                    if query in batch_results:
                        results = batch_results[query]
                        
                        for vector_id, vector, metadata, similarity in results:
                            if vector_id not in seen_ids:
                                seen_ids.add(vector_id)
                                
                                # Create an entry with the necessary information
                                entry = {
                                    "id": vector_id,
                                    "raw": metadata.get("raw", ""),
                                    "similarity": float(similarity)
                                }
                                combined_entries.append(entry)
                                
                                # Extract structured data for profiling_instructions
                                raw_text = metadata.get("raw", "")
                                if raw_text:
                                    structured_data = extract_structured_data_from_raw(raw_text)
                                    if structured_data:
                                        profiling_instructions[vector_id] = structured_data
                
                # Log the number of retrieved entries
                if combined_entries:
                    logger.info(f"Retrieved {len(combined_entries)} entries for context analysis")
                else:
                    logger.warning(f"No entries found for graph_version_id: {graph_version_id}.")
            
            except Exception as batch_error:
                logger.warning(f"Batch search failed: {batch_error}. Falling back to individual queries.")
                
                # Fall back to individual queries if batch search fails
                combined_entries = []
                seen_ids = set()
                
                for query in all_queries:
                    try:
                        results = await brain.get_similar_vectors_by_text(query, top_k=10)
                        
                        for vector_id, vector, metadata, similarity in results:
                            if vector_id not in seen_ids:
                                seen_ids.add(vector_id)
                                
                                # Create an entry with the necessary information
                                entry = {
                                    "id": vector_id,
                                    "raw": metadata.get("raw", ""),
                                    "similarity": float(similarity)
                                }
                                combined_entries.append(entry)
                                
                                # Extract structured data for profiling_instructions
                                raw_text = metadata.get("raw", "")
                                if raw_text:
                                    structured_data = extract_structured_data_from_raw(raw_text)
                                    if structured_data:
                                        profiling_instructions[vector_id] = structured_data
                                
                    except Exception as query_error:
                        logger.error(f"Individual query failed for '{query}': {query_error}")
                
                # Log the number of retrieved entries from individual queries
                if combined_entries:
                    logger.info(f"Retrieved {len(combined_entries)} entries for context analysis from individual queries")
                else:
                    logger.warning(f"No entries found for graph_version_id: {graph_version_id}.")
    
    except Exception as e:
        logger.error(f"Error fetching knowledge from brain: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Store profiling_instructions in params for potential later use
    params["profiling_instructions"] = profiling_instructions
    logger.info(f"Profiling instructions: {profiling_instructions}")
    analysis_prompt = build_context_analysis_prompt(conversation_context, profiling_instructions)
    
    # Stream the analysis
    thread_id = params.get("_thread_id")
    use_websocket = thread_id is not None
    
    async for chunk in stream_analysis(analysis_prompt, thread_id, use_websocket):
        yield chunk

async def next_actions_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Handle next actions generation with streaming
    
    Args:
        params: Dictionary containing conversation_context, context_analysis, and knowledge_context
        
    Yields:
        Dict events with streaming next action results
    """
    conversation_context = params.get("conversation_context", "")
    context_analysis = params.get("context_analysis", "")
    knowledge_context = params.get("knowledge_context", "")
    
    # Build the next actions prompt
    next_actions_prompt = build_next_actions_prompt(
        conversation_context, 
        context_analysis, 
        knowledge_context
    )
    
    # Use existing stream_next_action function
    thread_id = params.get("_thread_id")  # Internal param for WebSocket
    use_websocket = thread_id is not None
    
    async for chunk in stream_next_action(next_actions_prompt, thread_id, use_websocket):
        yield chunk


async def knowledge_query_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Handle knowledge query requests with direct brain access
    
    Args:
        params: Dictionary containing queries, graph_version_id, and optional top_k
        
    Yields:
        Dict events with streaming knowledge query results
    """
    queries = params.get("queries", [])
    graph_version_id = params.get("graph_version_id", "")
    top_k = params.get("top_k", 3)
    thread_id = params.get("_thread_id")  # Internal param for WebSocket
    
    # Handle empty queries
    if not queries:
        yield {"type": "knowledge", "content": [], "complete": True}
        return
    
    try:
        # Set graph version if provided and different from current
        if graph_version_id and graph_version_id != get_current_graph_version():
            logger.info(f"Setting graph version to: {graph_version_id}")
            version_changed = set_graph_version(graph_version_id)
            if version_changed and not is_brain_loaded():
                # Need to load vectors for new version
                logger.info(f"Loading vectors for graph version: {graph_version_id}")
                success = await load_brain_vectors()
                if not success:
                    logger.error(f"Failed to load vectors for graph version {graph_version_id}")
                    yield {"type": "knowledge", "content": [], "complete": True, 
                           "error": f"Failed to load vectors for graph version {graph_version_id}"}
                    return
        
        # Ensure brain is loaded
        if not is_brain_loaded():
            logger.info("Brain not loaded. Loading vectors...")
            success = await load_brain_vectors()
            if not success:
                logger.error("Failed to load brain vectors")
                yield {"type": "knowledge", "content": [], "complete": True, 
                       "error": "Brain not loaded. Please activate brain first."}
                return
        
        # Get global brain instance
        global brain
        brain = get_brain()
        
        # Process queries and collect results
        all_results = []
        seen_ids = set()
        
        try:
            # Use batch_similarity_search for efficient querying
            logger.info(f"Running batch similarity search for {len(queries)} queries")
            batch_results = await brain.batch_similarity_search(queries, top_k=top_k)
            
            # Process and stream results
            for query_idx, query in enumerate(queries):
                if query in batch_results:
                    results = batch_results[query]
                    new_results = []
                    
                    for vector_id, vector, metadata, similarity in results:
                        if vector_id not in seen_ids:
                            seen_ids.add(vector_id)
                            
                            # Format the result
                            result = {
                                "id": vector_id,
                                "similarity": float(similarity),  # Ensure JSON serializable
                                "metadata": metadata,
                                "vector_preview": vector[:10].tolist() if hasattr(vector, "tolist") else vector[:10],
                                "raw": metadata.get("raw", ""),
                                "query": query,
                                "query_idx": query_idx
                            }
                            new_results.append(result)
                            all_results.append(result)
                    
                    # Stream this batch of results
                    if new_results:
                        yield {"type": "knowledge", "content": new_results, "complete": False}
        
        except Exception as batch_error:
            logger.warning(f"Batch search failed: {batch_error}. Falling back to individual queries.")
            
            # Fall back to individual queries
            for query_idx, query in enumerate(queries):
                try:
                    results = await brain.get_similar_vectors_by_text(query, top_k=top_k)
                    new_results = []
                    
                    for vector_id, vector, metadata, similarity in results:
                        if vector_id not in seen_ids:
                            seen_ids.add(vector_id)
                            
                            # Format the result
                            result = {
                                "id": vector_id,
                                "similarity": float(similarity),
                                "metadata": metadata,
                                "vector_preview": vector[:10].tolist() if hasattr(vector, "tolist") else vector[:10],
                                "raw": metadata.get("raw", ""),
                                "query": query,
                                "query_idx": query_idx
                            }
                            
                            new_results.append(result)
                            all_results.append(result)
                    
                    # Stream this query's results
                    if new_results:
                        yield {"type": "knowledge", "content": new_results, "complete": False}
                        
                except Exception as e:
                    logger.error(f"Individual query failed for '{query}': {e}")
        
        # Final complete event with stats
        yield {
            "type": "knowledge", 
            "content": all_results, 
            "complete": True, 
            "stats": {
                "total_results": len(all_results),
                "query_count": len(queries),
                "graph_version": get_current_graph_version()
            }
        }
        logger.info(f"Completed knowledge search with {len(all_results)} total results")
        
    except Exception as e:
        logger.error(f"Error in knowledge query: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        yield {"type": "knowledge", "content": [], "complete": True, "error": str(e)}


async def response_generation_handler(params: Dict) -> AsyncGenerator[str, None]:
    """
    Handle response generation with streaming
    
    Args:
        params: Dictionary containing conversation_context, analysis, next_actions, 
                knowledge_context, personality_instructions, and knowledge_found
                
    Yields:
        Streamed response chunks as strings
    """
    conversation_context = params.get("conversation_context", "")
    analysis = params.get("analysis", "")
    next_actions = params.get("next_actions", "")
    knowledge_context = params.get("knowledge_context", "")
    personality_instructions = params.get("personality_instructions", "")
    knowledge_found = params.get("knowledge_found", False)
    
    # Detect language from analysis or conversation context
    detected_language = "unknown"
    if "Vietnamese" in analysis or "tiếng Việt" in analysis:
        detected_language = "vietnamese"
    elif "English" in analysis:
        detected_language = "english"
    else:
        # Simple language detection based on common Vietnamese words
        vietnamese_markers = ["của", "những", "và", "các", "là", "không", "có", "được", "người", "trong", "để", "anh", "chị", "em", "ơi", "nhé"]
        english_markers = ["the", "and", "is", "in", "to", "of", "that", "for", "you", "with", "have", "this", "are"]
        
        vietnamese_count = sum(1 for marker in vietnamese_markers if f" {marker} " in f" {conversation_context} ")
        english_count = sum(1 for marker in english_markers if f" {marker} " in f" {conversation_context} ")
        
        if vietnamese_count > english_count:
            detected_language = "vietnamese"
        else:
            detected_language = "english"
    
    # Extract user name or pronoun from conversation context
    user_name = None
    last_user_message = ""
    
    # Get the last user message
    for line in conversation_context.split('\n'):
        if line.startswith("User:"):
            last_user_message = line[5:].strip()
    
    # Build language and cultural awareness instructions
    cultural_instructions = """
        Respond in the detected language with a natural and culturally aware tone. Adapt to the cultural context implied by the language by following these principles:

        1. Choose forms of address or expressions that reflect the social and relational dynamics inferred from the user's communication style and context.
        2. Initiate or respond with a tone that feels appropriate to the conversation's flow, balancing respect and familiarity as needed.
        3. Incorporate linguistic nuances or conversational elements that enhance connection and align with natural speech patterns in the detected language.
        4. Adjust your tone and level of formality to match the user's style, evolving with the interaction.
        5. Present ideas or suggestions thoughtfully, respecting the user's perspective and cultural expectations.
        6. Ensure your language feels authentic and fluent, mirroring the natural rhythms and idioms of the detected language.

        Focus on embodying the cultural values and norms associated with the detected language, fostering rapport through empathy and adaptability while personalizing your tone to the user's cues.
        """
    
    # Build the response prompt
    prompt_parts = [
        "# Conversation Context",
        conversation_context,
        "# Analysis",
        analysis,
        "# Next Actions",
        next_actions
    ]
    
    if knowledge_context:
        prompt_parts.extend([
            "# Retrieved Knowledge",
            knowledge_context
        ])
    
    prompt_parts.extend([
        "# Personality Instructions",
        personality_instructions,
        "# Language and Cultural Guidelines",
        cultural_instructions,
        
        "# Your Task",
        """
            You are an empathetic AI assistant helping users with their concerns. Your goal is to provide a thoughtful, natural-sounding response tailored to this specific situation.
            
            Follow ALL of these requirements in your response:
            
            1. IMPLEMENT THE PRIMARY OBJECTIVE: Begin with the single most important goal identified in the next actions plan.
               
            2. FOLLOW THE COMMUNICATION APPROACH: Your message must:
               - Use exactly the TONE specified (formal, casual, empathetic, etc.)
               - Match the STYLE recommended (direct/indirect, structured/flexible)
               - Incorporate the specific CULTURAL CONSIDERATIONS mentioned
            
            3. APPLY THE KEY TECHNIQUES: For each technique mentioned in next actions:
               - Implement it exactly as described in the APPLICATION section
               - Incorporate relevant knowledge from the SOURCE quotes provided
               - Maintain the original intent and approach of each technique
            
            4. INCLUDE ALL RECOMMENDED ELEMENTS:
               - Begin with the specific OPENING approach outlined
               - Cover all KEY POINTS in the priority order listed
               - Ask the exact QUESTIONS recommended (when provided)
               - End with the suggested CLOSING approach

            5. PRIORITIZE KNOWLEDGE INTEGRATION:
               - Directly apply knowledge from the Retrieved Knowledge section
               - Use specific facts, methods, and information from knowledge sources
               - Reference relevant concepts or techniques mentioned in the knowledge
               - Prioritize domain-specific knowledge that addresses user needs
               - Maintain factual accuracy based on knowledge context
            
            6. PRIORITIZE ACTIONS: Implement the 3-5 specific actions listed in the PRIORITY NEXT ACTIONS section, maintaining their exact order of importance.
            
            7. SOUND NATURAL: While following all the above requirements:
               - Blend technical approaches and knowledge seamlessly into natural dialogue
               - Use the correct language (English or Vietnamese) as detected
               - Keep your response concise (80-100 words)
               - Avoid formulaic closings or repetitive reassurances
            
            Your response will be evaluated based on how faithfully you implement the specific next actions plan and integrate relevant knowledge while maintaining natural conversation flow.
        """
        ])
    
    prompt = "\n\n".join(prompt_parts)
    
    # Response optimization components
    response_processor = ResponseProcessor()
    response_filter = ResponseFilter()
    
    # Check if this is the first message
    is_first_message = "User:" in conversation_context and conversation_context.count("User:") <= 1
    
    # Collect the full response first
    full_response = ""
    buffer = ""
    
    try:
        async for chunk in StreamLLM.astream(prompt):
            content = chunk.content
            buffer += content
            full_response += content
            
            # Only yield periodically to reduce overhead
            if len(buffer) >= 20 or content.endswith((".", "!", "?", "\n")):
                yield buffer
                buffer = ""
        
        # Yield any remaining buffer
        if buffer:
            yield buffer
        
        # Process the full response for optimization
        processed_response = response_processor.process_response(
            full_response, 
            {"messages": conversation_context.split("\n")},
            conversation_context.split("\n")[-1] if conversation_context.split("\n") else "",
            knowledge_found
        )
        
        # Apply greeting filter if needed
        if not is_first_message:
            processed_response = response_filter.remove_greeting(processed_response)
        
        # We've already yielded the raw response, so we don't need to yield again
        
    except Exception as e:
        logger.error(f"Error in response generation: {str(e)}")
        # Fall back to language-specific error messages
        if detected_language == "vietnamese":
            yield "Xin lỗi, tôi đang gặp sự cố khi xử lý yêu cầu của bạn. Vui lòng thử lại."
        else:
            yield "I encountered an issue while generating a response. Let me try again."

# Register the tools
tool_registry.register_tool(
    "context_analysis_tool", 
    context_analysis_handler, 
    CONTEXT_ANALYSIS_SCHEMA
)

tool_registry.register_tool(
    "next_actions_tool", 
    next_actions_handler, 
    NEXT_ACTIONS_SCHEMA
)

tool_registry.register_tool(
    "knowledge_query_tool", 
    knowledge_query_handler, 
    KNOWLEDGE_QUERY_SCHEMA
)

tool_registry.register_tool(
    "response_generation_tool", 
    response_generation_handler, 
    RESPONSE_GENERATION_SCHEMA
)

async def execute_tool_call(tool_call: Dict) -> AsyncGenerator[Dict, None]:
    """
    Execute a tool call and return the result
    
    Args:
        tool_call: Dict with name and parameters for the tool call
        
    Yields:
        Tool results as they become available
    """
    tool_name = tool_call.get("name")
    parameters = tool_call.get("parameters", {})
    
    try:
        # Get the tool handler
        handler = tool_registry.get_tool_handler(tool_name)
        
        # Execute the tool
        async for result in handler(parameters):
            # If the handler already returns properly formatted results,
            # just pass them through
            if isinstance(result, dict) and result.get("type") in ["analysis", "knowledge", "next_actions"]:
                yield result
            else:
                # Otherwise, wrap the result
                yield {
                    "tool_call_id": tool_call.get("id", "unknown"),
                    "tool_name": tool_name,
                    "content": result
                }
            
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        yield {
            "tool_call_id": tool_call.get("id", "unknown"),
            "tool_name": tool_name,
            "content": {"error": str(e)},
            "error": True
        }

async def process_llm_with_tools(
    user_message: str,
    conversation_history: List[Dict],
    state: Dict,
    graph_version_id: str,
    thread_id: Optional[str] = None
) -> AsyncGenerator[Union[str, Dict], None]:
    """
    Process a user message using LLM with tool calling.
    This function mirrors the flow of MC._handle_request to ensure functional equivalence
    while using the tool-based architecture.
    
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
    
    # Prepare messages for LLM context
    messages = []
    
    # Convert conversation history to string format
    conversation_context = ""
    current_message_in_history = False
    
    # Check if the current message is already in the history
    for msg in conversation_history[-100:]:
        if msg.get("role") == "user" and msg.get("content") == user_message:
            current_message_in_history = True
            break
    
    # Format the conversation history
    prev_ai_msg = None
    for msg in conversation_history[-100:]:
        if msg.get("role") == "user":
            conversation_context += f"User: {msg.get('content', '')}\n"
            prev_ai_msg = None
        elif msg.get("role") == "assistant":
            # Skip duplicate consecutive AI messages
            current_content = msg.get('content', '')
            if prev_ai_msg == current_content:
                logger.info("Skipping duplicate consecutive AI message")
                continue
            conversation_context += f"AI: {current_content}\n"
            prev_ai_msg = current_content
    
    # Add current user message only if it's not already in the history
    if not current_message_in_history:
        conversation_context += f"User: {user_message}\n"
    
    use_websocket = thread_id is not None
    thread_id_for_analysis = thread_id
    
    try:
        # STEP 1: Load personality if needed
        from personality import PersonalityManager
        personality_manager = PersonalityManager()
        if not hasattr(personality_manager, 'personality_instructions') or not personality_manager.personality_instructions:
            await personality_manager.load_personality(graph_version_id)
        
        # STEP 2: Detect language (simplified - handled within tools)
        
        # STEP 3: Run context analysis
        logger.info(f"Starting context analysis for message: {user_message[:50]}...")
        context_params = {
            "conversation_context": conversation_context,
            "graph_version_id": graph_version_id
        }
        if thread_id:
            context_params["_thread_id"] = thread_id
            
        analysis_results = ""
        async for result in context_analysis_handler(context_params):
            # Pass through the analysis events
            yield {"type": "analysis", "content": result}
            
            # If this is a complete event, store the full analysis
            if isinstance(result, dict) and result.get("complete", False):
                analysis_results = result.get("content", "")
        
        # STEP 4: Extract search terms from analysis
        if analysis_results:
            logger.info(f"Context analysis complete, extracting search terms...")
            from analysis import extract_search_terms
            search_terms = extract_search_terms(analysis_results)
            logger.info(f"Extracted {len(search_terms)} search terms: {search_terms}")
        else:
            # Fallback to basic search terms
            search_terms = [user_message]
            logger.info(f"Using fallback search term: {user_message}")
        
        # STEP 5: Run initial knowledge query based on analysis using new approach
        logger.info(f"Starting initial knowledge query with {len(search_terms)} terms...")
        knowledge_params = {
            "queries": search_terms,
            "graph_version_id": graph_version_id,
            "top_k": 3
        }
        if thread_id:
            knowledge_params["_thread_id"] = thread_id
            
        all_knowledge_entries = []
        async for result in knowledge_query_handler(knowledge_params):
            # Pass through the knowledge events directly
            yield result
            
            # Collect knowledge entries
            if isinstance(result, dict) and "content" in result:
                entries = result.get("content", [])
                if isinstance(entries, list):
                    all_knowledge_entries.extend(entries)
        
        # Format initial knowledge context
        if all_knowledge_entries:
            knowledge_context = "\n\n".join(entry.get("raw", "") for entry in all_knowledge_entries)
            knowledge_found = True
            logger.info(f"Retrieved {len(all_knowledge_entries)} initial knowledge entries")
        else:
            knowledge_context = "No relevant knowledge found."
            knowledge_found = False
            logger.info("No initial knowledge entries found")
        
        # STEP 6: Run next actions
        logger.info(f"Starting next actions determination...")
        next_actions_params = {
            "conversation_context": conversation_context,
            "context_analysis": analysis_results,
            "knowledge_context": knowledge_context
        }
        if thread_id:
            next_actions_params["_thread_id"] = thread_id
            
        next_actions_results = ""
        next_action_error = False
        async for result in next_actions_handler(next_actions_params):
            # Pass through the next actions events directly without double-wrapping
            yield result
            
            # Check for error flag
            if result.get("error", False):
                next_action_error = True
                logger.warning(f"Received error in next action: {result.get('content', '')}")
            
            # If this is a complete event, store the full next actions
            if isinstance(result, dict) and result.get("complete", False):
                next_actions_results = result.get("content", "")
        
        # STEP 6.5: Extract search terms from next actions and retrieve additional targeted knowledge
        additional_knowledge_entries = []
        if not next_action_error and next_actions_results and len(next_actions_results) > 50:
            try:
                logger.info("Extracting search terms from next actions for targeted HOW-TO knowledge retrieval")
                from analysis import extract_search_terms_from_next_actions, process_next_actions_result
                
                # Process the next_actions_results
                next_actions_data = process_next_actions_result(next_actions_results)
                
                # Make sure we have valid data
                if not next_actions_data or not isinstance(next_actions_data, dict):
                    logger.warning(f"Invalid next_actions_data format: {next_actions_data}")
                    next_actions_data = {"next_action_english": "", "next_action_vietnamese": "", "next_action_full": next_actions_results}
                
                # Extract terms from the next actions
                logger.info(f"Using complete next actions for targeted knowledge queries...")
                action_search_terms = extract_search_terms_from_next_actions(next_actions_results)
                
                if action_search_terms:
                    logger.info(f"Extracted {len(action_search_terms)} search terms from next actions: {action_search_terms}")
                    
                    # Run a secondary knowledge query with terms from next actions - use the same handler
                    logger.info(f"Starting secondary knowledge query for HOW-TO knowledge...")
                    action_knowledge_params = {
                        "queries": action_search_terms,
                        "graph_version_id": graph_version_id,
                        "top_k": 3
                    }
                    if thread_id:
                        action_knowledge_params["_thread_id"] = thread_id
                    
                    # Query using the same improved handler
                    async for action_result in knowledge_query_handler(action_knowledge_params):
                        # Pass through the knowledge events directly
                        yield action_result
                        
                        # Collect additional knowledge entries
                        if isinstance(action_result, dict) and "content" in action_result:
                            add_entries = action_result.get("content", [])
                            if isinstance(add_entries, list):
                                additional_knowledge_entries.extend(add_entries)
                else:
                    logger.info("No additional search terms extracted from next actions")
            except Exception as e:
                logger.error(f"Error in extracting search terms from next actions: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Combine initial and additional knowledge
        if additional_knowledge_entries:
            logger.info(f"Retrieved {len(additional_knowledge_entries)} additional knowledge entries from next actions")
            
            # Deduplicate by ID
            all_entries_combined = all_knowledge_entries.copy()
            seen_ids = {entry.get("id", "unknown") for entry in all_knowledge_entries}
            
            for entry in additional_knowledge_entries:
                entry_id = entry.get("id", "unknown")
                if entry_id not in seen_ids:
                    seen_ids.add(entry_id)
                    all_entries_combined.append(entry)
            
            # Format combined knowledge context
            knowledge_context = "\n\n".join(entry.get("raw", "") for entry in all_entries_combined)
            knowledge_found = True
            logger.info(f"Using combined {len(all_entries_combined)} knowledge entries for response generation")
        
        # STEP 7: Assess knowledge coverage (simplified)
        # In the original _handle_request this would call assess_knowledge_coverage
        
        # STEP 8: Prepare and send final response prompt
        logger.info(f"KNOWLEDGE TO APPLY FOR FINAL RESPONSE: {knowledge_context}")
       
        # Build the final response prompt with all the components
        response_params = {
            "conversation_context": conversation_context,
            "analysis": analysis_results,
            "next_actions": next_actions_results,
            "knowledge_context": knowledge_context,
            "personality_instructions": personality_manager.personality_instructions,
            "knowledge_found": knowledge_found
        }
        
        # Stream the response
        response_buffer = ""
        async for chunk in response_generation_handler(response_params):
            # Response chunks are streamed as strings
            if isinstance(chunk, str):
                response_buffer += chunk
                yield chunk
        
        # After all processing, update the state
        if response_buffer:
            state["messages"].append({"role": "assistant", "content": response_buffer})
            state["prompt_str"] = response_buffer
        
        # Yield the final state
        yield {"state": state}
        
    except Exception as e:
        logger.error(f"Error in LLM tool calling: {str(e)}")
        logger.error(traceback.format_exc())
        yield "I encountered an issue while processing your request. Please try again." 