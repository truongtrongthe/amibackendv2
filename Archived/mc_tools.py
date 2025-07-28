"""
MC implementation with tool calling integration.
This modified version of the MC class uses LLM tool calling for enhanced capabilities.
"""

import asyncio
from typing import List, Dict, Optional, Any, AsyncGenerator, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from utilities import logger
import json

#from tools import process_llm_with_tools, tool_registry
#from tools_new import process_llm_with_tools
from tool6 import process_llm_with_tools
#from personality import PersonalityManager
from response_optimization import ResponseFilter, ResponseProcessor

# Use the same LLM instances as mc.py
LLM = ChatOpenAI(model="gpt-4o-mini", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o-mini", streaming=True)

def add_messages(existing_messages, new_messages):
    return existing_messages + new_messages

class ResponseBuilder:
    def __init__(self):
        self.parts = []
        self.error_flag = False

    def add(self, text: str, is_error: bool = False):
        if text:
            self.parts.append(text)
        self.error_flag = self.error_flag or is_error
        return self

    def build(self, separator: str = None) -> str:
        if not self.parts:
            return "Tôi không biết nói gì cả!"
        separator = separator or ""
        return separator.join(part for part in self.parts if part)

class MCWithTools:
    """
    MC implementation that uses LLM tool calling for enhanced capabilities.
    This class preserves the critical components from the original MC:
    1. Analysis with streaming
    2. Next action determination with streaming
    3. Response generation
    """

    def __init__(self, user_id: str = "thefusionlab", convo_id: str = None, 
                 similarity_threshold: float = 0.55, max_active_requests: int = 5):
        self.user_id = user_id
        self.convo_id = convo_id or "default_thread"
        self.similarity_threshold = similarity_threshold
        self.max_active_requests = max_active_requests
        self.personality_manager = ""
        self.response_filter = ResponseFilter()
        self.response_processor = ResponseProcessor()
        self.state = {
            "messages": [],
            "intent_history": [],
            "preset_memory": "Be friendly",
            "unresolved_requests": [],
            "convo_id": self.convo_id,
            "user_id": self.user_id,
            "prompt_str": "",
            "graph_version_id": "",
            "analysis": {
                "english": "",
                "vietnamese": ""
            }
        }
        self._cache = {
            'personality': {},
            'language': {},
            'knowledge': {},
            'analysis': {},
            'query_embeddings': {}
        }

    @property
    def name(self):
        return self.personality_manager.name

    @property
    def personality_instructions(self):
        return self.personality_manager.personality_instructions

    @property
    def instincts(self):
        return self.personality_manager.instincts

    async def initialize(self):
        if not hasattr(self, 'personality_manager'):
            self.personality_manager = PersonalityManager()
    
    async def trigger(self, state: Dict = None, user_id: str = None, graph_version_id: str = None, config: Dict = None):
        """
        Main entry point for processing a user message.
        Uses tool calling to handle the conversation flow.
        
        Args:
            state: Current conversation state
            user_id: User ID
            graph_version_id: Knowledge graph version ID
            config: Additional configuration
            
        Yields:
            Chunks of the response as they become available
        """
        state = state or self.state.copy()
        user_id = user_id or self.user_id
        config = config or {}
        graph_version_id = (
            graph_version_id if graph_version_id is not None 
            else state.get("graph_version_id", 
                config.get("configurable", {}).get("graph_version_id", 
                    self.graph_version_id if hasattr(self, 'graph_version_id') else ""))
        )
        
        # Get WebSocket configuration
        use_websocket = state.get("use_websocket", False) or config.get("configurable", {}).get("use_websocket", False)
        thread_id_for_analysis = state.get("thread_id_for_analysis") or config.get("configurable", {}).get("thread_id_for_analysis")
        
        if not graph_version_id:
            logger.warning(f"graph_version_id is empty! State: {state}, Config: {config}")
        
        # Sync with state
        self.state["graph_version_id"] = graph_version_id

        # Get the latest message
        latest_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        latest_msg_content = latest_msg.content.strip() if isinstance(latest_msg, HumanMessage) else latest_msg.strip()
        
        # Convert message history to a format that tool calling can use
        conversation_history = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                conversation_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation_history.append({"role": "AI", "content": msg.content})
            elif isinstance(msg, str):
                # Handle raw string messages (assuming they're from the user)
                conversation_history.append({"role": "user", "content": msg})
            else:
                # Handle dict-based messages
                conversation_history.append(msg)
        
        logger.info(f"Triggering - User: {user_id}, Graph Version: {graph_version_id}, latest_msg: {latest_msg_content}, WebSocket: {use_websocket}")
        
        # Use process_llm_with_tools to handle the conversation
        analysis_events_sent = 0
        knowledge_events_sent = 0
        next_action_events_sent = 0
        
        # Import socketio_manager functions outside the loop if using WebSockets
        if use_websocket and thread_id_for_analysis:
            try:
                from socketio_manager_async import emit_analysis_event, emit_knowledge_event, emit_next_action_event
                socket_imports_success = True
                logger.info(f"Successfully imported socketio_manager_async functions for thread {thread_id_for_analysis}")
            except Exception as socket_import_error:
                socket_imports_success = False
                logger.error(f"Failed to import socketio_manager_async functions: {socket_import_error}")
        else:
            socket_imports_success = False
        
        try:
            async for result in process_llm_with_tools(
                latest_msg_content, 
                conversation_history,
                state,
                graph_version_id,
                thread_id_for_analysis if use_websocket else None
            ):
                # Handle different types of results
                if isinstance(result, dict):
                    event_type = result.get("type")
                    
                    # Ensure thread_id is set for all events
                    if use_websocket and thread_id_for_analysis and "thread_id" not in result:
                        result["thread_id"] = thread_id_for_analysis
                    
                    # Log the event being processed
                    result_preview = str(result.get("content", ""))[:30]
                    #logger.info(f"MC received {event_type} event: {result_preview}...")
                    
                    if event_type == "analysis":
                        # For analysis chunks, handle WebSocket communication if configured
                        analysis_events_sent += 1
                        
                        if use_websocket and thread_id_for_analysis and socket_imports_success:
                            try:
                                # Check if we can get the sessions variable from the current state
                                from socketio_manager_async import ws_sessions, main_ws_sessions
                                
                                # Try to log session state for debugging
                                logger.info(f"Analysis event - Local session count: {len(ws_sessions) if ws_sessions else 0}, " +
                                           f"Main session count: {len(main_ws_sessions) if main_ws_sessions else 0}")
                                
                                # Make sure the chunk has the thread_id
                                if "thread_id" not in result:
                                    result["thread_id"] = thread_id_for_analysis
                                
                                # Emit the event asynchronously
                                await emit_analysis_event(thread_id_for_analysis, result)
                                logger.info(f"Emitted analysis event {analysis_events_sent} for thread {thread_id_for_analysis}")
                            except Exception as e:
                                logger.error(f"Error emitting analysis event {analysis_events_sent}: {str(e)}")
                                logger.error(f"Thread ID: {thread_id_for_analysis}, Event type: {event_type}")
                                # Don't raise exception here - continue processing
                                
                    elif event_type == "knowledge":
                        # For knowledge chunks, handle WebSocket communication if configured
                        knowledge_events_sent += 1
                        
                        if use_websocket and thread_id_for_analysis and socket_imports_success:
                            try:
                                # Check if we can get the sessions variable from the current state
                                from socketio_manager_async import ws_sessions, main_ws_sessions
                                
                                # Try to log session state for debugging
                                logger.info(f"Knowledge event - Local session count: {len(ws_sessions) if ws_sessions else 0}, " +
                                           f"Main session count: {len(main_ws_sessions) if main_ws_sessions else 0}")
                                
                                # Make sure the chunk has the thread_id
                                if "thread_id" not in result:
                                    result["thread_id"] = thread_id_for_analysis
                                
                                # Emit the event asynchronously
                                await emit_knowledge_event(thread_id_for_analysis, result)
                                logger.info(f"Emitted knowledge event {knowledge_events_sent} for thread {thread_id_for_analysis}")
                            except Exception as e:
                                logger.error(f"Error emitting knowledge event {knowledge_events_sent}: {str(e)}")
                                logger.error(f"Thread ID: {thread_id_for_analysis}, Event type: {event_type}")
                                # Don't raise exception here - continue processing
                                
                    elif event_type == "next_action":
                        # For next_action chunks, handle WebSocket communication if configured
                        next_action_events_sent += 1
                        
                        if use_websocket and thread_id_for_analysis and socket_imports_success:
                            try:
                                # Check if we can get the sessions variable from the current state
                                from socketio_manager_async import ws_sessions, main_ws_sessions
                                
                                # Try to log session state for debugging
                                logger.info(f"Next action event - Local session count: {len(ws_sessions) if ws_sessions else 0}, " +
                                           f"Main session count: {len(main_ws_sessions) if main_ws_sessions else 0}")
                                
                                # Make sure the chunk has the thread_id
                                if "thread_id" not in result:
                                    result["thread_id"] = thread_id_for_analysis
                                
                                # Emit the event asynchronously
                                await emit_next_action_event(thread_id_for_analysis, result)
                                logger.info(f"Emitted next_action event {next_action_events_sent} for thread {thread_id_for_analysis}")
                            except Exception as e:
                                logger.error(f"Error emitting next_action event {next_action_events_sent}: {str(e)}")
                                logger.error(f"Thread ID: {thread_id_for_analysis}, Event type: {event_type}")
                                # Don't raise exception here - continue processing
                            
                    elif "state" in result:
                        # This is the final state update
                        state.update(result["state"])
                        # Don't yield this to the caller
                        continue
                    else:
                        # Unknown dict type, just pass it through
                        yield result
                else:
                    # This is a direct response chunk - update the prompt string
                    if isinstance(result, str):
                        state["prompt_str"] += result
                    yield result
        
        except Exception as e:
            logger.error(f"Error in trigger: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            yield f"Error: {str(e)}"
        
        # Wrap response as AIMessage and append using add_messages if it wasn't already done
        if state["prompt_str"] and not any(
            isinstance(msg, AIMessage) and msg.content == state["prompt_str"] 
            for msg in state["messages"][-1:]
        ):
            state["messages"] = add_messages(state["messages"], [AIMessage(content=state["prompt_str"])])
        
        # Update the internal state 
        self.state.update(state)
        
        # Log event counts for debugging
        logger.info(f"Final response: {state['prompt_str']}")
        logger.info(f"Total analysis events: {analysis_events_sent}")
        logger.info(f"Total knowledge events: {knowledge_events_sent}")
        logger.info(f"Total next action events: {next_action_events_sent}")
        
        # Yield the final state as a special chunk
        yield {"state": state}

    async def load_personality_instructions(self, graph_version_id: str = ""):
        """Load personality instructions from the knowledge base"""
        # Use the existing personality manager's load_personality method
        await self.personality_manager.load_personality(graph_version_id)
        self.personality_manager.loaded_graph_version_id = graph_version_id

    async def stream_response(self, prompt: str, builder: ResponseBuilder, knowledge_found: bool = False):
        """
        Stream a response to the user.
        This is preserved from the original MC class for compatibility.
        
        Args:
            prompt: Prompt for the LLM
            builder: ResponseBuilder instance
            knowledge_found: Whether relevant knowledge was found
            
        Yields:
            Chunks of the response
        """
        if len(self.state["messages"]) > 20:
            prompt += "\nKeep it short and sweet."
        
        # Check if this is the first message or a continuation
        is_first_message = len(self.state["messages"]) <= 1
        buffer = ""
        
        try:
            # First collect the full response
            full_response = ""
            async for chunk in StreamLLM.astream(prompt):
                buffer += chunk.content
                full_response += chunk.content
            
            # Process the full response to optimize structure
            processed_response = self.response_processor.process_response(
                full_response, 
                self.state, 
                self.state["messages"][-1].content if self.state["messages"] else "",
                knowledge_found
            )
            
            # Then apply greeting filter if needed
            if not is_first_message:
                processed_response = self.response_filter.remove_greeting(processed_response)
            
            # Break into sentences for streaming
            sentences = processed_response.split(". ")
            for i, sentence in enumerate(sentences):
                # Add period back unless it's the last sentence and already has punctuation
                if i < len(sentences) - 1 or not sentence.endswith((".", "!", "?")):
                    sentence = sentence + "."
                
                builder.add(sentence.strip())
                yield builder.build(separator=" ")
            
            # Update conversation state after successful response
            self.response_filter.increment_turn()
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            builder.add("Có lỗi nhỏ, thử lại nhé!")
            yield builder.build(separator="\n") 