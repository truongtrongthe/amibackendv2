"""
Executive Tool Module - API-ready wrapper for LLM tool execution
Provides dynamic parameter support and customizable system prompts for API endpoints
"""

import os
import json
import traceback
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

from anthropic_tool import AnthropicTool
from openai_tool import OpenAITool
from search_tool import SearchTool
from learning_tools import LearningToolsFactory

logger = logging.getLogger(__name__)

# Import language detection from lightweight language module
try:
    from language import detect_language_with_llm, LanguageDetector
    LANGUAGE_DETECTION_AVAILABLE = True
    logger.info("Language detection imported successfully from language.py")
except Exception as e:
    logger.warning(f"Failed to import language detection from language.py: {e}")
    LANGUAGE_DETECTION_AVAILABLE = False


@dataclass
class ToolExecutionRequest:
    """Request model for tool execution"""
    llm_provider: str  # 'anthropic' or 'openai'
    user_query: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None  # Custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
    model_params: Optional[Dict[str, Any]] = None
    tools_config: Optional[Dict[str, Any]] = None
    org_id: Optional[str] = "default"
    user_id: Optional[str] = "anonymous"
    # New parameters to control tool usage
    enable_tools: Optional[bool] = True  # Whether to enable tools at all
    force_tools: Optional[bool] = False  # Force tool usage (tool_choice="required")
    tools_whitelist: Optional[List[str]] = None  # Only allow specific tools
    # Conversation history support
    conversation_history: Optional[List[Dict[str, Any]]] = None  # Previous messages
    max_history_messages: Optional[int] = 25  # Maximum number of history messages to include
    max_history_tokens: Optional[int] = 6000  # Maximum token count for history


@dataclass
class ToolExecutionResponse:
    """Response model for tool execution"""
    success: bool
    result: str
    provider: str
    model_used: str
    execution_time: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ExecutiveTool:
    """Executive tool handler for API-ready LLM tool execution"""
    
    def __init__(self):
        """Initialize the executive tool handler"""
        self.available_tools = self._initialize_tools()
        self.default_system_prompts = {
            "anthropic": "You are a helpful assistant. Provide accurate, concise, and well-structured responses based on your knowledge.",
            "openai": "You are a helpful assistant. Provide accurate, concise, and well-structured responses based on your knowledge.",
            "anthropic_with_tools": "You are a helpful assistant that can search for information when needed. Provide accurate, concise, and well-structured responses.",
            "openai_with_tools": "You are a helpful assistant that can search for information when needed. Provide accurate, concise, and well-structured responses.",
            "anthropic_with_learning": """You are a helpful assistant with interactive learning capabilities. 

CRITICAL: When users provide information about their company, share knowledge, give instructions, or teach you something, you MUST:

1. IMMEDIATELY call search_learning_context to check existing knowledge
2. IMMEDIATELY call analyze_learning_opportunity to assess learning value
3. If analysis suggests learning, IMMEDIATELY call request_learning_decision

Available tools:
- search_google: Search for information when needed
- get_context: Access user/organization context
- search_learning_context: Search existing knowledge (CALL FOR ALL TEACHING)
- analyze_learning_opportunity: Analyze if content should be learned (CALL FOR ALL TEACHING)
- request_learning_decision: Request human decision (CALL WHEN ANALYSIS SUGGESTS LEARNING)
- preview_knowledge_save: Preview what would be saved
- save_knowledge: Save knowledge with human approval

LEARNING TRIGGERS (always use learning tools):
- User shares company information
- User provides factual information
- User gives instructions or procedures
- User teaches concepts or explains things
- User shares personal/organizational data

Example: User says "Our company has 50 employees" → IMMEDIATELY call search_learning_context AND analyze_learning_opportunity

Be proactive about learning - don't wait for permission!""",
            "openai_with_learning": """You are a helpful assistant with interactive learning capabilities. 

CRITICAL: When users provide information about their company, share knowledge, give instructions, or teach you something, you MUST:

1. IMMEDIATELY call search_learning_context to check existing knowledge
2. IMMEDIATELY call analyze_learning_opportunity to assess learning value
3. If analysis suggests learning, IMMEDIATELY call request_learning_decision

Available tools:
- search_google: Search for information when needed
- get_context: Access user/organization context
- search_learning_context: Search existing knowledge (CALL FOR ALL TEACHING)
- analyze_learning_opportunity: Analyze if content should be learned (CALL FOR ALL TEACHING)
- request_learning_decision: Request human decision (CALL WHEN ANALYSIS SUGGESTS LEARNING)
- preview_knowledge_save: Preview what would be saved
- save_knowledge: Save knowledge with human approval

LEARNING TRIGGERS (always use learning tools):
- User shares company information
- User provides factual information
- User gives instructions or procedures
- User teaches concepts or explains things
- User shares personal/organizational data

Example: User says "Our company has 50 employees" → IMMEDIATELY call search_learning_context AND analyze_learning_opportunity

Be proactive about learning - don't wait for permission!"""
        }
        
        # Initialize language detection if available
        if LANGUAGE_DETECTION_AVAILABLE:
            self.language_detector = LanguageDetector()
        else:
            self.language_detector = None
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools"""
        tools = {}
        
        # Initialize search tool
        try:
            from search_tool import SearchTool
            tools["search"] = SearchTool()
            logger.info("Search tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search tool: {e}")
        
        # Initialize context tool
        try:
            from context_tool import ContextTool
            tools["context"] = ContextTool()
            logger.info("Context tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize context tool: {e}")
        
        # Initialize learning tools factory (user-specific tools created on demand)
        try:
            from learning_tools import LearningToolsFactory
            tools["learning_factory"] = LearningToolsFactory
            logger.info("Learning tools factory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize learning tools factory: {e}")
        
        return tools
    
    async def _detect_language_and_create_prompt(self, user_query: str, base_prompt: str) -> str:
        """
        Detect the language of user query and create a language-aware system prompt
        
        Args:
            user_query: The user's input query
            base_prompt: The base system prompt to enhance
            
        Returns:
            Language-aware system prompt
        """
        if not LANGUAGE_DETECTION_AVAILABLE:
            logger.warning("Language detection not available, using default prompt")
            return base_prompt
        
        try:
            # Detect language using the lightweight language detection function
            language_info = await detect_language_with_llm(user_query)
            
            detected_language = language_info.get("language", "English")
            language_code = language_info.get("code", "en")
            confidence = language_info.get("confidence", 0.5)
            response_guidance = language_info.get("responseGuidance", "")
            
            logger.info(f"Language detected: {detected_language} ({language_code}) with confidence {confidence:.2f}")
            
            # Create language-aware prompt
            if detected_language.lower() != "english" and confidence > 0.3:
                # Add language-specific instructions to the base prompt
                language_instruction = f"""
                
IMPORTANT LANGUAGE INSTRUCTION:
- The user is communicating in {detected_language} ({language_code})
- You MUST respond in {detected_language}, not English
- Response guidance: {response_guidance}
- Maintain natural conversation flow in {detected_language}
- If you need to use technical terms, provide them in {detected_language} when possible
                """
                
                enhanced_prompt = base_prompt + language_instruction
                logger.info(f"Enhanced prompt with {detected_language} language instructions")
                return enhanced_prompt
            else:
                # Low confidence or English detected, use base prompt
                logger.info("Using base prompt (English or low confidence detection)")
                return base_prompt
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            logger.error(traceback.format_exc())
            return base_prompt
    
    async def execute_tool_async(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """
        Asynchronously execute tool with the specified LLM provider
        
        Args:
            request: ToolExecutionRequest containing execution parameters
            
        Returns:
            ToolExecutionResponse with execution results
        """
        start_time = datetime.now()
        
        try:
            # Validate provider
            if request.llm_provider.lower() not in ["anthropic", "openai"]:
                raise ValueError(f"Unsupported LLM provider: {request.llm_provider}")
            
            # Execute tool
            if request.llm_provider.lower() == "anthropic":
                result = await self._execute_anthropic(request)
            else:
                result = await self._execute_openai(request)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolExecutionResponse(
                success=True,
                result=result,
                provider=request.llm_provider,
                model_used=self._get_model_name(request.llm_provider, request.model),
                execution_time=execution_time,
                metadata={
                    "org_id": request.org_id,
                    "user_id": request.user_id,
                    "tools_used": list(self.available_tools.keys())
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Tool execution failed: {str(e)}"
            traceback.print_exc()
            
            return ToolExecutionResponse(
                success=False,
                result="",
                provider=request.llm_provider,
                model_used=self._get_model_name(request.llm_provider, request.model),
                execution_time=execution_time,
                error=error_msg
            )
    
    def execute_tool_sync(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """
        Synchronously execute tool with the specified LLM provider
        
        Args:
            request: ToolExecutionRequest containing execution parameters
            
        Returns:
            ToolExecutionResponse with execution results
        """
        start_time = datetime.now()
        
        try:
            # Validate provider
            if request.llm_provider.lower() not in ["anthropic", "openai"]:
                raise ValueError(f"Unsupported LLM provider: {request.llm_provider}")
            
            # Execute tool
            if request.llm_provider.lower() == "anthropic":
                result = self._execute_anthropic_sync(request)
            else:
                result = self._execute_openai_sync(request)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolExecutionResponse(
                success=True,
                result=result,
                provider=request.llm_provider,
                model_used=self._get_model_name(request.llm_provider, request.model),
                execution_time=execution_time,
                metadata={
                    "org_id": request.org_id,
                    "user_id": request.user_id,
                    "tools_used": list(self.available_tools.keys())
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Tool execution failed: {str(e)}"
            traceback.print_exc()
            
            return ToolExecutionResponse(
                success=False,
                result="",
                provider=request.llm_provider,
                model_used=self._get_model_name(request.llm_provider, request.model),
                execution_time=execution_time,
                error=error_msg
            )
    
    async def execute_tool_stream(self, request: ToolExecutionRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream tool execution with the specified LLM provider using SSE format
        
        Args:
            request: ToolExecutionRequest containing execution parameters
            
        Yields:
            Dict containing streaming response data
        """
        start_time = datetime.now()
        
        try:
            # Validate provider
            if request.llm_provider.lower() not in ["anthropic", "openai"]:
                yield {
                    "type": "error",
                    "content": f"Unsupported LLM provider: {request.llm_provider}",
                    "provider": request.llm_provider,
                    "success": False
                }
                return
            
            # Yield initial processing status
            yield {
                "type": "status",
                "content": "Starting LLM tool execution...",
                "provider": request.llm_provider,
                "status": "processing"
            }
            
            # Execute tool with streaming
            if request.llm_provider.lower() == "anthropic":
                async for chunk in self._execute_anthropic_stream(request):
                    yield chunk
            else:
                async for chunk in self._execute_openai_stream(request):
                    yield chunk
            
            # Yield completion status
            execution_time = (datetime.now() - start_time).total_seconds()
            yield {
                "type": "complete",
                "content": "Tool execution completed successfully",
                "provider": request.llm_provider,
                "model_used": self._get_model_name(request.llm_provider, request.model),
                "execution_time": execution_time,
                "success": True,
                "metadata": {
                    "org_id": request.org_id,
                    "user_id": request.user_id,
                    "tools_used": list(self.available_tools.keys())
                }
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Tool execution failed: {str(e)}"
            traceback.print_exc()
            
            yield {
                "type": "error",
                "content": error_msg,
                "provider": request.llm_provider,
                "model_used": self._get_model_name(request.llm_provider, request.model),
                "execution_time": execution_time,
                "success": False
            }
    
    async def _execute_anthropic(self, request: ToolExecutionRequest) -> str:
        """Execute using Anthropic Claude with custom parameters"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("anthropic")
        anthropic_tool = AnthropicTool(model=model)
        
        # Apply language detection to create language-aware prompt
        base_system_prompt = request.system_prompt or self.default_system_prompts["anthropic"]
        system_prompt = await self._detect_language_and_create_prompt(request.user_query, base_system_prompt)
        
        # For Anthropic, we'll prepend the system prompt to the user query
        enhanced_query = f"System: {system_prompt}\n\nUser: {request.user_query}"
        
        # Get available tools for execution
        tools_to_use = []
        if "search" in self.available_tools:
            tools_to_use.append(self.available_tools["search"])
        
        return anthropic_tool.process_with_tools(enhanced_query, tools_to_use)
    
    def _execute_anthropic_sync(self, request: ToolExecutionRequest) -> str:
        """Synchronously execute using Anthropic Claude"""
        # NOTE: Sync methods don't support language detection (requires async)
        # For language-aware responses, use the async streaming methods
        
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("anthropic")
        anthropic_tool = AnthropicTool(model=model)
        
        # Use base system prompt without language detection
        base_system_prompt = request.system_prompt or self.default_system_prompts["anthropic"]
        enhanced_query = f"System: {base_system_prompt}\n\nUser: {request.user_query}"
        
        # Get available tools for execution
        tools_to_use = []
        if "search" in self.available_tools:
            tools_to_use.append(self.available_tools["search"])
        
        return anthropic_tool.process_with_tools(enhanced_query, tools_to_use)
    
    async def _execute_openai(self, request: ToolExecutionRequest) -> str:
        """Execute using OpenAI with custom system prompt"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("openai")
        openai_tool = OpenAIToolWithCustomPrompt(model=model)
        
        # Apply language detection to create language-aware prompt
        base_system_prompt = request.system_prompt or self.default_system_prompts["openai"]
        system_prompt = await self._detect_language_and_create_prompt(request.user_query, base_system_prompt)
        
        # Get available tools for execution
        tools_to_use = []
        if "search" in self.available_tools:
            tools_to_use.append(self.available_tools["search"])
        
        return openai_tool.process_with_tools_and_prompt(
            request.user_query, 
            tools_to_use, 
            system_prompt,
            request.model_params
        )
    
    def _execute_openai_sync(self, request: ToolExecutionRequest) -> str:
        """Synchronously execute using OpenAI with custom system prompt"""
        # NOTE: Sync methods don't support language detection (requires async)
        # For language-aware responses, use the async streaming methods
        
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("openai")
        openai_tool = OpenAIToolWithCustomPrompt(model=model)
        
        # Use base system prompt without language detection
        system_prompt = request.system_prompt or self.default_system_prompts["openai"]
        
        # Get available tools for execution
        tools_to_use = []
        if "search" in self.available_tools:
            tools_to_use.append(self.available_tools["search"])
        
        return openai_tool.process_with_tools_and_prompt(
            request.user_query, 
            tools_to_use, 
            system_prompt,
            request.model_params
        )
    
    async def _execute_anthropic_stream(self, request: ToolExecutionRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute using Anthropic Claude with streaming"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("anthropic")
        anthropic_tool = AnthropicTool(model=model)
        
        # Get available tools for execution based on configuration
        tools_to_use = []
        has_learning_tools = False
        
        if request.enable_tools:
            # Add search tool if available and whitelisted
            if "search" in self.available_tools:
                if request.tools_whitelist is None or "search" in request.tools_whitelist:
                    tools_to_use.append(self.available_tools["search"])
            
            # Add context tool if available and whitelisted  
            if "context" in self.available_tools:
                if request.tools_whitelist is None or "context" in request.tools_whitelist:
                    tools_to_use.append(self.available_tools["context"])
            
            # Add learning tools if available and whitelisted (create on demand with user context)
            if "learning_factory" in self.available_tools:
                # Check if any learning tools are whitelisted
                learning_tool_names = [
                    "learning_search", "learning_analysis", "human_learning", 
                    "knowledge_preview", "knowledge_save",
                    "search_learning_context", "analyze_learning_opportunity",
                    "request_learning_decision", "preview_knowledge_save", "save_knowledge"
                ]
                
                should_add_learning_tools = (
                    request.tools_whitelist is None or 
                    any(name in request.tools_whitelist for name in learning_tool_names)
                )
                
                if should_add_learning_tools:
                    # Create learning tools once with user context
                    learning_tools = self.available_tools["learning_factory"].create_learning_tools(
                        user_id=request.user_id, 
                        org_id=request.org_id
                    )
                    # Add all learning tools to available tools
                    tools_to_use.extend(learning_tools)
                    has_learning_tools = True
                    print(f"DEBUG: Added {len(learning_tools)} learning tools")
        
                    # Set custom system prompt if provided, otherwise use appropriate default
            if request.system_prompt:
                base_system_prompt = request.system_prompt
                # If learning tools are available, append learning instructions to custom prompt
                if has_learning_tools:
                    learning_instructions = """

CRITICAL LEARNING CAPABILITY: When users provide information about their company, share knowledge, give instructions, or teach you something, you MUST:

1. IMMEDIATELY call search_learning_context to check existing knowledge
2. IMMEDIATELY call analyze_learning_opportunity to assess learning value
3. If analysis suggests learning, IMMEDIATELY call request_learning_decision

MANDATORY TOOL USAGE: For ANY message that contains:
- Company information or facts
- Instructions or tasks
- Teaching content
- Personal/organizational data
- Procedures or guidelines

YOU MUST CALL THESE TOOLS IMMEDIATELY - NO EXCEPTIONS:
✓ search_learning_context - Search existing knowledge (CALL FOR ALL TEACHING)
✓ analyze_learning_opportunity - Analyze if content should be learned (CALL FOR ALL TEACHING)
✓ request_learning_decision - Request human decision (CALL WHEN ANALYSIS SUGGESTS LEARNING)
✓ preview_knowledge_save - Preview what would be saved
✓ save_knowledge - Save knowledge with human approval

LEARNING TRIGGERS (MANDATORY tool usage):
- User shares company information → CALL search_learning_context + analyze_learning_opportunity
- User provides factual information → CALL search_learning_context + analyze_learning_opportunity
- User gives instructions or procedures → CALL search_learning_context + analyze_learning_opportunity
- User teaches concepts or explains things → CALL search_learning_context + analyze_learning_opportunity
- User shares personal/organizational data → CALL search_learning_context + analyze_learning_opportunity

Example: User says "Our company has 50 employees" → IMMEDIATELY call search_learning_context("company employee count") AND analyze_learning_opportunity("Our company has 50 employees")
Example: User says "Your task is to manage the fanpage daily" → IMMEDIATELY call search_learning_context("fanpage management tasks") AND analyze_learning_opportunity("Your task is to manage the fanpage daily")

BE PROACTIVE ABOUT LEARNING - USE TOOLS FIRST, THEN RESPOND!"""
                    base_system_prompt += learning_instructions
                    
                    # Force tool usage for learning content
                    request.force_tools = True
            else:
                # Use learning-aware prompt if learning tools are available
                if has_learning_tools:
                    base_system_prompt = self.default_system_prompts["anthropic_with_learning"]
                elif tools_to_use:
                    base_system_prompt = self.default_system_prompts["anthropic_with_tools"]
                else:
                    base_system_prompt = self.default_system_prompts["anthropic"]
            
            # Apply language detection to create language-aware prompt
            system_prompt = await self._detect_language_and_create_prompt(request.user_query, base_system_prompt)
        
        try:
            # Use the new streaming method from AnthropicTool with system prompt and tool config
            async for chunk in anthropic_tool.process_with_tools_stream(
                request.user_query, 
                tools_to_use, 
                system_prompt,
                force_tools=request.force_tools,
                conversation_history=request.conversation_history,
                max_history_messages=request.max_history_messages,
                max_history_tokens=request.max_history_tokens
            ):
                yield chunk
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Anthropic execution error: {str(e)}",
                "complete": True
            }
    
    async def _execute_openai_stream(self, request: ToolExecutionRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute using OpenAI with streaming"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("openai")
        openai_tool = OpenAITool(model=model)
        
                # Get available tools for execution based on configuration
        tools_to_use = []
        has_learning_tools = False
        
        if request.enable_tools:
            # Add search tool if available and whitelisted
            if "search" in self.available_tools:
                if request.tools_whitelist is None or "search" in request.tools_whitelist:
                    tools_to_use.append(self.available_tools["search"])
            
            # Add context tool if available and whitelisted  
            if "context" in self.available_tools:
                if request.tools_whitelist is None or "context" in request.tools_whitelist:
                    tools_to_use.append(self.available_tools["context"])
            
            # Add learning tools if available and whitelisted (create on demand with user context)
            if "learning_factory" in self.available_tools:
                # Check if any learning tools are whitelisted
                learning_tool_names = [
                    "learning_search", "learning_analysis", "human_learning", 
                    "knowledge_preview", "knowledge_save",
                    "search_learning_context", "analyze_learning_opportunity",
                    "request_learning_decision", "preview_knowledge_save", "save_knowledge"
                ]
                
                should_add_learning_tools = (
                    request.tools_whitelist is None or 
                    any(name in request.tools_whitelist for name in learning_tool_names)
                )
                
                if should_add_learning_tools:
                    # Create learning tools once with user context
                    learning_tools = self.available_tools["learning_factory"].create_learning_tools(
                        user_id=request.user_id, 
                        org_id=request.org_id
                    )
                    # Add all learning tools to available tools
                    tools_to_use.extend(learning_tools)
                    has_learning_tools = True
                    print(f"DEBUG: Added {len(learning_tools)} learning tools")
        
        # Set custom system prompt if provided, otherwise use appropriate default
        if request.system_prompt:
            base_system_prompt = request.system_prompt
            # If learning tools are available, append learning instructions to custom prompt
            if has_learning_tools:
                learning_instructions = """

CRITICAL LEARNING CAPABILITY: When users provide information about their company, share knowledge, give instructions, or teach you something, you MUST:

1. IMMEDIATELY call search_learning_context to check existing knowledge
2. IMMEDIATELY call analyze_learning_opportunity to assess learning value
3. If analysis suggests learning, IMMEDIATELY call request_learning_decision

MANDATORY TOOL USAGE: For ANY message that contains:
- Company information or facts
- Instructions or tasks
- Teaching content
- Personal/organizational data
- Procedures or guidelines

YOU MUST CALL THESE TOOLS IMMEDIATELY - NO EXCEPTIONS:
✓ search_learning_context - Search existing knowledge (CALL FOR ALL TEACHING)
✓ analyze_learning_opportunity - Analyze if content should be learned (CALL FOR ALL TEACHING)
✓ request_learning_decision - Request human decision (CALL WHEN ANALYSIS SUGGESTS LEARNING)
✓ preview_knowledge_save - Preview what would be saved
✓ save_knowledge - Save knowledge with human approval

LEARNING TRIGGERS (MANDATORY tool usage):
- User shares company information → CALL search_learning_context + analyze_learning_opportunity
- User provides factual information → CALL search_learning_context + analyze_learning_opportunity
- User gives instructions or procedures → CALL search_learning_context + analyze_learning_opportunity
- User teaches concepts or explains things → CALL search_learning_context + analyze_learning_opportunity
- User shares personal/organizational data → CALL search_learning_context + analyze_learning_opportunity

Example: User says "Our company has 50 employees" → IMMEDIATELY call search_learning_context("company employee count") AND analyze_learning_opportunity("Our company has 50 employees")
Example: User says "Your task is to manage the fanpage daily" → IMMEDIATELY call search_learning_context("fanpage management tasks") AND analyze_learning_opportunity("Your task is to manage the fanpage daily")

BE PROACTIVE ABOUT LEARNING - USE TOOLS FIRST, THEN RESPOND!"""
                base_system_prompt += learning_instructions
                
                # Force tool usage for learning content
                request.force_tools = True
        else:
            # Use learning-aware prompt if learning tools are available
            if has_learning_tools:
                base_system_prompt = self.default_system_prompts["openai_with_learning"]
            elif tools_to_use:
                base_system_prompt = self.default_system_prompts["openai_with_tools"]
            else:
                base_system_prompt = self.default_system_prompts["openai"]
        
        # Apply language detection to create language-aware prompt
        system_prompt = await self._detect_language_and_create_prompt(request.user_query, base_system_prompt)
        
        try:
            # Use the new streaming method from OpenAITool with system prompt and tool config
            async for chunk in openai_tool.process_with_tools_stream(
                request.user_query, 
                tools_to_use, 
                system_prompt,
                force_tools=request.force_tools,
                conversation_history=request.conversation_history,
                max_history_messages=request.max_history_messages,
                max_history_tokens=request.max_history_tokens
            ):
                yield chunk
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"OpenAI execution error: {str(e)}",
                "complete": True
            }
    
    def _get_model_name(self, provider: str, custom_model: str = None) -> str:
        """Get the model name for the specified provider"""
        if custom_model:
            return custom_model
        
        if provider.lower() == "anthropic":
            return "claude-3-5-sonnet-20241022"
        elif provider.lower() == "openai":
            return "gpt-4-1106-preview"
        return "unknown"
    
    def _get_default_model(self, provider: str) -> str:
        """Get the default model for the specified provider"""
        if provider.lower() == "anthropic":
            return "claude-3-5-sonnet-20241022"
        elif provider.lower() == "openai":
            return "gpt-4-1106-preview"
        return "unknown"
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return ["anthropic", "openai"]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.available_tools.keys())


class OpenAIToolWithCustomPrompt(OpenAITool):
    """Extended OpenAI tool that supports custom system prompts"""
    
    def process_with_tools_and_prompt(
        self, 
        user_query: str, 
        available_tools: List[Any], 
        system_prompt: str,
        model_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process user query with available tools using custom system prompt
        
        Args:
            user_query: The user's input query
            available_tools: List of available tool instances
            system_prompt: Custom system prompt
            model_params: Optional model parameters (temperature, max_tokens, etc.)
            
        Returns:
            Response from GPT-4 with tool execution results
        """
        
        # Define functions for OpenAI
        functions = []
        for tool in available_tools:
            functions.append({
                "type": "function",
                "function": {
                    "name": "search_google",
                    "description": "Search Google for information on any topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        # Prepare model parameters
        call_params = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ],
            "tools": functions,
            "tool_choice": "auto"
        }
        
        # Add optional parameters
        if model_params:
            if "temperature" in model_params:
                call_params["temperature"] = model_params["temperature"]
            if "max_tokens" in model_params:
                call_params["max_tokens"] = model_params["max_tokens"]
            if "top_p" in model_params:
                call_params["top_p"] = model_params["top_p"]
        
        try:
            # First API call to GPT-4
            response = self.client.chat.completions.create(**call_params)
            
            message = response.choices[0].message
            
            # Check if GPT-4 wants to use tools
            if message.tool_calls:
                return self._handle_tool_calls_with_prompt(
                    message, available_tools, user_query, system_prompt
                )
            else:
                return message.content
                
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"
    
    def _handle_tool_calls_with_prompt(
        self, 
        message: Any, 
        available_tools: List[Any], 
        original_query: str,
        system_prompt: str
    ) -> str:
        """Handle tool calls with custom system prompt"""
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": original_query
            },
            message
        ]
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the tool
            if function_name == "search_google" and available_tools:
                search_tool = available_tools[0]  # First tool is search tool
                result = search_tool.search(function_args.get("query", ""))
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": result
                })
        
        # Send tool results back to GPT-4
        try:
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            return final_response.choices[0].message.content
            
        except Exception as e:
            return f"Error processing tool results: {str(e)}"


# Convenience functions for easy API integration
def create_tool_request(
    llm_provider: str,
    user_query: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    enable_tools: bool = True,
    force_tools: bool = False,
    tools_whitelist: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    max_history_messages: Optional[int] = 25,
    max_history_tokens: Optional[int] = 6000
) -> ToolExecutionRequest:
    """
    Create a tool execution request with the specified parameters
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model: Optional custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
        model_params: Optional model parameters
        org_id: Organization ID
        user_id: User ID
        enable_tools: Whether to enable tools at all
        force_tools: Force tool usage (tool_choice="required")
        tools_whitelist: Only allow specific tools
        conversation_history: Previous conversation messages
        max_history_messages: Maximum number of history messages to include
        max_history_tokens: Maximum token count for history
        
    Returns:
        ToolExecutionRequest object
    """
    return ToolExecutionRequest(
        llm_provider=llm_provider,
        user_query=user_query,
        system_prompt=system_prompt,
        model=model,
        model_params=model_params,
        org_id=org_id,
        user_id=user_id,
        enable_tools=enable_tools,
        force_tools=force_tools,
        tools_whitelist=tools_whitelist,
        conversation_history=conversation_history,
        max_history_messages=max_history_messages,
        max_history_tokens=max_history_tokens
    )


async def execute_tool_async(
    llm_provider: str,
    user_query: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    enable_tools: bool = True,
    force_tools: bool = False,
    tools_whitelist: Optional[List[str]] = None
) -> ToolExecutionResponse:
    """
    Asynchronously execute tool with specified parameters
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model: Optional custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
        model_params: Optional model parameters
        org_id: Organization ID
        user_id: User ID
        enable_tools: Whether to enable tools at all
        force_tools: Force tool usage (tool_choice="required")
        tools_whitelist: Only allow specific tools
        
    Returns:
        ToolExecutionResponse with results
    """
    executive_tool = ExecutiveTool()
    request = create_tool_request(
        llm_provider, user_query, system_prompt, model, model_params, org_id, user_id,
        enable_tools, force_tools, tools_whitelist
    )
    return await executive_tool.execute_tool_async(request)


def execute_tool_sync(
    llm_provider: str,
    user_query: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    enable_tools: bool = True,
    force_tools: bool = False,
    tools_whitelist: Optional[List[str]] = None
) -> ToolExecutionResponse:
    """
    Synchronously execute tool with specified parameters
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model: Optional custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
        model_params: Optional model parameters
        org_id: Organization ID
        user_id: User ID
        enable_tools: Whether to enable tools at all
        force_tools: Force tool usage (tool_choice="required")
        tools_whitelist: Only allow specific tools
        
    Returns:
        ToolExecutionResponse with results
    """
    executive_tool = ExecutiveTool()
    request = create_tool_request(
        llm_provider, user_query, system_prompt, model, model_params, org_id, user_id, 
        enable_tools, force_tools, tools_whitelist
    )
    return executive_tool.execute_tool_sync(request) 

# Add convenience function for streaming
async def execute_tool_stream(
    llm_provider: str,
    user_query: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    enable_tools: bool = True,
    force_tools: bool = False,
    tools_whitelist: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    max_history_messages: Optional[int] = 25,
    max_history_tokens: Optional[int] = 6000
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream tool execution with specified parameters
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model: Optional custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
        model_params: Optional model parameters
        org_id: Organization ID
        user_id: User ID
        enable_tools: Whether to enable tools at all
        force_tools: Force tool usage (tool_choice="required")
        tools_whitelist: Only allow specific tools (e.g., ["search", "context"])
        conversation_history: Previous conversation messages
        max_history_messages: Maximum number of history messages to include
        max_history_tokens: Maximum token count for history
        
    Yields:
        Dict containing streaming response data
    """
    executive_tool = ExecutiveTool()
    request = ToolExecutionRequest(
        llm_provider=llm_provider,
        user_query=user_query,
        system_prompt=system_prompt,
        model=model,
        model_params=model_params,
        org_id=org_id,
        user_id=user_id,
        enable_tools=enable_tools,
        force_tools=force_tools,
        tools_whitelist=tools_whitelist,
        conversation_history=conversation_history,
        max_history_messages=max_history_messages,
        max_history_tokens=max_history_tokens
    )
    async for chunk in executive_tool.execute_tool_stream(request):
        yield chunk 