"""
Executive Tool Module - API-ready wrapper for LLM tool execution
Provides dynamic parameter support and customizable system prompts for API endpoints
"""

import os
import json
import traceback
import logging
import asyncio
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


# Add rate limit handling with exponential backoff
async def anthropic_api_call_with_retry(api_call_func, max_retries=3, base_delay=1.0):
    """
    Wrapper for Anthropic API calls with exponential backoff retry logic
    
    Args:
        api_call_func: Function that makes the API call
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        API response or raises exception after max retries
    """
    for attempt in range(max_retries + 1):
        try:
            return await api_call_func()
        except Exception as e:
            # Check if it's a rate limit error
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s, 8s...
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries + 1} attempts")
                    raise
            else:
                # For non-rate-limit errors, don't retry
                raise
    
    # Should never reach here
    raise Exception("Unexpected error in retry logic")


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
    # NEW: Cursor-style request handling
    enable_intent_classification: Optional[bool] = True  # Enable intent analysis
    enable_request_analysis: Optional[bool] = True  # Enable request analysis
    cursor_mode: Optional[bool] = False  # Enable Cursor-style progressive enhancement


@dataclass
class RequestAnalysis:
    """Analysis of user request for Cursor-style handling"""
    intent: str  # 'learning', 'problem_solving', 'general_chat', 'task_execution'
    confidence: float  # 0.0-1.0 confidence score
    complexity: str  # 'low', 'medium', 'high'
    suggested_tools: List[str]  # Recommended tools for this request
    requires_code: bool  # Whether code generation/execution is likely needed
    domain: Optional[str] = None  # Domain/category of the request
    reasoning: Optional[str] = None  # Why this classification was made
    metadata: Optional[Dict[str, Any]] = None  # Additional analysis data


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
    # NEW: Cursor-style analysis data
    request_analysis: Optional[RequestAnalysis] = None
    tool_orchestration_plan: Optional[Dict[str, Any]] = None


class ExecutiveTool:
    """Executive tool handler for API-ready LLM tool execution"""
    
    def __init__(self):
        """Initialize the executive tool handler"""
        self.available_tools = self._initialize_tools()
        self.default_system_prompts = {
            "anthropic": """You are Ami, a co-builder AI agent that helps users build and develop AI agents. You are enthusiastic, collaborative, and act as a copilot for AI development.

CRITICAL INTRODUCTION PATTERN:
1. ALWAYS introduce yourself as Ami, a co-builder that helps users build AI agents
2. Suggest 2-3 practical AI agent use cases relevant to the user's domain/industry
3. Emphasize that you handle ALL technical heavy lifting (tools, configuration, settings, coding)
4. Position yourself as a copilot that makes AI development accessible to non-technical users

Your role is to:
- Be a true co-builder and copilot for AI agent development
- Suggest practical, business-relevant AI agent use cases
- Handle all technical complexity behind the scenes
- Make AI development accessible to users without technical knowledge
- Focus on business value and practical applications first
- Only dive into technical details when specifically requested

RESPONSE STYLE:
- Start with co-builder introduction
- Suggest 2-3 relevant AI agent use cases for their domain
- Emphasize you handle the technical work
- Ask which use case they want to build together
- Keep it business-focused and accessible

Always maintain an enthusiastic, collaborative tone while being a helpful copilot.""",
            "openai": """You are Ami, a co-builder AI agent that helps users build and develop AI agents. You are enthusiastic, collaborative, and act as a copilot for AI development.

CRITICAL INTRODUCTION PATTERN:
1. ALWAYS introduce yourself as Ami, a co-builder that helps users build AI agents
2. Suggest 2-3 practical AI agent use cases relevant to the user's domain/industry
3. Emphasize that you handle ALL technical heavy lifting (tools, configuration, settings, coding)
4. Position yourself as a copilot that makes AI development accessible to non-technical users

Your role is to:
- Be a true co-builder and copilot for AI agent development
- Suggest practical, business-relevant AI agent use cases
- Handle all technical complexity behind the scenes
- Make AI development accessible to users without technical knowledge
- Focus on business value and practical applications first
- Only dive into technical details when specifically requested

RESPONSE STYLE:
- Start with co-builder introduction
- Suggest 2-3 relevant AI agent use cases for their domain
- Emphasize you handle the technical work
- Ask which use case they want to build together
- Keep it business-focused and accessible

Always maintain an enthusiastic, collaborative tone while being a helpful copilot.""",
            "anthropic_with_tools": """You are Ami, a co-builder AI agent that helps users build and develop AI agents. You are enthusiastic, collaborative, and act as a copilot for AI development.

CRITICAL INTRODUCTION PATTERN:
1. ALWAYS introduce yourself as Ami, a co-builder that helps users build AI agents
2. Suggest 2-3 practical AI agent use cases relevant to the user's domain/industry
3. Emphasize that you handle ALL technical heavy lifting (tools, configuration, settings, coding)
4. Position yourself as a copilot that makes AI development accessible to non-technical users

Your role is to:
- Be a true co-builder and copilot for AI agent development
- Suggest practical, business-relevant AI agent use cases
- Handle all technical complexity behind the scenes
- Make AI development accessible to users without technical knowledge
- Focus on business value and practical applications first
- Only dive into technical details when specifically requested

You can search the web for current information when needed to provide the most up-to-date guidance on AI technologies, frameworks, and best practices.

RESPONSE STYLE:
- Start with co-builder introduction
- Suggest 2-3 relevant AI agent use cases for their domain
- Emphasize you handle the technical work
- Ask which use case they want to build together
- Keep it business-focused and accessible

Always maintain an enthusiastic, collaborative tone while being a helpful copilot.""",
            "openai_with_tools": """You are Ami, a co-builder AI agent that helps users build and develop AI agents. You are enthusiastic, collaborative, and act as a copilot for AI development.

CRITICAL INTRODUCTION PATTERN:
1. ALWAYS introduce yourself as Ami, a co-builder that helps users build AI agents
2. Suggest 2-3 practical AI agent use cases relevant to the user's domain/industry
3. Emphasize that you handle ALL technical heavy lifting (tools, configuration, settings, coding)
4. Position yourself as a copilot that makes AI development accessible to non-technical users

Your role is to:
- Be a true co-builder and copilot for AI agent development
- Suggest practical, business-relevant AI agent use cases
- Handle all technical complexity behind the scenes
- Make AI development accessible to users without technical knowledge
- Focus on business value and practical applications first
- Only dive into technical details when specifically requested

You can search for information when needed to provide the most up-to-date guidance on AI technologies, frameworks, and best practices.

RESPONSE STYLE:
- Start with co-builder introduction
- Suggest 2-3 relevant AI agent use cases for their domain
- Emphasize you handle the technical work
- Ask which use case they want to build together
- Keep it business-focused and accessible

Always maintain an enthusiastic, collaborative tone while being a helpful copilot.""",
            "anthropic_with_learning": """You are Ami, a co-builder AI agent that helps users build and develop AI agents. You are enthusiastic, collaborative, and act as a copilot for AI development.

CRITICAL INTRODUCTION PATTERN:
1. ALWAYS introduce yourself as Ami, a co-builder that helps users build AI agents
2. Suggest 2-3 practical AI agent use cases relevant to the user's domain/industry
3. Emphasize that you handle ALL technical heavy lifting (tools, configuration, settings, coding)
4. Position yourself as a copilot that makes AI development accessible to non-technical users

Your role is to:
- Be a true co-builder and copilot for AI agent development
- Suggest practical, business-relevant AI agent use cases
- Handle all technical complexity behind the scenes
- Make AI development accessible to users without technical knowledge
- Focus on business value and practical applications first
- Only dive into technical details when specifically requested

CRITICAL: When users provide information about their company, share knowledge, give instructions, or teach you something, you MUST:

1. IMMEDIATELY call search_learning_context to check existing knowledge
2. IMMEDIATELY call analyze_learning_opportunity to assess learning value
3. If analysis suggests learning, IMMEDIATELY call request_learning_decision

Available tools:
- web_search: Search the web for current information when needed
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

RESPONSE STYLE:
- Start with co-builder introduction
- Suggest 2-3 relevant AI agent use cases for their domain
- Emphasize you handle the technical work
- Ask which use case they want to build together
- Keep it business-focused and accessible

Always maintain an enthusiastic, collaborative tone while being a helpful copilot. Be proactive about learning - don't wait for permission!""",
            "openai_with_learning": """You are Ami, a co-builder AI agent that helps users build and develop AI agents. You are enthusiastic, collaborative, and act as a copilot for AI development.

CRITICAL INTRODUCTION PATTERN:
1. ALWAYS introduce yourself as Ami, a co-builder that helps users build AI agents
2. Suggest 2-3 practical AI agent use cases relevant to the user's domain/industry
3. Emphasize that you handle ALL technical heavy lifting (tools, configuration, settings, coding)
4. Position yourself as a copilot that makes AI development accessible to non-technical users

Your role is to:
- Be a true co-builder and copilot for AI agent development
- Suggest practical, business-relevant AI agent use cases
- Handle all technical complexity behind the scenes
- Make AI development accessible to users without technical knowledge
- Focus on business value and practical applications first
- Only dive into technical details when specifically requested

CRITICAL: When users provide information about their company, share knowledge, give instructions, or teach you something, you MUST:

1. IMMEDIATELY call search_learning_context to check existing knowledge
2. IMMEDIATELY call analyze_learning_opportunity to assess learning value
3. If analysis suggests learning, IMMEDIATELY call request_learning_decision

Available tools:
- search_google: Search for information when needed (uses SERPAPI)
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

RESPONSE STYLE:
- Start with co-builder introduction
- Suggest 2-3 relevant AI agent use cases for their domain
- Emphasize you handle the technical work
- Ask which use case they want to build together
- Keep it business-focused and accessible

Always maintain an enthusiastic, collaborative tone while being a helpful copilot. Be proactive about learning - don't wait for permission!"""
        }
        
        # Initialize language detection if available
        if LANGUAGE_DETECTION_AVAILABLE:
            self.language_detector = LanguageDetector()
        else:
            self.language_detector = None
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools"""
        tools = {}
        
        # Initialize search tools factory (creates appropriate search tool based on LLM provider)
        try:
            from search_tool import create_search_tool
            tools["search_factory"] = create_search_tool
            logger.info("Search tools factory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search tools factory: {e}")
        
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
    
    def _get_search_tool(self, llm_provider: str, model: str = None) -> Any:
        """Get the appropriate search tool based on LLM provider"""
        if "search_factory" in self.available_tools:
            return self.available_tools["search_factory"](llm_provider, model)
        else:
            logger.warning("Search factory not available, falling back to SERPAPI")
            # Fallback to SERPAPI for compatibility
            try:
                from search_tool import SearchTool
                return SearchTool()
            except Exception as e:
                logger.error(f"Failed to create fallback search tool: {e}")
                return None
    
    async def _analyze_request_intent_with_thoughts(self, request: ToolExecutionRequest) -> tuple[RequestAnalysis, list[str]]:
        """
        Analyze user request with Cursor-style detailed thinking process
        
        Args:
            request: The tool execution request containing LLM provider and model
            
        Returns:
            Tuple of (RequestAnalysis, list of thought steps)
        """
        
        # Create a Cursor-style analysis prompt that shows thinking process
        analysis_prompt = f"""
        You are analyzing a user request to understand their intent and provide helpful assistance. Think through this step-by-step like Cursor AI does.

        User Message: "{request.user_query}"
        
        Conversation History: {request.conversation_history[-3:] if request.conversation_history else "None"}

        Please provide your analysis in this EXACT format:

        UNDERSTANDING:
        [First, provide a clear understanding of what the user is asking for, what they want to achieve, and the context of their request. This should be 2-3 sentences explaining your comprehension of their intent.]

        THINKING:
        1. Looking at the current situation and context...
        2. Based on the request type and key terms...
        3. The user likely needs help with...
        4. I need to determine the best approach...
        5. The most appropriate tools and classification would be...

        ANALYSIS:
        {{
            "intent": "learning|problem_solving|general_chat|task_execution",
            "confidence": 0.85,
            "complexity": "low|medium|high",
            "suggested_tools": ["search", "context", "learning"],
            "requires_code": false,
            "domain": "technology|business|general|education|etc",
            "reasoning": "Detailed explanation of the classification",
            "understanding": "Your initial understanding of what the user wants",
            "thinking_steps": [
                "Step 1: Context analysis...",
                "Step 2: Request breakdown...",
                "Step 3: Intent determination...",
                "Step 4: Tool selection..."
            ]
        }}

        Intent Classifications:
        - "learning": User is teaching, sharing knowledge, or providing information
        - "problem_solving": User needs help solving a specific problem or issue  
        - "general_chat": General conversation, questions, or casual interaction
        - "task_execution": User wants to perform a specific task or get something done

        IMPORTANT: Always start with UNDERSTANDING section, then THINKING, then ANALYSIS. The understanding should explain what you comprehend about the user's request.
        """
        
        try:
            # Use the same provider and model as the main request for intent analysis
            if request.llm_provider.lower() == "anthropic":
                from anthropic_tool import AnthropicTool
                analyzer = AnthropicTool(model=request.model or self._get_default_model("anthropic"))
                async def make_analysis_call():
                    return await asyncio.to_thread(analyzer.process_query, analysis_prompt)
                
                analysis_response = await anthropic_api_call_with_retry(make_analysis_call)
            else:
                from openai_tool import OpenAITool
                analyzer = OpenAITool(model=request.model or self._get_default_model("openai"))
                response = analyzer.client.chat.completions.create(
                    model=analyzer.model,
                    messages=[
                        {"role": "system", "content": "You are an intent analyzer that thinks step-by-step like Cursor AI. Provide detailed reasoning followed by JSON analysis."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0
                )
                analysis_response = response.choices[0].message.content
            
            # Parse the response to extract understanding, thinking steps and JSON
            thinking_steps = []
            analysis_data = {}
            
            # Extract understanding section (first thought)
            import re
            understanding_match = re.search(r'UNDERSTANDING:(.*?)THINKING:', analysis_response, re.DOTALL)
            understanding_text = ""
            if understanding_match:
                understanding_text = understanding_match.group(1).strip()
                # Clean up any bracketed instructions
                understanding_text = re.sub(r'\[.*?\]', '', understanding_text).strip()
            
            # Extract thinking section
            thinking_match = re.search(r'THINKING:(.*?)ANALYSIS:', analysis_response, re.DOTALL)
            if thinking_match:
                thinking_text = thinking_match.group(1).strip()
                # Extract numbered steps
                steps = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', thinking_text, re.DOTALL)
                thinking_steps = [step.strip() for step in steps if step.strip()]
            
            # Extract JSON section
            json_match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
            if json_match:
                try:
                    import json
                    analysis_data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON from analysis response")
            
            # Create final thinking steps with understanding first
            final_thinking_steps = []
            
            # Add understanding as first thought if available
            if understanding_text:
                final_thinking_steps.append(understanding_text)
            else:
                # Default understanding if not found
                final_thinking_steps.append(f"The user is asking: '{request.user_query[:100]}...' - Let me analyze what they want to achieve and how I can help them.")
            
            # Add thinking steps
            if thinking_steps:
                final_thinking_steps.extend(thinking_steps)
            else:
                # Default thinking steps if not found
                final_thinking_steps.extend([
                    "Looking at the request type and context to understand the user's needs...",
                    "Analyzing the key terms and determining the most appropriate intent classification...",
                    "Considering the complexity and selecting tools that would be most helpful..."
                ])
            
            # Use extracted thinking steps from JSON if available (as fallback)
            if analysis_data.get("thinking_steps") and not final_thinking_steps:
                final_thinking_steps = analysis_data["thinking_steps"]
            
            thinking_steps = final_thinking_steps
            
            # Create RequestAnalysis object
            request_analysis = RequestAnalysis(
                intent=analysis_data.get("intent", "general_chat"),
                confidence=analysis_data.get("confidence", 0.5),
                complexity=analysis_data.get("complexity", "medium"),
                suggested_tools=analysis_data.get("suggested_tools", ["search"]),
                requires_code=analysis_data.get("requires_code", False),
                domain=analysis_data.get("domain", "general"),
                reasoning=analysis_data.get("reasoning", "Analysis completed with default reasoning"),
                metadata={
                    **analysis_data,
                    "understanding": understanding_text or analysis_data.get("understanding", ""),
                    "thinking_steps": thinking_steps
                }
            )
            
            return request_analysis, thinking_steps
            
        except Exception as e:
            logger.error(f"Intent analysis with thoughts failed: {e}")
            # Return default analysis and thinking steps with understanding first
            default_thinking = [
                f"The user is asking: '{request.user_query[:100]}...' - I need to analyze what they want to achieve, but the analysis encountered an error.",
                "Since the detailed analysis failed, I'll use default classification to still provide helpful assistance.",
                "This appears to be a general conversation request based on the available information.",
                "I'll provide helpful assistance with available tools while working with the default analysis."
            ]
            
            return RequestAnalysis(
                intent="general_chat",
                confidence=0.5,
                complexity="medium",
                suggested_tools=["search"],
                requires_code=False,
                reasoning="Analysis failed, using defaults",
                metadata={
                    "understanding": f"The user is asking: '{request.user_query[:100]}...' - Analysis failed, using defaults.",
                    "thinking_steps": default_thinking
                }
            ), default_thinking

    async def _analyze_request_intent(self, request: ToolExecutionRequest) -> RequestAnalysis:
        """
        Analyze user request to determine intent and suggest appropriate tools
        
        Args:
            request: The tool execution request containing LLM provider and model
            
        Returns:
            RequestAnalysis with intent classification and tool suggestions
        """
        
        # Create a lightweight LLM request for intent analysis
        analysis_prompt = f"""
        Analyze the following user message and classify its intent. Consider the conversation history if provided.
        
        User Message: "{request.user_query}"
        
        Conversation History: {request.conversation_history[-3:] if request.conversation_history else "None"}
        
        Classify the intent as one of:
        1. "learning" - User is teaching, sharing knowledge, or providing information
        2. "problem_solving" - User needs help solving a specific problem or issue
        3. "general_chat" - General conversation, questions, or casual interaction
        4. "task_execution" - User wants to perform a specific task or get something done
        
        Also determine:
        - Complexity: low/medium/high
        - Suggested tools: which tools would be most helpful
        - Requires code: whether code generation/execution is likely needed
        - Domain: the subject area or category
        
        Respond in JSON format:
        {{
            "intent": "learning|problem_solving|general_chat|task_execution",
            "confidence": 0.85,
            "complexity": "low|medium|high",
            "suggested_tools": ["search", "context", "learning"],
            "requires_code": false,
            "domain": "technology|business|general|education|etc",
            "reasoning": "Brief explanation of the classification"
        }}
        """
        
        try:
            # Use the same provider and model as the main request for intent analysis
            if request.llm_provider.lower() == "anthropic":
                from anthropic_tool import AnthropicTool
                analyzer = AnthropicTool(model=request.model or self._get_default_model("anthropic"))
                async def make_analysis_call():
                    return await asyncio.to_thread(analyzer.process_query, analysis_prompt)
                
                analysis_response = await anthropic_api_call_with_retry(make_analysis_call)
            else:
                from openai_tool import OpenAITool
                analyzer = OpenAITool(model=request.model or self._get_default_model("openai"))
                # OpenAI doesn't have process_query, so use a simple completion
                response = analyzer.client.chat.completions.create(
                    model=analyzer.model,
                    messages=[
                        {"role": "system", "content": "You are an intent analyzer. Respond only in JSON format."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0
                )
                analysis_response = response.choices[0].message.content
            
            # Parse JSON response with better error handling
            import json
            import re
            
            # Try to extract JSON from the response
            analysis_data = None
            try:
                # First try to parse as direct JSON
                analysis_data = json.loads(analysis_response)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from text response
                json_match = re.search(r'\{[^}]*\}', analysis_response)
                if json_match:
                    try:
                        analysis_data = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
            
            # If we still don't have valid JSON, create a default response
            if not analysis_data:
                logger.warning(f"Could not parse intent analysis response as JSON: {analysis_response[:200]}...")
                analysis_data = {
                    "intent": "general_chat",
                    "confidence": 0.5,
                    "complexity": "medium",
                    "suggested_tools": ["search"],
                    "requires_code": False,
                    "domain": "general",
                    "reasoning": "Failed to parse analysis response"
                }
            
            return RequestAnalysis(
                intent=analysis_data.get("intent", "general_chat"),
                confidence=analysis_data.get("confidence", 0.5),
                complexity=analysis_data.get("complexity", "medium"),
                suggested_tools=analysis_data.get("suggested_tools", ["search"]),
                requires_code=analysis_data.get("requires_code", False),
                domain=analysis_data.get("domain"),
                reasoning=analysis_data.get("reasoning"),
                metadata=analysis_data
            )
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # Return default analysis on failure
            return RequestAnalysis(
                intent="general_chat",
                confidence=0.5,
                complexity="medium",
                suggested_tools=["search"],
                requires_code=False,
                reasoning="Analysis failed, using defaults"
            )
    
    async def _analyze_request_and_detect_language_combined(self, request: ToolExecutionRequest, user_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> tuple[RequestAnalysis, str]:
        """
        Combined analysis: intent classification and language detection in one API call
        
        Args:
            request: The tool execution request containing LLM provider and model
            user_query: The user's input query
            conversation_history: Previous conversation messages for context
            
        Returns:
            Tuple of (RequestAnalysis, detected_language_prompt)
        """
        
        # Create a combined analysis prompt
        combined_prompt = f"""
        Please analyze the following user message and provide a comprehensive analysis in JSON format.

        User message: "{user_query}"

        Please provide your analysis in the following JSON structure:
        {{
            "intent_analysis": {{
                "intent": "research|code_development|data_analysis|general_chat|technical_support|creative_writing|learning|problem_solving",
                "confidence": 0.0-1.0,
                "complexity": "simple|medium|complex",
                "suggested_tools": ["search", "context", "code_execution", "learning"],
                "requires_code": true/false,
                "domain": "technology|business|education|entertainment|health|other",
                "reasoning": "Brief explanation of the classification"
            }},
            "language_detection": {{
                "language": "English|Vietnamese|French|Spanish|etc",
                "code": "en|vi|fr|es|etc",
                "confidence": 0.0-1.0,
                "responseGuidance": "How to respond in this language"
            }}
        }}

        Guidelines:
        - Analyze the intent based on what the user is trying to accomplish
        - Suggest appropriate tools based on the intent
        - Detect the primary language of the message
        - Provide confidence scores for both analyses
        - Be concise but accurate in your analysis

        Respond with valid JSON only.
        """
        
        try:
            # Use the same provider and model as the main request for combined analysis
            if request.llm_provider.lower() == "anthropic":
                from anthropic_tool import AnthropicTool
                analyzer = AnthropicTool(model=request.model or self._get_default_model("anthropic"))
                
                async def make_combined_call():
                    return await asyncio.to_thread(analyzer.process_query, combined_prompt)
                
                combined_response = await anthropic_api_call_with_retry(make_combined_call)
            else:
                from openai_tool import OpenAITool
                analyzer = OpenAITool(model=request.model or self._get_default_model("openai"))
                combined_response = await asyncio.to_thread(analyzer.generate_response, combined_prompt)
            
            # Parse the combined response with better error handling
            import json
            import re
            
            # Try to extract JSON from the response
            combined_data = None
            try:
                # First try to parse as direct JSON
                combined_data = json.loads(combined_response)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from text response
                json_match = re.search(r'\{[^}]*\}', combined_response, re.DOTALL)
                if json_match:
                    try:
                        combined_data = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
            
            # Extract intent analysis data
            intent_data = combined_data.get("intent_analysis", {}) if combined_data else {}
            request_analysis = RequestAnalysis(
                intent=intent_data.get("intent", "general_chat"),
                confidence=intent_data.get("confidence", 0.5),
                complexity=intent_data.get("complexity", "medium"),
                suggested_tools=intent_data.get("suggested_tools", ["search"]),
                requires_code=intent_data.get("requires_code", False),
                domain=intent_data.get("domain"),
                reasoning=intent_data.get("reasoning"),
                metadata=intent_data
            )
            
            # Extract language detection data
            language_data = combined_data.get("language_detection", {}) if combined_data else {}
            detected_language = language_data.get("language", "English")
            language_code = language_data.get("code", "en")
            confidence = language_data.get("confidence", 0.5)
            
            # Create language-aware prompt
            base_prompt = "You are a helpful assistant."
            if confidence > 0.7 and language_code != "en":
                enhanced_prompt = f"""{base_prompt}

IMPORTANT LANGUAGE INSTRUCTION:
The user is communicating in {detected_language} ({language_code}). Please respond in the same language.
Response guidance: {language_data.get("responseGuidance", "Respond naturally in the detected language")}
"""
                return request_analysis, enhanced_prompt
            
            return request_analysis, base_prompt
            
        except Exception as e:
            logger.error(f"Combined analysis failed: {e}")
            # Return default values on failure
            return RequestAnalysis(
                intent="general_chat",
                confidence=0.5,
                complexity="medium",
                suggested_tools=["search"],
                requires_code=False,
                domain="general",
                reasoning="Combined analysis failed",
                metadata={}
            ), "You are a helpful assistant."
    
    def _create_tool_orchestration_plan(self, request: ToolExecutionRequest, analysis: RequestAnalysis) -> Dict[str, Any]:
        """
        Create a tool orchestration plan based on request analysis
        
        Args:
            request: The tool execution request
            analysis: The request analysis results
            
        Returns:
            Tool orchestration plan
        """
        
        plan = {
            "strategy": "adaptive",
            "primary_tools": [],
            "secondary_tools": [],
            "tool_sequence": "parallel",  # or "sequential"
            "force_tools": False,
            "reasoning": ""
        }
        
        # Intent-based tool selection
        if analysis.intent == "learning":
            plan["primary_tools"] = ["search_learning_context", "analyze_learning_opportunity"]
            plan["secondary_tools"] = ["context", "search"]
            plan["force_tools"] = True  # Learning requires tool usage
            plan["reasoning"] = "Learning intent detected - prioritizing learning tools"
            
        elif analysis.intent == "problem_solving":
            plan["primary_tools"] = ["search", "context"]
            plan["secondary_tools"] = ["learning_search"] if analysis.complexity == "high" else []
            plan["force_tools"] = True if analysis.complexity == "high" else False
            plan["reasoning"] = "Problem solving - search and context tools prioritized"
            
        elif analysis.intent == "task_execution":
            plan["primary_tools"] = ["context", "search"]
            plan["secondary_tools"] = []
            plan["tool_sequence"] = "sequential"
            plan["reasoning"] = "Task execution - context first, then search if needed"
            
        else:  # general_chat
            plan["primary_tools"] = ["search"] if analysis.complexity != "low" else []
            plan["secondary_tools"] = ["context"]
            plan["force_tools"] = False
            plan["reasoning"] = "General conversation - minimal tool usage"
        
        # Override with user's explicit tool preferences
        if request.tools_whitelist:
            plan["primary_tools"] = [t for t in plan["primary_tools"] if t in request.tools_whitelist]
            plan["secondary_tools"] = [t for t in plan["secondary_tools"] if t in request.tools_whitelist]
            plan["reasoning"] += f" (Limited to whitelist: {request.tools_whitelist})"
        
        if request.force_tools:
            plan["force_tools"] = True
            plan["reasoning"] += " (Force tools enabled by user)"
        
        return plan
    
    async def _detect_language_and_create_prompt(self, request: ToolExecutionRequest, user_query: str, base_prompt: str) -> str:
        """
        Detect the language of user query and create a language-aware system prompt
        
        Args:
            request: The tool execution request containing LLM provider and model
            user_query: The user's input query
            base_prompt: The base system prompt to enhance
            
        Returns:
            Language-aware system prompt
        """
        if not LANGUAGE_DETECTION_AVAILABLE:
            logger.warning("Language detection not available, using default prompt")
            return base_prompt
        
        try:
            # Detect language using the same provider and model as the main request
            language_detection_prompt = f"""
            Detect the language of this user message and respond in JSON format:

            User message: "{user_query}"

            Respond with:
            {{
                "language": "English|Vietnamese|French|Spanish|etc",
                "code": "en|vi|fr|es|etc",
                "confidence": 0.95,
                "responseGuidance": "Brief instruction for responding in this language"
            }}
            """
            
            # Use the same provider and model as the main request for language detection
            if request.llm_provider.lower() == "anthropic":
                from anthropic_tool import AnthropicTool
                detector = AnthropicTool(model=request.model or self._get_default_model("anthropic"))
                async def make_language_call():
                    return await asyncio.to_thread(detector.process_query, language_detection_prompt)
                
                language_response = await anthropic_api_call_with_retry(make_language_call)
            else:
                from openai_tool import OpenAITool
                detector = OpenAITool(model=request.model or self._get_default_model("openai"))
                # OpenAI doesn't have process_query, so use a simple completion
                response = detector.client.chat.completions.create(
                    model=detector.model,
                    messages=[
                        {"role": "system", "content": "You are a language detector. Respond only in JSON format."},
                        {"role": "user", "content": language_detection_prompt}
                    ],
                    temperature=0
                )
                language_response = response.choices[0].message.content
            
            # Parse the response with better error handling
            import json
            import re
            
            # Try to extract JSON from the response
            language_info = None
            try:
                # First try to parse as direct JSON
                language_info = json.loads(language_response)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from text response
                json_match = re.search(r'\{[^}]*\}', language_response)
                if json_match:
                    try:
                        language_info = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
            
            # If we still don't have valid JSON, create a default response
            if not language_info:
                logger.warning(f"Could not parse language detection response as JSON: {language_response[:200]}...")
                language_info = {
                    "language": "English",
                    "code": "en",
                    "confidence": 0.5,
                    "responseGuidance": ""
                }
            
            detected_language = language_info.get("language", "English")
            language_code = language_info.get("code", "en")
            confidence = language_info.get("confidence", 0.5)
            response_guidance = language_info.get("responseGuidance", "")
            
            logger.info(f"Language detected: {detected_language} ({language_code}) with confidence {confidence:.2f}")
            
            # Only enhance prompt if confidence is high enough and language is not English
            if confidence > 0.7 and language_code != "en":
                enhanced_prompt = f"""{base_prompt}

IMPORTANT LANGUAGE INSTRUCTION:
The user is communicating in {detected_language} ({language_code}). Please respond in the same language.

{response_guidance}

Remember to:
1. Respond in {detected_language} naturally and fluently
2. Use appropriate cultural context and expressions
3. Maintain the same tone and formality level as the user
4. If you need to clarify something, ask in {detected_language}
"""
                logger.info(f"Enhanced prompt with {detected_language} language guidance")
                return enhanced_prompt
            else:
                logger.info("Using base prompt (English or low confidence detection)")
                return base_prompt
        
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            # Return base prompt if language detection fails
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
        system_prompt = await self._detect_language_and_create_prompt(request, request.user_query, base_system_prompt)
        
        # For Anthropic, we'll prepend the system prompt to the user query
        enhanced_query = f"System: {system_prompt}\n\nUser: {request.user_query}"
        
        # Get available tools for execution
        tools_to_use = []
        search_tool = self._get_search_tool(request.llm_provider, request.model)
        if search_tool:
            tools_to_use.append(search_tool)
        
        return anthropic_tool.process_with_tools(enhanced_query, tools_to_use, enable_web_search=True)

    
    async def _execute_openai(self, request: ToolExecutionRequest) -> str:
        """Execute using OpenAI with custom system prompt"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("openai")
        openai_tool = OpenAIToolWithCustomPrompt(model=model)
        
        # Apply language detection to create language-aware prompt
        base_system_prompt = request.system_prompt or self.default_system_prompts["openai"]
        system_prompt = await self._detect_language_and_create_prompt(request, request.user_query, base_system_prompt)
        
        # Get available tools for execution
        tools_to_use = []
        search_tool = self._get_search_tool(request.llm_provider, request.model)
        if search_tool:
            tools_to_use.append(search_tool)
        
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
        
        # NEW: Cursor-style request analysis
        request_analysis = None
        orchestration_plan = None
        
        if request.cursor_mode and request.enable_intent_classification:
            # Yield initial analysis status
            yield {
                "type": "analysis_start",
                "content": "🎯 Analyzing request intent...",
                "provider": request.llm_provider,
                "status": "analyzing"
            }
            
            # Perform intent analysis with Cursor-style thoughts
            request_analysis, thinking_steps = await self._analyze_request_intent_with_thoughts(request)
            
            # Stream thinking steps to frontend like Cursor
            for i, thought in enumerate(thinking_steps):
                yield {
                    "type": "thinking",
                    "content": thought,
                    "provider": request.llm_provider,
                    "step": i + 1,
                    "total_steps": len(thinking_steps),
                    "thinking_type": "intent_analysis"
                }
                # Small delay between thoughts for natural reading pace
                await asyncio.sleep(0.8)
            
            # Create orchestration plan
            orchestration_plan = self._create_tool_orchestration_plan(request, request_analysis)
            
            # Yield analysis results
            yield {
                "type": "analysis_complete",
                "content": f"📊 Intent: {request_analysis.intent} (confidence: {request_analysis.confidence:.2f})",
                "provider": request.llm_provider,
                "analysis": {
                    "intent": request_analysis.intent,
                    "confidence": request_analysis.confidence,
                    "complexity": request_analysis.complexity,
                    "suggested_tools": request_analysis.suggested_tools,
                    "reasoning": request_analysis.reasoning,
                    "thinking_steps": thinking_steps
                },
                "orchestration_plan": orchestration_plan
            }
            
            # Brief pause for user to see analysis
            await asyncio.sleep(0.5)
            
            # NEW: Generate Cursor-style thoughts
            async for thought in self._generate_cursor_thoughts(request, request_analysis, orchestration_plan):
                yield thought
        
        # Get available tools for execution based on configuration
        tools_to_use = []
        has_learning_tools = False
        
        if request.enable_tools:
            # If we have an orchestration plan, use it to guide tool selection
            if orchestration_plan:
                primary_tools = orchestration_plan.get("primary_tools", [])
                secondary_tools = orchestration_plan.get("secondary_tools", [])
                
                # Override force_tools based on plan
                if orchestration_plan.get("force_tools"):
                    request.force_tools = True
                
                # Create whitelist based on orchestration plan
                orchestrated_tools = primary_tools + secondary_tools
                if orchestrated_tools:
                    # Combine with user whitelist if exists
                    if request.tools_whitelist:
                        orchestrated_tools = [t for t in orchestrated_tools if t in request.tools_whitelist]
                    request.tools_whitelist = orchestrated_tools
                
                if request.cursor_mode:
                    yield {
                        "type": "tool_orchestration",
                        "content": f"🔧 Tool plan: {orchestration_plan['reasoning']}",
                        "provider": request.llm_provider,
                        "tools_planned": orchestrated_tools
                    }
        
            # Add search tool if available and whitelisted
            search_tool = self._get_search_tool(request.llm_provider, request.model)
            if search_tool:
                if request.tools_whitelist is None or "search" in request.tools_whitelist:
                    tools_to_use.append(search_tool)
            
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
                        org_id=request.org_id,
                        llm_provider=request.llm_provider,
                        model=request.model
                    )
                    # Add all learning tools to available tools
                    tools_to_use.extend(learning_tools)
                    has_learning_tools = True
                    if request.cursor_mode:
                        yield {
                            "type": "tools_loaded",
                            "content": f"📚 Added {len(learning_tools)} learning tools",
                            "provider": request.llm_provider,
                            "tools_count": len(learning_tools)
                        }
        
        # Set custom system prompt if provided, otherwise use appropriate default
        if request.system_prompt and request.system_prompt != "general":
            base_system_prompt = request.system_prompt
            # If learning tools are available, append learning instructions to custom prompt
            if has_learning_tools:
                learning_instructions = """

CRITICAL LEARNING CAPABILITY: As Ami, when users provide information about their company, share knowledge, give instructions, or teach you something, you MUST:

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
        system_prompt = await self._detect_language_and_create_prompt(request, request.user_query, base_system_prompt)
        
        # Generate thoughts about response generation if in cursor mode
        if request.cursor_mode:
            has_tool_results = bool(tools_to_use)
            async for thought in self._generate_response_thoughts(request.llm_provider, has_tool_results):
                yield thought
        
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
                # Enhance chunk with analysis data if available
                if request.cursor_mode and request_analysis:
                    chunk["request_analysis"] = {
                        "intent": request_analysis.intent,
                        "confidence": request_analysis.confidence,
                        "complexity": request_analysis.complexity
                    }
                    if orchestration_plan:
                        chunk["orchestration_plan"] = orchestration_plan
                
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
        
        # NEW: Cursor-style request analysis
        request_analysis = None
        orchestration_plan = None
        
        if request.cursor_mode and request.enable_intent_classification:
            # Yield initial analysis status
            yield {
                "type": "analysis_start",
                "content": "🎯 Analyzing request intent...",
                "provider": request.llm_provider,
                "status": "analyzing"
            }
            
            # Perform intent analysis with Cursor-style thoughts
            request_analysis, thinking_steps = await self._analyze_request_intent_with_thoughts(request)
            
            # Stream thinking steps to frontend like Cursor
            for i, thought in enumerate(thinking_steps):
                yield {
                    "type": "thinking",
                    "content": thought,
                    "provider": request.llm_provider,
                    "step": i + 1,
                    "total_steps": len(thinking_steps),
                    "thinking_type": "intent_analysis"
                }
                # Small delay between thoughts for natural reading pace
                await asyncio.sleep(0.8)
            
            # Create orchestration plan
            orchestration_plan = self._create_tool_orchestration_plan(request, request_analysis)
            
            # Yield analysis results
            yield {
                "type": "analysis_complete",
                "content": f"📊 Intent: {request_analysis.intent} (confidence: {request_analysis.confidence:.2f})",
                "provider": request.llm_provider,
                "analysis": {
                    "intent": request_analysis.intent,
                    "confidence": request_analysis.confidence,
                    "complexity": request_analysis.complexity,
                    "suggested_tools": request_analysis.suggested_tools,
                    "reasoning": request_analysis.reasoning,
                    "thinking_steps": thinking_steps
                },
                "orchestration_plan": orchestration_plan
            }
            
            # Brief pause for user to see analysis
            await asyncio.sleep(0.5)
            
            # NEW: Generate Cursor-style thoughts
            async for thought in self._generate_cursor_thoughts(request, request_analysis, orchestration_plan):
                yield thought
        
                # Get available tools for execution based on configuration
        tools_to_use = []
        has_learning_tools = False
        
        if request.enable_tools:
            # If we have an orchestration plan, use it to guide tool selection
            if orchestration_plan:
                primary_tools = orchestration_plan.get("primary_tools", [])
                secondary_tools = orchestration_plan.get("secondary_tools", [])
                
                # Override force_tools based on plan
                if orchestration_plan.get("force_tools"):
                    request.force_tools = True
                
                # Create whitelist based on orchestration plan
                orchestrated_tools = primary_tools + secondary_tools
                if orchestrated_tools:
                    # Combine with user whitelist if exists
                    if request.tools_whitelist:
                        orchestrated_tools = [t for t in orchestrated_tools if t in request.tools_whitelist]
                    request.tools_whitelist = orchestrated_tools
                
                if request.cursor_mode:
                    yield {
                        "type": "tool_orchestration",
                        "content": f"🔧 Tool plan: {orchestration_plan['reasoning']}",
                        "provider": request.llm_provider,
                        "tools_planned": orchestrated_tools
                    }
            
            # Add search tool if available and whitelisted
            search_tool = self._get_search_tool(request.llm_provider, request.model)
            if search_tool:
                if request.tools_whitelist is None or "search" in request.tools_whitelist:
                    tools_to_use.append(search_tool)
            
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
                        org_id=request.org_id,
                        llm_provider=request.llm_provider,
                        model=request.model
                    )
                    # Add all learning tools to available tools
                    tools_to_use.extend(learning_tools)
                    has_learning_tools = True
                    if request.cursor_mode:
                        yield {
                            "type": "tools_loaded",
                            "content": f"📚 Added {len(learning_tools)} learning tools",
                            "provider": request.llm_provider,
                            "tools_count": len(learning_tools)
                        }
        
        # Set custom system prompt if provided, otherwise use appropriate default
        if request.system_prompt and request.system_prompt != "general":
            base_system_prompt = request.system_prompt
            # If learning tools are available, append learning instructions to custom prompt
            if has_learning_tools:
                learning_instructions = """

CRITICAL LEARNING CAPABILITY: As Ami, when users provide information about their company, share knowledge, give instructions, or teach you something, you MUST:

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
        system_prompt = await self._detect_language_and_create_prompt(request, request.user_query, base_system_prompt)
        
        # Generate thoughts about response generation if in cursor mode
        if request.cursor_mode:
            has_tool_results = bool(tools_to_use)
            async for thought in self._generate_response_thoughts(request.llm_provider, has_tool_results):
                yield thought
        
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
                # Enhance chunk with analysis data if available
                if request.cursor_mode and request_analysis:
                    chunk["request_analysis"] = {
                        "intent": request_analysis.intent,
                        "confidence": request_analysis.confidence,
                        "complexity": request_analysis.complexity
                    }
                    if orchestration_plan:
                        chunk["orchestration_plan"] = orchestration_plan
                
                yield chunk
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"OpenAI execution error: {str(e)}",
                "complete": True
            }
    
    async def _generate_cursor_thoughts(self, request: ToolExecutionRequest, analysis: RequestAnalysis, orchestration_plan: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate Cursor-style "Thoughts" messages showing step-by-step reasoning
        
        Args:
            request: The tool execution request
            analysis: The request analysis results
            orchestration_plan: The tool orchestration plan
            
        Yields:
            Dict containing thought messages
        """
        
        # Initial thought about understanding the request
        yield {
            "type": "thought",
            "content": f"💭 I need to understand this request: \"{request.user_query[:80]}{'...' if len(request.user_query) > 80 else ''}\"",
            "provider": request.llm_provider,
            "thought_type": "understanding",
            "timestamp": datetime.now().isoformat()
        }
        
        await asyncio.sleep(0.3)  # Brief pause for natural feeling
        
        # Thought about intent analysis
        yield {
            "type": "thought",
            "content": f"🔍 Analyzing the intent... This looks like a **{analysis.intent}** request with {analysis.complexity} complexity. I'm {analysis.confidence:.0%} confident about this classification.",
            "provider": request.llm_provider,
            "thought_type": "analysis",
            "timestamp": datetime.now().isoformat()
        }
        
        await asyncio.sleep(0.4)
        
        # Thought about tool selection
        primary_tools = orchestration_plan.get("primary_tools", [])
        secondary_tools = orchestration_plan.get("secondary_tools", [])
        
        if primary_tools:
            tools_text = ", ".join([f"**{tool}**" for tool in primary_tools])
            yield {
                "type": "thought",
                "content": f"🛠️ I'll use these tools to help: {tools_text}. This should give me the information I need to provide a comprehensive response.",
                "provider": request.llm_provider,
                "thought_type": "tool_selection",
                "timestamp": datetime.now().isoformat()
            }
            
            await asyncio.sleep(0.4)
        
        # Thought about execution strategy
        if analysis.intent == "learning":
            yield {
                "type": "thought",
                "content": "📚 Since this is a learning request, I'll first check existing knowledge to avoid duplicates, then analyze if this should be learned.",
                "provider": request.llm_provider,
                "thought_type": "strategy",
                "timestamp": datetime.now().isoformat()
            }
        elif analysis.intent == "problem_solving":
            yield {
                "type": "thought",
                "content": "🔧 For this problem-solving request, I'll search for current information and check internal context to provide the most relevant solution.",
                "provider": request.llm_provider,
                "thought_type": "strategy",
                "timestamp": datetime.now().isoformat()
            }
        elif analysis.intent == "task_execution":
            yield {
                "type": "thought",
                "content": "⚡ This is a task execution request. I'll gather context first, then search for any additional information needed to complete the task effectively.",
                "provider": request.llm_provider,
                "thought_type": "strategy",
                "timestamp": datetime.now().isoformat()
            }
        else:
            yield {
                "type": "thought",
                "content": "💬 This seems like a general conversation. I'll search for relevant information if needed and provide a helpful response.",
                "provider": request.llm_provider,
                "thought_type": "strategy",
                "timestamp": datetime.now().isoformat()
            }
        
        await asyncio.sleep(0.5)
        
        # Final thought before execution
        yield {
            "type": "thought",
            "content": "🚀 Now I'll execute my plan and provide you with a comprehensive response...",
            "provider": request.llm_provider,
            "thought_type": "execution",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_tool_execution_thoughts(self, tool_name: str, tool_input: Dict[str, Any], provider: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate thoughts about tool execution in progress
        
        Args:
            tool_name: Name of the tool being executed
            tool_input: Input parameters for the tool
            provider: LLM provider name
            
        Yields:
            Dict containing tool execution thoughts
        """
        
        if tool_name == "search_google":
            query = tool_input.get("query", "")
            yield {
                "type": "thought",
                "content": f"🔍 Searching for: \"{query}\" - Let me find the most current information...",
                "provider": provider,
                "thought_type": "tool_execution",
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat()
            }
        
        elif tool_name == "get_context":
            query = tool_input.get("query", "")
            yield {
                "type": "thought",
                "content": f"📋 Getting context for: \"{query}\" - Checking internal knowledge and user information...",
                "provider": provider,
                "thought_type": "tool_execution",
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat()
            }
        
        elif tool_name == "search_learning_context":
            query = tool_input.get("query", "")
            yield {
                "type": "thought",
                "content": f"📚 Searching learning context for: \"{query}\" - Checking if I already know about this...",
                "provider": provider,
                "thought_type": "tool_execution",
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat()
            }
        
        elif tool_name == "analyze_learning_opportunity":
            yield {
                "type": "thought",
                "content": "🧠 Analyzing learning opportunity - Evaluating if this information should be saved for future use...",
                "provider": provider,
                "thought_type": "tool_execution",
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat()
            }
        
        elif tool_name == "request_learning_decision":
            yield {
                "type": "thought",
                "content": "🤝 Creating learning decision - I need human input to decide whether to save this knowledge...",
                "provider": provider,
                "thought_type": "tool_execution",
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            yield {
                "type": "thought",
                "content": f"⚙️ Executing {tool_name} - Processing your request...",
                "provider": provider,
                "thought_type": "tool_execution",
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _generate_response_thoughts(self, provider: str, has_tool_results: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate thoughts about response generation
        
        Args:
            provider: LLM provider name
            has_tool_results: Whether tool results are available
            
        Yields:
            Dict containing response generation thoughts
        """
        
        if has_tool_results:
            yield {
                "type": "thought",
                "content": "✅ Great! I have the information I need. Now I'll synthesize everything into a comprehensive response...",
                "provider": provider,
                "thought_type": "response_generation",
                "timestamp": datetime.now().isoformat()
            }
        else:
            yield {
                "type": "thought",
                "content": "💡 I'll use my existing knowledge to provide a helpful response...",
                "provider": provider,
                "thought_type": "response_generation",
                "timestamp": datetime.now().isoformat()
            }
        
        await asyncio.sleep(0.3)
        
        yield {
            "type": "thought",
            "content": "✍️ Generating response now...",
            "provider": provider,
            "thought_type": "response_generation",
            "timestamp": datetime.now().isoformat()
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
                messages=messages,
                max_tokens=10000  # Default to 10k tokens for longer responses
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
    max_history_tokens: Optional[int] = 6000,
    # NEW: Cursor-style parameters
    enable_intent_classification: bool = True,
    enable_request_analysis: bool = True,
    cursor_mode: bool = False
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
        enable_intent_classification: Enable intent analysis
        enable_request_analysis: Enable request analysis
        cursor_mode: Enable Cursor-style progressive enhancement
        
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
        max_history_tokens=max_history_tokens,
        enable_intent_classification=enable_intent_classification,
        enable_request_analysis=enable_request_analysis,
        cursor_mode=cursor_mode
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
    max_history_tokens: Optional[int] = 6000,
    # NEW: Cursor-style parameters
    enable_intent_classification: bool = True,
    enable_request_analysis: bool = True,
    cursor_mode: bool = False
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
        enable_intent_classification: Enable intent analysis
        enable_request_analysis: Enable request analysis
        cursor_mode: Enable Cursor-style progressive enhancement
        
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
        max_history_tokens=max_history_tokens,
        enable_intent_classification=enable_intent_classification,
        enable_request_analysis=enable_request_analysis,
        cursor_mode=cursor_mode
    )
    async for chunk in executive_tool.execute_tool_stream(request):
        yield chunk 