"""
Executive Tool Module - API-ready wrapper for LLM tool execution
Provides dynamic parameter support and customizable system prompts for API endpoints
"""

import os
import json
import re
import traceback
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

from search_tool import SearchTool
from learning_tools import LearningToolsFactory

# Import provider-specific executors
from exec_anthropic import AnthropicExecutor
from exec_openai import OpenAIExecutor

logger = logging.getLogger(__name__)

# Configure detailed logging for tool calls
tool_logger = logging.getLogger("tool_calls")
tool_logger.setLevel(logging.INFO)
if not tool_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('🔧 [TOOL] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    tool_logger.addHandler(handler)

# Import language detection from lightweight language module
try:
    from language import detect_language_with_llm, LanguageDetector
    LANGUAGE_DETECTION_AVAILABLE = True
    logger.info("Language detection imported successfully from language.py")
except Exception as e:
    logger.warning(f"Failed to import language detection from language.py: {e}")
    LANGUAGE_DETECTION_AVAILABLE = False


@dataclass
class InvestigationStep:
    """Single step in the deep reasoning investigation chain"""
    key: str
    search_query: str
    thought_description: str
    discovery_template: str
    requires_search: bool = True
    requires_analysis: bool = True
    analysis_type: str = "capability_analysis"
    context_vars: Optional[Dict[str, Any]] = None
    order: int = 1


@dataclass
class InvestigationPlan:
    """Complete plan for contextual investigation"""
    initial_thought: str
    steps: List[InvestigationStep]
    total_steps: int
    reasoning_focus: str
    
    @classmethod
    def from_llm_response(cls, llm_response: str) -> 'InvestigationPlan':
        """Parse LLM JSON response into structured plan"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                steps = []
                for i, step_data in enumerate(data.get("steps", []), 1):
                    step = InvestigationStep(
                        key=step_data.get("key", f"step_{i}"),
                        search_query=step_data.get("search_query", ""),
                        thought_description=step_data.get("thought_description", ""),
                        discovery_template=step_data.get("discovery_template", ""),
                        requires_search=step_data.get("requires_search", True),
                        requires_analysis=step_data.get("requires_analysis", True),
                        analysis_type=step_data.get("analysis_type", "capability_analysis"),
                        context_vars=step_data.get("context_vars", {}),
                        order=i
                    )
                    steps.append(step)
                
                return cls(
                    initial_thought=data.get("initial_thought", "Starting investigation..."),
                    steps=steps,
                    total_steps=len(steps),
                    reasoning_focus=data.get("reasoning_focus", "general")
                )
            else:
                # Fallback plan if parsing fails
                return cls._create_fallback_plan()
                
        except Exception as e:
            logger.error(f"Failed to parse investigation plan: {e}")
            return cls._create_fallback_plan()
    
    @classmethod
    def _create_fallback_plan(cls) -> 'InvestigationPlan':
        """Create a basic fallback investigation plan"""
        fallback_step = InvestigationStep(
            key="brain_knowledge",
            search_query="agent capabilities and knowledge",
            thought_description="Let me check your agent's knowledge base to understand its capabilities...",
            discovery_template="Found {count} knowledge pieces about your agent - analyzing capabilities...",
            analysis_type="capability_analysis",
            order=1
        )
        
        return cls(
            initial_thought="I need to understand your agent's capabilities first...",
            steps=[fallback_step],
            total_steps=1,
            reasoning_focus="capability_analysis"
        )


@dataclass
class ContextualStrategy:
    """Strategy synthesized from investigation findings"""
    reasoning_summary: str
    action_plan: List[str]
    context_specific_recommendations: List[str]
    tailored_guidance: str
    
    @classmethod
    def from_llm_response(cls, llm_response: str) -> 'ContextualStrategy':
        """Parse LLM strategy response"""
        try:
            import json
            import re
            
            # Try to extract JSON structure
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return cls(
                    reasoning_summary=data.get("reasoning_summary", ""),
                    action_plan=data.get("action_plan", []),
                    context_specific_recommendations=data.get("context_specific_recommendations", []),
                    tailored_guidance=data.get("tailored_guidance", "")
                )
            else:
                # Parse from text format
                return cls._parse_text_strategy(llm_response)
                
        except Exception as e:
            logger.error(f"Failed to parse contextual strategy: {e}")
            return cls._create_fallback_strategy(llm_response)
    
    @classmethod
    def _parse_text_strategy(cls, response: str) -> 'ContextualStrategy':
        """Parse strategy from text format"""
        lines = response.split('\n')
        return cls(
            reasoning_summary=response[:200] + "..." if len(response) > 200 else response,
            action_plan=["Follow the guidance provided in the response"],
            context_specific_recommendations=["Review the specific recommendations in the detailed response"],
            tailored_guidance=response
        )
    
    @classmethod
    def _create_fallback_strategy(cls, response: str) -> 'ContextualStrategy':
        """Create fallback strategy"""
        return cls(
            reasoning_summary="Based on the analysis, here's the recommended approach",
            action_plan=["Review the provided guidance", "Implement step by step"],
            context_specific_recommendations=["Follow best practices for your specific setup"],
            tailored_guidance=response or "Please refer to the detailed guidance provided"
        )


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
    # NEW: Deep reasoning parameters
    enable_deep_reasoning: Optional[bool] = False  # Enable multi-step reasoning
    reasoning_depth: Optional[str] = "standard"    # "light", "standard", "deep"
    brain_reading_enabled: Optional[bool] = True   # Read user's brain vectors
    max_investigation_steps: Optional[int] = 5     # Limit reasoning steps
    # Internal state (set during execution)
    contextual_strategy: Optional[ContextualStrategy] = None


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
        
        # Initialize provider-specific executors
        self.anthropic_executor = AnthropicExecutor(self)
        self.openai_executor = OpenAIExecutor(self)
        
        # Keep minimal fallback prompts (main prompts are now in provider executors)
        self.default_system_prompts = {
            "anthropic": "You are Ami, a helpful AI assistant.",
            "openai": "You are Ami, a helpful AI assistant."
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
        
        # Initialize brain vector tool for accessing brain knowledge during conversations
        try:
            from brain_vector_tool import BrainVectorTool
            tools["brain_vector"] = BrainVectorTool()
            logger.info("Brain vector tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize brain vector tool: {e}")
        
        # Initialize learning tools factory (user-specific tools created on demand)
        try:
            from learning_tools import LearningToolsFactory
            tools["learning_factory"] = LearningToolsFactory
            logger.info("Learning tools factory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize learning tools factory: {e}")
        
        # Initialize human context tool (Mom Test-inspired discovery)
        try:
            from human_context_tool import create_human_context_tool
            tools["human_context"] = create_human_context_tool()
            logger.info("Human context tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize human context tool: {e}")
        
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
        Analyze user request intent and return analysis data and raw thinking steps
        (No longer streams thoughts directly - unified method handles that)
        
        Args:
            request: The tool execution request containing LLM provider and model
            
        Returns:
            Tuple of (RequestAnalysis, list of raw thinking steps from LLM)
        """
        
        # Create analysis prompt that generates structured thinking data
        analysis_prompt = f"""
        You are analyzing a user request to understand their intent. Think through this step-by-step and provide structured analysis.

        User Message: "{request.user_query}"
        
        Conversation History: {request.conversation_history[-3:] if request.conversation_history else "None"}

        Please provide your analysis in this EXACT format:

        UNDERSTANDING:
        [Provide a clear understanding of what the user is asking for and context]

        DETAILED_THINKING:
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
            "reasoning": "Detailed explanation of the classification"
        }}

        Intent Classifications:
        - "learning": User is teaching, sharing knowledge, or providing information
        - "problem_solving": User needs help solving a specific problem or issue  
        - "general_chat": General conversation, questions, or casual interaction
        - "task_execution": User wants to perform a specific task or get something done
        """
        
        try:
            # Use the same provider and model as the main request for intent analysis
            if request.llm_provider.lower() == "anthropic":
                analysis_response = await self.anthropic_executor._analyze_with_anthropic(analysis_prompt, request.model)
            else:
                analysis_response = await self.openai_executor._analyze_with_openai(analysis_prompt, request.model)
            
            # Parse the response to extract components
            thinking_steps = []
            analysis_data = {}
            understanding_text = ""
            
            import re
            # Extract understanding section
            understanding_match = re.search(r'UNDERSTANDING:(.*?)DETAILED_THINKING:', analysis_response, re.DOTALL)
            if understanding_match:
                understanding_text = understanding_match.group(1).strip()
                # Clean up any bracketed instructions
                understanding_text = re.sub(r'\[.*?\]', '', understanding_text).strip()
            
            # Extract detailed thinking section
            thinking_match = re.search(r'DETAILED_THINKING:(.*?)ANALYSIS:', analysis_response, re.DOTALL)
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
                    "understanding": understanding_text,
                    "detailed_thinking_steps": thinking_steps
                }
            )
            
            return request_analysis, thinking_steps
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # Return default analysis and thinking steps
            default_thinking = [
                "Looking at the request to understand what the user wants to achieve...",
                "Since detailed analysis failed, I'll use default classification to provide helpful assistance...",
                "This appears to be a general conversation request based on available information...",
                "I'll provide assistance with available tools while working with default analysis..."
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
                    "detailed_thinking_steps": default_thinking
                }
            ), default_thinking

    async def _generate_unified_cursor_thoughts(self, request: ToolExecutionRequest, analysis: RequestAnalysis, orchestration_plan: Dict[str, Any], thinking_steps: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate unified Cursor-style thoughts in logical order with UI/UX enhancements
        
        Args:
            request: The tool execution request
            analysis: The request analysis results
            orchestration_plan: The tool orchestration plan
            thinking_steps: Raw thinking steps from LLM analysis
            
        Yields:
            Dict containing thought messages in natural logical order
        """
        
        # 1. FIRST: Initial understanding (always first)
        understanding = analysis.metadata.get("understanding", "")
        if understanding:
            yield {
                "type": "thinking",
                "content": f"💭 {understanding}",
                "provider": request.llm_provider,
                "thought_type": "understanding",
                "step": 1,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Fallback understanding
            yield {
                "type": "thinking",
                "content": f"💭 I need to understand this request: \"{request.user_query[:80]}{'...' if len(request.user_query) > 80 else ''}\" - Let me analyze what they want to achieve and how I can help them.",
                "provider": request.llm_provider,
                "thought_type": "understanding",
                "step": 1,
                "timestamp": datetime.now().isoformat()
            }
        
        await asyncio.sleep(0.4)  # Natural reading pace
        
        # 2. SECOND: Intent analysis result (UI/UX enhancement)
        yield {
            "type": "thinking",
            "content": f"🔍 Analyzing the intent... This looks like a **{analysis.intent}** request with {analysis.complexity} complexity. I'm {analysis.confidence:.0%} confident about this classification.",
            "provider": request.llm_provider,
            "thought_type": "intent_analysis",
            "step": 2,
            "timestamp": datetime.now().isoformat()
        }
        
        await asyncio.sleep(0.3)
        
        # 3. THIRD: Detailed thinking steps from LLM (the core reasoning)
        for i, step in enumerate(thinking_steps, start=3):
            yield {
                "type": "thinking",
                "content": f"🧠 {step}",
                "provider": request.llm_provider,
                "thought_type": "detailed_analysis",
                "step": i,
                "timestamp": datetime.now().isoformat()
            }
            await asyncio.sleep(0.3)  # Brief pause between detailed thoughts
        
        # 4. FOURTH: Tool selection (UI/UX enhancement)
        current_step = len(thinking_steps) + 3
        primary_tools = orchestration_plan.get("primary_tools", [])
        secondary_tools = orchestration_plan.get("secondary_tools", [])
        
        if primary_tools:
            tools_text = ", ".join([f"**{tool}**" for tool in primary_tools])
            yield {
                "type": "thinking",
                "content": f"🛠️ I'll use these tools to help: {tools_text}. This should give me the information I need to provide a comprehensive response.",
                "provider": request.llm_provider,
                "thought_type": "tool_selection",
                "step": current_step,
                "timestamp": datetime.now().isoformat()
            }
            await asyncio.sleep(0.4)
            current_step += 1
        
        # 5. FIFTH: Strategy explanation (UI/UX enhancement)
        strategy_content = ""
        if analysis.intent == "learning":
            strategy_content = "📚 Since this is a learning request, I'll first check existing knowledge to avoid duplicates, then analyze if this should be learned."
        elif analysis.intent == "problem_solving":
            strategy_content = "🔧 For this problem-solving request, I'll search for current information and check internal context to provide the most relevant solution."
        elif analysis.intent == "task_execution":
            strategy_content = "⚡ This is a task execution request. I'll gather context first, then search for any additional information needed to complete the task effectively."
        else:
            strategy_content = "💬 This seems like a general conversation. I'll search for relevant information if needed and provide a helpful response."
        
        yield {
            "type": "thinking",
            "content": strategy_content,
            "provider": request.llm_provider,
            "thought_type": "strategy",
            "step": current_step,
            "timestamp": datetime.now().isoformat()
        }
        
        await asyncio.sleep(0.4)
        current_step += 1
        
        # 6. FINALLY: Execution readiness (UI/UX enhancement)
        yield {
            "type": "thinking",
            "content": "🚀 Now I'll execute my plan and provide you with a comprehensive response...",
            "provider": request.llm_provider,
            "thought_type": "execution_ready",
            "step": current_step,
            "timestamp": datetime.now().isoformat()
        }

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
                language_response = await self.anthropic_executor._analyze_with_anthropic(language_detection_prompt, request.model)
            else:
                language_response = await self.openai_executor._analyze_with_openai(language_detection_prompt, request.model)
            
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
                "type": "thinking",
                "content": "✅ Great! I have the information I need. Now I'll synthesize everything into a comprehensive response...",
                "provider": provider,
                "thought_type": "response_generation",
                "timestamp": datetime.now().isoformat()
            }
        else:
            yield {
                "type": "thinking",
                "content": "💡 I'll use my existing knowledge to provide a helpful response...",
                "provider": provider,
                "thought_type": "response_generation",
                "timestamp": datetime.now().isoformat()
            }
        
        await asyncio.sleep(0.3)
        
        yield {
            "type": "thinking",
            "content": "✍️ Generating response now...",
            "provider": provider,
            "thought_type": "response_generation",
            "timestamp": datetime.now().isoformat()
        }

    async def _read_agent_brain_vectors(self, request: ToolExecutionRequest, search_context: str) -> List[Dict]:
        """Read user's brain vectors like Cursor reads source files"""
        
        brain_vectors = []
        try:
            # Query brain vectors with user-specific context
            if "brain_vector" in self.available_tools:
                brain_vectors = await asyncio.to_thread(
                    self.available_tools["brain_vector"].query_knowledge,
                    user_id=request.user_id,
                    org_id=request.org_id,
                    query=search_context,
                    limit=50
                )
            
        except Exception as e:
            logger.error(f"Failed to read brain vectors: {e}")
            brain_vectors = []
            
        return brain_vectors

    async def _stream_brain_reading_thoughts(self, request: ToolExecutionRequest, search_context: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream thoughts during brain vector reading"""
        
        yield {
            "type": "thinking",
            "content": f"🧠 Reading your agent's brain vectors to understand its knowledge...",
            "thought_type": "brain_reading_start",
            "reasoning_step": "brain_access",
            "timestamp": datetime.now().isoformat()
        }
        
        brain_vectors = await self._read_agent_brain_vectors(request, search_context)
        
        if brain_vectors:
            yield {
                "type": "thinking", 
                "content": f"📚 Found {len(brain_vectors)} knowledge vectors - analyzing your agent's capabilities...",
                "thought_type": "brain_vectors_loaded",
                "reasoning_step": "brain_analysis",
                "metadata": {
                    "vector_count": len(brain_vectors),
                    "search_context": search_context
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            yield {
                "type": "thinking",
                "content": f"⚠️ Could not access brain vectors - proceeding with available knowledge...",
                "thought_type": "brain_reading_error",
                "reasoning_step": "brain_fallback",
                "timestamp": datetime.now().isoformat()
            }

    async def _analyze_brain_vector_contents(self, brain_vectors: List[Dict], request: ToolExecutionRequest, analysis_focus: str) -> Dict[str, Any]:
        """Analyze brain contents like Cursor analyzes code structure"""
        
        if not brain_vectors:
            return {"summary": "No specific agent knowledge found", "core_capabilities": [], "knowledge_domains": []}
        
        # Create analysis prompt
        analysis_prompt = f"""
        I'm analyzing an agent's brain vectors to understand its capabilities, similar to how Cursor analyzes source code.
        
        Analysis Focus: {analysis_focus}
        User Query: "{request.user_query}"
        
        Brain Vector Contents (first 200 chars each):
        {json.dumps([str(v).get('content', str(v))[:200] + '...' if len(str(v)) > 200 else str(v) for v in brain_vectors[:10]], indent=2)}
        
        Total vectors: {len(brain_vectors)}
        
        Provide analysis in this JSON format:
        {{
            "core_capabilities": ["capability1", "capability2"],
            "knowledge_domains": ["domain1", "domain2"], 
            "specific_processes": ["process1", "process2"],
            "integration_points": ["integration1", "integration2"],
            "strengths": ["strength1", "strength2"],
            "potential_gaps": ["gap1", "gap2"],
            "summary": "Brief summary of agent's specialization"
        }}
        
        Analyze like you're reading source code to understand system architecture.
        """
        
        try:
            analysis_response = await self._get_reasoning_llm_response(analysis_prompt, request)
            
            # Parse analysis response
            import re
            json_match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group(0))
            else:
                analysis_data = {"summary": "Agent capabilities analyzed", "core_capabilities": ["general assistance"]}
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Brain analysis failed: {e}")
            return {"summary": "Agent analysis completed with general approach", "core_capabilities": ["general assistance"]}

    async def _stream_brain_analysis_thoughts(self, brain_vectors: List[Dict], request: ToolExecutionRequest, analysis_focus: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream thoughts during brain vector analysis"""
        
        if not brain_vectors:
            yield {
                "type": "thinking",
                "content": "💡 No brain vectors found - will provide general guidance based on best practices...",
                "thought_type": "brain_analysis_empty",
                "reasoning_step": "capability_understanding",
                "timestamp": datetime.now().isoformat()
            }
            return
        
        # Perform analysis
        analysis_data = await self._analyze_brain_vector_contents(brain_vectors, request, analysis_focus)
        
        # Stream the results
        yield {
            "type": "thinking",
            "content": f"💡 Your agent specializes in: {analysis_data.get('summary', 'multiple areas')}",
            "thought_type": "brain_analysis_complete",
            "reasoning_step": "capability_understanding",
            "timestamp": datetime.now().isoformat()
        }

    async def _get_reasoning_llm_response(self, prompt: str, request: ToolExecutionRequest) -> str:
        """Get LLM response for reasoning analysis"""
        try:
            if request.llm_provider.lower() == "anthropic":
                return await self.anthropic_executor._analyze_with_anthropic(prompt, request.model)
            else:
                return await self.openai_executor._analyze_with_openai(prompt, request.model)
        except Exception as e:
            logger.error(f"Reasoning LLM response failed: {e}")
            return "Analysis completed with available information."

    async def _extract_structured_knowledge(self, response_content: str, user_query: str, llm_provider: str) -> List[Dict[str, Any]]:
        """
        Extract structured, actionable knowledge pieces from LLM response
        
        Args:
            response_content: The LLM's response content
            user_query: Original user query for context
            llm_provider: LLM provider for extraction analysis
            
        Returns:
            List of structured knowledge pieces with titles, content, categories, and quality scores
        """
        
        try:
            # Use GPT-4o for knowledge extraction analysis
            extraction_prompt = f"""
Analyze this AI response and extract discrete, actionable knowledge pieces that would be valuable to save for future reference.

Original User Query: "{user_query}"

AI Response to Analyze:
"{response_content}"

EXTRACTION CRITERIA:
Focus ONLY on actionable, reusable knowledge pieces such as:
• Specific instructions or procedures
• Tool configurations or settings  
• Technical requirements or specifications
• Process workflows or steps
• API endpoints or integration details
• Command sequences or code snippets
• Business rules or policies
• Contact information or resources

EXCLUDE:
• Conversational elements ("Let's build", "Great choice!")
• Generic encouragement or motivational text
• Questions back to the user
• Vague or non-actionable content

For each valuable knowledge piece found, extract:

Response format (JSON array):
[
  {{
    "title": "Concise, descriptive title (max 60 chars)",
    "content": "The actual actionable knowledge/instruction",
    "category": "instruction|configuration|requirement|process|integration|resource|policy",
    "quality_score": 0.0-1.0,
    "actionability": "high|medium|low",
    "reusability": "high|medium|low",
    "specificity": "high|medium|low"
  }}
]

Only include pieces with quality_score >= 0.6 and high actionability.
If no valuable knowledge pieces are found, return an empty array [].

EXAMPLE GOOD EXTRACTIONS:
- "API Integration Setup": "To connect Zalo API, use endpoint https://zalo.me/api/v2/messages with authentication token"
- "File Processing Workflow": "Read Excel files from /Daily Report folder, check 'mô tả công việc' column for empty cells"
- "Error Handling Protocol": "When file not found, retry 3 times with 5-second intervals, then send alert to admin"

Respond with ONLY the JSON array, no additional text.
"""
            
            # Use the same LLM provider for consistency
            if llm_provider.lower() == "anthropic":
                extraction_result = await self.anthropic_executor._analyze_with_anthropic(extraction_prompt, None)
            else:
                extraction_result = await self.openai_executor._analyze_with_openai(extraction_prompt, None)
            
            # Parse the JSON response
            import json
            import re
            
            # Clean up the response to extract JSON
            if "```json" in extraction_result:
                extraction_result = extraction_result.split("```json")[1].split("```")[0].strip()
            elif "```" in extraction_result:
                extraction_result = extraction_result.split("```")[1].strip()
            
            try:
                knowledge_pieces = json.loads(extraction_result)
                
                # Validate and filter the results
                validated_pieces = []
                for piece in knowledge_pieces:
                    if (isinstance(piece, dict) and 
                        all(key in piece for key in ["title", "content", "category", "quality_score"]) and
                        piece["quality_score"] >= 0.6 and
                        len(piece["content"].strip()) > 20):  # Minimum content length
                        
                        validated_pieces.append({
                            "title": piece["title"][:60],  # Truncate title
                            "content": piece["content"].strip(),
                            "category": piece["category"],
                            "quality_score": float(piece["quality_score"]),
                            "actionability": piece.get("actionability", "medium"),
                            "reusability": piece.get("reusability", "medium"),
                            "specificity": piece.get("specificity", "medium")
                        })
                
                logger.info(f"Knowledge extraction: Found {len(knowledge_pieces)} pieces, {len(validated_pieces)} passed validation")
                return validated_pieces
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse knowledge extraction JSON: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Knowledge extraction error: {e}")
            return []

    async def _extract_user_input_knowledge(self, user_query: str, request: 'ToolExecutionRequest') -> List[Dict[str, Any]]:
        """
        Extract structured, actionable knowledge pieces using AI Agent Implementation Reasoning
        
        Flow:
        1. User drops request/shares knowledge
        2. LLM reasons "How to perform it as an AI Agent"
        3. Break into smaller implementation parts
        4. Generate knowledge instructions for each part
        5. For technical parts: Focus on info collection for Ami, not theory
        
        Args:
            user_query: The user's input to analyze
            request: ToolExecutionRequest with LLM provider and model info
            
        Returns:
            List of structured knowledge pieces with titles, content, categories, and quality scores
        """
        
        try:
            logger.info(f"Starting AI Agent implementation reasoning for query: {user_query[:100]}...")
            
            # STEP 1: AI Agent Implementation Reasoning
            implementation_reasoning_prompt = f"""
You are Ami, an AI agent builder. A human has shared a request or knowledge with you. Your job is to reason through "How would I implement this as an AI Agent?" and break it down into actionable parts.

Human Input:
"{user_query}"

REASONING PROCESS:
1. **UNDERSTAND THE REQUEST**: What does the human want their AI agent to do?
2. **AGENT IMPLEMENTATION ANALYSIS**: How would an AI agent actually perform this task?
3. **BREAK DOWN INTO PARTS**: What are the smaller implementation components?
4. **IDENTIFY TECHNICAL VS BUSINESS LOGIC**: Separate technical integrations from business processes
5. **AMI'S ROLE CLARITY**: For technical parts, what info does Ami need to collect to help the agent?

RESPONSE FORMAT (JSON):
{{
  "agent_goal": "What the AI agent needs to accomplish",
  "implementation_breakdown": [
    {{
      "part_name": "Clear name for this implementation part",
      "part_type": "business_logic|technical_integration|data_processing|communication|workflow",
      "agent_actions": "What the AI agent will actually do for this part",
      "ami_collection_needed": "What info Ami needs to collect from human (for technical parts)",
      "priority": "high|medium|low"
    }}
  ],
  "technical_dependencies": ["List of technical systems/APIs the agent will need"],
  "business_logic_requirements": ["List of business rules the agent must follow"]
}}

EXAMPLES:

Input: "I need agent to read financial reports and send alerts"
Output:
{{
  "agent_goal": "Automatically monitor financial reports and send alerts based on findings",
  "implementation_breakdown": [
    {{
      "part_name": "Financial Report Reading",
      "part_type": "data_processing", 
      "agent_actions": "Access designated folder, read Excel/PDF files, extract key financial metrics",
      "ami_collection_needed": "Folder path, file formats, which metrics to monitor, alert thresholds",
      "priority": "high"
    }},
    {{
      "part_name": "Alert System Integration",
      "part_type": "technical_integration",
      "agent_actions": "Send notifications via chosen platform when conditions are met",
      "ami_collection_needed": "Notification platform (Zalo/Slack), API credentials, message templates",
      "priority": "high"
    }}
  ],
  "technical_dependencies": ["File system access", "Zalo/Slack API"],
  "business_logic_requirements": ["Alert threshold rules", "Report analysis criteria"]
}}

RESPOND WITH ONLY THE JSON OBJECT, NO OTHER TEXT.
"""

            # Get implementation reasoning
            logger.info("Getting AI Agent implementation reasoning...")
            reasoning_response = await self._get_reasoning_llm_response(implementation_reasoning_prompt, request)
            logger.info(f"Received implementation reasoning: {len(reasoning_response)} chars")
            
            # Parse implementation reasoning
            try:
                json_match = re.search(r'\{.*\}', reasoning_response, re.DOTALL)
                if not json_match:
                    logger.warning("No JSON found in implementation reasoning response")
                    return []
                
                implementation_data = json.loads(json_match.group(0))
                logger.info(f"Implementation breakdown: {len(implementation_data.get('implementation_breakdown', []))} parts")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in implementation reasoning: {e}")
                return []
            
            # STEP 2: Retrieve Existing Relevant Knowledge
            logger.info("Retrieving existing relevant knowledge...")
            try:
                # Search for existing knowledge related to the implementation parts
                search_queries = []
                
                # Add the agent goal as a search query
                if implementation_data.get("agent_goal"):
                    search_queries.append(implementation_data["agent_goal"])
                
                # Add each implementation part as search queries
                for part in implementation_data.get("implementation_breakdown", []):
                    search_queries.extend([
                        part.get("part_name", ""),
                        part.get("agent_actions", ""),
                        part.get("ami_collection_needed", "")
                    ])
                
                # Add technical dependencies and business logic requirements
                search_queries.extend(implementation_data.get("technical_dependencies", []))
                search_queries.extend(implementation_data.get("business_logic_requirements", []))
                
                # Filter out empty queries
                search_queries = [q.strip() for q in search_queries if q and q.strip()]
                logger.info(f"Generated {len(search_queries)} search queries for existing knowledge")
                
                # Search for existing knowledge using the brain vector tool
                existing_knowledge_context = ""
                if hasattr(self, 'brain_vector_tool') and self.brain_vector_tool:
                    try:
                        # Use the first few most relevant queries to avoid overwhelming the search
                        top_queries = search_queries[:3]
                        for query in top_queries:
                            knowledge_results = await self.brain_vector_tool.search_knowledge(
                                query=query,
                                user_id=getattr(request, 'user_id', 'unknown'),
                                org_id=getattr(request, 'org_id', 'unknown'),
                                limit=3
                            )
                            
                            if knowledge_results:
                                for result in knowledge_results:
                                    existing_knowledge_context += f"\n--- Existing Knowledge ---\n"
                                    existing_knowledge_context += f"Title: {result.get('title', 'Unknown')}\n"
                                    existing_knowledge_context += f"Content: {result.get('content', result.get('raw', ''))}\n"
                                    existing_knowledge_context += f"Relevance Score: {result.get('score', 0.0):.3f}\n"
                        
                        logger.info(f"Retrieved existing knowledge context: {len(existing_knowledge_context)} chars")
                        
                    except Exception as e:
                        logger.warning(f"Knowledge retrieval failed, continuing without existing knowledge: {e}")
                        existing_knowledge_context = ""
                else:
                    logger.info("Brain vector tool not available, proceeding without existing knowledge")
                    
            except Exception as e:
                logger.error(f"Error during knowledge retrieval: {e}")
                existing_knowledge_context = ""
            
            # STEP 3: Generate Knowledge Instructions for Each Part
            knowledge_generation_prompt = f"""
Based on the AI Agent implementation breakdown, generate specific knowledge instructions for each part.

IMPLEMENTATION DATA:
{json.dumps(implementation_data, indent=2)}

ORIGINAL USER INPUT:
"{user_query}"

EXISTING RELEVANT KNOWLEDGE:
{existing_knowledge_context if existing_knowledge_context else "No existing relevant knowledge found."}

KNOWLEDGE GENERATION RULES:
1. **BUSINESS LOGIC PARTS**: Create direct instructions for the agent's decision-making
2. **TECHNICAL PARTS**: Create info collection instructions for Ami (not theory/academic)  
3. **DATA PROCESSING PARTS**: Create specific data handling instructions
4. **COMMUNICATION PARTS**: Create message templates and trigger conditions
5. **WORKFLOW PARTS**: Create step-by-step process instructions

EXISTING KNOWLEDGE INTEGRATION:
- **BUILD UPON EXISTING**: If existing knowledge covers similar topics, enhance/extend rather than duplicate
- **REFERENCE EXISTING**: When relevant existing knowledge exists, reference it in new knowledge pieces
- **FILL GAPS**: Focus on generating knowledge for areas NOT covered by existing knowledge
- **AVOID DUPLICATION**: Don't create knowledge pieces that essentially repeat existing knowledge

TECHNICAL PARTS FOCUS:
- DON'T generate theoretical knowledge like "Google Drive API documentation"
- DO generate: "Ami needs to collect: Google Drive folder path, authentication method, file types to monitor"
- The knowledge should help Ami know WHAT INFO TO COLLECT, not how to code

RESPONSE FORMAT (JSON array):
[
  {{
    "title": "Concise, descriptive title (max 60 chars)",
    "content": "Specific instruction for the agent OR info collection task for Ami",
    "category": "agent_instruction|ami_collection|business_rule|workflow_step|message_template",
    "implementation_part": "Which part from breakdown this relates to",
    "builds_upon_existing": "Reference to existing knowledge this builds upon (if any)",
    "quality_score": 0.0-1.0,
    "actionability": "high|medium|low",
    "reusability": "high|medium|low",
    "specificity": "high|medium|low"
  }}
]

EXAMPLES:

For "Financial Report Reading" part with existing knowledge about file processing:
{{
  "title": "Daily Report Folder Access",
  "content": "Building on existing file processing knowledge: Ami needs to collect specific folder path for financial reports, file naming pattern, and which columns to check for empty values",
  "category": "ami_collection",
  "implementation_part": "Financial Report Reading",
  "builds_upon_existing": "File processing workflow knowledge",
  "quality_score": 0.9,
  "actionability": "high",
  "reusability": "high", 
  "specificity": "high"
}}

For "Alert System" part with no existing knowledge:
{{
  "title": "Zalo Alert Message Template",
  "content": "When empty cells found, send: '⚠️ [File Name] có ô trống ở cột [Column] - cần kiểm tra ngay'",
  "category": "message_template",
  "implementation_part": "Alert System Integration",
  "builds_upon_existing": null,
  "quality_score": 0.85,
  "actionability": "high",
  "reusability": "medium",
  "specificity": "high"
}}

RESPOND WITH ONLY THE JSON ARRAY, NO OTHER TEXT.
"""

            # Generate knowledge instructions
            logger.info("Generating knowledge instructions from implementation breakdown with existing knowledge context...")
            knowledge_response = await self._get_reasoning_llm_response(knowledge_generation_prompt, request)
            logger.info(f"Received knowledge instructions: {len(knowledge_response)} chars")
            
            # Parse and validate knowledge instructions
            try:
                json_match = re.search(r'\[.*\]', knowledge_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    knowledge_data = json.loads(json_str)
                else:
                    logger.warning("No JSON array found in knowledge generation response")
                    return []
                
                # Validate and structure the knowledge pieces
                validated_pieces = []
                for piece in knowledge_data:
                    if isinstance(piece, dict) and "title" in piece and "content" in piece:
                        validated_pieces.append({
                            "title": piece.get("title", "")[:60],  # Limit title length
                            "content": piece.get("content", ""),
                            "category": piece.get("category", "agent_instruction"),
                            "implementation_part": piece.get("implementation_part", "unknown"),
                            "builds_upon_existing": piece.get("builds_upon_existing"),
                            "quality_score": float(piece.get("quality_score", 0.7)),
                            "actionability": piece.get("actionability", "medium"),
                            "reusability": piece.get("reusability", "medium"),
                            "specificity": piece.get("specificity", "medium")
                        })
                
                logger.info(f"AI Agent knowledge extraction: Found {len(knowledge_data)} pieces, {len(validated_pieces)} passed validation")
                return validated_pieces
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in knowledge generation: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error in AI Agent implementation knowledge extraction: {e}")
            return []

    async def _stream_knowledge_approval_request(self, knowledge_pieces: List[Dict[str, Any]], request: 'ToolExecutionRequest') -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream knowledge approval request to frontend for human-in-the-loop flow
        
        Args:
            knowledge_pieces: List of extracted knowledge pieces to approve
            request: ToolExecutionRequest for context
            
        Yields:
            Dict chunks for streaming the approval request
        """
        
        try:
            # Create the approval request data structure
            approval_data = {
                "type": "knowledge_approval_request",
                "content": f"📚 I've extracted {len(knowledge_pieces)} knowledge pieces from your input. Please review and approve which ones should be learned:",
                "knowledge_pieces": knowledge_pieces,
                "requires_human_input": True,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "extraction_method": "user_input",
                    "total_pieces": len(knowledge_pieces),
                    "user_id": getattr(request, 'user_id', 'unknown'),
                    "org_id": getattr(request, 'org_id', 'unknown')
                }
            }
            
            # Stream the approval request
            yield approval_data
            
            # Also yield individual knowledge pieces for detailed review
            for i, piece in enumerate(knowledge_pieces):
                yield {
                    "type": "knowledge_piece",
                    "content": f"**{piece['title']}**\n{piece['content']}",
                    "piece_index": i,
                    "piece_data": piece,
                    "category": piece.get('category', 'unknown'),
                    "quality_score": piece.get('quality_score', 0.0),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error streaming knowledge approval request: {e}")
            yield {
                "type": "error",
                "content": f"Error preparing knowledge approval: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def handle_knowledge_approval(self, approved_knowledge_ids: List[str], all_knowledge_pieces: List[Dict[str, Any]], request: 'ToolExecutionRequest') -> AsyncGenerator[str, None]:
        """
        Handle human approval of knowledge pieces and generate copilot-style summary
        
        Args:
            approved_knowledge_ids: List of approved knowledge piece IDs
            all_knowledge_pieces: All knowledge pieces that were presented for approval
            request: Original ToolExecutionRequest for context
            
        Yields:
            String chunks (JSON) for streaming the copilot summary response
        """
        
        try:
            # Filter approved knowledge pieces - handle multiple possible ID formats
            approved_pieces = []
            for i, piece in enumerate(all_knowledge_pieces):
                # Check various possible ID formats
                piece_matched = (
                    str(i) in approved_knowledge_ids or  # Index as string
                    i in approved_knowledge_ids or      # Index as number
                    piece.get('id') in approved_knowledge_ids or  # Piece ID
                    piece.get('title') in approved_knowledge_ids  # Title match
                )
                if piece_matched:
                    approved_pieces.append(piece)
            
            # If no pieces matched but we have approval IDs, assume all are approved
            if not approved_pieces and approved_knowledge_ids and all_knowledge_pieces:
                logger.warning(f"No pieces matched approval IDs {approved_knowledge_ids}, assuming all pieces approved")
                approved_pieces = all_knowledge_pieces
            
            logger.info(f"Processing {len(approved_pieces)} approved knowledge pieces out of {len(all_knowledge_pieces)} total")
            
            # Save approved knowledge pieces (if any)
            if approved_pieces:
                yield f"data: {json.dumps({
                    'type': 'thinking',
                    'content': f'💾 Saving {len(approved_pieces)} approved knowledge pieces to your agent\'s brain...',
                    'thought_type': 'knowledge_saving',
                    'timestamp': datetime.now().isoformat()
                })}\n\n"
                
                # Actually save knowledge pieces to brain vectors using pccontroller
                from pccontroller import save_knowledge
                
                saved_count = 0
                for piece in approved_pieces:
                    try:
                        result = await save_knowledge(
                            input=piece['content'],
                            user_id=request.user_id,
                            org_id=request.org_id,
                            title=piece['title'],
                            thread_id=None,  # No specific thread for knowledge extraction
                            topic=piece.get('category', 'user_teaching'),
                            categories=['human_approved', 'teaching_intent', piece.get('category', 'process')],
                            ttl_days=365  # Keep for 1 year
                        )
                        
                        if result and result.get('success'):
                            saved_count += 1
                            logger.info(f"✅ SAVED KNOWLEDGE TO BRAIN: {piece['title']} -> {result.get('vector_id')}")
                        else:
                            logger.error(f"❌ FAILED TO SAVE: {piece['title']} -> {result}")
                            
                    except Exception as e:
                        logger.error(f"❌ ERROR SAVING KNOWLEDGE: {piece['title']} -> {str(e)}")
                
                logger.info(f"Successfully saved {saved_count}/{len(approved_pieces)} knowledge pieces to brain vectors")
                
                yield f"data: {json.dumps({
                    'type': 'thinking',
                    'content': f'✅ Successfully saved {saved_count}/{len(approved_pieces)} knowledge pieces to your agent\'s brain',
                    'thought_type': 'knowledge_saved',
                    'timestamp': datetime.now().isoformat()
                })}\n\n"
            
            # Generate copilot-style summary
            yield f"data: {json.dumps({
                'type': 'thinking',
                'content': '🤖 Generating copilot-style summary based on your request and approved knowledge...',
                'thought_type': 'summary_generation',
                'timestamp': datetime.now().isoformat()
            })}\n\n"
            
            # Create summary prompt
            approved_knowledge_text = "\n".join([
                f"- {piece['title']}: {piece['content']}" 
                for piece in approved_pieces
            ])
            
            summary_prompt = f"""
Generate a copilot-style summary that shows Ami's role as an AI assistant who has just completed a task from the user.

User Request: "{request.user_query}"

Approved Knowledge Pieces:
{approved_knowledge_text}

Write a summary in this copilot style:
- Start with "I've added [knowledge description] to your agent's brain"
- Follow with "Now your agent will be able to [capability gained]"
- Use first person (I/me) to show Ami's active role
- Be specific about what was learned
- Show how this knowledge enhances the agent's capabilities

Example format:
"I've added your financial monitoring workflow and Slack integration requirements to your agent's brain. Now your agent will be able to recognize similar automation requests and provide more targeted guidance for financial report processing and alert systems."

Make it as detailed as possible. Write in a confident, helpful copilot tone.
"""
            
            # Get summary from LLM
            summary_response = await self._get_reasoning_llm_response(summary_prompt, request)
            
            # Stream the summary as response chunks
            sentences = summary_response.split('. ')
            for sentence in sentences:
                if sentence.strip():
                    yield f"data: {json.dumps({
                        'type': 'response_chunk',
                        'content': sentence.strip() + ('. ' if not sentence.endswith('.') else ' '),
                        'complete': False,
                        'timestamp': datetime.now().isoformat()
                    })}\n\n"
                    await asyncio.sleep(0.1)  # Small delay for streaming effect
            
            # Final completion
            yield f"data: {json.dumps({
                'type': 'response_complete',
                'content': '',
                'complete': True,
                'metadata': {
                    'approved_knowledge_count': len(approved_pieces),
                    'total_knowledge_count': len(all_knowledge_pieces),
                    'summary_generated': True
                },
                'timestamp': datetime.now().isoformat()
            })}\n\n"
            
        except Exception as e:
            logger.error(f"Error handling knowledge approval: {e}")
            yield f"data: {json.dumps({
                'type': 'error',
                'content': f'Error generating summary after approval: {str(e)}',
                'complete': True,
                'timestamp': datetime.now().isoformat()
            })}\n\n"
    
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
            
            # Execute tool using provider-specific executor
            if request.llm_provider.lower() == "anthropic":
                result = await self.anthropic_executor.execute(request)
            else:
                result = await self.openai_executor.execute(request)
            
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
            
            # Execute tool with streaming using provider-specific executor
            if request.llm_provider.lower() == "anthropic":
                async for chunk in self.anthropic_executor.execute_stream(request):
                    yield chunk
            else:
                async for chunk in self.openai_executor.execute_stream(request):
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
    
    async def _plan_contextual_investigation(self, request: ToolExecutionRequest, analysis: RequestAnalysis) -> InvestigationPlan:
        """Plan investigation steps based on specific request type"""
        
        planning_prompt = f"""
        Plan a contextual investigation for this specific request:
        
        User Query: "{request.user_query}"
        Intent: {analysis.intent}
        Complexity: {analysis.complexity}
        Domain: {analysis.domain}
        
        Analyze what type of request this is:
        - Information retrieval ("tell me about...", "what is...")
        - Agent assessment ("test my agent", "evaluate performance") 
        - Problem diagnosis ("X isn't working", "how to fix...")
        - Knowledge sharing ("our company does...", "we use...")
        - Creation/action ("create...", "build...", "do...")
        - Learning guidance ("how to learn...", "teach me...")
        
        Based on the specific request type, create an investigation plan in this EXACT JSON format:
        
        {{
            "initial_thought": "I need to understand their specific situation first...",
            "reasoning_focus": "agent_testing|information_gathering|problem_solving|knowledge_sharing|task_creation|learning_guidance",
            "steps": [
                {{
                    "key": "brain_knowledge",
                    "search_query": "relevant brain vector search terms based on the request",
                    "thought_description": "What I'm investigating and why",
                    "discovery_template": "Found {{count}} knowledge pieces about {{domain}} - analyzing capabilities...",
                    "analysis_type": "capability_analysis|information_synthesis|problem_diagnosis|knowledge_assessment",
                    "requires_search": true,
                    "requires_analysis": true,
                    "order": 1
                }}
            ]
        }}
        
        Make investigation steps specific to the request type! 
        For agent testing: focus on capabilities and testing strategies
        For information requests: focus on finding and synthesizing relevant information
        For problem solving: focus on diagnosis and solution finding
        
        Respond with ONLY the JSON, no additional text.
        """
        
        try:
            plan_response = await self._get_reasoning_llm_response(planning_prompt, request)
            return InvestigationPlan.from_llm_response(plan_response)
        except Exception as e:
            logger.error(f"Investigation planning failed: {e}")
            return InvestigationPlan._create_fallback_plan()

    async def _execute_investigation_step(self, step: InvestigationStep, request: ToolExecutionRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute any type of investigation step and yield thoughts"""
        
        findings = {}
        
        # Brain vector reading (primary source)
        if step.requires_search and step.key == "brain_knowledge":
            # Stream brain reading thoughts
            async for thought in self._stream_brain_reading_thoughts(request, step.search_query):
                yield thought
            
            # Get brain vectors data
            brain_vectors = await self._read_agent_brain_vectors(request, step.search_query)
            findings["brain_vectors"] = brain_vectors
            
            if step.requires_analysis and brain_vectors:
                # Stream brain analysis thoughts
                async for thought in self._stream_brain_analysis_thoughts(brain_vectors, request, step.analysis_type):
                    yield thought
                
                # Get analysis data
                analysis_data = await self._analyze_brain_vector_contents(brain_vectors, request, step.analysis_type)
                findings["brain_analysis"] = analysis_data
        
        # External search (secondary source)
        elif step.requires_search and step.key == "external_knowledge":
            search_tool = self._get_search_tool(request.llm_provider, request.model)
            if search_tool:
                try:
                    search_results = await asyncio.to_thread(search_tool.search, step.search_query)
                    findings["search_results"] = search_results
                except Exception as e:
                    logger.error(f"External search failed: {e}")
                    findings["search_results"] = []
        
        # Context gathering (tertiary source)
        elif step.requires_search and step.key == "context_knowledge":
            if "context" in self.available_tools:
                try:
                    context_results = await asyncio.to_thread(
                        self.available_tools["context"].get_context,
                        query=step.search_query,
                        source_types=["user_profile", "system_status", "org_info"]
                    )
                    findings["context_results"] = context_results
                except Exception as e:
                    logger.error(f"Context gathering failed: {e}")
                    findings["context_results"] = {}
        
        # Yield the findings as the final result
        yield {
            "type": "investigation_result",
            "findings": findings,
            "step_key": step.key,
            "timestamp": datetime.now().isoformat()
        }

    async def _synthesize_contextual_strategy(self, context_findings: Dict[str, Any], request: ToolExecutionRequest, analysis: RequestAnalysis) -> ContextualStrategy:
        """Synthesize findings into actionable strategy - works for any request type"""
        
        synthesis_prompt = f"""
        Based on my investigation findings, create a tailored strategy:
        
        Original Request: "{request.user_query}"
        Intent: {analysis.intent}
        Complexity: {analysis.complexity}
        
        Investigation Findings:
        {json.dumps({k: str(v)[:500] + "..." if len(str(v)) > 500 else str(v) for k, v in context_findings.items()}, indent=2)}
        
        Now synthesize this into a specific, actionable strategy that addresses their exact situation.
        Focus on their specific context and capabilities I discovered.
        
        Provide response in this JSON format:
        {{
            "reasoning_summary": "Why this approach fits their specific situation",
            "action_plan": ["step1", "step2", "step3"],
            "context_specific_recommendations": ["rec1", "rec2"],
            "tailored_guidance": "Detailed guidance based on their specific setup"
        }}
        
        Make it specific to what I found about their agent/situation!
        """
        
        try:
            strategy_response = await self._get_reasoning_llm_response(synthesis_prompt, request)
            return ContextualStrategy.from_llm_response(strategy_response)
        except Exception as e:
            logger.error(f"Strategy synthesis failed: {e}")
            return ContextualStrategy._create_fallback_strategy("Strategy created based on available information")

    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.available_tools.keys())


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
    # Cursor-style parameters
    enable_intent_classification: bool = True,
    enable_request_analysis: bool = True,
    cursor_mode: bool = False,
    # NEW: Deep reasoning parameters
    enable_deep_reasoning: bool = False,
    reasoning_depth: str = "standard",
    brain_reading_enabled: bool = True,
    max_investigation_steps: int = 5
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream tool execution with specified parameters including deep reasoning
    
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
        enable_deep_reasoning: Enable multi-step reasoning with brain reading
        reasoning_depth: Depth of reasoning ("light", "standard", "deep")
        brain_reading_enabled: Whether to read user's brain vectors
        max_investigation_steps: Maximum number of investigation steps
        
    Yields:
        Dict containing streaming response data with deep reasoning thoughts
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
        cursor_mode=cursor_mode,
        enable_deep_reasoning=enable_deep_reasoning,
        reasoning_depth=reasoning_depth,
        brain_reading_enabled=brain_reading_enabled,
        max_investigation_steps=max_investigation_steps
    )
    async for chunk in executive_tool.execute_tool_stream(request):
        yield chunk 