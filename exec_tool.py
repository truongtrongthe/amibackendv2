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
                analysis_response = await self.anthropic_executor.executive_tool._analyze_with_anthropic(analysis_prompt, request.model)
            else:
                analysis_response = await self.openai_executor.executive_tool._analyze_with_openai(analysis_prompt, request.model)
            
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
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return ["anthropic", "openai"]
    
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