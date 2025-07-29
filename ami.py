"""
Ami - The Agent Creator Module
Simple, focused agent creator that integrates with the dynamic agent management system
"""

import os
import json
import re
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)

# Configure Ami creator logging
ami_logger = logging.getLogger("ami_creator")
ami_logger.setLevel(logging.INFO)
if not ami_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('🤖 [AMI] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    ami_logger.addHandler(handler)


class ConversationState(Enum):
    """States in the collaborative agent creation process"""
    INITIAL_IDEA = "initial_idea"           # Human provides initial idea
    UNDERSTANDING = "understanding"         # Ami analyzes and refines
    SKELETON_REVIEW = "skeleton_review"     # Human reviews skeleton plan
    REFINEMENT = "refinement"              # Ami refines based on feedback
    APPROVED = "approved"                  # Human approves, ready to build
    BUILDING = "building"                  # Ami is building the agent
    COMPLETED = "completed"                # Agent creation completed


@dataclass
class AgentIdea:
    """Initial human idea for an agent"""
    raw_idea: str                          # Original human input
    conversation_id: str                   # Unique conversation ID
    org_id: str
    user_id: str
    created_at: datetime
    state: ConversationState = ConversationState.INITIAL_IDEA


@dataclass
class AgentSkeleton:
    """Refined agent plan for human review"""
    conversation_id: str
    
    # Refined understanding
    agent_name: str
    agent_purpose: str
    target_users: str
    use_cases: List[str]
    
    # Technical specs
    agent_type: str
    language: str
    personality_traits: Dict[str, str]
    
    # Capabilities
    key_capabilities: List[str]
    required_tools: List[str]
    knowledge_domains: List[str]
    
    # Implementation details
    success_criteria: List[str]
    potential_challenges: List[str]
    
    # Conversation context
    created_at: datetime
    state: ConversationState = ConversationState.SKELETON_REVIEW


@dataclass
class CollaborativeAgentRequest:
    """Request for collaborative agent creation"""
    user_input: str                        # Human input at any stage
    conversation_id: Optional[str] = None  # Existing conversation ID
    org_id: str = ""
    user_id: str = ""
    llm_provider: str = "anthropic"
    model: Optional[str] = None
    current_state: ConversationState = ConversationState.INITIAL_IDEA


@dataclass
class CollaborativeAgentResponse:
    """Response from collaborative agent creation"""
    success: bool
    conversation_id: str
    current_state: ConversationState
    ami_message: str                       # Ami's response to human
    data: Optional[Dict[str, Any]] = None  # State-specific data
    next_actions: List[str] = None         # What human can do next
    error: Optional[str] = None


# Legacy model for backward compatibility
@dataclass
class AgentCreationRequest:
    """Request model for direct agent creation (legacy)"""
    user_request: str           # "Create a sales agent for Vietnamese market"
    org_id: str
    user_id: str
    llm_provider: str = "anthropic"
    model: Optional[str] = None


@dataclass
class AgentCreationResult:
    """Response model for agent creation"""
    success: bool
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None


@dataclass
class SimpleAgentConfig:
    """Internal config model - keep simple"""
    name: str
    description: str
    agent_type: str             # "sales", "support", "analyst", "document_analysis", "general"
    tools_needed: List[str]
    language: str = "english"
    specialization: List[str] = None


class AmiAgentCreator:
    """
    Enhanced AI Agent Creator with Chief Product Officer approach
    Collaborates with humans to refine requirements before building agents
    """
    
    def __init__(self):
        """Initialize with conversation state management"""
        self.anthropic_client = self._init_anthropic()
        self.openai_client = self._init_openai()
        
        # Simple tool mapping based on agent types
        self.available_tools = {
            "document_analysis": ["file_access", "business_logic"],
            "sales": ["search", "context", "business_logic"],
            "support": ["brain_vector", "context", "search"],
            "analyst": ["file_access", "business_logic", "search"],
            "automation": ["file_access", "business_logic"],
            "research": ["search", "brain_vector", "context"],
            "general": ["search", "context"]
        }
        
        # Conversation state storage (in production, use Redis or database)
        self.conversations: Dict[str, Dict[str, Any]] = {}
        
        ami_logger.info("Ami Agent Creator initialized with collaborative capabilities")
    
    async def create_agent(self, request: AgentCreationRequest) -> AgentCreationResult:
        """
        Main creation method - simple and direct
        
        Args:
            request: Agent creation request containing user requirements
            
        Returns:
            AgentCreationResult with success status and agent details
        """
        ami_logger.info(f"Starting agent creation for request: '{request.user_request[:100]}...'")
        
        try:
            # Step 1: Simple analysis of what kind of agent is needed
            ami_logger.info("Step 1: Analyzing agent requirements...")
            agent_config = await self._analyze_agent_needs(request.user_request, request.llm_provider)
            ami_logger.info(f"Agent analysis complete: {agent_config.name} ({agent_config.agent_type})")
            
            # Step 2: Generate comprehensive system prompt
            ami_logger.info("Step 2: Generating system prompt...")
            system_prompt_data = await self._generate_system_prompt(agent_config, request.llm_provider)
            ami_logger.info("System prompt generated successfully")
            
            # Step 3: Select optimal tools for this agent type
            ami_logger.info("Step 3: Selecting tools...")
            tools_list = self._select_tools(agent_config.agent_type)
            ami_logger.info(f"Selected tools: {tools_list}")
            
            # Step 4: Determine knowledge requirements
            ami_logger.info("Step 4: Setting up knowledge access...")
            knowledge_list = self._determine_knowledge_needs(agent_config)
            ami_logger.info(f"Knowledge domains: {knowledge_list}")
            
            # Step 5: Save to database via org_agent system
            ami_logger.info("Step 5: Saving agent to database...")
            agent_id = await self._save_to_database(
                config={
                    "name": agent_config.name,
                    "description": agent_config.description,
                    "system_prompt": system_prompt_data,
                    "tools_list": tools_list,
                    "knowledge_list": knowledge_list
                },
                org_id=request.org_id,
                user_id=request.user_id
            )
            ami_logger.info(f"Agent saved successfully with ID: {agent_id}")
            
            return AgentCreationResult(
                success=True,
                agent_id=agent_id,
                agent_name=agent_config.name,
                message=f"✅ Created '{agent_config.name}' successfully! The agent is now ready to use.",
                agent_config={
                    "name": agent_config.name,
                    "description": agent_config.description,
                    "agent_type": agent_config.agent_type,
                    "language": agent_config.language,
                    "tools": tools_list,
                    "knowledge": knowledge_list
                }
            )
            
        except Exception as e:
            ami_logger.error(f"Agent creation failed: {str(e)}")
            return AgentCreationResult(
                success=False,
                error=str(e),
                message="❌ Failed to create agent. Please try again or contact support."
            )
    
    async def collaborate_on_agent(self, request: CollaborativeAgentRequest) -> CollaborativeAgentResponse:
        """
        Main collaborative method - acts like a Chief Product Officer
        Guides the human through agent creation with iterative refinement
        
        Args:
            request: Collaborative request with user input and conversation state
            
        Returns:
            CollaborativeAgentResponse with Ami's guidance and next steps
        """
        ami_logger.info(f"Collaborative session: {request.current_state.value} - '{request.user_input[:100]}...'")
        
        try:
            # Generate conversation ID if new conversation
            if not request.conversation_id:
                request.conversation_id = str(uuid4())
                ami_logger.info(f"Started new collaborative session: {request.conversation_id}")
            
            # Route to appropriate handler based on current state
            if request.current_state == ConversationState.INITIAL_IDEA:
                return await self._understand_and_refine_idea(request)
            
            elif request.current_state == ConversationState.SKELETON_REVIEW:
                return await self._handle_skeleton_feedback(request)
            
            elif request.current_state == ConversationState.APPROVED:
                return await self._build_approved_agent(request)
            
            else:
                return CollaborativeAgentResponse(
                    success=False,
                    conversation_id=request.conversation_id,
                    current_state=request.current_state,
                    ami_message="I'm not sure how to handle this state. Let's start over!",
                    error=f"Unknown state: {request.current_state}",
                    next_actions=["Start a new conversation"]
                )
                
        except Exception as e:
            ami_logger.error(f"Collaborative session error: {e}")
            return CollaborativeAgentResponse(
                success=False,
                conversation_id=request.conversation_id or "unknown",
                current_state=request.current_state,
                ami_message="Something went wrong. Let me help you start fresh!",
                error=str(e),
                next_actions=["Try again with a new request"]
            )
    
    async def _understand_and_refine_idea(self, request: CollaborativeAgentRequest) -> CollaborativeAgentResponse:
        """
        Step 1: Understand human's initial idea and create refined skeleton plan
        Acts like a product manager gathering requirements
        """
        ami_logger.info(f"Understanding idea: '{request.user_input}'")
        
        # Create comprehensive analysis prompt
        understanding_prompt = f"""
        You are Ami, a Chief Product Officer for AI agents. A human has shared their initial idea for an AI agent. Your job is to:

        1. UNDERSTAND their vision deeply
        2. ASK CLARIFYING QUESTIONS 
        3. REFINE the idea into a concrete plan
        4. CREATE a detailed agent skeleton

        Human's Initial Idea: "{request.user_input}"

        Act like an experienced product manager. Think through:
        - What problem are they trying to solve?
        - Who will use this agent? 
        - What should it actually DO?
        - What are the success criteria?
        - What challenges might we face?

        Respond with this EXACT JSON format:
        {{
            "understanding": "What I understand you want to achieve...",
            "clarifying_questions": [
                "Question 1 about their needs?",
                "Question 2 about usage scenarios?", 
                "Question 3 about constraints?"
            ],
            "agent_skeleton": {{
                "agent_name": "Descriptive Agent Name",
                "agent_purpose": "Clear purpose statement",
                "target_users": "Who will use this agent",
                "use_cases": ["Use case 1", "Use case 2", "Use case 3"],
                "agent_type": "sales|support|analyst|document_analysis|research|automation|general",
                "language": "english|vietnamese|french|spanish",
                "personality_traits": {{
                    "tone": "professional|friendly|authoritative|consultative",
                    "style": "concise|detailed|conversational|technical",
                    "approach": "proactive|reactive|analytical|creative"
                }},
                "key_capabilities": ["Capability 1", "Capability 2", "Capability 3"],
                "required_tools": ["tool1", "tool2", "tool3"],
                "knowledge_domains": ["domain1", "domain2"],
                "success_criteria": ["Success measure 1", "Success measure 2"],
                "potential_challenges": ["Challenge 1", "Challenge 2"]
            }},
            "ami_message": "Here's what I understand... [natural explanation of the plan and questions]"
        }}

        Be conversational, helpful, and thorough. Ask smart questions that show you understand their business context.
        """
        
        try:
            analysis_response = await self._call_llm(understanding_prompt, request.llm_provider)
            
            # Parse the structured response
            json_match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                # Create agent skeleton object
                skeleton_data = data.get("agent_skeleton", {})
                skeleton = AgentSkeleton(
                    conversation_id=request.conversation_id,
                    agent_name=skeleton_data.get("agent_name", "Custom Agent"),
                    agent_purpose=skeleton_data.get("agent_purpose", "Specialized AI assistant"),
                    target_users=skeleton_data.get("target_users", "General users"),
                    use_cases=skeleton_data.get("use_cases", ["General assistance"]),
                    agent_type=skeleton_data.get("agent_type", "general"),
                    language=skeleton_data.get("language", "english"),
                    personality_traits=skeleton_data.get("personality_traits", {"tone": "professional"}),
                    key_capabilities=skeleton_data.get("key_capabilities", ["General assistance"]),
                    required_tools=skeleton_data.get("required_tools", ["search", "context"]),
                    knowledge_domains=skeleton_data.get("knowledge_domains", []),
                    success_criteria=skeleton_data.get("success_criteria", ["User satisfaction"]),
                    potential_challenges=skeleton_data.get("potential_challenges", ["None identified"]),
                    created_at=datetime.now()
                )
                
                # Store conversation state
                self.conversations[request.conversation_id] = {
                    "original_idea": request.user_input,
                    "skeleton": skeleton,
                    "state": ConversationState.SKELETON_REVIEW,
                    "org_id": request.org_id,
                    "user_id": request.user_id
                }
                
                ami_logger.info(f"Created skeleton for {skeleton.agent_name}")
                
                return CollaborativeAgentResponse(
                    success=True,
                    conversation_id=request.conversation_id,
                    current_state=ConversationState.SKELETON_REVIEW,
                    ami_message=data.get("ami_message", "Here's my understanding of your agent idea. What do you think?"),
                    data={
                        "understanding": data.get("understanding", ""),
                        "clarifying_questions": data.get("clarifying_questions", []),
                        "agent_skeleton": {
                            "agent_name": skeleton.agent_name,
                            "agent_purpose": skeleton.agent_purpose,
                            "target_users": skeleton.target_users,
                            "use_cases": skeleton.use_cases,
                            "agent_type": skeleton.agent_type,
                            "language": skeleton.language,
                            "personality_traits": skeleton.personality_traits,
                            "key_capabilities": skeleton.key_capabilities,
                            "required_tools": skeleton.required_tools,
                            "knowledge_domains": skeleton.knowledge_domains,
                            "success_criteria": skeleton.success_criteria,
                            "potential_challenges": skeleton.potential_challenges
                        }
                    },
                    next_actions=[
                        "Approve this plan to build the agent",
                        "Request changes to any part of the plan", 
                        "Ask for clarification on any aspect",
                        "Add more requirements or constraints"
                    ]
                )
                
        except Exception as e:
            ami_logger.error(f"Idea understanding failed: {e}")
        
        # Fallback response if parsing fails
        return CollaborativeAgentResponse(
            success=False,
            conversation_id=request.conversation_id,
            current_state=ConversationState.INITIAL_IDEA,
            ami_message="I had trouble understanding your idea. Could you tell me more about what kind of agent you'd like to create?",
            error="Failed to parse agent requirements",
            next_actions=["Describe your agent idea in more detail"]
        )
    
    async def _handle_skeleton_feedback(self, request: CollaborativeAgentRequest) -> CollaborativeAgentResponse:
        """
        Step 2: Handle human feedback on the skeleton plan
        Refine the plan or approve for building
        """
        ami_logger.info(f"Handling skeleton feedback: '{request.user_input}'")
        
        # Get conversation state
        conversation = self.conversations.get(request.conversation_id)
        if not conversation:
            return CollaborativeAgentResponse(
                success=False,
                conversation_id=request.conversation_id,
                current_state=ConversationState.INITIAL_IDEA,
                ami_message="I lost track of our conversation. Let's start fresh!",
                error="Conversation state not found",
                next_actions=["Start a new agent creation conversation"]
            )
        
        skeleton = conversation["skeleton"]
        
        # Check if this is approval
        approval_keywords = ["approve", "approved", "looks good", "build it", "proceed", "yes", "perfect", "go ahead", "create it"]
        is_approval = any(keyword in request.user_input.lower() for keyword in approval_keywords)
        
        if is_approval:
            # Update state to approved
            conversation["state"] = ConversationState.APPROVED
            self.conversations[request.conversation_id] = conversation
            
            return CollaborativeAgentResponse(
                success=True,
                conversation_id=request.conversation_id,
                current_state=ConversationState.APPROVED,
                ami_message=f"Perfect! I'll now build '{skeleton.agent_name}' exactly as planned. This will take a moment...",
                data={"approved_skeleton": skeleton.__dict__},
                next_actions=["Wait for agent creation to complete"]
            )
        
        # Handle refinement request
        refinement_prompt = f"""
        You are Ami, a Chief Product Officer. The human is providing feedback on an agent plan. Your job is to:

        1. UNDERSTAND their feedback and concerns
        2. REFINE the agent skeleton based on their input
        3. EXPLAIN the changes you're making
        4. ASK if they need any other adjustments

        Original Agent Skeleton:
        - Name: {skeleton.agent_name}
        - Purpose: {skeleton.agent_purpose}
        - Target Users: {skeleton.target_users}
        - Use Cases: {skeleton.use_cases}
        - Agent Type: {skeleton.agent_type}
        - Language: {skeleton.language}
        - Capabilities: {skeleton.key_capabilities}
        - Tools: {skeleton.required_tools}
        - Success Criteria: {skeleton.success_criteria}

        Human Feedback: "{request.user_input}"

        Respond with this EXACT JSON format:
        {{
            "feedback_understanding": "What I understand from your feedback...",
            "changes_made": ["Change 1", "Change 2", "Change 3"],
            "updated_skeleton": {{
                "agent_name": "Updated Agent Name",
                "agent_purpose": "Updated purpose statement",
                "target_users": "Updated target users",
                "use_cases": ["Updated use case 1", "Updated use case 2"],
                "agent_type": "sales|support|analyst|document_analysis|research|automation|general",
                "language": "english|vietnamese|french|spanish",
                "personality_traits": {{
                    "tone": "professional|friendly|authoritative|consultative",
                    "style": "concise|detailed|conversational|technical",
                    "approach": "proactive|reactive|analytical|creative"
                }},
                "key_capabilities": ["Updated capability 1", "Updated capability 2"],
                "required_tools": ["tool1", "tool2", "tool3"],
                "knowledge_domains": ["domain1", "domain2"],
                "success_criteria": ["Updated success measure 1", "Updated success measure 2"],
                "potential_challenges": ["Updated challenge 1", "Updated challenge 2"]
            }},
            "ami_message": "I've updated the plan based on your feedback... [explanation of changes and questions]"
        }}

        Be responsive to their specific concerns and explain your reasoning.
        """
        
        try:
            refinement_response = await self._call_llm(refinement_prompt, request.llm_provider)
            
            # Parse the refinement response
            json_match = re.search(r'\{.*\}', refinement_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                # Update skeleton with refined data
                updated_skeleton_data = data.get("updated_skeleton", {})
                refined_skeleton = AgentSkeleton(
                    conversation_id=request.conversation_id,
                    agent_name=updated_skeleton_data.get("agent_name", skeleton.agent_name),
                    agent_purpose=updated_skeleton_data.get("agent_purpose", skeleton.agent_purpose),
                    target_users=updated_skeleton_data.get("target_users", skeleton.target_users),
                    use_cases=updated_skeleton_data.get("use_cases", skeleton.use_cases),
                    agent_type=updated_skeleton_data.get("agent_type", skeleton.agent_type),
                    language=updated_skeleton_data.get("language", skeleton.language),
                    personality_traits=updated_skeleton_data.get("personality_traits", skeleton.personality_traits),
                    key_capabilities=updated_skeleton_data.get("key_capabilities", skeleton.key_capabilities),
                    required_tools=updated_skeleton_data.get("required_tools", skeleton.required_tools),
                    knowledge_domains=updated_skeleton_data.get("knowledge_domains", skeleton.knowledge_domains),
                    success_criteria=updated_skeleton_data.get("success_criteria", skeleton.success_criteria),
                    potential_challenges=updated_skeleton_data.get("potential_challenges", skeleton.potential_challenges),
                    created_at=datetime.now()
                )
                
                # Update conversation state
                conversation["skeleton"] = refined_skeleton
                self.conversations[request.conversation_id] = conversation
                
                ami_logger.info(f"Refined skeleton for {refined_skeleton.agent_name}")
                
                return CollaborativeAgentResponse(
                    success=True,
                    conversation_id=request.conversation_id,
                    current_state=ConversationState.SKELETON_REVIEW,
                    ami_message=data.get("ami_message", "I've updated the plan based on your feedback. How does this look now?"),
                    data={
                        "feedback_understanding": data.get("feedback_understanding", ""),
                        "changes_made": data.get("changes_made", []),
                        "updated_skeleton": {
                            "agent_name": refined_skeleton.agent_name,
                            "agent_purpose": refined_skeleton.agent_purpose,
                            "target_users": refined_skeleton.target_users,
                            "use_cases": refined_skeleton.use_cases,
                            "agent_type": refined_skeleton.agent_type,
                            "language": refined_skeleton.language,
                            "personality_traits": refined_skeleton.personality_traits,
                            "key_capabilities": refined_skeleton.key_capabilities,
                            "required_tools": refined_skeleton.required_tools,
                            "knowledge_domains": refined_skeleton.knowledge_domains,
                            "success_criteria": refined_skeleton.success_criteria,
                            "potential_challenges": refined_skeleton.potential_challenges
                        }
                    },
                    next_actions=[
                        "Approve this updated plan",
                        "Request further changes",
                        "Ask questions about the updates"
                    ]
                )
                
        except Exception as e:
            ami_logger.error(f"Skeleton refinement failed: {e}")
        
        # Fallback for refinement issues
        return CollaborativeAgentResponse(
            success=False,
            conversation_id=request.conversation_id,
            current_state=ConversationState.SKELETON_REVIEW,
            ami_message="I had trouble processing your feedback. Could you tell me specifically what you'd like me to change about the agent plan?",
            error="Failed to process refinement request",
            next_actions=["Be more specific about what to change", "Approve the current plan"]
        )
    
    async def _build_approved_agent(self, request: CollaborativeAgentRequest) -> CollaborativeAgentResponse:
        """
        Step 3: Build the final agent from approved skeleton
        Convert skeleton into actual agent configuration and save to database
        """
        ami_logger.info(f"Building approved agent for conversation: {request.conversation_id}")
        
        # Get conversation state
        conversation = self.conversations.get(request.conversation_id)
        if not conversation:
            return CollaborativeAgentResponse(
                success=False,
                conversation_id=request.conversation_id,
                current_state=ConversationState.INITIAL_IDEA,
                ami_message="I lost track of our conversation. Let's start fresh!",
                error="Conversation state not found",
                next_actions=["Start a new agent creation conversation"]
            )
        
        skeleton = conversation["skeleton"]
        
        try:
            # Generate comprehensive system prompt from skeleton
            system_prompt_data = await self._generate_system_prompt_from_skeleton(skeleton, request.llm_provider)
            
            # Select tools based on skeleton requirements
            tools_list = skeleton.required_tools
            
            # Set up knowledge domains
            knowledge_list = skeleton.knowledge_domains
            
            # Save to database
            agent_id = await self._save_to_database(
                config={
                    "name": skeleton.agent_name,
                    "description": skeleton.agent_purpose,
                    "system_prompt": system_prompt_data,
                    "tools_list": tools_list,
                    "knowledge_list": knowledge_list
                },
                org_id=conversation["org_id"],
                user_id=conversation["user_id"]
            )
            
            # Update conversation state to completed
            conversation["state"] = ConversationState.COMPLETED
            conversation["final_agent_id"] = agent_id
            self.conversations[request.conversation_id] = conversation
            
            ami_logger.info(f"Successfully built agent: {skeleton.agent_name} (ID: {agent_id})")
            
            return CollaborativeAgentResponse(
                success=True,
                conversation_id=request.conversation_id,
                current_state=ConversationState.COMPLETED,
                ami_message=f"🎉 Perfect! I've successfully created '{skeleton.agent_name}'!\n\nYour agent is now ready to use with specialized capabilities for {skeleton.agent_purpose.lower()}. You can start using it right away!",
                data={
                    "agent_id": agent_id,
                    "agent_name": skeleton.agent_name,
                    "agent_config": {
                        "name": skeleton.agent_name,
                        "description": skeleton.agent_purpose,
                        "agent_type": skeleton.agent_type,
                        "language": skeleton.language,
                        "capabilities": skeleton.key_capabilities,
                        "tools": tools_list,
                        "knowledge": knowledge_list
                    }
                },
                next_actions=[
                    f"Start using '{skeleton.agent_name}' for your tasks",
                    "Create another agent",
                    "Test the agent with a sample task"
                ]
            )
            
        except Exception as e:
            ami_logger.error(f"Agent building failed: {e}")
            return CollaborativeAgentResponse(
                success=False,
                conversation_id=request.conversation_id,
                current_state=ConversationState.APPROVED,
                ami_message="I encountered an issue while building your agent. Let me try again or we can refine the plan.",
                error=str(e),
                next_actions=["Try building again", "Go back to refine the plan"]
            )
    
    async def _generate_system_prompt_from_skeleton(self, skeleton: AgentSkeleton, provider: str = "anthropic") -> Dict[str, Any]:
        """Generate system prompt from detailed skeleton plan"""
        
        prompt_generation_request = f"""
        Create a comprehensive system prompt for this AI agent based on the detailed plan:
        
        Agent Skeleton:
        - Name: {skeleton.agent_name}
        - Purpose: {skeleton.agent_purpose}
        - Target Users: {skeleton.target_users}
        - Use Cases: {skeleton.use_cases}
        - Agent Type: {skeleton.agent_type}
        - Language: {skeleton.language}
        - Personality: {skeleton.personality_traits}
        - Key Capabilities: {skeleton.key_capabilities}
        - Success Criteria: {skeleton.success_criteria}
        
        Create a system prompt that:
        1. Embodies the agent's specific purpose and personality
        2. Reflects the target users and use cases
        3. Includes appropriate language instructions
        4. Emphasizes the key capabilities
        5. Sets the right tone and approach
        6. Is comprehensive yet natural (300-500 words)
        
        Make the agent feel truly specialized for its intended purpose.
        
        Return only the system prompt text, no explanation or formatting.
        """
        
        try:
            system_prompt_text = await self._call_llm(prompt_generation_request, provider)
            system_prompt_text = system_prompt_text.strip()
            
            # Create comprehensive prompt data structure
            prompt_data = {
                "base_instruction": system_prompt_text,
                "agent_type": skeleton.agent_type,
                "language": skeleton.language,
                "specialization": skeleton.key_capabilities,
                "personality": skeleton.personality_traits,
                "target_users": skeleton.target_users,
                "use_cases": skeleton.use_cases,
                "success_criteria": skeleton.success_criteria,
                "created_at": datetime.now().isoformat(),
                "created_from": "collaborative_skeleton"
            }
            
            return prompt_data
            
        except Exception as e:
            ami_logger.error(f"System prompt generation from skeleton failed: {e}")
            # Fallback prompt structure
            fallback_prompt = f"You are {skeleton.agent_name}. {skeleton.agent_purpose} You specialize in {', '.join(skeleton.key_capabilities)}. Your target users are {skeleton.target_users}. Be {skeleton.personality_traits.get('tone', 'professional')} and {skeleton.personality_traits.get('style', 'helpful')}."
            
            return {
                "base_instruction": fallback_prompt,
                "agent_type": skeleton.agent_type,
                "language": skeleton.language,
                "specialization": skeleton.key_capabilities,
                "personality": skeleton.personality_traits,
                "created_at": datetime.now().isoformat(),
                "created_from": "collaborative_skeleton_fallback"
            }
    
    async def _analyze_agent_needs(self, user_request: str, provider: str = "anthropic") -> SimpleAgentConfig:
        """
        Simple LLM analysis to understand what kind of agent is needed
        
        Args:
            user_request: User's description of what they want
            provider: LLM provider to use
            
        Returns:
            SimpleAgentConfig with analyzed requirements
        """
        
        analysis_prompt = f"""
        Analyze this agent creation request and respond with JSON only.
        
        User Request: "{user_request}"
        
        Determine:
        1. Professional agent name (descriptive, specific to purpose)
        2. Clear description of what this agent does
        3. Agent type from: sales, support, analyst, document_analysis, research, automation, general
        4. Primary language: english, vietnamese, french, spanish, chinese
        5. Key specialization areas (2-4 specific areas)
        
        Agent Types:
        - sales: Sales assistance, lead qualification, customer communication
        - support: Customer support, troubleshooting, FAQ assistance  
        - analyst: Data analysis, business intelligence, report generation
        - document_analysis: Document processing, content analysis, file management
        - research: Information gathering, market research, competitive analysis
        - automation: Process automation, workflow management, task execution
        - general: Multi-purpose assistant for various tasks
        
        Respond with this exact JSON format:
        {{
            "name": "Specific Agent Name",
            "description": "Clear description of agent's purpose and capabilities",
            "agent_type": "document_analysis",
            "language": "english",
            "specialization": ["area1", "area2", "area3"]
        }}
        
        Examples:
        Request: "I need help with Vietnamese sales documents"
        Response: {{"name": "Vietnamese Sales Document Specialist", "description": "Analyzes Vietnamese sales documents and provides insights for deal progression", "agent_type": "document_analysis", "language": "vietnamese", "specialization": ["sales_document_analysis", "vietnamese_business_context", "deal_assessment"]}}
        
        Make the agent name specific and professional. Focus on the primary task.
        """
        
        try:
            response = await self._call_llm(analysis_prompt, provider)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return SimpleAgentConfig(
                    name=data.get("name", "Custom Agent"),
                    description=data.get("description", "A specialized AI agent"),
                    agent_type=data.get("agent_type", "general"),
                    tools_needed=[],
                    language=data.get("language", "english"),
                    specialization=data.get("specialization", ["general_assistance"])
                )
        except Exception as e:
            ami_logger.warning(f"Agent analysis parsing failed: {e}, using fallback")
        
        # Fallback configuration if parsing fails
        return SimpleAgentConfig(
            name="Custom Agent",
            description="A specialized AI agent tailored to your needs",
            agent_type="general",
            tools_needed=[],
            language="english",
            specialization=["general_assistance"]
        )
    
    async def _generate_system_prompt(self, config: SimpleAgentConfig, provider: str = "anthropic") -> Dict[str, Any]:
        """
        Generate comprehensive system prompt configuration
        
        Args:
            config: Agent configuration
            provider: LLM provider
            
        Returns:
            Dictionary with system prompt data for database storage
        """
        
        prompt_generation_request = f"""
        Create a comprehensive system prompt for this AI agent:
        
        Agent Details:
        - Name: {config.name}
        - Type: {config.agent_type}
        - Purpose: {config.description}
        - Language: {config.language}
        - Specialization: {config.specialization}
        
        Create a system prompt that:
        1. Clearly defines the agent's role and expertise
        2. Sets appropriate professional tone and personality
        3. Includes specific language instructions if not English
        4. Mentions specialization areas naturally
        5. Encourages use of available tools when relevant
        6. Is concise but comprehensive (200-400 words)
        
        Make it sound natural and professional. The agent should feel specialized and capable.
        
        Return only the system prompt text, no explanation or formatting.
        """
        
        try:
            system_prompt_text = await self._call_llm(prompt_generation_request, provider)
            
            # Clean up the response
            system_prompt_text = system_prompt_text.strip()
            
            # Create comprehensive prompt data structure
            prompt_data = {
                "base_instruction": system_prompt_text,
                "agent_type": config.agent_type,
                "language": config.language,
                "specialization": config.specialization,
                "personality": {
                    "tone": "professional",
                    "style": "helpful",
                    "approach": "solution-oriented"
                },
                "created_at": datetime.now().isoformat()
            }
            
            return prompt_data
            
        except Exception as e:
            ami_logger.error(f"System prompt generation failed: {e}")
            # Return fallback prompt structure
            fallback_prompt = f"You are {config.name}, a specialized {config.agent_type} agent. Your purpose is to {config.description.lower()}. You are professional, helpful, and focus on providing accurate, relevant assistance."
            
            return {
                "base_instruction": fallback_prompt,
                "agent_type": config.agent_type,
                "language": config.language,
                "specialization": config.specialization,
                "personality": {"tone": "professional", "style": "helpful", "approach": "solution-oriented"},
                "created_at": datetime.now().isoformat()
            }
    
    def _select_tools(self, agent_type: str) -> List[str]:
        """
        Simple tool selection based on agent type
        
        Args:
            agent_type: Type of agent
            
        Returns:
            List of tool names this agent should have access to
        """
        tools = self.available_tools.get(agent_type, self.available_tools["general"])
        ami_logger.info(f"Selected tools for {agent_type}: {tools}")
        return tools
    
    def _determine_knowledge_needs(self, config: SimpleAgentConfig) -> List[str]:
        """
        Determine what knowledge domains this agent needs access to
        
        Args:
            config: Agent configuration
            
        Returns:
            List of knowledge domain identifiers
        """
        # Start with empty knowledge list - can be populated later
        # This allows for manual knowledge assignment or future AI-driven knowledge mapping
        knowledge_domains = []
        
        # Add basic knowledge based on agent type
        if config.agent_type == "sales":
            knowledge_domains.extend(["sales_techniques", "product_information"])
        elif config.agent_type == "support":
            knowledge_domains.extend(["faq_database", "troubleshooting_guides"])
        elif config.agent_type == "document_analysis":
            knowledge_domains.extend(["document_processing", "business_intelligence"])
        elif config.agent_type == "analyst":
            knowledge_domains.extend(["data_analysis", "business_metrics"])
        
        # Add language-specific knowledge if needed
        if config.language != "english":
            knowledge_domains.append(f"{config.language}_context")
        
        return knowledge_domains
    
    async def _save_to_database(self, config: dict, org_id: str, user_id: str) -> str:
        """
        Save agent configuration to database via org_agent system
        
        Args:
            config: Agent configuration dictionary
            org_id: Organization ID
            user_id: User ID who created the agent
            
        Returns:
            Agent ID of created agent
        """
        try:
            from orgdb import create_agent
            
            agent = create_agent(
                org_id=org_id,
                created_by=user_id,
                name=config["name"],
                description=config["description"],
                system_prompt=config["system_prompt"],
                tools_list=config["tools_list"],
                knowledge_list=config["knowledge_list"]
            )
            
            ami_logger.info(f"Agent saved to database: {agent.name} (ID: {agent.id})")
            return agent.id
            
        except Exception as e:
            ami_logger.error(f"Database save failed: {e}")
            raise Exception(f"Failed to save agent to database: {str(e)}")
    
    async def _call_llm(self, prompt: str, provider: str = "anthropic") -> str:
        """
        Simple LLM call - no complex features
        
        Args:
            prompt: Prompt to send to LLM
            provider: LLM provider ('anthropic' or 'openai')
            
        Returns:
            LLM response text
        """
        try:
            if provider == "anthropic" and self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1500,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif provider == "openai" and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=1500,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            
            else:
                raise Exception(f"LLM provider {provider} not available or not configured")
                
        except Exception as e:
            ami_logger.error(f"LLM call failed: {e}")
            raise Exception(f"LLM call failed: {str(e)}")
    
    def _init_anthropic(self):
        """Simple Anthropic client initialization"""
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                return anthropic.AsyncAnthropic(api_key=api_key)
            else:
                ami_logger.warning("ANTHROPIC_API_KEY not found in environment")
                return None
        except ImportError:
            ami_logger.warning("Anthropic library not installed")
            return None
        except Exception as e:
            ami_logger.warning(f"Failed to initialize Anthropic client: {e}")
            return None
    
    def _init_openai(self):
        """Simple OpenAI client initialization"""
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return openai.AsyncOpenAI(api_key=api_key)
            else:
                ami_logger.warning("OPENAI_API_KEY not found in environment")
                return None
        except ImportError:
            ami_logger.warning("OpenAI library not installed")
            return None
        except Exception as e:
            ami_logger.warning(f"Failed to initialize OpenAI client: {e}")
            return None


# Convenience functions for easy integration
async def create_agent_simple(user_request: str, org_id: str, user_id: str, provider: str = "anthropic") -> AgentCreationResult:
    """
    Simple function to create an agent
    
    Args:
        user_request: Description of what kind of agent is needed
        org_id: Organization ID
        user_id: User ID
        provider: LLM provider to use
        
    Returns:
        AgentCreationResult with creation status and details
    """
    creator = AmiAgentCreator()
    request = AgentCreationRequest(
        user_request=user_request,
        org_id=org_id,
        user_id=user_id,
        llm_provider=provider
    )
    return await creator.create_agent(request)


# Initialize global creator instance for API usage
ami_creator_instance = None

def get_ami_creator() -> AmiAgentCreator:
    """Get global Ami creator instance (singleton pattern)"""
    global ami_creator_instance
    if ami_creator_instance is None:
        ami_creator_instance = AmiAgentCreator()
    return ami_creator_instance


# FastAPI Integration Models
from pydantic import BaseModel, Field

class CreateAgentAPIRequest(BaseModel):
    """API request model for direct agent creation (legacy)"""
    user_request: str = Field(..., description="Description of what kind of agent is needed", min_length=10, max_length=1000)
    llm_provider: str = Field("anthropic", description="LLM provider to use for agent creation")
    model: Optional[str] = Field(None, description="Specific model to use (optional)")


class CollaborativeAgentAPIRequest(BaseModel):
    """API request model for collaborative agent creation"""
    user_input: str = Field(..., description="Human input at any stage of conversation", min_length=5, max_length=2000)
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID (for continuing conversations)")
    current_state: Optional[str] = Field("initial_idea", description="Current conversation state")
    llm_provider: str = Field("anthropic", description="LLM provider to use")
    model: Optional[str] = Field(None, description="Specific model to use")


class CollaborativeAgentAPIResponse(BaseModel):
    """API response model for collaborative agent creation"""
    success: bool
    conversation_id: str
    current_state: str
    ami_message: str
    data: Optional[Dict[str, Any]] = None
    next_actions: Optional[List[str]] = None
    error: Optional[str] = None

class CreateAgentAPIResponse(BaseModel):
    """API response model for agent creation"""
    success: bool
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    message: str
    error: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None


async def create_agent_via_api(api_request: CreateAgentAPIRequest, org_id: str, user_id: str) -> CreateAgentAPIResponse:
    """
    Create agent via API call
    
    Args:
        api_request: API request with agent requirements
        org_id: Organization ID
        user_id: User ID
        
    Returns:
        API response with creation results
    """
    try:
        ami_creator = get_ami_creator()
        
        creation_request = AgentCreationRequest(
            user_request=api_request.user_request,
            org_id=org_id,
            user_id=user_id,
            llm_provider=api_request.llm_provider,
            model=api_request.model
        )
        
        result = await ami_creator.create_agent(creation_request)
        
        return CreateAgentAPIResponse(
            success=result.success,
            agent_id=result.agent_id,
            agent_name=result.agent_name,
            message=result.message,
            error=result.error,
            agent_config=result.agent_config
        )
        
    except Exception as e:
        ami_logger.error(f"API agent creation failed: {e}")
        return CreateAgentAPIResponse(
            success=False,
            message="❌ Agent creation failed",
            error=str(e)
        )


async def collaborate_on_agent_via_api(api_request: CollaborativeAgentAPIRequest, org_id: str, user_id: str) -> CollaborativeAgentAPIResponse:
    """
    Collaborative agent creation via API - Chief Product Officer approach
    
    Args:
        api_request: API request with user input and conversation state
        org_id: Organization ID
        user_id: User ID
        
    Returns:
        API response with Ami's guidance and conversation state
    """
    try:
        ami_creator = get_ami_creator()
        
        # Convert API request to internal request
        collaboration_request = CollaborativeAgentRequest(
            user_input=api_request.user_input,
            conversation_id=api_request.conversation_id,
            org_id=org_id,
            user_id=user_id,
            llm_provider=api_request.llm_provider,
            model=api_request.model,
            current_state=ConversationState(api_request.current_state)
        )
        
        result = await ami_creator.collaborate_on_agent(collaboration_request)
        
        return CollaborativeAgentAPIResponse(
            success=result.success,
            conversation_id=result.conversation_id,
            current_state=result.current_state.value,
            ami_message=result.ami_message,
            data=result.data,
            next_actions=result.next_actions,
            error=result.error
        )
        
    except Exception as e:
        ami_logger.error(f"API collaborative agent creation failed: {e}")
        return CollaborativeAgentAPIResponse(
            success=False,
            conversation_id=api_request.conversation_id or "unknown",
            current_state=api_request.current_state,
            ami_message="Something went wrong. Let me help you start fresh!",
            error=str(e)
        )


"""
FastAPI Integration Examples:

# Add to your FastAPI routes (main.py or similar):

from ami import (
    CreateAgentAPIRequest, CreateAgentAPIResponse, create_agent_via_api,
    CollaborativeAgentAPIRequest, CollaborativeAgentAPIResponse, collaborate_on_agent_via_api
)

# APPROACH 1: Direct Agent Creation (Legacy - Simple but less precise)
@app.post("/ami/create-agent", response_model=CreateAgentAPIResponse)
async def create_agent_endpoint(
    request: CreateAgentAPIRequest,
    current_user: dict = Depends(get_current_user)
):
    '''Create a new AI agent directly - fast but basic'''
    
    from organization import get_my_organization
    org_response = await get_my_organization(current_user)
    
    result = await create_agent_via_api(
        api_request=request,
        org_id=org_response.id,
        user_id=current_user["id"]
    )
    return result

# APPROACH 2: Collaborative Agent Creation (NEW - Chief Product Officer approach)
@app.post("/ami/collaborate", response_model=CollaborativeAgentAPIResponse)
async def collaborate_agent_endpoint(
    request: CollaborativeAgentAPIRequest,
    current_user: dict = Depends(get_current_user)
):
    '''Collaborate with Ami like a Chief Product Officer to build precise agents'''
    
    from organization import get_my_organization
    org_response = await get_my_organization(current_user)
    
    result = await collaborate_on_agent_via_api(
        api_request=request,
        org_id=org_response.id,
        user_id=current_user["id"]
    )
    return result

# Usage Examples:

# COLLABORATIVE APPROACH (RECOMMENDED):

# Step 1: Start conversation with initial idea
POST /ami/collaborate
{
    "user_input": "I want a sales agent for my Vietnamese business",
    "current_state": "initial_idea"
}
# Response: Ami analyzes, asks questions, proposes skeleton plan

# Step 2: Review and provide feedback
POST /ami/collaborate  
{
    "user_input": "I like it but make it more technical and add email capabilities",
    "conversation_id": "uuid-from-step-1",
    "current_state": "skeleton_review"
}
# Response: Ami refines plan based on feedback

# Step 3: Approve the plan
POST /ami/collaborate
{
    "user_input": "Perfect! Build it!",
    "conversation_id": "uuid-from-step-1", 
    "current_state": "skeleton_review"
}
# Response: Ami builds the agent

# DIRECT APPROACH (Legacy):
POST /ami/create-agent
{
    "user_request": "Create a Vietnamese sales agent that can analyze documents"
}

Complete Collaborative Flow:
1. Human: "I want a sales agent" (initial_idea)
2. Ami: "Here's what I understand... questions?" (skeleton_review)
3. Human: "Change X, add Y" (skeleton_review)
4. Ami: "Updated plan with changes..." (skeleton_review)
5. Human: "Perfect! Build it!" (approved)
6. Ami: "Agent created! Ready to use!" (completed)

Benefits of Collaborative Approach:
- ✅ More precise agents that match exact needs
- ✅ Clear understanding of requirements before building
- ✅ Ability to refine and adjust before creation
- ✅ Better success criteria and capabilities
- ✅ Chief Product Officer experience
""" 