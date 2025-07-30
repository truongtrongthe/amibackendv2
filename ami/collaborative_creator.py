"""
Ami Collaborative Creator - Chief Product Officer Approach
=========================================================

Handles the collaborative agent creation process with iterative refinement.
Acts like a Chief Product Officer guiding humans through agent creation.
"""

import json
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from .models import (
    ConversationState, AgentSkeleton, CollaborativeAgentRequest, 
    CollaborativeAgentResponse
)
from .knowledge_manager import AmiKnowledgeManager

logger = logging.getLogger(__name__)

# Configure collaborative session logging
collab_logger = logging.getLogger("ami_collaborative")
collab_logger.setLevel(logging.INFO)
if not collab_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('🤝 [COLLAB] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    collab_logger.addHandler(handler)


class CollaborativeCreator:
    """
    Handles collaborative agent creation with Chief Product Officer approach
    Guides humans through iterative refinement before building agents
    """
    
    def __init__(self, anthropic_executor, openai_executor, knowledge_manager: AmiKnowledgeManager):
        """Initialize collaborative creator with required dependencies"""
        self.anthropic_executor = anthropic_executor
        self.openai_executor = openai_executor
        self.knowledge_manager = knowledge_manager
        
        # Conversation state storage (in production, use Redis or database)
        self.conversations: Dict[str, Dict[str, Any]] = {}
        
        collab_logger.info("Collaborative Creator initialized")
    
    async def handle_collaborative_request(self, request: CollaborativeAgentRequest) -> CollaborativeAgentResponse:
        """
        Main collaborative method - acts like a Chief Product Officer
        Guides the human through agent creation with iterative refinement
        """
        collab_logger.info(f"Collaborative session: {request.current_state.value} - '{request.user_input[:100]}...'")
        
        try:
            # Generate conversation ID if new conversation
            if not request.conversation_id:
                request.conversation_id = str(uuid4())
                collab_logger.info(f"Started new collaborative session: {request.conversation_id}")
            
            # Route to appropriate handler based on current state
            if request.current_state == ConversationState.INITIAL_IDEA:
                return await self._understand_and_refine_idea(request)
            
            elif request.current_state == ConversationState.SKELETON_REVIEW:
                return await self._handle_skeleton_feedback(request)
            
            elif request.current_state == ConversationState.APPROVED:
                # This will be handled by the orchestrator
                return CollaborativeAgentResponse(
                    success=True,
                    conversation_id=request.conversation_id,
                    current_state=ConversationState.APPROVED,
                    ami_message="Ready to build! Please proceed with agent creation.",
                    data={"ready_for_build": True},
                    next_actions=["Build the approved agent"]
                )
            
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
            collab_logger.error(f"Collaborative session error: {e}")
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
        collab_logger.info(f"Understanding idea: '{request.user_input}'")
        
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
            
            # 🧠 NEW: Detect and save user domain knowledge during understanding phase
            if self.knowledge_manager.contains_domain_knowledge(request.user_input):
                domain_knowledge_result = await self.knowledge_manager.save_user_domain_knowledge(
                    user_input=request.user_input,
                    agent_context="Agent Planning Session",
                    user_id=request.user_id,
                    org_id=request.org_id
                )
                collab_logger.info(f"Domain knowledge detected and saved: {domain_knowledge_result.get('success')}")
            
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
                
                collab_logger.info(f"Created skeleton for {skeleton.agent_name}")
                
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
            collab_logger.error(f"Idea understanding failed: {e}")
        
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
        collab_logger.info(f"Handling skeleton feedback: '{request.user_input}'")
        
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
                
                collab_logger.info(f"Refined skeleton for {refined_skeleton.agent_name}")
                
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
            collab_logger.error(f"Skeleton refinement failed: {e}")
        
        # Fallback for refinement issues
        return CollaborativeAgentResponse(
            success=False,
            conversation_id=request.conversation_id,
            current_state=ConversationState.SKELETON_REVIEW,
            ami_message="I had trouble processing your feedback. Could you tell me specifically what you'd like me to change about the agent plan?",
            error="Failed to process refinement request",
            next_actions=["Be more specific about what to change", "Approve the current plan"]
        )
    
    def get_conversation_data(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation data for building phase"""
        return self.conversations.get(conversation_id)
    
    def update_conversation_state(self, conversation_id: str, state: ConversationState, **updates):
        """Update conversation state"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["state"] = state
            self.conversations[conversation_id].update(updates)
    
    async def _call_llm(self, prompt: str, provider: str = "anthropic") -> str:
        """Call LLM through executors"""
        if not self.anthropic_executor and not self.openai_executor:
            raise Exception("No LLM executors available")
            
        try:
            if provider == "anthropic" and self.anthropic_executor:
                response = await self.anthropic_executor.call_anthropic_direct(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.7
                )
                return response.content[0].text
            
            elif provider == "openai" and self.openai_executor:
                response = await self.openai_executor.call_openai_direct(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            else:
                raise Exception(f"LLM provider {provider} not available or not configured")
                
        except Exception as e:
            collab_logger.error(f"LLM call failed: {e}")
            raise Exception(f"LLM call failed: {str(e)}")