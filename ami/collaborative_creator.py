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
    
    def __init__(self, anthropic_executor, openai_executor, knowledge_manager: AmiKnowledgeManager, orchestrator=None):
        """Initialize collaborative creator with required dependencies"""
        self.anthropic_executor = anthropic_executor
        self.openai_executor = openai_executor
        self.knowledge_manager = knowledge_manager
        self.orchestrator = orchestrator
        
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
        4. CREATE a complete human-friendly agent blueprint following the 7-part structure

        Human's Initial Idea: "{request.user_input}"

        Create a blueprint that follows the 7-part Agent Pitch structure:
        1. Meet Me: Personal introduction in agent's voice
        2. What I Can Do: Specific tasks with conversation examples
        3. Where I Get My Smarts: Specific data sources and update frequency
        4. Apps I Team Up With: Tool integrations with triggers/actions
        5. How I Keep You in the Loop: Monitoring and fallback behaviors
        6. Try Me Out: Test questions with expected responses
        7. How I Work: Step-by-step workflow process

        Respond with this EXACT JSON format:
        {{
            "understanding": "What I understand you want to achieve...",
            "clarifying_questions": [
                "Question 1 about their needs?",
                "Question 2 about usage scenarios?", 
                "Question 3 about constraints?"
            ],
            "agent_blueprint": {{
                "agent_name": "Descriptive Agent Name",
                "agent_purpose": "Clear purpose statement",
                "meet_me": {{
                    "introduction": "Hi, I'm [Agent Name]! My main job is [role]. I'm here to [benefit].",
                    "value_proposition": "Think of me as your [analogy] that [key benefit]."
                }},
                "what_i_do": {{
                    "primary_tasks": [
                        {{"task": "Task name", "description": "What I do for this task"}},
                        {{"task": "Task name", "description": "What I do for this task"}}
                    ],
                    "personality": {{
                        "tone": "professional|friendly|authoritative|consultative", 
                        "style": "concise|detailed|conversational|technical",
                        "analogy": "like a helpful pal, cheerful receptionist, etc."
                    }},
                    "sample_conversation": {{
                        "user_question": "Sample question a user might ask",
                        "agent_response": "How I would respond in my voice"
                    }}
                }},
                "knowledge_sources": [
                    {{
                        "source": "Specific source name (e.g., 'Customer Database', 'Product Catalog Google Sheet')",
                        "type": "spreadsheet|database|api|website",
                        "update_frequency": "hourly|daily|weekly|real-time",
                        "content_examples": ["Example data 1", "Example data 2"]
                    }}
                ],
                "integrations": [
                    {{
                        "app_name": "Specific app (e.g., Slack, Gmail)",
                        "trigger": "When this happens",
                        "action": "I do this specific action"
                    }}
                ],
                "monitoring": {{
                    "reporting_method": "How I report progress (e.g., weekly email, dashboard)",
                    "metrics_tracked": ["Metric 1", "Metric 2"],
                    "fallback_response": "What I say when unsure",
                    "escalation_method": "How I get help (e.g., email team, Slack alert)"
                }},
                "test_scenarios": [
                    {{
                        "question": "Test question users can ask",
                        "expected_response": "Exactly how I should respond"
                    }}
                ],
                "workflow_steps": [
                    "Step 1: What happens first",
                    "Step 2: What I do next", 
                    "Step 3: How I complete the task"
                ],
                "visual_flow": "Simple text description for workflow diagram",
                "target_users": "Who will use this agent",
                "agent_type": "sales|support|analyst|document_analysis|research|automation|general",
                "language": "english|vietnamese|french|spanish",
                "success_criteria": ["Success measure 1", "Success measure 2"],
                "potential_challenges": ["Challenge 1", "Challenge 2"]
            }},
            "ami_message": "Here's what I understand... [natural explanation of the plan and questions]"
        }}

        Make the blueprint human-friendly, avoid technical jargon, and ensure it reads like the agent is introducing itself to a human user.
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
                
                # Create agent blueprint object
                blueprint_data = data.get("agent_blueprint", {})
                skeleton = AgentSkeleton(
                    conversation_id=request.conversation_id,
                    agent_name=blueprint_data.get("agent_name", "Custom Agent"),
                    agent_purpose=blueprint_data.get("agent_purpose", "Specialized AI assistant"),
                    target_users=blueprint_data.get("target_users", "General users"),
                    agent_type=blueprint_data.get("agent_type", "general"),
                    language=blueprint_data.get("language", "english"),
                    meet_me=blueprint_data.get("meet_me", {
                        "introduction": f"Hi, I'm {blueprint_data.get('agent_name', 'Custom Agent')}!",
                        "value_proposition": "I'm here to help you succeed."
                    }),
                    what_i_do=blueprint_data.get("what_i_do", {
                        "primary_tasks": [],
                        "personality": {"tone": "professional", "style": "helpful"},
                        "sample_conversation": {"user_question": "", "agent_response": ""}
                    }),
                    knowledge_sources=blueprint_data.get("knowledge_sources", []),
                    integrations=blueprint_data.get("integrations", []),
                    monitoring=blueprint_data.get("monitoring", {
                        "reporting_method": "Weekly updates",
                        "metrics_tracked": ["User satisfaction"],
                        "fallback_response": "Let me help you with that",
                        "escalation_method": "I'll get help from the team"
                    }),
                    test_scenarios=blueprint_data.get("test_scenarios", []),
                    workflow_steps=blueprint_data.get("workflow_steps", []),
                    visual_flow=blueprint_data.get("visual_flow", "User asks → I analyze → I respond"),
                    success_criteria=blueprint_data.get("success_criteria", ["User satisfaction"]),
                    potential_challenges=blueprint_data.get("potential_challenges", ["None identified"]),
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
                        "agent_blueprint": {
                            "agent_name": skeleton.agent_name,
                            "agent_purpose": skeleton.agent_purpose,
                            "target_users": skeleton.target_users,
                            "agent_type": skeleton.agent_type,
                            "language": skeleton.language,
                            "meet_me": skeleton.meet_me,
                            "what_i_do": skeleton.what_i_do,
                            "knowledge_sources": skeleton.knowledge_sources,
                            "integrations": skeleton.integrations,
                            "monitoring": skeleton.monitoring,
                            "test_scenarios": skeleton.test_scenarios,
                            "workflow_steps": skeleton.workflow_steps,
                            "visual_flow": skeleton.visual_flow,
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
            collab_logger.error(f"Conversation not found for ID: {request.conversation_id}")
            return CollaborativeAgentResponse(
                success=False,
                conversation_id=request.conversation_id,
                current_state=ConversationState.INITIAL_IDEA,
                ami_message="I lost track of our conversation. Let's start fresh!",
                error="Conversation state not found",
                next_actions=["Start a new agent creation conversation"]
            )
        
        collab_logger.info(f"Found conversation for {request.conversation_id}")
        skeleton = conversation["skeleton"]
        collab_logger.info(f"Got skeleton: {skeleton.agent_name if skeleton else 'None'}")
        
        # Check if this is approval (smarter contextual detection)
        user_input_lower = request.user_input.lower().strip()
        
        # Explicit approval keywords (strong signals)
        strong_approval_keywords = [
            "approve", "approved", "looks good", "build it", "proceed", "perfect", "go ahead", "create it",
            "build đi", "xây dựng đi", "làm đi", "tiến hành", "đồng ý", "được", "hoàn hảo",
            "build nó đi", "tạo đi", "thực hiện đi"
        ]
        
        # Simple affirmatives (only count if message is short and direct)
        simple_affirmatives = ["yes", "ok", "được", "tốt"]
        
        # Check for strong approval keywords
        has_strong_approval = any(keyword in user_input_lower for keyword in strong_approval_keywords)
        
        # Check for simple affirmatives (only if message is short and doesn't contain additional requirements)
        has_simple_approval = False
        if any(keyword in user_input_lower for keyword in simple_affirmatives):
            # Only count as approval if:
            # 1. Message is short (< 50 characters)
            # 2. Doesn't contain requirement indicators
            requirement_indicators = ["cần", "phải", "thêm", "need", "should", "add", "also", "và", "then", "sau khi"]
            message_is_short = len(request.user_input) < 50
            has_requirements = any(indicator in user_input_lower for indicator in requirement_indicators)
            
            has_simple_approval = message_is_short and not has_requirements
        
        is_approval = has_strong_approval or has_simple_approval
        
        collab_logger.info(f"Checking approval for input: '{request.user_input}' → is_approval: {is_approval}")
        
        if is_approval:
            # Update state to approved
            conversation["state"] = ConversationState.APPROVED
            self.conversations[request.conversation_id] = conversation
            
            collab_logger.info(f"Agent approved! Triggering build process for '{skeleton.agent_name}'")
            
            # Create an approved request to trigger the actual building
            approved_request = CollaborativeAgentRequest(
                user_input=request.user_input,
                conversation_id=request.conversation_id,
                org_id=request.org_id,
                user_id=request.user_id,
                llm_provider=request.llm_provider,
                model=request.model,
                current_state=ConversationState.APPROVED
            )
            
            # Trigger the actual agent building process using the orchestrator instance
            if self.orchestrator:
                return await self.orchestrator._build_approved_agent(approved_request)
            else:
                # Fallback if orchestrator not available
                collab_logger.error("Orchestrator not available for building agent")
                return CollaborativeAgentResponse(
                    success=False,
                    conversation_id=request.conversation_id,
                    current_state=ConversationState.APPROVED,
                    ami_message="I had trouble building the agent. Please try again.",
                    error="Orchestrator not available",
                    next_actions=["Try again", "Start a new conversation"]
                )
        
        # Handle refinement request
        refinement_prompt = f"""
        You are Ami, a Chief Product Officer. The human is providing feedback on an agent blueprint. Your job is to:

        1. UNDERSTAND their feedback and concerns
        2. REFINE the complete 7-part agent blueprint based on their input
        3. TRACK DETAILED CHANGES for collaborative review
        4. EXPLAIN the changes you're making with specific field-level tracking
        5. ASK if they need any other adjustments

        Original Agent Blueprint:
        - Name: {skeleton.agent_name}
        - Purpose: {skeleton.agent_purpose}
        - Meet Me: {skeleton.meet_me}
        - What I Do: {skeleton.what_i_do}
        - Knowledge Sources: {skeleton.knowledge_sources}
        - Integrations: {skeleton.integrations}
        - Test Scenarios: {skeleton.test_scenarios}
        - Workflow: {skeleton.workflow_steps}

        Human Feedback: "{request.user_input}"

        Respond with this EXACT JSON format with DETAILED change tracking:
        {{
            "feedback_understanding": "What I understand from your feedback...",
            "tracked_changes": {{
                "summary": {{
                    "total_changes": 2,
                    "modified_sections": ["integrations", "knowledge_sources"]
                }},
                "changes": [
                    {{
                        "change_id": "change_1",
                        "type": "addition|modification|deletion",
                        "section": "integrations|knowledge_sources|what_i_do|meet_me|test_scenarios|workflow_steps|monitoring",
                        "field_path": "specific.field.path (e.g., integrations[0].action)",
                        "change_description": "Human-readable description of what changed",
                        "before": "previous value or null if addition",
                        "after": "new value or null if deletion",
                        "reasoning": "Why this change was made based on user feedback"
                    }}
                ]
            }},
            "blueprint_diff": {{
                "previous_version": {{
                    "note": "Include ONLY the sections that changed, with their original values"
                }},
                "updated_version": {{
                    "note": "Include ONLY the sections that changed, with their new values"
                }}
            }},
            "updated_blueprint": {{
                "agent_name": "Updated Agent Name",
                "agent_purpose": "Updated purpose statement",
                "target_users": "Updated target users",
                "agent_type": "sales|support|analyst|document_analysis|research|automation|general",
                "language": "english|vietnamese|french|spanish",
                "meet_me": {{
                    "introduction": "Hi, I'm [Agent Name]! My main job is [role]. I'm here to [benefit].",
                    "value_proposition": "Think of me as your [analogy] that [key benefit]."
                }},
                "what_i_do": {{
                    "primary_tasks": [
                        {{"task": "Task name", "description": "What I do for this task"}}
                    ],
                    "personality": {{
                        "tone": "professional|friendly|authoritative|consultative", 
                        "style": "concise|detailed|conversational|technical",
                        "analogy": "like a helpful pal, cheerful receptionist, etc."
                    }},
                    "sample_conversation": {{
                        "user_question": "Sample question a user might ask",
                        "agent_response": "How I would respond in my voice"
                    }}
                }},
                "knowledge_sources": [
                    {{
                        "source": "Specific source name",
                        "type": "spreadsheet|database|api|website",
                        "update_frequency": "hourly|daily|weekly|real-time",
                        "content_examples": ["Example data 1", "Example data 2"]
                    }}
                ],
                "integrations": [
                    {{
                        "app_name": "Specific app name",
                        "trigger": "When this happens",
                        "action": "I do this specific action"
                    }}
                ],
                "monitoring": {{
                    "reporting_method": "How I report progress",
                    "metrics_tracked": ["Metric 1", "Metric 2"],
                    "fallback_response": "What I say when unsure",
                    "escalation_method": "How I get help"
                }},
                "test_scenarios": [
                    {{
                        "question": "Test question users can ask",
                        "expected_response": "Exactly how I should respond"
                    }}
                ],
                "workflow_steps": ["Step 1", "Step 2", "Step 3"],
                "visual_flow": "Simple workflow description",
                "success_criteria": ["Success measure 1", "Success measure 2"],
                "potential_challenges": ["Challenge 1", "Challenge 2"]
            }},
            "ui_hints": {{
                "highlight_sections": ["list of sections that changed"],
                "changed_fields": ["list of specific field paths that changed"],
                "animation_sequence": ["order of sections to animate/highlight"]
            }},
            "ami_message": "I've updated the blueprint based on your feedback... [explanation of changes and questions]"
        }}

        IMPORTANT: Be very specific about changes. Compare the original blueprint with your updates and track every single modification in the tracked_changes array. Include the exact before/after values and clear reasoning for each change.
        """
        
        try:
            refinement_response = await self._call_llm(refinement_prompt, request.llm_provider)
            
            # Parse the refinement response with robust JSON extraction
            data = self._extract_and_parse_json(refinement_response, "skeleton_refinement")
            if not data:
                raise ValueError("Failed to extract valid JSON from LLM response")
            
            collab_logger.info(f"Successfully extracted JSON with keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                
            # Update blueprint with refined data
            updated_blueprint_data = data.get("updated_blueprint", {})
            collab_logger.info(f"Creating refined skeleton with data keys: {list(updated_blueprint_data.keys())}")
            
            refined_skeleton = AgentSkeleton(
                conversation_id=request.conversation_id,
                agent_name=updated_blueprint_data.get("agent_name", skeleton.agent_name),
                agent_purpose=updated_blueprint_data.get("agent_purpose", skeleton.agent_purpose),
                target_users=updated_blueprint_data.get("target_users", skeleton.target_users),
                agent_type=updated_blueprint_data.get("agent_type", skeleton.agent_type),
                language=updated_blueprint_data.get("language", skeleton.language),
                meet_me=updated_blueprint_data.get("meet_me", skeleton.meet_me),
                what_i_do=updated_blueprint_data.get("what_i_do", skeleton.what_i_do),
                knowledge_sources=updated_blueprint_data.get("knowledge_sources", skeleton.knowledge_sources),
                integrations=updated_blueprint_data.get("integrations", skeleton.integrations),
                monitoring=updated_blueprint_data.get("monitoring", skeleton.monitoring),
                test_scenarios=updated_blueprint_data.get("test_scenarios", skeleton.test_scenarios),
                workflow_steps=updated_blueprint_data.get("workflow_steps", skeleton.workflow_steps),
                visual_flow=updated_blueprint_data.get("visual_flow", skeleton.visual_flow),
                success_criteria=updated_blueprint_data.get("success_criteria", skeleton.success_criteria),
                potential_challenges=updated_blueprint_data.get("potential_challenges", skeleton.potential_challenges),
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
                    # ✅ Enhanced change tracking
                    "tracked_changes": data.get("tracked_changes", {
                        "summary": {"total_changes": 0, "modified_sections": []},
                        "changes": []
                    }),
                    "blueprint_diff": data.get("blueprint_diff", {
                        "previous_version": {},
                        "updated_version": {}
                    }),
                    "ui_hints": data.get("ui_hints", {
                        "highlight_sections": [],
                        "changed_fields": [],
                        "animation_sequence": []
                    }),
                    # ✅ Legacy support (backwards compatible)
                    "changes_made": data.get("changes_made", []),
                    "updated_blueprint": {
                        "agent_name": refined_skeleton.agent_name,
                        "agent_purpose": refined_skeleton.agent_purpose,
                        "target_users": refined_skeleton.target_users,
                        "agent_type": refined_skeleton.agent_type,
                        "language": refined_skeleton.language,
                        "meet_me": refined_skeleton.meet_me,
                        "what_i_do": refined_skeleton.what_i_do,
                        "knowledge_sources": refined_skeleton.knowledge_sources,
                        "integrations": refined_skeleton.integrations,
                        "monitoring": refined_skeleton.monitoring,
                        "test_scenarios": refined_skeleton.test_scenarios,
                        "workflow_steps": refined_skeleton.workflow_steps,
                        "visual_flow": refined_skeleton.visual_flow,
                        "success_criteria": refined_skeleton.success_criteria,
                        "potential_challenges": refined_skeleton.potential_challenges
                    }
                },
                next_actions=[
                    "Approve all changes",
                    "Approve individual changes",
                    "Request further changes",
                    "Revert specific changes",
                    "Ask questions about the updates"
                ]
            )
                
        except Exception as e:
            collab_logger.error(f"Skeleton refinement failed: {e}")
            collab_logger.error(f"Error type: {type(e).__name__}")
            collab_logger.error(f"Raw LLM response: {refinement_response[:500] if 'refinement_response' in locals() else 'No response received'}")
            import traceback
            collab_logger.error(f"Full traceback: {traceback.format_exc()}")
        
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
                    max_tokens=3000,
                    temperature=0.7
                )
                return response.content[0].text
            
            elif provider == "openai" and self.openai_executor:
                response = await self.openai_executor.call_openai_direct(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            else:
                raise Exception(f"LLM provider {provider} not available or not configured")
                
        except Exception as e:
            collab_logger.error(f"LLM call failed: {e}")
            raise Exception(f"LLM call failed: {str(e)}")
    
    def _extract_and_parse_json(self, response: str, context: str = "unknown") -> Optional[Dict[str, Any]]:
        """
        Robust JSON extraction and parsing from LLM responses
        """
        import re
        import json
        
        collab_logger.info(f"[{context}] Starting JSON extraction from response length: {len(response)}")
        collab_logger.info(f"[{context}] Response preview: {response[:200]}...")
        
        try:
            # Method 1: Try to find JSON block with ```json markers
            json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_block_match:
                json_text = json_block_match.group(1)
                collab_logger.info(f"[{context}] Found JSON block, attempting to parse...")
                collab_logger.info(f"[{context}] JSON text: {json_text[:300]}...")
                return json.loads(json_text)
            
            # Method 2: Find JSON object by balancing braces
            json_text = self._extract_balanced_json(response)
            if json_text:
                collab_logger.info(f"[{context}] Found balanced JSON, attempting to parse...")
                collab_logger.info(f"[{context}] JSON text: {json_text[:300]}...")
                return json.loads(json_text)
            
            # Method 3: Try regex extraction (fallback)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                # Clean up common JSON issues
                json_text = self._clean_json_text(json_text)
                collab_logger.info(f"[{context}] Found regex JSON, attempting to parse...")
                collab_logger.info(f"[{context}] JSON text: {json_text[:300]}...")
                return json.loads(json_text)
            
            collab_logger.error(f"[{context}] No JSON found in response")
            return None
            
        except json.JSONDecodeError as e:
            collab_logger.error(f"[{context}] JSON decode error: {e}")
            collab_logger.error(f"[{context}] Problematic JSON: {json_text[:200] if 'json_text' in locals() else 'N/A'}...")
            return None
        except Exception as e:
            collab_logger.error(f"[{context}] JSON extraction error: {e}")
            return None
    
    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Extract JSON by balancing braces"""
        try:
            start_idx = text.find('{')
            if start_idx == -1:
                return None
            
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text[start_idx:], start_idx):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return text[start_idx:i+1]
            
            return None
        except Exception:
            return None
    
    def _clean_json_text(self, json_text: str) -> str:
        """Clean common JSON formatting issues"""
        try:
            # Remove trailing commas before closing braces/brackets
            json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
            
            # Fix unescaped quotes in strings (basic attempt)
            # This is a simplified approach - more complex cases might need additional handling
            json_text = re.sub(r'(?<!\\)"(?![,}\]:])(?![^"]*"[,}\]:])', r'\\"', json_text)
            
            return json_text
        except Exception:
            return json_text