"""
Ami Orchestrator - Main Coordination Module
===========================================

Coordinates all Ami components for streamlined agent creation.
Following the same pattern as agent/orchestrator.py for consistency.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .models import (
    AgentCreationRequest, AgentCreationResult, 
    CollaborativeAgentRequest, CollaborativeAgentResponse,
    ConversationState, AgentSkeleton
)
from .knowledge_manager import AmiKnowledgeManager
from .collaborative_creator import CollaborativeCreator
from .direct_creator import DirectCreator

logger = logging.getLogger(__name__)

# Configure ami orchestrator logging
ami_logger = logging.getLogger("ami_orchestrator")
ami_logger.setLevel(logging.INFO)
if not ami_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('🤖 [AMI] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    ami_logger.addHandler(handler)


class AmiOrchestrator:
    """
    Main Ami orchestration engine with modular architecture
    Coordinates direct creation, collaborative creation, and knowledge management
    """
    
    def __init__(self):
        """Initialize the Ami orchestrator with modular components"""
        # Initialize shared infrastructure (same as agent system)
        try:
            from exec_anthropic import AnthropicExecutor
            from exec_openai import OpenAIExecutor
            self.anthropic_executor = AnthropicExecutor(self)
            self.openai_executor = OpenAIExecutor(self)
        except ImportError as e:
            ami_logger.warning(f"LLM executors import failed: {e}")
            self.anthropic_executor = None
            self.openai_executor = None
        
        # Initialize modular components
        self.knowledge_manager = AmiKnowledgeManager()
        self.collaborative_creator = CollaborativeCreator(
            self.anthropic_executor, self.openai_executor, self.knowledge_manager, self
        )
        self.direct_creator = DirectCreator(
            self.anthropic_executor, self.openai_executor
        )
        
        ami_logger.info("Ami Orchestrator initialized with modular architecture")
    
    async def create_agent(self, request: AgentCreationRequest) -> AgentCreationResult:
        """
        Direct agent creation - simple and fast (legacy approach)
        """
        ami_logger.info(f"Direct agent creation requested: '{request.user_request[:100]}...'")
        return await self.direct_creator.create_agent_direct(request)
    
    async def collaborate_on_agent(self, request: CollaborativeAgentRequest) -> CollaborativeAgentResponse:
        """
        Collaborative agent creation - Chief Product Officer approach
        """
        ami_logger.info(f"Collaborative session: {request.current_state.value}")
        
        # Handle collaborative request
        if request.current_state != ConversationState.APPROVED:
            return await self.collaborative_creator.handle_collaborative_request(request)
        
        # If approved, build the agent
        else:
            return await self._build_approved_agent(request)
    
    async def _build_approved_agent(self, request: CollaborativeAgentRequest) -> CollaborativeAgentResponse:
        """
        Build the final agent from approved skeleton
        Convert skeleton into actual agent configuration and save to database
        """
        ami_logger.info(f"Building approved agent for conversation: {request.conversation_id}")
        
        # Get conversation state
        conversation = self.collaborative_creator.get_conversation_data(request.conversation_id)
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
            # Convert skeleton to blueprint format
            blueprint_data = await self._convert_skeleton_to_blueprint(skeleton)
            
            # Create agent with blueprint using new architecture
            agent_id, blueprint_id = await self._create_agent_with_blueprint(
                name=skeleton.agent_name,
                description=skeleton.agent_purpose,
                blueprint_data=blueprint_data,
                org_id=conversation["org_id"],
                user_id=conversation["user_id"],
                conversation_id=request.conversation_id
            )
            
            # 🆕 STEP 7: Generate Implementation Todos based on blueprint analysis
            ami_logger.info("🔧 Analyzing blueprint and generating implementation todos...")
            todos_result = await self._generate_implementation_todos(
                blueprint_id=blueprint_id,
                user_id=conversation["user_id"]
            )
            
            # 🧠 Save agent expertise knowledge to Pinecone
            expertise_result = await self.knowledge_manager.save_agent_creation_knowledge(
                skeleton=skeleton,
                conversation_id=request.conversation_id,
                user_id=conversation["user_id"],
                org_id=conversation["org_id"]
            )
            
            # 🧠 Save collaborative session insights
            collaboration_result = await self.knowledge_manager.save_collaborative_insights(
                conversation_data=conversation,
                user_id=conversation["user_id"],
                org_id=conversation["org_id"]
            )
            
            # Log knowledge saving results
            ami_logger.info(f"Knowledge saved - Expertise: {expertise_result.get('success')}, "
                           f"Insights: {collaboration_result.get('success')}")
            
            # Update conversation state to completed
            self.collaborative_creator.update_conversation_state(
                request.conversation_id, 
                ConversationState.COMPLETED,
                final_agent_id=agent_id
            )
            
            ami_logger.info(f"Successfully created agent with blueprint and todos: {skeleton.agent_name} (ID: {agent_id})")
            
            return CollaborativeAgentResponse(
                success=True,
                conversation_id=request.conversation_id,
                current_state=ConversationState.COMPLETED,
                ami_message=f"🎉 Perfect! I've successfully created '{skeleton.agent_name}' with a comprehensive Agent Architecture!\n\n**What I've built for you:**\n✅ Agent Blueprint with detailed specifications\n✅ {todos_result.get('todos_generated', 0)} Implementation Todos for proper setup\n✅ Knowledge base integration\n\n**Next Steps:**\n1. Complete the implementation todos to ensure your agent works optimally\n2. Once todos are done, compile the blueprint to activate the agent\n3. Your agent will then be ready for production use!\n\nThe todos will guide you through setting up integrations, configurations, and testing to make sure '{skeleton.agent_name}' achieves its goals effectively.",
                data={
                    "agent_id": agent_id,
                    "blueprint_id": blueprint_id,
                    "agent_name": skeleton.agent_name,
                    "todos_generated": todos_result.get('todos_generated', 0),
                    "agent_architecture": {
                        "blueprint": {
                            "name": skeleton.agent_name,
                            "description": skeleton.agent_purpose,
                            "agent_type": skeleton.agent_type,
                            "language": skeleton.language,
                            "capabilities": [task.get('task', '') for task in skeleton.what_i_do.get('primary_tasks', [])],
                            "integrations": [integration.get('app_name', '') for integration in skeleton.integrations],
                            "knowledge_sources": [source.get('source', '') for source in skeleton.knowledge_sources]
                        },
                        "implementation_todos": todos_result.get('todos', [])
                    }
                },
                next_actions=[
                    f"View and complete implementation todos for '{skeleton.agent_name}'",
                    "Compile blueprint once todos are completed",
                    "Create another agent",
                    "Test the agent architecture"
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
        
        Agent Blueprint:
        - Name: {skeleton.agent_name}
        - Purpose: {skeleton.agent_purpose}
        - Target Users: {skeleton.target_users}
        - Primary Tasks: {[task.get('task', '') for task in skeleton.what_i_do.get('primary_tasks', [])]}
        - Agent Type: {skeleton.agent_type}
        - Language: {skeleton.language}
        - Personality: {skeleton.what_i_do.get('personality', {})}
        - Workflow Steps: {skeleton.workflow_steps}
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
            primary_tasks = [task.get('task', '') for task in skeleton.what_i_do.get('primary_tasks', [])]
            prompt_data = {
                "base_instruction": system_prompt_text,
                "agent_type": skeleton.agent_type,
                "language": skeleton.language,
                "specialization": primary_tasks,
                "personality": skeleton.what_i_do.get('personality', {}),
                "target_users": skeleton.target_users,
                "use_cases": primary_tasks,
                "success_criteria": skeleton.success_criteria,
                "created_at": datetime.now().isoformat(),
                "created_from": "collaborative_blueprint"
            }
            
            return prompt_data
            
        except Exception as e:
            ami_logger.error(f"System prompt generation from skeleton failed: {e}")
            # Fallback prompt structure
            primary_tasks = [task.get('task', '') for task in skeleton.what_i_do.get('primary_tasks', [])]
            personality = skeleton.what_i_do.get('personality', {})
            fallback_prompt = f"You are {skeleton.agent_name}. {skeleton.agent_purpose} You specialize in {', '.join(primary_tasks)}. Your target users are {skeleton.target_users}. Be {personality.get('tone', 'professional')} and {personality.get('style', 'helpful')}."
            
            return {
                "base_instruction": fallback_prompt,
                "agent_type": skeleton.agent_type,
                "language": skeleton.language,
                "specialization": primary_tasks,
                "personality": personality,
                "created_at": datetime.now().isoformat(),
                "created_from": "collaborative_blueprint_fallback"
            }
    
    async def _save_to_database(self, config: dict, org_id: str, user_id: str) -> str:
        """Save agent configuration to database via org_agent system"""
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
            ami_logger.error(f"LLM call failed: {e}")
            raise Exception(f"LLM call failed: {str(e)}")
    
    async def _convert_skeleton_to_blueprint(self, skeleton: AgentSkeleton) -> dict:
        """
        Convert complete 7-part agent blueprint to internal format for the new architecture
        """
        blueprint_data = {
            "identity": {
                "name": skeleton.agent_name,
                "role": skeleton.agent_purpose,
                "primary_purpose": skeleton.agent_purpose,
                "language": skeleton.language,
                "introduction": skeleton.meet_me.get("introduction", f"Hi, I'm {skeleton.agent_name}!"),
                "value_proposition": skeleton.meet_me.get("value_proposition", "I'm here to help you succeed.")
            },
            "capabilities": [
                {
                    "task": task.get("task", "General assistance"),
                    "description": task.get("description", "Handle tasks efficiently"),
                    "examples": []
                }
                for task in skeleton.what_i_do.get("primary_tasks", [])
            ],
            "personality": skeleton.what_i_do.get("personality", {
                "tone": "professional",
                "style": "helpful",
                "analogy": "helpful assistant"
            }),
            "sample_conversation": skeleton.what_i_do.get("sample_conversation", {}),
            "knowledge_sources": skeleton.knowledge_sources,
            "tools": [
                {
                    "name": integration.get("app_name", "Unknown Tool"),
                    "purpose": f"Integration with {integration.get('app_name', 'external service')}",
                    "trigger": integration.get("trigger", "When needed"),
                    "action": integration.get("action", "Perform action")
                }
                for integration in skeleton.integrations
            ],
            "monitoring": skeleton.monitoring,
            "test_scenarios": skeleton.test_scenarios,
            "workflow": {
                "steps": skeleton.workflow_steps,
                "visual_flow": skeleton.visual_flow
            },
            "success_criteria": skeleton.success_criteria,
            "potential_challenges": skeleton.potential_challenges,
            "agent_type": skeleton.agent_type,
            "target_users": skeleton.target_users
        }
        
        return blueprint_data
    
    async def _create_agent_with_blueprint(self, name: str, description: str, blueprint_data: dict, 
                                         org_id: str, user_id: str, conversation_id: str) -> tuple[str, str]:
        """
        Create agent with blueprint using the new architecture
        """
        try:
            from orgdb import create_agent_with_blueprint
            
            agent, blueprint = create_agent_with_blueprint(
                org_id=org_id,
                created_by=user_id,
                name=name,
                blueprint_data=blueprint_data,
                description=description,
                conversation_id=conversation_id
            )
            
            ami_logger.info(f"Agent with blueprint created: {agent.name} (Agent ID: {agent.id}, Blueprint ID: {blueprint.id})")
            return agent.id, blueprint.id
            
        except Exception as e:
            ami_logger.error(f"Failed to create agent with blueprint: {e}")
            raise Exception(f"Failed to create agent with blueprint: {str(e)}")
    
    async def _generate_implementation_todos(self, blueprint_id: str, user_id: str) -> dict:
        """
        Generate implementation todos for the blueprint using Ami's analysis
        """
        try:
            from orgdb import generate_implementation_todos
            
            updated_blueprint = generate_implementation_todos(blueprint_id, user_id)
            if not updated_blueprint:
                raise Exception("Failed to generate implementation todos")
            
            todos_count = len(updated_blueprint.implementation_todos)
            ami_logger.info(f"Generated {todos_count} implementation todos for blueprint {blueprint_id}")
            
            return {
                "success": True,
                "todos_generated": todos_count,
                "todos": updated_blueprint.implementation_todos,
                "blueprint_id": blueprint_id
            }
            
        except Exception as e:
            ami_logger.error(f"Failed to generate todos: {e}")
            return {
                "success": False,
                "todos_generated": 0,
                "todos": [],
                "error": str(e)
            }


# Global orchestrator instance (singleton pattern)
_ami_orchestrator_instance = None

def get_ami_orchestrator() -> AmiOrchestrator:
    """Get global Ami orchestrator instance (singleton pattern)"""
    global _ami_orchestrator_instance
    if _ami_orchestrator_instance is None:
        _ami_orchestrator_instance = AmiOrchestrator()
    return _ami_orchestrator_instance