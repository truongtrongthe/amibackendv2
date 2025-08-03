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
        
        collab_logger.info("Collaborative Creator initialized - now works with database records")
    
    async def handle_collaborative_request(self, request: CollaborativeAgentRequest) -> CollaborativeAgentResponse:
        """
        Main collaborative method - now works directly with database blueprint records
        Guides the human through agent blueprint refinement with iterative improvement
        """
        collab_logger.info(f"Collaborative session on agent {request.agent_id}: {request.current_state.value} - '{request.user_input[:100]}...'")
        
        try:
            # Validate that agent and blueprint exist
            from orgdb import get_agent, get_blueprint
            agent = get_agent(request.agent_id)
            blueprint = get_blueprint(request.blueprint_id)
            
            if not agent:
                return CollaborativeAgentResponse(
                    success=False,
                    conversation_id=request.conversation_id or "unknown",
                    current_state=request.current_state,
                    ami_message="I can't find the agent you're referring to. Please check the agent ID.",
                    error="Agent not found"
                )
            
            if not blueprint:
                return CollaborativeAgentResponse(
                    success=False,
                    conversation_id=request.conversation_id or "unknown", 
                    current_state=request.current_state,
                    ami_message="I can't find the blueprint for this agent. Please check the blueprint ID.",
                    error="Blueprint not found"
                )
            
            # Ensure the blueprint belongs to the agent
            if blueprint.agent_id != agent.id:
                return CollaborativeAgentResponse(
                    success=False,
                    conversation_id=request.conversation_id or "unknown",
                    current_state=request.current_state,
                    ami_message="The blueprint doesn't belong to this agent. Please check your IDs.",
                    error="Blueprint-agent mismatch"
                )
            
            # Generate conversation ID if new conversation
            if not request.conversation_id:
                request.conversation_id = str(uuid4())
                collab_logger.info(f"Started new collaborative session: {request.conversation_id}")
            
            # Route to appropriate handler - now always refinement since agent exists
            return await self._handle_blueprint_refinement(request, agent, blueprint)
                
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
    
    async def _handle_blueprint_refinement(self, request: CollaborativeAgentRequest, agent, blueprint) -> CollaborativeAgentResponse:
        """
        New unified method for handling blueprint refinement
        Works directly with database blueprint records - now with rich context awareness
        """
        collab_logger.info(f"Refining blueprint {blueprint.id} for agent {agent.name}")
        
        # Create rich context for AMI responses
        context = self._build_rich_context(agent, blueprint)
        
        # Get conversation history for this agent/blueprint
        conversation_history = await self._get_conversation_history(agent.id, blueprint.id)
        
        # Check if this is an approval
        is_approval = self._detect_approval(request.user_input)
        collab_logger.info(f"Checking approval for input: '{request.user_input}' → is_approval: {is_approval}")
        
        if is_approval:
            # Handle approval - trigger compilation
            return await self._handle_blueprint_approval(request, agent, blueprint, context, conversation_history)
        else:
            # Handle refinement request
            return await self._refine_blueprint_with_feedback(request, agent, blueprint, context, conversation_history)
    
    def _detect_approval(self, user_input: str) -> bool:
        """Enhanced approval detection with better logic"""
        user_input_lower = user_input.lower().strip()
        
        # Explicit approval keywords (strong signals)
        strong_approval_keywords = [
            'approve', 'approved', 'looks good', 'build it', 'proceed', 'perfect', 'go ahead', 'create it',
            'compile', 'activate', 'ready', 'done', 'finalize', 'complete',
            'build đi', 'xây dựng đi', 'làm đi', 'tiến hành', 'đồng ý', 'được', 'hoàn hảo',
            'build nó đi', 'tạo đi', 'thực hiện đi', 'compile đi', 'hoàn thành'
        ]
        
        # Simple affirmatives (only count if message is short and direct)
        simple_affirmatives = ['yes', 'ok', 'được', 'tốt', 'good']
        
        # Check for strong approval keywords
        has_strong_approval = any(keyword in user_input_lower for keyword in strong_approval_keywords)
        
        # Check for simple affirmatives (only if message is short and doesn't contain additional requirements)
        has_simple_approval = False
        if any(keyword in user_input_lower for keyword in simple_affirmatives):
            # Only count as approval if:
            # 1. Message is short (< 50 characters)
            # 2. Doesn't contain requirement indicators
            requirement_indicators = ['cần', 'phải', 'thêm', 'need', 'should', 'add', 'also', 'và', 'then', 'sau khi', 'but', 'however', 'nhưng']
            message_is_short = len(user_input) < 50
            has_requirements = any(indicator in user_input_lower for indicator in requirement_indicators)
            
            has_simple_approval = message_is_short and not has_requirements
        
        return has_strong_approval or has_simple_approval
    
    def _build_rich_context(self, agent, blueprint) -> dict:
        """
        Build rich context for AMI responses - similar to how Cursor understands source code
        This makes AMI fully context-aware of the agent and blueprint being worked on
        """
        # Get blueprint data
        blueprint_data = blueprint.agent_blueprint
        
        # Extract key information
        agent_name = blueprint_data.get("agent_name", agent.name)
        agent_purpose = blueprint_data.get("agent_purpose", agent.description)
        agent_type = blueprint_data.get("agent_type", "general")
        language = blueprint_data.get("language", "english")
        
        # Get current capabilities
        primary_tasks = blueprint_data.get("what_i_do", {}).get("primary_tasks", [])
        personality = blueprint_data.get("what_i_do", {}).get("personality", {})
        knowledge_sources = blueprint_data.get("knowledge_sources", [])
        integrations = blueprint_data.get("integrations", [])
        
        # Get current challenges and gaps
        potential_challenges = blueprint_data.get("potential_challenges", [])
        success_criteria = blueprint_data.get("success_criteria", [])
        
        # Build comprehensive context
        context = {
            "agent_identity": {
                "name": agent_name,
                "purpose": agent_purpose,
                "type": agent_type,
                "language": language,
                "personality": personality
            },
            "current_capabilities": {
                "tasks": primary_tasks,
                "task_count": len(primary_tasks),
                "knowledge_sources": knowledge_sources,
                "knowledge_count": len(knowledge_sources),
                "integrations": integrations,
                "integration_count": len(integrations)
            },
            "blueprint_status": {
                "compilation_status": blueprint.compilation_status,
                "has_system_prompt": bool(blueprint.compiled_system_prompt),
                "version": blueprint.version,
                "created_date": blueprint.created_date.strftime("%Y-%m-%d")
            },
            "development_context": {
                "challenges": potential_challenges,
                "success_criteria": success_criteria,
                "completeness_score": self._calculate_blueprint_completeness(blueprint_data)
            },
            "conversation_hints": {
                "agent_reference": f"your agent '{agent_name}'",
                "capability_summary": self._summarize_capabilities(primary_tasks),
                "next_logical_improvements": self._suggest_improvements(blueprint_data)
            }
        }
        
        return context
    
    def _calculate_blueprint_completeness(self, blueprint_data: dict) -> float:
        """Calculate how complete the blueprint is (0-100%)"""
        required_fields = [
            "agent_name", "agent_purpose", "target_users", "what_i_do", 
            "knowledge_sources", "integrations", "monitoring", "test_scenarios"
        ]
        
        completed = 0
        for field in required_fields:
            if field in blueprint_data and blueprint_data[field]:
                if isinstance(blueprint_data[field], list) and len(blueprint_data[field]) > 0:
                    completed += 1
                elif isinstance(blueprint_data[field], dict) and blueprint_data[field]:
                    completed += 1
                elif isinstance(blueprint_data[field], str) and blueprint_data[field].strip():
                    completed += 1
        
        return (completed / len(required_fields)) * 100
    
    def _summarize_capabilities(self, tasks: list) -> str:
        """Create a brief summary of current capabilities"""
        if not tasks:
            return "basic capabilities"
        
        if len(tasks) == 1:
            return f"focuses on {tasks[0].get('task', 'general tasks').lower()}"
        elif len(tasks) <= 3:
            task_names = [task.get('task', 'task').lower() for task in tasks[:3]]
            return f"handles {', '.join(task_names[:-1])} and {task_names[-1]}"
        else:
            return f"handles {len(tasks)} different capabilities including {tasks[0].get('task', 'various tasks').lower()}"
    
    def _suggest_improvements(self, blueprint_data: dict) -> list:
        """Suggest logical next improvements based on current state"""
        suggestions = []
        
        # Check for missing integrations
        integrations = blueprint_data.get("integrations", [])
        if len(integrations) == 0:
            suggestions.append("add integrations to external tools")
        
        # Check for vague knowledge sources
        knowledge_sources = blueprint_data.get("knowledge_sources", [])
        if len(knowledge_sources) == 0:
            suggestions.append("specify knowledge sources and data")
        
        # Check for basic personality
        personality = blueprint_data.get("what_i_do", {}).get("personality", {})
        if not personality or personality.get("tone") == "professional":
            suggestions.append("enhance personality and communication style")
        
        # Check for test scenarios
        test_scenarios = blueprint_data.get("test_scenarios", [])
        if len(test_scenarios) == 0:
            suggestions.append("add test scenarios for quality assurance")
        
        # Check for monitoring
        monitoring = blueprint_data.get("monitoring", {})
        if not monitoring or not monitoring.get("metrics_tracked"):
            suggestions.append("define monitoring and success metrics")
        
        return suggestions[:3]  # Return max 3 suggestions
    
    async def _get_conversation_history(self, agent_id: str, blueprint_id: str) -> list:
        """
        Retrieve conversation history for this agent/blueprint collaboration
        This maintains conversational context across AMI interactions
        """
        try:
            # Import here to avoid circular imports
            from orgdb import get_agent_collaboration_history
            
            # Get recent conversation history (last 20 messages to keep context manageable)
            history = get_agent_collaboration_history(agent_id, blueprint_id, limit=20)
            
            if not history:
                return []
                
            # Format history for LLM context
            formatted_history = []
            for entry in history:
                formatted_history.append({
                    "timestamp": entry.get("created_at", ""),
                    "type": entry.get("message_type", "user"),  # "user" or "ami"
                    "message": entry.get("message_content", ""),
                    "context": entry.get("context_data", {})
                })
            
            collab_logger.info(f"Retrieved {len(formatted_history)} conversation history entries for agent {agent_id}")
            return formatted_history
            
        except Exception as e:
            collab_logger.warning(f"Could not retrieve conversation history: {e}")
            return []
    
    def _format_conversation_history_for_llm(self, conversation_history: list, agent_name: str) -> str:
        """Format conversation history for inclusion in LLM prompts"""
        if not conversation_history:
            return "**CONVERSATION HISTORY:** This is the start of our collaboration on this agent."
        
        history_text = f"**CONVERSATION HISTORY for {agent_name}:**\n"
        for entry in conversation_history[-10:]:  # Use last 10 messages for context
            timestamp = entry.get("timestamp", "")
            message_type = entry.get("type", "user")
            message = entry.get("message", "")
            
            if message_type == "user":
                history_text += f"👤 **Human** ({timestamp}): {message}\n"
            else:
                history_text += f"🤖 **AMI** ({timestamp}): {message[:200]}{'...' if len(message) > 200 else ''}\n"
        
        history_text += "\n**CURRENT CONVERSATION:**\n"
        return history_text
    
    async def _save_conversation_message(self, agent_id: str, blueprint_id: str, message_type: str, message_content: str, context_data: dict = None):
        """Save conversation message to maintain history"""
        try:
            from orgdb import save_agent_collaboration_message
            
            save_agent_collaboration_message(
                agent_id=agent_id,
                blueprint_id=blueprint_id,
                message_type=message_type,  # "user" or "ami"
                message_content=message_content,
                context_data=context_data or {}
            )
            
        except Exception as e:
            collab_logger.warning(f"Could not save conversation message: {e}")
    
    async def _handle_blueprint_approval(self, request: CollaborativeAgentRequest, agent, blueprint, context, conversation_history) -> CollaborativeAgentResponse:
        """Handle blueprint approval and trigger compilation"""
        collab_logger.info(f"Blueprint approved! Triggering compilation for {agent.name}")
        
        # Save user's approval message to conversation history
        await self._save_conversation_message(
            agent_id=agent.id,
            blueprint_id=blueprint.id,
            message_type="user",
            message_content=request.user_input,
            context_data={"action": "approval", "current_state": "refinement"}
        )
        
        # Update blueprint conversation_id if needed
        if blueprint.conversation_id != request.conversation_id and request.conversation_id:
            from orgdb import update_blueprint_conversation_id
            update_blueprint_conversation_id(blueprint.id, request.conversation_id)
        
        # Trigger compilation directly (no need to wait for todos in this simplified flow)
        try:
            from orgdb import compile_blueprint
            compiled_blueprint = compile_blueprint(blueprint.id, request.user_id)
            
            if compiled_blueprint:
                # Activate the compiled blueprint
                from orgdb import activate_blueprint
                success = activate_blueprint(agent.id, blueprint.id)
                
                if success:
                    # Build context-rich success message
                    agent_name = context["agent_identity"]["name"]
                    capability_summary = context["conversation_hints"]["capability_summary"]
                    completeness_score = context["development_context"]["completeness_score"]
                    
                    success_message = f"🎉 Excellent! I've successfully compiled and activated **{agent_name}**!\n\n"
                    success_message += f"**Your {context['agent_identity']['type']} agent is now production-ready:**\n"
                    success_message += f"✅ **Identity**: {agent_name} - {context['agent_identity']['purpose']}\n"
                    success_message += f"✅ **Capabilities**: {capability_summary}\n"
                    success_message += f"✅ **Knowledge Sources**: {context['current_capabilities']['knowledge_count']} configured\n"
                    success_message += f"✅ **Integrations**: {context['current_capabilities']['integration_count']} connected\n"
                    success_message += f"✅ **Blueprint Completeness**: {completeness_score:.0f}%\n"
                    success_message += f"✅ **System Prompt**: Generated and compiled\n\n"
                    success_message += f"**{agent_name} is now live and ready to help users!** The compiled blueprint contains all our collaborative refinements and is optimized for {context['agent_identity']['language']} communication."
                    
                    # Save AMI's success response to conversation history
                    await self._save_conversation_message(
                        agent_id=agent.id,
                        blueprint_id=blueprint.id,
                        message_type="ami",
                        message_content=success_message,
                        context_data={
                            "action": "compilation_success", 
                            "compilation_status": "compiled",
                            "activation_status": "active",
                            "context": context
                        }
                    )
                    
                    return CollaborativeAgentResponse(
                        success=True,
                        conversation_id=request.conversation_id,
                        current_state=ConversationState.COMPLETED,
                        ami_message=success_message,
                        data={
                            "agent_id": agent.id,
                            "blueprint_id": blueprint.id,
                            "compilation_status": "compiled",
                            "activation_status": "active",
                            "context": context  # Include rich context in response
                        },
                        next_actions=[
                            f"Start using {agent_name}",
                            "Test agent responses",
                            "Create another agent",
                            "Monitor agent performance"
                        ]
                    )
                else:
                    # Compilation successful but activation failed
                    return CollaborativeAgentResponse(
                        success=True,
                        conversation_id=request.conversation_id,
                        current_state=ConversationState.COMPLETED,
                        ami_message=f"✅ Great! I've compiled '{agent.name}' successfully, but there was an issue activating it. The blueprint is ready - you can activate it manually from the agent dashboard.",
                        data={
                            "agent_id": agent.id,
                            "blueprint_id": blueprint.id,
                            "compilation_status": "compiled",
                            "activation_status": "pending"
                        },
                        next_actions=[
                            "Activate the agent manually",
                            "Check agent status",
                            "Create another agent"
                        ]
                    )
            else:
                # Compilation failed
                return CollaborativeAgentResponse(
                    success=False,
                    conversation_id=request.conversation_id,
                    current_state=ConversationState.SKELETON_REVIEW,
                    ami_message="I had trouble compiling the blueprint. There might be some configuration issues. Let's review and fix them together.",
                    error="Blueprint compilation failed",
                    next_actions=[
                        "Review blueprint configuration",
                        "Try a different approach", 
                        "Start over"
                    ]
                )
                
        except Exception as e:
            collab_logger.error(f"Compilation/activation error: {e}")
            return CollaborativeAgentResponse(
                success=False,
                conversation_id=request.conversation_id,
                current_state=ConversationState.SKELETON_REVIEW,
                ami_message="I encountered an error while compiling the blueprint. Let's refine it further and try again.",
                error=str(e),
                next_actions=[
                    "Refine the blueprint further",
                    "Try again",
                    "Start over"
                ]
            )
    
    async def _refine_blueprint_with_feedback(self, request: CollaborativeAgentRequest, agent, blueprint, context, conversation_history) -> CollaborativeAgentResponse:
        """Refine the blueprint based on human feedback - now with rich context awareness and conversation history"""
        collab_logger.info(f"Refining blueprint based on feedback: '{request.user_input[:100]}...'")
        
        # Save user's refinement message to conversation history
        await self._save_conversation_message(
            agent_id=agent.id,
            blueprint_id=blueprint.id,
            message_type="user",
            message_content=request.user_input,
            context_data={"action": "refinement_request", "current_state": "refinement"}
        )
        
        # Build context-rich refinement prompt
        agent_name = context["agent_identity"]["name"]
        capability_summary = context["conversation_hints"]["capability_summary"] 
        completeness_score = context["development_context"]["completeness_score"]
        suggested_improvements = context["conversation_hints"]["next_logical_improvements"]
        
        # Format conversation history for LLM context
        conversation_context = self._format_conversation_history_for_llm(conversation_history, agent_name)
        
        refinement_prompt = f"""
        You are Ami, an expert AI agent designer. You're working with a human to refine {agent_name}, their {context['agent_identity']['type']} agent.

        {conversation_context}

        AGENT CONTEXT (like Cursor's file context):
        - **Agent**: {agent_name}
        - **Purpose**: {context['agent_identity']['purpose']}
        - **Current State**: {capability_summary}
        - **Completeness**: {completeness_score:.0f}%
        - **Language**: {context['agent_identity']['language']}
        - **Knowledge Sources**: {context['current_capabilities']['knowledge_count']} configured
        - **Integrations**: {context['current_capabilities']['integration_count']} connected
        
        CURRENT BLUEPRINT:
        {json.dumps(blueprint.agent_blueprint, indent=2)}
        
        HUMAN FEEDBACK: "{request.user_input}"
        
        CONTEXT-AWARE SUGGESTIONS:
        Based on current state and our conversation, logical improvements might include: {', '.join(suggested_improvements)}
        
        Your task:
        1. Review our conversation history to understand the refinement journey
        2. Use your deep understanding of {agent_name}'s current state and our discussions
        3. Analyze the human's feedback in context of previous refinements and agent purpose
        4. Make intelligent updates that build on existing strengths and previous feedback
        5. Explain changes using the agent's name, context, and conversation history
        6. Avoid repeating suggestions or changes we've already discussed
        
        Return a JSON response with:
        {{
            "updated_blueprint": {{...complete updated blueprint...}},
            "changes_made": ["list of specific changes with context and history awareness"],  
            "explanation": "Context-aware explanation mentioning {agent_name}, current capabilities, and how this builds on our conversation",
            "questions": ["Smart follow-up questions based on agent context and conversation flow"],
            "suggested_next_steps": ["New contextual suggestions that haven't been covered yet"]
        }}
        
        Be intelligent, context-aware, and conversation-aware like Cursor with source code.
        """
        
        try:
            # Get LLM response
            executor = self._get_executor(request.llm_provider)
            raw_response = await executor.execute(
                user_query=refinement_prompt,
                system_prompt="You are an expert agent blueprint designer.",
                max_tokens=3000,
                model=request.model
            )
            
            # Parse the response
            refinement_result = self._extract_and_parse_json(raw_response)
            
            if not refinement_result or "updated_blueprint" not in refinement_result:
                raise ValueError("Invalid refinement response format")
            
            # Update the blueprint in the database
            from orgdb import update_blueprint
            updated_blueprint_data = refinement_result["updated_blueprint"]
            updated_blueprint = update_blueprint(blueprint.id, updated_blueprint_data)
            
            if updated_blueprint:
                collab_logger.info(f"Blueprint updated successfully: {blueprint.id}")
                
                # Prepare context-rich response
                changes_made = refinement_result.get("changes_made", [])
                explanation = refinement_result.get("explanation", f"I've updated {agent_name} based on your feedback.")
                questions = refinement_result.get("questions", [])
                suggested_next_steps = refinement_result.get("suggested_next_steps", [])
                
                # Calculate new completeness score
                updated_completeness = self._calculate_blueprint_completeness(updated_blueprint_data)
                
                ami_message = f"Perfect! I've enhanced **{agent_name}** based on your feedback.\n\n"
                ami_message += f"**Changes made to your {context['agent_identity']['type']} agent:**\n"
                for change in changes_made:
                    ami_message += f"✅ {change}\n"
                
                ami_message += f"\n{explanation}\n"
                ami_message += f"\n**Updated Status:**\n"
                ami_message += f"📊 **Blueprint Completeness**: {completeness_score:.0f}% → {updated_completeness:.0f}%\n"
                ami_message += f"🎯 **Current Focus**: {self._summarize_capabilities(updated_blueprint_data.get('what_i_do', {}).get('primary_tasks', []))}\n"
                
                if questions:
                    ami_message += f"\n**I have some smart questions to make {agent_name} even better:**\n"
                    for question in questions:
                        ami_message += f"❓ {question}\n"
                
                if suggested_next_steps:
                    ami_message += f"\n**Contextual suggestions for {agent_name}:**\n"
                    for step in suggested_next_steps:
                        ami_message += f"💡 {step}\n"
                
                ami_message += f"\nWhat do you think? Ready to compile {agent_name}, or would you like to refine further?"
                
                # Save AMI's refinement response to conversation history
                await self._save_conversation_message(
                    agent_id=agent.id,
                    blueprint_id=blueprint.id,
                    message_type="ami",
                    message_content=ami_message,
                    context_data={
                        "action": "refinement_response",
                        "changes_made": changes_made,
                        "completeness_improvement": updated_completeness - completeness_score,
                        "context": context,
                        "suggested_next_steps": suggested_next_steps
                    }
                )
                
                return CollaborativeAgentResponse(
                    success=True,
                    conversation_id=request.conversation_id,
                    current_state=ConversationState.SKELETON_REVIEW,
                    ami_message=ami_message,
                    data={
                        "agent_id": agent.id,
                        "blueprint_id": blueprint.id,  
                        "changes_made": changes_made,
                        "updated_blueprint": updated_blueprint_data,
                        "context": context,  # Include rich context
                        "completeness_improvement": updated_completeness - completeness_score,
                        "suggested_next_steps": suggested_next_steps,
                        "conversation_aware": True  # Flag indicating this response is conversation-aware
                    },
                    next_actions=[
                        f"Compile {agent_name}",
                        "Make more refinements",
                        "Ask questions about the updates",
                        "Review agent capabilities"
                    ]
                )
            else:
                raise ValueError("Failed to update blueprint in database")
                
        except Exception as e:
            collab_logger.error(f"Blueprint refinement failed: {e}")
            agent_name = context["agent_identity"]["name"]
            return CollaborativeAgentResponse(
                success=False,
                conversation_id=request.conversation_id,
                current_state=ConversationState.SKELETON_REVIEW,
                ami_message=f"I had trouble updating {agent_name}'s blueprint. This might be due to the complexity of your request or a technical issue. Could you try rephrasing your feedback more specifically about what aspect of {agent_name} you'd like to improve?",
                error=str(e),
                data={"context": context},  # Include context even in errors
                next_actions=[
                    f"Be more specific about {agent_name}'s changes",
                    "Try a simpler modification",
                    "Ask me about current capabilities",
                    "Start with a fresh approach"
                ]
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