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
from datetime import datetime, timezone
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
        self.conversations = {}  # Store conversation states for input collection phase
        
        collab_logger.info("Collaborative Creator initialized - now works with database records")
    
    async def handle_collaborative_request(self, request: CollaborativeAgentRequest) -> CollaborativeAgentResponse:
        """
        Main collaborative method - handles conversation, creation, and refinement
        - No agent/blueprint: Conversation mode (explore ideas, ask questions)
        - Has agent/blueprint: Refinement mode (refine existing agent)
        - Conversation + approval detected: Creation mode (create agent from conversation)
        """
        collab_logger.info(f"Collaborative request: {request.current_state.value} - '{request.user_input[:100]}...' (Agent: {request.agent_id or 'CONVERSATION'})")
        
        try:
            # Determine the mode based on request content
            if not request.agent_id or not request.blueprint_id:
                # No agent/blueprint - could be conversation or creation time
                
                # Get conversation context from frontend-provided history (following exec_tool.py pattern)
                conversation_context = self._get_conversation_context_from_history(request)
                
                # Use LLM to detect if user is ready to create agent
                is_approval = await self._detect_approval_intent_via_llm(request.user_input, conversation_context, request)
                collab_logger.info(f"LLM approval detection for: '{request.user_input[:50]}...' → {is_approval}")
                
                if is_approval:
                    # User is ready to create - switch to creation mode
                    collab_logger.info("Approval detected - creating agent from conversation")
                    return await self._create_agent_from_conversation(request, conversation_context)
                else:
                    # Still in conversation mode - ask questions, explore ideas
                    collab_logger.info("Conversation mode - exploring ideas and asking questions")
                    return await self._handle_idea_conversation(request, conversation_context)
            else:
                collab_logger.info("Refinement mode - has existing agent/blueprint")
            
            # Validate that agent and blueprint exist (should exist now)
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
            
            # Route to appropriate handler based on current state
            if request.current_state == ConversationState.BUILDING:
                # Handle input collection phase (Steps 4-5)
                response = await self._handle_input_collection(request, agent, blueprint)
            else:
                # Default to refinement mode (SKELETON_REVIEW, etc.)
                response = await self._handle_blueprint_refinement(request, agent, blueprint)
            
            # Always include agent_id and blueprint_id in response for frontend
            response.agent_id = agent.id
            response.blueprint_id = blueprint.id
            
            return response
                
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
    
    async def _create_draft_from_input(self, request: CollaborativeAgentRequest):
        """
        Create a draft agent and blueprint from initial user input
        """
        try:
            # Extract agent name from user input using simple heuristics
            user_input = request.user_input.strip()
            
            # Try to extract a meaningful name
            if "agent" in user_input.lower():
                # Look for patterns like "sales agent", "customer support agent"
                words = user_input.split()
                agent_idx = next(i for i, word in enumerate(words) if "agent" in word.lower())
                if agent_idx > 0:
                    agent_name = f"{words[agent_idx-1].title()} Agent"
                else:
                    agent_name = "New Agent"
            elif len(user_input) < 50:
                # Short input, use as-is with "Agent" suffix
                agent_name = f"{user_input.title()} Agent"
            else:
                # Long input, create generic name
                agent_name = "New Agent"
            
            # Create minimal initial blueprint
            initial_blueprint = {
                "identity": {
                    "name": agent_name,
                    "purpose": user_input[:200] + "..." if len(user_input) > 200 else user_input,
                    "type": "custom",
                    "language": "english",
                    "personality": {
                        "tone": "professional",
                        "style": "helpful",
                        "analogy": "like a helpful assistant"
                    }
                },
                "capabilities": {
                    "tasks": [
                        {
                            "task": "Initial Task",
                            "description": "To be defined during collaboration with AMI"
                        }
                    ],
                    "knowledge_sources": [],
                    "integrations": [],
                    "tools": []
                },
                "configuration": {
                    "communication_style": "conversational",
                    "response_length": "appropriate",
                    "confidence_level": "balanced",
                    "escalation_method": "To be defined"
                },
                "test_scenarios": [],
                "workflow_steps": ["To be defined during collaboration"],
                "visual_flow": "To be defined",
                "success_criteria": ["To be defined"],
                "potential_challenges": ["To be defined"],
                "created_from_input": user_input,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Create agent with initial blueprint
            from orgdb import create_agent_with_blueprint
            agent, blueprint = create_agent_with_blueprint(
                org_id=request.org_id,
                created_by=request.user_id,
                name=agent_name,
                blueprint_data=initial_blueprint,
                description=f"Agent created from: {user_input[:100]}...",
                conversation_id=request.conversation_id
            )
            
            collab_logger.info(f"Created draft agent from input: {agent.name} (ID: {agent.id})")
            return agent, blueprint
            
        except Exception as e:
            collab_logger.error(f"Failed to create draft from input: {e}")
            raise Exception(f"Failed to create draft agent: {str(e)}")
    
    def _get_conversation_context_from_history(self, request: CollaborativeAgentRequest) -> list:
        """Get conversation context from frontend-provided history (following exec_tool.py pattern)"""
        try:
            if not request.conversation_history:
                collab_logger.info("No conversation history provided")
                return []
            
            # Limit history to max_history_messages (default 25)
            max_messages = request.max_history_messages or 25
            limited_history = request.conversation_history[-max_messages:] if len(request.conversation_history) > max_messages else request.conversation_history
            
            # Format messages for LLM context - frontend should provide in consistent format
            formatted_messages = []
            for msg in limited_history:
                formatted_messages.append({
                    "role": msg.get("role", msg.get("sender", "user")),  # Support both 'role' and 'sender' keys
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp", msg.get("created_at", ""))
                })
            
            collab_logger.info(f"Using {len(formatted_messages)} messages from frontend conversation history")
            return formatted_messages
            
        except Exception as e:
            collab_logger.warning(f"Could not process conversation history: {e}")
            return []
    
    async def _detect_approval_intent_via_llm(self, user_input: str, conversation_context: list, request: CollaborativeAgentRequest) -> bool:
        """Use human-centric logic to detect if user is ready to create an agent"""
        try:
            user_input_lower = user_input.lower().strip()
            
            # PRIORITY 1: Direct creation commands - always approve these
            direct_creation_phrases = [
                "just create", "create agent", "build agent", "make agent", 
                "just build", "create it", "build it", "make it",
                "go ahead and create", "please create", "create now"
            ]
            
            if any(phrase in user_input_lower for phrase in direct_creation_phrases):
                collab_logger.info(f"Direct creation detected: '{user_input[:30]}...' → APPROVED")
                return True
            
            # PRIORITY 2: If conversation history exists, be more permissive
            has_conversation_context = conversation_context and len(conversation_context) > 1
            
            if has_conversation_context:
                # User has already been discussing the agent - be more permissive
                permissive_keywords = [
                    "yes", "ok", "okay", "sure", "sounds good", "perfect", 
                    "let's do", "proceed", "go ahead", "that works", "approved"
                ]
                
                if any(keyword in user_input_lower for keyword in permissive_keywords):
                    collab_logger.info(f"Permissive approval with context: '{user_input[:30]}...' → APPROVED")
                    return True
            
            # PRIORITY 3: Use LLM for nuanced cases, but with better prompt
            conversation_text = ""
            if conversation_context:
                recent_messages = conversation_context[-20:]  # Last 4 messages for context
                conversation_text = "\n".join([
                    f"{msg['role']}: {msg['content']}" for msg in recent_messages
                ])
            
            approval_prompt = f"""
CONTEXT: User has been discussing an agent with AMI. Determine if they want to CREATE it now.

Conversation:
{conversation_text}

User: "{user_input}"

IMPORTANT: If user has already described what they want and now says anything that could mean "create it", return true.

BE PERMISSIVE - err on the side of creating the agent. Users can always refine later.

Return ONLY "true" or "false".
"""

            if request.llm_provider == "anthropic":
                response = await self.anthropic_executor.call_anthropic_direct(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": approval_prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                result = response.content[0].text.strip().lower()
            else:
                response = await self.openai_executor.call_openai_direct(
                    model="gpt-4",
                    messages=[{"role": "user", "content": approval_prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                result = response.choices[0].message.content.strip().lower()
            
            is_approval = result == "true"
            collab_logger.info(f"LLM approval detection: '{user_input[:30]}...' → {result} → {is_approval}")
            return is_approval
            
        except Exception as e:
            collab_logger.error(f"Approval detection failed: {e}")
            # Fallback: if there's conversation context and any creation intent, approve
            if conversation_context and len(conversation_context) > 1:
                creation_keywords = ["create", "build", "make", "yes", "ok", "sure", "go ahead"]
                if any(keyword in user_input.lower() for keyword in creation_keywords):
                    collab_logger.info(f"Fallback approval with context: '{user_input[:30]}...' → APPROVED")
                    return True
            
            return False
    
    async def _handle_idea_conversation(self, request: CollaborativeAgentRequest, conversation_context: list) -> CollaborativeAgentResponse:
        """Handle conversation mode - ask questions and explore ideas without creating agent"""
        try:
            # Format conversation for LLM context
            conversation_text = ""
            if conversation_context:
                recent_messages = conversation_context[-8:]
                conversation_text = "\n".join([
                    f"{msg['role']}: {msg['content']}" for msg in recent_messages
                ])
            
            # 🚀 BOLD MOVE: Research unknown tools IMMEDIATELY!
            research_context = await self._research_tools_in_conversation(request.user_input, conversation_text)
            
            conversation_prompt = f"""
You are AMI, an expert AI agent designer. You're having a conversation with a human to understand what kind of agent they want to build.

Your role: Ask thoughtful questions to understand their needs before building anything.

Conversation History:
{conversation_text}

Human's Latest Input: "{request.user_input}"

{research_context}

GUIDELINES FOR IDEATION PHASE:
1. **SHOW UNDERSTANDING FIRST** - Demonstrate you researched their tools by mentioning specific capabilities you discovered
2. **PROVIDE CREATIVE IDEAS** - Based on research, suggest innovative ways the agent could work beyond basic requirements
3. **EXPAND THEIR VISION** - Use research insights to propose features they might not have considered
4. **BE INSPIRATIONAL** - Help them see the full potential of what's possible with these tools
5. **ASK EXPLORATORY QUESTIONS** - Focus on "What if we could..." rather than technical implementation details
6. **REFERENCE SPECIFIC CAPABILITIES** - Show you understand their business context by mentioning researched features

EXAMPLE IDEATION RESPONSES:
- SHOW UNDERSTANDING: "I see you're using 1Office.vn - that's Vietnam's comprehensive enterprise management platform with automated reporting capabilities, and Shopee, the largest e-commerce platform in Southeast Asia with $47.9B GMV..."

- EXPAND VISION: "What if your agent could go beyond basic reconciliation? Since 1Office.vn has automated Excel formulas, we could create predictive analytics to flag potential issues before they happen..."

- CREATIVE IDEAS: "Given Shopee's escrow service and 1Office.vn's flexible integrations, we could build an intelligent agent that not only reconciles but also predicts cash flow patterns and suggests optimal inventory levels..."

RESPONSE FORMAT:
You MUST return a valid JSON object with these exact fields:
{{
    "ami_message": "Start by showing understanding of their researched tools, then provide 2-3 creative ideas or 'What if we could...' questions to expand their vision",
    "suggestions": ["2-3 innovative suggestions based on research insights, not basic technical questions"],
    "agent_concept": "Enhanced concept summary that incorporates research findings and suggests advanced capabilities"
}}

CRITICAL: 
- Return ONLY valid JSON - no markdown, no extra text
- Properly escape all quotes in strings using \"
- Ensure all strings are properly quoted
- No trailing commas
- Test your JSON before responding

Be conversational, curious, and helpful. Focus on understanding their vision deeply.
"""
            
            if request.llm_provider == "anthropic":
                response = await self.anthropic_executor.call_anthropic_direct(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": conversation_prompt}],
                    max_tokens=800,
                    temperature=0.7
                )
                raw_response = response.content[0].text
            else:
                response = await self.openai_executor.call_openai_direct(
                    model="gpt-4",
                    messages=[{"role": "user", "content": conversation_prompt}],
                    max_tokens=800,
                    temperature=0.7
                )
                raw_response = response.choices[0].message.content
            
            # Parse LLM response - should be clean JSON now!
            import json
            try:
                conversation_result = json.loads(raw_response)
            except json.JSONDecodeError as e:
                collab_logger.error(f"LLM returned invalid JSON: {e}")
                conversation_result = None
            
            if not conversation_result:
                # Fallback response
                conversation_result = {
                    "ami_message": "I'd love to help you build an agent! Can you tell me more about what specific tasks you'd like it to handle?",
                    "suggestions": ["What problem should this agent solve?", "Who will be using this agent?"],
                    "agent_concept": "Exploring agent ideas"
                }
            
            # Log AMI's response (frontend handles saving to chat)
            self._log_response_for_debugging(request, conversation_result["ami_message"])
            
            return CollaborativeAgentResponse(
                success=True,
                conversation_id=request.conversation_id or str(uuid4()),
                current_state=ConversationState.INITIAL_IDEA,
                ami_message=conversation_result["ami_message"],
                data={
                    "mode": "conversation",
                    "suggestions": conversation_result.get("suggestions", []),
                    "agent_concept": conversation_result.get("agent_concept", ""),
                    "context": "exploring_ideas"
                },
                next_actions=[
                    "Answer AMI's questions",
                    "Provide more details about your needs", 
                    "Say 'let's build this' when ready to create"
                ]
            )
            
        except Exception as e:
            collab_logger.error(f"Conversation handling failed: {e}")
            return CollaborativeAgentResponse(
                success=False,
                conversation_id=request.conversation_id or "unknown",
                current_state=ConversationState.INITIAL_IDEA,
                ami_message="I'm having trouble processing that. Could you tell me more about what kind of agent you'd like to build?",
                error=str(e)
            )
    
    async def _research_tools_in_conversation(self, user_input: str, conversation_text: str) -> str:
        """
        🚀 BOLD MOVE: Research unknown tools immediately when user mentions them
        Returns research context to inject into the conversation prompt
        """
        try:
            # Create a mini blueprint from conversation to detect tools
            combined_text = f"{conversation_text}\n{user_input}"
            
            # Use the same LLM-based detection from orgdb.py
            from orgdb import _identify_important_terms
            
            # Create a mock blueprint structure for tool detection
            mock_blueprint = {
                'conversation_requirements': {
                    'concept': combined_text,
                    'purpose': user_input,
                    'key_tasks': [user_input]
                }
            }
            
            important_terms = _identify_important_terms(mock_blueprint)
            
            if not important_terms:
                collab_logger.info("🔍 No important terms detected in conversation - no research needed")
                return ""
            
            collab_logger.info(f"🚀 BOLD MOVE: Detected important terms in conversation: {important_terms}")
            collab_logger.info(f"   Starting immediate research for better conversation...")
            
            # Use Anthropic's native web search to research these tools
            import anthropic
            import os
            
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            research_prompt = f"""Research these important business terms/concepts that the user mentioned: {', '.join(important_terms)}

For each term, find:
1. What it is (brief description and business context)
2. Main use cases and industry applications
3. Technical requirements or integration methods (if applicable)
4. Common implementation approaches
5. Typical setup or configuration needs

Keep each term summary to 2-3 sentences. Focus on information relevant to building AI agents.

Format as:
**Term**: Brief description, business context, technical details, implementation notes."""

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5
                    # No domain restrictions - Claude can search the entire web!
                    # This allows research of any business software: misa.vn, 1office.vn, etc.
                }],
                messages=[{"role": "user", "content": research_prompt}]
            )
            
            # 🚨 DEBUG: Log the full response structure to understand what we got
            collab_logger.info(f"🔍 ANTHROPIC RESPONSE STRUCTURE:")
            collab_logger.info(f"   Response type: {type(response)}")
            collab_logger.info(f"   Content blocks: {len(response.content)}")
            for i, block in enumerate(response.content):
                collab_logger.info(f"   Block {i}: type={getattr(block, 'type', 'unknown')}")
                if hasattr(block, 'text'):
                    collab_logger.info(f"   Block {i}: text_length={len(block.text)}")
            
            # Extract ALL text content from the response, not just the first block
            research_results = ""
            for i, block in enumerate(response.content):
                if hasattr(block, 'text') and block.text:
                    research_results += block.text
                    collab_logger.info(f"   Block {i} text: {block.text[:100]}...")
                elif hasattr(block, 'type'):
                    collab_logger.info(f"   Block {i}: {block.type} (no text content)")
            
            # If no text found, use a fallback
            if not research_results.strip():
                research_results = "Research completed but no detailed results were extracted."
            
            collab_logger.info(f"🌐 Research completed for conversation terms")
            collab_logger.info(f"   Terms researched: {', '.join(important_terms)}")
            collab_logger.info(f"🔍 RESEARCH RESULTS:")
            collab_logger.info(f"{research_results}")
            collab_logger.info(f"🔍 END RESEARCH RESULTS")
            
            # Return formatted research context
            return f"""
RESEARCH INSIGHTS (use this to ask specific questions):
{research_results}

ACTIONABLE GUIDANCE:
- Reference specific API capabilities, rate limits, or data formats you learned
- Ask about integration patterns mentioned in the research
- Suggest implementation approaches based on discovered technical details
- Show you understand their business context from the research findings
"""
            
        except Exception as e:
            collab_logger.warning(f"Conversation research failed: {e}")
            return ""  # Fail gracefully - conversation continues without research

    def _extract_and_parse_json(self, response_text: str, context: str = "response") -> dict:
        """
        Extract and parse JSON from LLM response, handling markdown code blocks
        """
        import json
        import re
        
        # First try direct parsing (for clean JSON)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks
        try:
            # Look for ```json...``` or ```...``` blocks
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                return json.loads(json_content)
        except json.JSONDecodeError:
            pass
        
        # Final attempt: look for JSON-like content between { and }
        try:
            # Find the first { and last } to extract JSON object
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_content = response_text[start:end+1]
                return json.loads(json_content)
        except json.JSONDecodeError:
            pass
        
        collab_logger.error(f"LLM returned unparseable JSON for {context}")
        collab_logger.error(f"Raw response: {response_text}")
        return None

    def _generate_contextual_refinement_suggestions(self, requirements: dict, agent_name: str) -> list:
        """
        Generate contextual suggestions for agent refinement
        """
        suggestions = []
        
        # Add suggestions based on agent requirements
        if requirements:
            if 'integrations' in requirements:
                suggestions.append("Consider additional integrations for complete workflow")
            if 'tasks' in requirements:
                suggestions.append("Define more specific tasks and responsibilities")
        
        # Default suggestions
        suggestions.extend([
            f"Preview {agent_name}'s full blueprint configuration",
            f"Test and approve the agent for production use"
        ])
        
        return suggestions

    def _generate_change_deltas(self, original_blueprint: dict, updated_blueprint: dict) -> dict:
        """
        Generate change deltas to help frontend highlight what changed
        """
        deltas = {
            "added": {},
            "modified": {},
            "removed": {}
        }
        
        # Input validation
        if original_blueprint is None or updated_blueprint is None:
            collab_logger.warning("🔄 One of the blueprints is None, returning empty deltas")
            return deltas
        
        try:
            # Compare tasks
            original_tasks = original_blueprint.get("capabilities", {}).get("tasks", [])
            updated_tasks = updated_blueprint.get("capabilities", {}).get("tasks", [])
            
            # Find added tasks
            original_task_names = {task.get("task", "") for task in original_tasks if isinstance(task, dict)}
            added_tasks = []
            for i, task in enumerate(updated_tasks):
                if isinstance(task, dict) and task.get("task", "") not in original_task_names:
                    added_tasks.append({
                        "task": task.get("task", ""),
                        "description": task.get("description", ""),
                        "position": i
                    })
            
            if added_tasks:
                deltas["added"]["tasks"] = added_tasks
            
            # Compare tools
            original_tools = original_blueprint.get("capabilities", {}).get("tools", [])
            updated_tools = updated_blueprint.get("capabilities", {}).get("tools", [])
            
            # Find added tools
            original_tool_set = set(original_tools) if isinstance(original_tools, list) else set()
            updated_tool_set = set(updated_tools) if isinstance(updated_tools, list) else set()
            added_tools = list(updated_tool_set - original_tool_set)
            
            if added_tools:
                deltas["added"]["tools"] = added_tools
            
            # Compare integrations
            original_integrations = original_blueprint.get("capabilities", {}).get("integrations", [])
            updated_integrations = updated_blueprint.get("capabilities", {}).get("integrations", [])
            
            original_integration_names = {
                integration.get("tool", "") if isinstance(integration, dict) else str(integration)
                for integration in original_integrations
            }
            
            added_integrations = []
            for integration in updated_integrations:
                integration_name = integration.get("tool", "") if isinstance(integration, dict) else str(integration)
                if integration_name not in original_integration_names:
                    added_integrations.append(integration)
            
            if added_integrations:
                deltas["added"]["integrations"] = added_integrations
            
            # Compare knowledge sources
            original_knowledge = original_blueprint.get("capabilities", {}).get("knowledge_sources", [])
            updated_knowledge = updated_blueprint.get("capabilities", {}).get("knowledge_sources", [])
            
            original_knowledge_domains = {
                ks.get("domain", "") if isinstance(ks, dict) else str(ks)
                for ks in original_knowledge
            }
            
            added_knowledge = []
            for ks in updated_knowledge:
                domain = ks.get("domain", "") if isinstance(ks, dict) else str(ks)
                if domain not in original_knowledge_domains:
                    added_knowledge.append(ks)
            
            if added_knowledge:
                deltas["added"]["knowledge_sources"] = added_knowledge
                
        except Exception as e:
            collab_logger.error(f"❌ Exception in _generate_change_deltas: {e}")
            collab_logger.error(f"❌ Exception type: {type(e)}")
            import traceback
            collab_logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # Return empty deltas on error
            pass
        
        return deltas
    
    async def _create_agent_from_conversation(self, request: CollaborativeAgentRequest, conversation_context: list) -> CollaborativeAgentResponse:
        """Create agent when user approves after conversation"""
        try:
            collab_logger.info("Creating agent from conversation context")
            
            # Analyze conversation to extract agent requirements  
            agent_requirements = await self._analyze_conversation_for_agent_creation(conversation_context, request.user_input, request)
            
            # Create agent and blueprint with conversation insights
            agent, blueprint = await self._create_agent_from_requirements(agent_requirements, request)
            
            # Creation message (frontend handles saving to chat)
            creation_message = f"Great! I've created '{agent.name}' based on our conversation. The agent is ready to use, and you can refine any details as needed. What would you like to adjust?"
            
            # Generate contextual suggestions based on what was created
            contextual_suggestions = self._generate_contextual_refinement_suggestions(agent_requirements, agent.name)
            
            return CollaborativeAgentResponse(
                success=True,
                conversation_id=request.conversation_id or str(uuid4()),
                current_state=ConversationState.SKELETON_REVIEW,
                ami_message=creation_message,
                agent_id=agent.id,
                blueprint_id=blueprint.id,
                data={
                    "mode": "created",
                    "agent_name": agent.name,
                    "agent_type": agent_requirements.get("agent_type", "assistant"),
                    "agent_concept": agent_requirements.get("concept", ""),
                    "agent_purpose": agent_requirements.get("purpose", ""),
                    "key_tasks": agent_requirements.get("key_tasks", []),
                    "integrations": agent_requirements.get("integrations", []),
                    "target_users": agent_requirements.get("target_users", ""),
                    "next_phase": "refinement",
                    "blueprint_summary": {
                        "tasks_count": len(agent_requirements.get("key_tasks", [])),
                        "integrations_count": len(agent_requirements.get("integrations", [])),
                        "has_monitoring": bool(agent_requirements.get("business_context"))
                    },
                    "frontend_actions": {
                        "load_blueprint": f"/org-agents/{agent.id}/blueprint",
                        "load_agent_details": f"/org-agents/{agent.id}",
                        "suggested_api_calls": [
                            {"method": "GET", "endpoint": f"/org-agents/{agent.id}/blueprint", "purpose": "Load full blueprint details"},
                            {"method": "GET", "endpoint": f"/org-agents/{agent.id}", "purpose": "Load agent configuration"}
                        ]
                    }
                },
                next_actions=contextual_suggestions
            )
            
        except Exception as e:
            collab_logger.error(f"Agent creation from conversation failed: {e}")
            return CollaborativeAgentResponse(
                success=False,
                conversation_id=request.conversation_id or "unknown",
                current_state=ConversationState.INITIAL_IDEA,
                ami_message="I had trouble creating the agent. Let's continue our conversation to clarify the requirements.",
                error=str(e)
            )
    
    def _log_response_for_debugging(self, request: CollaborativeAgentRequest, ami_message: str):
        """Log AMI's response for debugging (frontend handles saving to chat)"""
        collab_logger.info(f"AMI response for conversation {request.conversation_id}: {ami_message[:100]}...")
        # Note: Frontend handles saving messages to chat sessions
    
    async def _analyze_conversation_for_agent_creation(self, conversation_context: list, latest_input: str, request: CollaborativeAgentRequest) -> dict:
        """Analyze conversation to extract agent creation requirements"""
        try:
            # Format conversation for analysis
            conversation_text = ""
            if conversation_context:
                conversation_text = "\n".join([
                    f"{msg['role']}: {msg['content']}" for msg in conversation_context
                ])
            
            analysis_prompt = f"""
You are analyzing a conversation to extract requirements for creating an AI agent.

Conversation History:
{conversation_text}

Latest User Input: "{latest_input}"

IMPORTANT: The user has requested agent creation. Create a functional agent based on available information.

Extract the following information from the conversation:

RESPONSE FORMAT (JSON):
{{
    "agent_name": "Suggested name for the agent",
    "agent_type": "Type of agent (e.g., assistant, analyst, support)",
    "purpose": "Clear description of what the agent should do",
    "key_tasks": ["List of main tasks the agent should handle"],
    "integrations": ["Required tools/systems (e.g., Gmail, Slack, CRM)"],
    "knowledge_domains": ["Areas of expertise needed"],
    "target_users": "Who will use this agent",
    "business_context": "What problem it solves",
    "concept": "One-sentence summary of the agent"
}}

GUIDELINES:
- If limited information is provided, infer reasonable capabilities from context
- Create a working agent that can be refined later
- Don't leave fields empty - provide sensible defaults
- Focus on what the user has expressed interest in

Analyze the conversation and create agent requirements now.
"""
            
            if request.llm_provider == "anthropic":
                response = await self.anthropic_executor.call_anthropic_direct(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    max_tokens=1000,
                    temperature=0.3
                )
                raw_response = response.content[0].text
            else:
                response = await self.openai_executor.call_openai_direct(
                    model="gpt-4",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    max_tokens=1000,
                    temperature=0.3
                )
                raw_response = response.choices[0].message.content
            
            # Parse the analysis
            requirements = self._extract_and_parse_json(raw_response, "requirements")
            
            if not requirements:
                # Fallback requirements
                requirements = {
                    "agent_name": "Custom Agent",
                    "agent_type": "assistant",
                    "purpose": latest_input[:200],
                    "key_tasks": ["To be defined"],
                    "integrations": [],
                    "knowledge_domains": [],
                    "target_users": "Team members",
                    "business_context": "Automation and assistance",
                    "concept": "AI assistant for custom tasks"
                }
            
            collab_logger.info(f"Analyzed conversation for agent: {requirements.get('agent_name', 'Unknown')}")
            return requirements
            
        except Exception as e:
            collab_logger.error(f"Conversation analysis failed: {e}")
            return {
                "agent_name": "Custom Agent",
                "agent_type": "assistant", 
                "purpose": latest_input[:200],
                "key_tasks": ["Custom tasks"],
                "integrations": [],
                "knowledge_domains": [],
                "target_users": "Users",
                "business_context": "Automation",
                "concept": "Custom AI agent"
            }
    
    async def _create_agent_from_requirements(self, requirements: dict, request: CollaborativeAgentRequest):
        """Create agent and blueprint from analyzed requirements"""
        try:
            agent_name = requirements.get("agent_name", "Custom Agent")
            
            # Create rich blueprint from conversation requirements
            initial_blueprint = {
                "identity": {
                    "name": agent_name,
                    "purpose": requirements.get("purpose", "Custom AI agent"),
                    "type": requirements.get("agent_type", "assistant"),
                    "language": "english",
                    "personality": {
                        "tone": "professional",
                        "style": "helpful",
                        "analogy": f"like a {requirements.get('agent_type', 'helpful assistant')}"
                    }
                },
                "capabilities": {
                    "tasks": [
                        {"task": task, "description": f"Handle {task}"} 
                        for task in requirements.get("key_tasks", ["Custom tasks"])
                    ],
                    "knowledge_sources": [
                        {"domain": domain, "description": f"Expertise in {domain}"}
                        for domain in requirements.get("knowledge_domains", [])
                    ],
                    "integrations": [
                        {"tool": integration, "purpose": f"Connect with {integration}"}
                        for integration in requirements.get("integrations", [])
                    ],
                    "tools": []
                },
                "configuration": {
                    "communication_style": "conversational",
                    "response_length": "appropriate",
                    "confidence_level": "balanced",
                    "escalation_method": "To be defined"
                },
                "business_context": {
                    "problem_solved": requirements.get("business_context", "Automation and assistance"),
                    "target_users": requirements.get("target_users", "Team members"),
                    "success_metrics": ["User satisfaction", "Task completion rate"]
                },
                "test_scenarios": [],
                "workflow_steps": ["To be refined during collaboration"],
                "visual_flow": "To be defined",
                "success_criteria": ["Successfully complete assigned tasks"],
                "potential_challenges": ["To be identified"],
                "created_from_conversation": True,
                "conversation_requirements": requirements,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Create agent with conversation-derived blueprint
            from orgdb import create_agent_with_blueprint
            agent, blueprint = create_agent_with_blueprint(
                org_id=request.org_id,
                created_by=request.user_id,
                name=agent_name,
                blueprint_data=initial_blueprint,
                description=f"Agent created from conversation: {requirements.get('concept', 'Custom agent')}",
                conversation_id=request.conversation_id
            )
            
            collab_logger.info(f"Created agent from conversation: {agent.name} (ID: {agent.id})")
            return agent, blueprint
            
        except Exception as e:
            collab_logger.error(f"Agent creation from requirements failed: {e}")
            raise Exception(f"Failed to create agent from conversation: {str(e)}")
    
    async def _handle_input_collection(self, request: CollaborativeAgentRequest, agent, blueprint) -> CollaborativeAgentResponse:
        """
        Handle input collection phase (Steps 4-5 of the 8-step process)
        - User provides API keys, credentials, configuration details
        - System validates inputs and updates blueprint
        - When all inputs collected, allows compilation to final agent
        """
        collab_logger.info(f"Input collection phase for agent {agent.name} (Blueprint: {blueprint.id})")
        
        # Check if user wants to compile (all inputs collected)
        compile_intent = self._detect_compile_intent(request.user_input)
        
        if compile_intent:
            # Check if all required todos are completed
            todos_complete = self._check_todos_completion(blueprint)
            
            if todos_complete:
                # All inputs collected - proceed to final compilation (Steps 6-8)
                collab_logger.info(f"All inputs collected for {agent.name} - proceeding to final compilation")
                
                # Create approved request for final building
                approved_request = CollaborativeAgentRequest(
                    user_input=request.user_input,
                    conversation_id=request.conversation_id,
                    org_id=request.org_id,
                    user_id=request.user_id,
                    agent_id=request.agent_id,
                    blueprint_id=request.blueprint_id,
                    llm_provider=request.llm_provider,
                    model=request.model,
                    current_state=ConversationState.APPROVED
                )
                
                # Final compilation
                return await self.orchestrator._build_approved_agent(approved_request)
            else:
                # Not all inputs collected yet
                pending_todos = self._get_pending_todos(blueprint)
                return CollaborativeAgentResponse(
                    success=True,
                    conversation_id=request.conversation_id,
                    current_state=ConversationState.BUILDING,
                    ami_message=f"I'd love to compile **{agent.name}** for you! However, I still need some information to complete the setup.\n\n**Pending todos:** {len(pending_todos)} remaining\n\nPlease provide the required inputs for the pending todos, then I can compile your agent.",
                    agent_id=agent.id,
                    blueprint_id=blueprint.id,
                    data={
                        "pending_todos": pending_todos,
                        "completion_status": "inputs_pending"
                    },
                    next_actions=[
                        "Complete pending todos",
                        "Provide required API keys and credentials",
                        "Try compiling again once inputs are provided"
                    ]
                )
        else:
            # Handle input provision or todo updates
            # Parse user input for todo completion
            todo_updated = await self._parse_and_update_todo_completion(request, blueprint)
            
            if todo_updated:
                # Reload blueprint to get updated todos
                from orgdb import get_agent_blueprint
                blueprint = get_agent_blueprint(blueprint.id)
                
                # Check if all todos are now complete
                todos_complete = self._check_todos_completion(blueprint)
                
                if todos_complete:
                    # All inputs collected - proceed to final compilation
                    collab_logger.info(f"All inputs collected for {agent.name} after todo update - proceeding to final compilation")
                    
                    # Create approved request for final building
                    approved_request = CollaborativeAgentRequest(
                        user_input=request.user_input,
                        conversation_id=request.conversation_id,
                        org_id=request.org_id,
                        user_id=request.user_id,
                        agent_id=request.agent_id,
                        blueprint_id=request.blueprint_id,
                        llm_provider=request.llm_provider,
                        model=request.model,
                        current_state=ConversationState.APPROVED
                    )
                    
                    # Final compilation
                    return await self.orchestrator._build_approved_agent(approved_request)
            
            # Get current todos status
            todos = blueprint.implementation_todos if blueprint.implementation_todos else []
            pending_todos = [todo for todo in todos if todo.get('status') != 'completed']
            
            return CollaborativeAgentResponse(
                success=True,
                conversation_id=request.conversation_id,
                current_state=ConversationState.BUILDING,
                ami_message=f"Great! I'm ready to help you set up **{agent.name}**.\n\n**Current Status:**\n✅ Blueprint approved and created\n📋 {len(todos)} implementation todos generated\n⏳ {len(pending_todos)} todos pending completion\n\n**Next Steps:**\n1. Complete the implementation todos by providing required information\n2. Once all todos are done, say 'compile' to build your final agent\n\nWhich todo would you like to work on first?",
                agent_id=agent.id,
                blueprint_id=blueprint.id,
                data={
                    "todos": todos,
                    "pending_todos": pending_todos,
                    "phase": "input_collection"
                },
                next_actions=[
                    "Complete implementation todos",
                    "Provide API keys and credentials",
                    "Say 'compile' when ready to build final agent"
                ]
            )
    
    def _detect_compile_intent(self, user_input: str) -> bool:
        """Detect if user wants to compile/finalize the agent"""
        user_input_lower = user_input.lower().strip()
        compile_keywords = [
            'compile', 'build', 'finalize', 'complete', 'activate', 'deploy',
            'ready', 'go', 'finish', 'done', 'create final', 'make it live'
        ]
        return any(keyword in user_input_lower for keyword in compile_keywords)
    
    def _check_todos_completion(self, blueprint) -> bool:
        """Check if all required todos are completed"""
        if not blueprint.implementation_todos:
            return True  # No todos means ready to compile
        
        required_todos = [todo for todo in blueprint.implementation_todos 
                         if todo.get('priority') in ['high', 'critical']]
        
        if not required_todos:
            return True  # No high/critical todos
        
        completed_todos = [todo for todo in required_todos 
                          if todo.get('status') == 'completed']
        
        return len(completed_todos) == len(required_todos)
    
    async def _parse_and_update_todo_completion(self, request: CollaborativeAgentRequest, blueprint) -> bool:
        """
        Parse user input for todo completion and update the blueprint
        Returns True if a todo was updated, False otherwise
        """
        user_input = request.user_input.lower()
        
        # Check if user is providing todo completion
        if "completed" in user_input and ("task" in user_input or "todo" in user_input):
            # Extract todo information using simple parsing
            lines = request.user_input.split('\n')
            
            todo_title = None
            notes = None
            
            for line in lines:
                if 'review agent configuration' in line.lower():
                    todo_title = "Review Agent Configuration"
                elif 'configure required tools' in line.lower():
                    todo_title = "Configure Required Tools"  
                elif 'validate agent setup' in line.lower():
                    todo_title = "Validate Agent Setup"
                elif line.strip().startswith('notes:'):
                    notes = line.split('notes:', 1)[1].strip()
            
            if todo_title:
                # Update the todo in the blueprint
                updated = self._update_blueprint_todo(blueprint, todo_title, notes or "Completed")
                if updated:
                    collab_logger.info(f"Updated todo '{todo_title}' to completed for blueprint {blueprint.id}")
                    return True
        
        return False
    
    def _update_blueprint_todo(self, blueprint, todo_title: str, notes: str) -> bool:
        """Update a specific todo in the blueprint"""
        if not blueprint.implementation_todos:
            return False
            
        # Find and update the todo
        for todo in blueprint.implementation_todos:
            if todo.get('title') == todo_title:
                todo['status'] = 'completed'
                todo['collected_inputs'] = {'notes': notes}
                
                # Update in database
                from orgdb import update_agent_blueprint
                update_agent_blueprint(blueprint.id, {'implementation_todos': blueprint.implementation_todos})
                return True
        
        return False

    def _get_pending_todos(self, blueprint) -> list:
        """Get list of pending todos"""
        if not blueprint.implementation_todos:
            return []
        
        return [todo for todo in blueprint.implementation_todos 
                if todo.get('status') != 'completed']
    
    def _convert_blueprint_to_skeleton(self, blueprint):
        """Convert blueprint back to skeleton format for input collection phase"""
        from .models import AgentSkeleton
        from datetime import datetime
        
        blueprint_data = blueprint.agent_blueprint
        
        return AgentSkeleton(
            conversation_id=blueprint.conversation_id or "unknown",
            agent_name=blueprint_data.get("identity", {}).get("name", "Unknown Agent"),
            agent_purpose=blueprint_data.get("identity", {}).get("purpose", ""),
            target_users=blueprint_data.get("business_context", {}).get("target_users", ""),
            agent_type=blueprint_data.get("identity", {}).get("type", "assistant"),
            language=blueprint_data.get("identity", {}).get("language", "english"),
            meet_me={"introduction": "AI Agent", "value_proposition": "Helpful assistant"},
            what_i_do={
                "primary_tasks": blueprint_data.get("capabilities", {}).get("tasks", []),
                "personality": blueprint_data.get("identity", {}).get("personality", {}),
                "sample_conversation": "Sample conversation"
            },
            knowledge_sources=blueprint_data.get("capabilities", {}).get("knowledge_sources", []),
            integrations=blueprint_data.get("capabilities", {}).get("integrations", []),
            monitoring={"reporting_method": "standard", "metrics_tracked": [], "fallback_response": "standard", "escalation_method": "standard"},
            test_scenarios=[],
            workflow_steps=blueprint_data.get("workflow_steps", []),
            visual_flow="Standard workflow",
            success_criteria=blueprint_data.get("success_criteria", []),
            potential_challenges=blueprint_data.get("potential_challenges", []),
            created_at=datetime.now()
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
            # Handle approval - start input collection phase instead of immediate compilation
            collab_logger.info(f"Blueprint approved! Starting input collection phase for {agent.name}")
            
            # Create skeleton from blueprint for input collection
            skeleton = self._convert_blueprint_to_skeleton(blueprint)
            
            # Store conversation state for input collection
            conversation = {
                "skeleton": skeleton,
                "state": ConversationState.BUILDING,
                "org_id": request.org_id,
                "user_id": request.user_id,
                "agent_id": agent.id,
                "blueprint_id": blueprint.id
            }
            
            # Start input collection phase
            if self.orchestrator:
                return await self.orchestrator._start_input_collection_phase(
                    request, skeleton, conversation
                )
            else:
                collab_logger.error("Orchestrator not available for input collection")
                return CollaborativeAgentResponse(
                    success=False,
                    conversation_id=request.conversation_id,
                    current_state=ConversationState.SKELETON_REVIEW,
                    ami_message="I had trouble starting the input collection phase. Please try again.",
                    error="Orchestrator not available",
                    next_actions=["Try again", "Start a new conversation"]
                )
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
        5. You CAN change the agent name (identity.name) if requested - this will update both blueprint and agent records
        6. Explain changes using the agent's name, context, and conversation history
        7. Avoid repeating suggestions or changes we've already discussed
        
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
            if request.llm_provider == "anthropic":
                response = await self.anthropic_executor.call_anthropic_direct(
                    model=request.model or "claude-3-5-sonnet-20241022",
                    messages=[
                        {"role": "user", "content": refinement_prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.7
                )
                raw_response = response.content[0].text
            elif request.llm_provider == "openai":
                response = await self.openai_executor.call_openai_direct(
                    model=request.model or "gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert agent blueprint designer."},
                        {"role": "user", "content": refinement_prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.7
                )
                raw_response = response.choices[0].message.content
            else:
                raise Exception(f"Unknown LLM provider: {request.llm_provider}")
            
            # Parse the response
            refinement_result = self._extract_and_parse_json(raw_response)
            
            if not refinement_result or "updated_blueprint" not in refinement_result:
                # More specific error for JSON parsing failures
                if not refinement_result:
                    collab_logger.error("JSON parsing completely failed - LLM returned unparseable response")
                    raise ValueError("The AI response couldn't be parsed - please try a simpler request")
                else:
                    collab_logger.error(f"Missing updated_blueprint in response: {list(refinement_result.keys())}")
                    raise ValueError("Invalid refinement response format")
            
            # Update the blueprint in the database
            from orgdb import update_blueprint, update_agent
            updated_blueprint_data = refinement_result["updated_blueprint"]
            
            collab_logger.info(f"🔄 Attempting to update blueprint {blueprint.id} in database...")
            collab_logger.info(f"🔄 Updated blueprint data keys: {list(updated_blueprint_data.keys())}")
            
            # Generate change deltas for frontend highlighting
            try:
                change_deltas = self._generate_change_deltas(blueprint.agent_blueprint, updated_blueprint_data)
                collab_logger.info(f"🔄 Generated change deltas: {len(change_deltas.get('added', {}))} additions, {len(change_deltas.get('modified', {}))} modifications")
            except Exception as e:
                collab_logger.error(f"❌ Failed to generate change deltas: {e}")
                change_deltas = {"added": {}, "modified": {}, "removed": {}}  # Fallback empty deltas
            
            updated_blueprint = update_blueprint(blueprint.id, updated_blueprint_data)
            
            if updated_blueprint:
                collab_logger.info(f"✅ Blueprint database update successful: {blueprint.id}")
            else:
                collab_logger.error(f"❌ Blueprint database update FAILED: {blueprint.id}")
            
            if updated_blueprint:
                collab_logger.info(f"Blueprint updated successfully: {blueprint.id}")
                
                # ✅ NEW: Check if agent name was changed and sync with agent table
                new_agent_name = updated_blueprint_data.get("identity", {}).get("name")
                if new_agent_name and new_agent_name != agent.name:
                    collab_logger.info(f"Agent name changed from '{agent.name}' to '{new_agent_name}' - syncing agent table")
                    updated_agent = update_agent(agent.id, name=new_agent_name)
                    if updated_agent:
                        collab_logger.info(f"Agent table updated successfully with new name: {new_agent_name}")
                        agent.name = new_agent_name  # Update local object for response
                    else:
                        collab_logger.warning(f"Failed to update agent table with new name: {new_agent_name}")
                        # Continue anyway - blueprint was updated successfully
                
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
                
                # Log the response data for debugging
                response_data = {
                    "agent_id": agent.id,
                    "blueprint_id": blueprint.id,
                    "changes_made": changes_made,
                    "change_deltas": change_deltas,  # Frontend highlighting support
                    "updated_blueprint": updated_blueprint_data,
                    "context": context,  # Include rich context
                    "completeness_improvement": updated_completeness - completeness_score,
                    "suggested_next_steps": suggested_next_steps,
                    "conversation_aware": True  # Flag indicating this response is conversation-aware
                }
                
                collab_logger.info(f"🔄 Response includes change deltas: {bool(response_data.get('change_deltas'))}")
                
                return CollaborativeAgentResponse(
                    success=True,
                    conversation_id=request.conversation_id,
                    current_state=ConversationState.SKELETON_REVIEW,
                    ami_message=ami_message,
                    data=response_data,
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
            # Update state to building (not approved - we need input collection first)
            conversation["state"] = ConversationState.BUILDING
            self.conversations[request.conversation_id] = conversation
            
            collab_logger.info(f"Agent approved! Starting input collection phase for '{skeleton.agent_name}'")
            
            # Generate todos for input collection instead of building immediately
            if self.orchestrator:
                return await self.orchestrator._start_input_collection_phase(
                    request, skeleton, conversation
                )
            else:
                # Fallback if orchestrator not available
                collab_logger.error("Orchestrator not available for input collection")
                return CollaborativeAgentResponse(
                    success=False,
                    conversation_id=request.conversation_id,
                    current_state=ConversationState.SKELETON_REVIEW,
                    ami_message="I had trouble starting the input collection phase. Please try again.",
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
    
    # DELETED: All JSON cleanup functions - no longer needed!
    # The LLM now returns proper JSON from the start.
