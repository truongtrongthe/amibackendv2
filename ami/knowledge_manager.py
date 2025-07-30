"""
Ami Knowledge Manager - Pinecone Integration
===========================================

Handles knowledge management for agent creation, including saving agent expertise,
user domain knowledge, and collaboration insights directly to Pinecone.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .models import AgentSkeleton

logger = logging.getLogger(__name__)

try:
    from pccontroller import save_knowledge
except ImportError as e:
    logger.warning(f"pccontroller import failed: {e}")
    save_knowledge = None


class AmiKnowledgeManager:
    """
    Direct Pinecone knowledge integration for Ami
    Uses pccontroller.py directly - no wrapper layers like ava.py!
    """
    
    def __init__(self):
        """Initialize knowledge manager with direct pccontroller access"""
        self.logger = logging.getLogger("ami_knowledge")
    
    async def save_agent_creation_knowledge(self, skeleton: AgentSkeleton, conversation_id: str, 
                                          user_id: str, org_id: str) -> Dict[str, Any]:
        """
        Save agent expertise knowledge when agent is created
        
        This knowledge becomes discoverable by the agent execution system
        for skill discovery and multi-step planning!
        """
        try:
            if save_knowledge is None:
                self.logger.warning("⚠️ pccontroller not available, skipping knowledge save")
                return {"success": False, "error": "pccontroller not available"}
            
            # Build rich agent expertise content
            knowledge_content = self._build_agent_expertise_knowledge(skeleton)
            
            # 🚀 DIRECT pccontroller.save_knowledge call
            result = await save_knowledge(
                input=knowledge_content,
                user_id=user_id,
                org_id=org_id,
                title=f"Agent Expertise: {skeleton.agent_name}",
                topic=f"agent_creation_{skeleton.agent_type}",
                categories=[
                    "agent_expertise",
                    skeleton.agent_type,
                    f"capabilities_{skeleton.agent_name.lower().replace(' ', '_')}",
                    "created_by_ami"
                ],
                ttl_days=None  # Permanent agent knowledge
            )
            
            if result.get("success"):
                self.logger.info(f"✅ Agent expertise saved: {skeleton.agent_name} → {result.get('vector_id')}")
            else:
                self.logger.error(f"❌ Failed to save agent expertise: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error saving agent creation knowledge: {e}")
            return {"success": False, "error": str(e)}
    
    async def save_user_domain_knowledge(self, user_input: str, agent_context: str,
                                       user_id: str, org_id: str) -> Dict[str, Any]:
        """
        Save domain knowledge provided by user during agent building
        
        This captures user expertise that agents can later discover and apply!
        """
        try:
            if save_knowledge is None:
                self.logger.warning("⚠️ pccontroller not available, skipping domain knowledge save")
                return {"success": False, "error": "pccontroller not available"}
            
            # Structure user domain knowledge
            domain_knowledge = f"""
Domain Knowledge for {agent_context}:

User Input: {user_input}

Context: Provided during collaborative agent building for {agent_context}
Source: User expertise and domain knowledge
Application: Can be used by {agent_context} and similar agents for specialized tasks

This knowledge represents user's domain expertise that should be considered
when the agent handles related tasks or similar domains.
"""
            
            # 🚀 DIRECT pccontroller.save_knowledge call
            result = await save_knowledge(
                input=domain_knowledge,
                user_id=user_id,
                org_id=org_id,
                title=f"Domain Knowledge: {agent_context}",
                topic="user_domain_knowledge",
                categories=[
                    "domain_knowledge", 
                    "user_provided", 
                    agent_context.lower().replace(' ', '_'),
                    "agent_building_session"
                ],
                ttl_days=730  # 2 years retention for domain knowledge
            )
            
            if result.get("success"):
                self.logger.info(f"✅ Domain knowledge saved: {agent_context} → {result.get('vector_id')}")
            else:
                self.logger.error(f"❌ Failed to save domain knowledge: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error saving user domain knowledge: {e}")
            return {"success": False, "error": str(e)}
    
    async def save_collaborative_insights(self, conversation_data: Dict, 
                                        user_id: str, org_id: str) -> Dict[str, Any]:
        """
        Save insights from collaborative agent building sessions
        
        This helps improve future agent creation and builds organizational knowledge!
        """
        try:
            if save_knowledge is None:
                self.logger.warning("⚠️ pccontroller not available, skipping insights save")
                return {"success": False, "error": "pccontroller not available"}
            
            insights = self._extract_collaboration_insights(conversation_data)
            
            # 🚀 DIRECT pccontroller.save_knowledge call
            result = await save_knowledge(
                input=insights,
                user_id=user_id,
                org_id=org_id,
                title="Agent Collaboration Insights",
                topic="collaborative_agent_building",
                categories=[
                    "collaboration_insights",
                    "agent_building", 
                    "user_patterns",
                    "ami_learning"
                ],
                ttl_days=365  # 1 year retention for collaboration patterns
            )
            
            if result.get("success"):
                self.logger.info(f"✅ Collaboration insights saved → {result.get('vector_id')}")
            else:
                self.logger.error(f"❌ Failed to save collaboration insights: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error saving collaborative insights: {e}")
            return {"success": False, "error": str(e)}
    
    def _build_agent_expertise_knowledge(self, skeleton: AgentSkeleton) -> str:
        """Build structured knowledge content for agent expertise"""
        
        return f"""
AGENT EXPERTISE PROFILE: {skeleton.agent_name}

PURPOSE & MISSION:
{skeleton.agent_purpose}

CORE SPECIALIZATIONS:
{chr(10).join(f"• {capability}" for capability in skeleton.key_capabilities)}

TARGET USERS & AUDIENCE:
{skeleton.target_users}

SPECIALIZED USE CASES:
{chr(10).join(f"• {use_case}" for use_case in skeleton.use_cases)}

REQUIRED TOOLS & CAPABILITIES:
{chr(10).join(f"• {tool}" for tool in skeleton.required_tools)}

KNOWLEDGE DOMAINS:
{chr(10).join(f"• {domain}" for domain in skeleton.knowledge_domains)}

PERSONALITY & APPROACH:
{chr(10).join(f"• {trait}: {value}" for trait, value in skeleton.personality_traits.items())}

SUCCESS CRITERIA:
{chr(10).join(f"• {criteria}" for criteria in skeleton.success_criteria)}

POTENTIAL CHALLENGES & SOLUTIONS:
{chr(10).join(f"• {challenge}" for challenge in skeleton.potential_challenges)}

AGENT TYPE: {skeleton.agent_type}
LANGUAGE: {skeleton.language}

EXPERTISE SUMMARY:
This agent excels at {skeleton.agent_purpose.lower()} with deep specialization in {', '.join(skeleton.key_capabilities)}. 
Designed specifically for {skeleton.target_users} with focus on {', '.join(skeleton.use_cases[:3])}. 
The agent combines {skeleton.personality_traits.get('tone', 'professional')} communication style with 
{skeleton.personality_traits.get('approach', 'solution-oriented')} problem-solving approach.

This expertise profile was created through collaborative planning and represents the agent's 
core competencies and specialized knowledge areas.
"""
    
    def _extract_collaboration_insights(self, conversation_data: Dict) -> str:
        """Extract insights from collaborative session"""
        
        skeleton = conversation_data.get("skeleton")
        conversation_id = conversation_data.get("conversation_id", "unknown")
        
        return f"""
COLLABORATIVE AGENT BUILDING INSIGHTS

Session ID: {conversation_id}
Agent Created: {skeleton.agent_name if skeleton else 'Unknown'}
Agent Type: {skeleton.agent_type if skeleton else 'Unknown'}

COLLABORATION PATTERNS:
• User requested agent for: {skeleton.agent_purpose if skeleton else 'Not specified'}
• Primary focus areas: {', '.join(skeleton.key_capabilities) if skeleton else 'Not specified'}
• Target audience: {skeleton.target_users if skeleton else 'Not specified'}

USER PREFERENCES OBSERVED:
• Preferred agent type: {skeleton.agent_type if skeleton else 'Unknown'}
• Communication style: {skeleton.personality_traits.get('tone', 'Not specified') if skeleton else 'Unknown'}
• Tool preferences: {', '.join(skeleton.required_tools) if skeleton else 'Not specified'}

SESSION INSIGHTS:
This collaborative session demonstrates user preferences for {skeleton.agent_type if skeleton else 'specialized'} agents
with focus on {', '.join(skeleton.key_capabilities[:2]) if skeleton else 'various capabilities'}.

The user showed interest in agents that can handle {', '.join(skeleton.use_cases[:2]) if skeleton else 'diverse tasks'}
with {skeleton.personality_traits.get('approach', 'professional') if skeleton else 'effective'} approach.

These patterns can inform future agent recommendations and collaborative sessions.
"""
    
    def contains_domain_knowledge(self, user_input: str) -> bool:
        """Check if user input contains domain knowledge worth saving"""
        
        domain_indicators = [
            "in my industry", "in my field", "in my experience", 
            "typically we", "usually we", "in our company",
            "best practice", "common approach", "standard procedure",
            "industry standard", "professional experience", "expertise in"
        ]
        
        user_input_lower = user_input.lower()
        return any(indicator in user_input_lower for indicator in domain_indicators)