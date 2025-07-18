"""
Human Context Tool - Mom Test-inspired discovery system
Combines organizational data, braingraph info, and conversational discovery principles
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class MomTestEngine:
    """Generates Mom Test-compliant discovery questions using LLM"""
    
    def __init__(self):
        # Mom Test principles for LLM generation
        self.mom_test_principles = """
        The Mom Test principles:
        1. Ask about past behavior, not future hypotheticals
        2. Focus on specific, concrete examples from recent work
        3. Ask about time, effort, and actual experiences
        4. Avoid leading questions about solutions
        5. Be genuinely curious about their real work
        6. Ask follow-up questions like "How long did that take?" and "Can you walk me through what you did?"
        """
    
    async def generate_contextual_questions(self, org_context: Dict, user_context: Dict, llm_provider: str = "openai") -> List[str]:
        """Generate Mom Test questions using LLM based on context"""
        
        # Create context-aware prompt for LLM
        context_prompt = f"""
        You are a Mom Test expert generating discovery questions for a co-builder AI named Ami.
        
        {self.mom_test_principles}
        
        CONTEXT:
        Organization: {org_context.get('name', 'Unknown')} ({org_context.get('industry', 'unknown industry')})
        User Role: {user_context.get('role', 'unknown role')}
        Department: {user_context.get('department', 'unknown')}
        Organization Challenges: {', '.join(org_context.get('challenges', []))}
        User Interests: {', '.join(user_context.get('interests', []))}
        
        Generate 3 Mom Test questions that:
        - Are specific to their industry and role
        - Ask about past behavior and concrete examples
        - Focus on time, effort, and actual experiences
        - Are conversational and natural
        - Avoid mentioning AI or automation
        
        Format as JSON array:
        ["question 1", "question 2", "question 3"]
        """
        
        try:
            # Use LLM to generate questions
            if llm_provider.lower() == "anthropic":
                from anthropic_tool import AnthropicTool
                generator = AnthropicTool()
                response = await asyncio.to_thread(generator.process_query, context_prompt)
            else:
                from openai_tool import OpenAITool
                generator = OpenAITool()
                response = await asyncio.to_thread(generator.generate_response, context_prompt)
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group(0))
                return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error(f"Failed to generate contextual questions: {e}")
        
        # Fallback to generic Mom Test questions
        return self._get_fallback_questions(org_context, user_context)
    
    def _get_fallback_questions(self, org_context: Dict, user_context: Dict) -> List[str]:
        """Fallback questions if LLM generation fails"""
        industry = org_context.get("industry", "").lower()
        
        if "finance" in industry:
            return [
                "What took up most of your time in your last project?",
                "What was the most tedious part of your recent work?",
                "How long did your last analysis actually take?"
            ]
        elif "healthcare" in industry:
            return [
                "What did you spend most of your time on during your last shift?",
                "What was the most time-consuming part of your recent work?",
                "When did you last think 'there has to be a faster way'?"
            ]
        else:
            return [
                "What did you spend most of your time on yesterday?",
                "What was the last task that took longer than expected?",
                "Can you walk me through your typical workday?"
            ]
    
    async def generate_natural_opener(self, context: Dict, llm_provider: str = "openai") -> str:
        """Generate a natural, conversational opener using LLM based on context"""
        
        org_profile = context.get("org_profile", {})
        user_profile = context.get("user_profile", {})
        
        # Create prompt for LLM to generate natural opener
        opener_prompt = f"""
        You are Ami, an AI co-builder. Generate a natural, conversational introduction that:
        
        CONTEXT:
        Organization: {org_profile.get('name', 'Unknown')} ({org_profile.get('industry', 'unknown industry')})
        User Role: {user_profile.get('role', 'unknown role')}
        Organization Size: {org_profile.get('size', 'unknown size')}
        
        Requirements:
        1. Introduce yourself as Ami, an AI co-builder
        2. Acknowledge their context naturally (if available)
        3. Be warm and conversational
        4. Keep it concise (1-2 sentences)
        5. Set up for genuine curiosity about their work
        
        Don't mention AI agents or building solutions yet - just introduce yourself warmly.
        """
        
        try:
            # Use LLM to generate opener
            if llm_provider.lower() == "anthropic":
                from anthropic_tool import AnthropicTool
                generator = AnthropicTool()
                response = await asyncio.to_thread(generator.process_query, opener_prompt)
            else:
                from openai_tool import OpenAITool
                generator = OpenAITool()
                response = await asyncio.to_thread(generator.generate_response, opener_prompt)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate natural opener: {e}")
            
            # Fallback to simple opener
            org_name = org_profile.get("name", "")
            industry = org_profile.get("industry", "")
            
            if org_name and industry:
                return f"Hi! I'm Ami, your AI co-builder. I help people at {industry} companies like {org_name} build practical AI agents."
            elif industry:
                return f"Hi! I'm Ami, your AI co-builder. I help people in {industry} build practical AI agents."
            else:
                return "Hi! I'm Ami, your AI co-builder. I help people build practical AI agents."


class ContextDataRetriever:
    """Retrieves context from org and braingraph databases"""
    
    def __init__(self, db_connection=None, braingraph_service=None):
        self.db = db_connection
        self.braingraph = braingraph_service
    
    async def get_org_profile(self, org_id: str) -> Dict:
        """Get organization profile from database"""
        
        try:
            if not self.db:
                # Fallback to mock data for testing
                return self._get_mock_org_profile(org_id)
            
            # In real implementation, query the database
            query = """
            SELECT name, description, industry, size, focus_areas, 
                   key_challenges, tech_stack, business_model
            FROM organizations 
            WHERE org_id = %s
            """
            
            org_data = await self.db.fetch_one(query, (org_id,))
            
            if org_data:
                return {
                    "name": org_data.get("name"),
                    "industry": org_data.get("industry"),
                    "size": org_data.get("size"),
                    "challenges": org_data.get("key_challenges", []),
                    "focus_areas": org_data.get("focus_areas", []),
                    "tech_maturity": org_data.get("tech_stack")
                }
            
        except Exception as e:
            logger.error(f"Error retrieving org profile: {e}")
        
        return self._get_mock_org_profile(org_id)
    
    def _get_mock_org_profile(self, org_id: str) -> Dict:
        """Mock org profile for testing"""
        mock_profiles = {
            "test_org": {
                "name": "TechCorp",
                "industry": "finance",
                "size": "50-100 employees",
                "challenges": ["manual processes", "compliance reporting", "data analysis"],
                "focus_areas": ["fintech", "M&A consulting", "risk management"],
                "tech_maturity": "intermediate"
            },
            "default": {
                "name": "Your Organization",
                "industry": "business",
                "size": "unknown",
                "challenges": [],
                "focus_areas": [],
                "tech_maturity": "unknown"
            }
        }
        return mock_profiles.get(org_id, mock_profiles["default"])
    
    async def get_user_profile(self, user_id: str) -> Dict:
        """Get user profile from database"""
        
        try:
            if not self.db:
                return self._get_mock_user_profile(user_id)
            
            # In real implementation, query the database
            query = """
            SELECT name, role, department, interests, skills, 
                   previous_projects, preferences
            FROM users 
            WHERE user_id = %s
            """
            
            user_data = await self.db.fetch_one(query, (user_id,))
            
            if user_data:
                return {
                    "name": user_data.get("name"),
                    "role": user_data.get("role"),
                    "department": user_data.get("department"),
                    "interests": user_data.get("interests", []),
                    "skills": user_data.get("skills", []),
                    "previous_projects": user_data.get("previous_projects", [])
                }
            
        except Exception as e:
            logger.error(f"Error retrieving user profile: {e}")
        
        return self._get_mock_user_profile(user_id)
    
    def _get_mock_user_profile(self, user_id: str) -> Dict:
        """Mock user profile for testing"""
        return {
            "name": "User",
            "role": "analyst",
            "department": "operations",
            "interests": ["automation", "data analysis"],
            "skills": ["Excel", "Python", "SQL"],
            "previous_projects": ["financial modeling", "process optimization"]
        }
    
    async def get_braingraph_context(self, user_id: str, org_id: str) -> Dict:
        """Get insights from braingraph"""
        
        try:
            if not self.braingraph:
                return self._get_mock_braingraph_context(user_id, org_id)
            
            # Get user's braingraph nodes and connections
            user_nodes = await self.braingraph.get_user_nodes(user_id)
            org_nodes = await self.braingraph.get_org_nodes(org_id)
            
            # Analyze patterns
            context = {
                "user_interests": self._extract_interests(user_nodes),
                "work_patterns": self._analyze_work_patterns(user_nodes),
                "org_knowledge": self._summarize_org_knowledge(org_nodes),
                "recent_topics": self._get_recent_topics(user_nodes)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving braingraph context: {e}")
        
        return self._get_mock_braingraph_context(user_id, org_id)
    
    def _get_mock_braingraph_context(self, user_id: str, org_id: str) -> Dict:
        """Mock braingraph context for testing"""
        return {
            "user_interests": ["AI automation", "financial analysis", "process improvement"],
            "work_patterns": ["data-heavy tasks", "repetitive analysis", "client reporting"],
            "org_knowledge": ["M&A processes", "compliance requirements", "financial modeling"],
            "recent_topics": ["due diligence", "valuation models", "risk assessment"]
        }
    
    


class HumanContextTool:
    """
    Mom Test-inspired human context discovery tool
    Combines organizational data, braingraph info, and Mom Test principles
    """
    
    def __init__(self, db_connection=None, braingraph_service=None):
        self.data_retriever = ContextDataRetriever(db_connection, braingraph_service)
        self.mom_test_engine = MomTestEngine()
    
    async def get_human_context(self, user_id: str, org_id: str, conversation_history: List = None) -> Dict:
        """Get comprehensive human context for Mom Test discovery"""
        
        # Get all context data
        org_profile = await self.data_retriever.get_org_profile(org_id)
        user_profile = await self.data_retriever.get_user_profile(user_id)
        braingraph_insights = await self.data_retriever.get_braingraph_context(user_id, org_id)
        
        # Analyze conversation for additional clues
        conversation_patterns = self._analyze_conversation_patterns(conversation_history)
        
        context = {
            "org_profile": org_profile,
            "user_profile": user_profile,
            "braingraph_insights": braingraph_insights,
            "conversation_patterns": conversation_patterns
        }
        
        return context
    
    async def generate_discovery_strategy(self, context: Dict, llm_provider: str = "openai") -> Dict:
        """Generate Mom Test-based discovery strategy"""
        
        # Generate natural opener using LLM
        opener = await self.mom_test_engine.generate_natural_opener(context, llm_provider)
        
        # Generate contextual Mom Test questions using LLM
        questions = await self.mom_test_engine.generate_contextual_questions(
            context.get("org_profile", {}),
            context.get("user_profile", {}),
            llm_provider
        )
        
        # Create conversational discovery approach using LLM
        discovery_approach = await self._create_conversational_approach(context, questions, llm_provider)
        
        return {
            "opener": opener,
            "discovery_questions": questions,
            "conversational_approach": discovery_approach,
            "context_summary": self._create_context_summary(context)
        }
    
    def _analyze_conversation_patterns(self, conversation_history: List) -> Dict:
        """Analyze conversation history for domain clues"""
        
        if not conversation_history:
            return {"patterns": [], "topics": [], "clues": []}
        
        # Extract patterns from conversation
        patterns = []
        topics = []
        clues = []
        
        for message in conversation_history[-5:]:  # Last 5 messages
            content = message.get("content", "").lower()
            
            # Look for industry clues
            if any(word in content for word in ["finance", "financial", "money", "investment"]):
                clues.append("finance")
            elif any(word in content for word in ["patient", "medical", "healthcare", "doctor"]):
                clues.append("healthcare")
            elif any(word in content for word in ["customer", "sales", "retail", "store"]):
                clues.append("retail")
            elif any(word in content for word in ["consulting", "client", "advisory"]):
                clues.append("consulting")
            
            # Look for task patterns
            if any(word in content for word in ["analyze", "report", "data", "spreadsheet"]):
                patterns.append("analytical_work")
            elif any(word in content for word in ["meeting", "presentation", "client"]):
                patterns.append("client_facing")
        
        return {
            "patterns": list(set(patterns)),
            "topics": list(set(topics)),
            "clues": list(set(clues))
        }
    
    async def _create_conversational_approach(self, context: Dict, questions: List[str], llm_provider: str = "openai") -> str:
        """Create a conversational approach using LLM based on context"""
        
        org_profile = context.get("org_profile", {})
        user_profile = context.get("user_profile", {})
        
        # Create prompt for LLM to generate conversational approach
        approach_prompt = f"""
        You are a Mom Test expert creating a conversational discovery approach for Ami, an AI co-builder.
        
        CONTEXT:
        Organization: {org_profile.get('name', 'Unknown')} ({org_profile.get('industry', 'unknown industry')})
        User Role: {user_profile.get('role', 'unknown role')}
        Generated Questions: {questions}
        
        Create a brief, natural conversational approach that:
        1. Starts with genuine curiosity (not AI pitching)
        2. Uses the generated questions naturally
        3. Focuses on listening and follow-up questions
        4. Maintains conversational flow
        5. Avoids being salesy or robotic
        
        Keep it concise (3-4 sentences) and conversational. Focus on being genuinely curious about their work.
        """
        
        try:
            # Use LLM to generate conversational approach
            if llm_provider.lower() == "anthropic":
                from anthropic_tool import AnthropicTool
                generator = AnthropicTool()
                response = await asyncio.to_thread(generator.process_query, approach_prompt)
            else:
                from openai_tool import OpenAITool
                generator = OpenAITool()
                response = await asyncio.to_thread(generator.generate_response, approach_prompt)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate conversational approach: {e}")
            
            # Fallback to simple approach
            return f"""
            Start with genuine curiosity: "{questions[0] if questions else 'What did you spend most of your time on yesterday?'}"
            
            Listen for concrete examples, then dig deeper with follow-up questions.
            
            Stay conversational - you're learning about their real problems, not pitching solutions.
            """
    
    def _create_context_summary(self, context: Dict) -> str:
        """Create a concise context summary for the AI"""
        
        org = context.get("org_profile", {})
        user = context.get("user_profile", {})
        brain = context.get("braingraph_insights", {})
        
        summary = f"""
        Context: {org.get('name', 'Unknown org')} ({org.get('industry', 'unknown industry')})
        User: {user.get('role', 'unknown role')} interested in {', '.join(brain.get('user_interests', [])[:2])}
        Recent focus: {', '.join(brain.get('recent_topics', [])[:2])}
        """
        
        return summary.strip()


# Tool function for integration with exec_tool.py
def create_human_context_tool(db_connection=None, braingraph_service=None):
    """Factory function to create human context tool"""
    return HumanContextTool(db_connection, braingraph_service)


# Example usage for testing
if __name__ == "__main__":
    async def test_human_context_tool():
        tool = HumanContextTool()
        
        context = await tool.get_human_context("test_user", "test_org")
        strategy = await tool.generate_discovery_strategy(context)
        
        print("Context:", json.dumps(context, indent=2))
        print("\nStrategy:", json.dumps(strategy, indent=2))
    
    asyncio.run(test_human_context_tool()) 