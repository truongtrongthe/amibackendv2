"""
Agent Task Complexity Analyzer
==============================

Handles analysis of task complexity to determine appropriate execution approach.
"""

import logging
from typing import Dict, Any
from .models import TaskComplexityAnalysis, DiscoveredSkills
from .skill_discovery import SkillDiscoveryEngine

logger = logging.getLogger(__name__)


class TaskComplexityAnalyzer:
    """Analyzes task complexity and determines execution approach"""
    
    def __init__(self, available_tools: Dict[str, Any], anthropic_executor, openai_executor):
        self.available_tools = available_tools
        self.anthropic_executor = anthropic_executor
        self.openai_executor = openai_executor
        self.skill_discovery = SkillDiscoveryEngine(available_tools)
        
    async def analyze_task_complexity(self, user_request: str, agent_config: Dict[str, Any], 
                                    llm_provider: str, user_id: str, org_id: str, 
                                    model: str = None) -> TaskComplexityAnalysis:
        """
        Analyze task complexity to determine if multi-step planning is needed
        WITH skill discovery and knowledge-aware enhancement
        
        Args:
            user_request: The user's task request
            agent_config: Agent configuration with capabilities
            llm_provider: LLM provider for analysis
            user_id: User ID for knowledge access
            org_id: Organization ID for knowledge access
            model: Optional model name
            
        Returns:
            TaskComplexityAnalysis with complexity scoring and recommendations enhanced by discovered skills
        """
        
        # Get agent's execution capabilities
        system_prompt_data = agent_config.get("system_prompt", {})
        execution_capabilities = system_prompt_data.get("execution_capabilities", {})
        max_complexity = execution_capabilities.get("max_complexity_score", 10)
        complexity_thresholds = execution_capabilities.get("complexity_thresholds", {
            "simple": {"min": 1, "max": 3, "steps": 3},
            "standard": {"min": 4, "max": 7, "steps": 5},
            "complex": {"min": 8, "max": 10, "steps": 7}
        })
        
        # Discover relevant skills from agent's knowledge base
        discovered_skills = await self._discover_skills_safe(user_request, agent_config, user_id, org_id)
        
        # Create complexity analysis prompt enhanced with discovered skills
        skills_context = self._build_skills_context(discovered_skills)
        
        analysis_prompt = f"""
        You are an expert task complexity analyzer with access to the agent's specific knowledge and skills. 
        Analyze this task request and determine its complexity level, considering the agent's discovered capabilities.

        AGENT CAPABILITIES:
        - Agent Name: {agent_config.get('name', 'Unknown')}
        - Agent Type: {system_prompt_data.get('agent_type', 'general')}
        - Available Tools: {', '.join(agent_config.get('tools_list', []))}
        - Knowledge Domains: {', '.join(agent_config.get('knowledge_list', []))}
        - Max Complexity Score: {max_complexity}
        - Domain Specialization: {system_prompt_data.get('domain_specialization', ['general'])}

        {skills_context}

        TASK REQUEST: "{user_request}"

        COMPLEXITY SCORING CRITERIA (Enhanced with Skills):
        1-3 (Simple): Single tool usage, direct information retrieval, straightforward Q&A
        4-7 (Standard): Multiple tool coordination, analysis tasks, document processing
        8-10 (Complex): Multi-step reasoning, cross-domain analysis, complex problem solving
        
        SKILL-BASED ADJUSTMENTS:
        - If agent has relevant methodologies/frameworks: +1-2 complexity points (can handle more complex tasks)
        - If agent has specific experience: More sophisticated execution steps
        - If agent has best practices: Enhanced quality standards

        Consider the agent's discovered skills when scoring. An agent with relevant experience can handle 
        more complex approaches to tasks that might otherwise be simpler.

        Provide your analysis in this EXACT JSON format:
        {{
            "complexity_score": 6,
            "complexity_level": "standard",
            "task_type": "information_retrieval|problem_solving|analysis|communication|automation",
            "required_steps": 5,
            "estimated_duration": "short|medium|long",
            "key_challenges": ["challenge1", "challenge2"],
            "recommended_approach": "Detailed description of recommended execution approach incorporating discovered skills",
            "confidence": 0.85,
            "skill_integration": "How discovered skills enhance the execution approach",
            "reasoning": "Explanation of complexity scoring including skill-based adjustments"
        }}
        """
        
        try:
            # Get complexity analysis from LLM
            if llm_provider.lower() == "anthropic":
                analysis_response = await self.anthropic_executor._analyze_with_anthropic(analysis_prompt, model)
            else:
                analysis_response = await self.openai_executor._analyze_with_openai(analysis_prompt, model)
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group(0))
                
                # Apply skill-based complexity boost
                base_complexity_score = analysis_data.get("complexity_score", 5)
                skill_boost = discovered_skills.capabilities_enhancement.get("complexity_boost", 0)
                complexity_score = min(base_complexity_score + skill_boost, max_complexity)
                
                # Determine complexity level based on thresholds (with skill enhancement)
                complexity_level = "standard"
                required_steps = 5
                
                for level, threshold in complexity_thresholds.items():
                    if threshold["min"] <= complexity_score <= threshold["max"]:
                        complexity_level = level
                        required_steps = threshold["steps"]
                        break
                
                # Add skill-based additional steps
                additional_steps = len(discovered_skills.capabilities_enhancement.get("additional_steps", []))
                if additional_steps > 0:
                    required_steps += additional_steps
                
                # Enhanced recommended approach
                base_approach = analysis_data.get("recommended_approach", "Standard execution approach")
                if discovered_skills.methodologies:
                    base_approach += f" Enhanced with discovered methodologies: {', '.join(discovered_skills.methodologies[:2])}"
                
                return TaskComplexityAnalysis(
                    complexity_score=complexity_score,
                    complexity_level=complexity_level,
                    task_type=analysis_data.get("task_type", "general"),
                    required_steps=required_steps,
                    estimated_duration=analysis_data.get("estimated_duration", "medium"),
                    key_challenges=analysis_data.get("key_challenges", []),
                    recommended_approach=base_approach,
                    confidence=analysis_data.get("confidence", 0.7)
                )
            else:
                # Fallback to heuristic-based analysis with skill enhancement
                return self._heuristic_complexity_analysis_with_skills(user_request, agent_config, discovered_skills)
                
        except Exception as e:
            logger.warning(f"LLM complexity analysis failed: {e}, using heuristic analysis")
            return self._heuristic_complexity_analysis_with_skills(user_request, agent_config, discovered_skills)
    
    async def _discover_skills_safe(self, user_request: str, agent_config: Dict[str, Any], 
                                   user_id: str, org_id: str) -> DiscoveredSkills:
        """Safely discover skills with error handling"""
        try:
            discovered_skills = await self.skill_discovery.discover_agent_skills(
                user_request, agent_config, user_id, org_id
            )
            
            # Log discovered skills
            if discovered_skills.skills or discovered_skills.methodologies:
                logger.info(f"Discovered {len(discovered_skills.skills)} skills, {len(discovered_skills.methodologies)} methodologies")
                
            return discovered_skills
                
        except Exception as e:
            logger.warning(f"Skill discovery failed during complexity analysis: {e}")
            return DiscoveredSkills(
                skills=[], methodologies=[], experience=[], frameworks=[], 
                best_practices=[], capabilities_enhancement={}
            )
    
    def _build_skills_context(self, discovered_skills: DiscoveredSkills) -> str:
        """Build skills context for complexity analysis prompt"""
        if not (discovered_skills.skills or discovered_skills.methodologies or discovered_skills.frameworks):
            return ""
            
        return f"""
DISCOVERED AGENT SKILLS & EXPERIENCE:
- Skills: {', '.join(discovered_skills.skills[:3])}
- Methodologies: {', '.join(discovered_skills.methodologies[:3])}
- Frameworks: {', '.join(discovered_skills.frameworks[:3])}
- Best Practices: {', '.join(discovered_skills.best_practices[:2])}

CAPABILITY ENHANCEMENT:
- Complexity Boost: +{discovered_skills.capabilities_enhancement.get('complexity_boost', 0)} points
- Additional Capabilities: {len(discovered_skills.capabilities_enhancement.get('additional_steps', []))} specialized steps available
"""
    
    def _heuristic_complexity_analysis_with_skills(self, user_request: str, agent_config: Dict[str, Any], 
                                                 discovered_skills: DiscoveredSkills) -> TaskComplexityAnalysis:
        """Fallback heuristic-based complexity analysis enhanced with discovered skills"""
        
        request_lower = user_request.lower()
        complexity_indicators = {
            'simple': ['what', 'who', 'when', 'where', 'define', 'explain', 'list'],
            'standard': ['analyze', 'compare', 'evaluate', 'process', 'review', 'optimize'],
            'complex': ['integrate', 'coordinate', 'transform', 'redesign', 'implement', 'multiple', 'comprehensive']
        }
        
        # Count indicators
        simple_count = sum(1 for indicator in complexity_indicators['simple'] if indicator in request_lower)
        standard_count = sum(1 for indicator in complexity_indicators['standard'] if indicator in request_lower)
        complex_count = sum(1 for indicator in complexity_indicators['complex'] if indicator in request_lower)
        
        # Determine base complexity
        if complex_count >= 2 or len(user_request) > 200:
            complexity_score = 8
            complexity_level = "complex"
            required_steps = 7
        elif standard_count >= 1 or simple_count == 0:
            complexity_score = 5
            complexity_level = "standard"
            required_steps = 5
        else:
            complexity_score = 3
            complexity_level = "simple"
            required_steps = 3
        
        # Apply skill-based enhancements
        skill_boost = discovered_skills.capabilities_enhancement.get("complexity_boost", 0)
        complexity_score = min(complexity_score + skill_boost, 10)
        
        # Add skill-based steps
        additional_steps = len(discovered_skills.capabilities_enhancement.get("additional_steps", []))
        required_steps += additional_steps
        
        # Update complexity level based on enhanced score
        if complexity_score >= 8:
            complexity_level = "complex"
        elif complexity_score >= 4:
            complexity_level = "standard"
        
        # Enhanced approach description
        approach = "Standard execution with complexity-appropriate steps"
        if discovered_skills.methodologies:
            approach += f" enhanced with {len(discovered_skills.methodologies)} discovered methodologies"
        
        return TaskComplexityAnalysis(
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            task_type="general",
            required_steps=required_steps,
            estimated_duration="medium",
            key_challenges=["Heuristic analysis - enhanced with discovered skills"],
            recommended_approach=approach,
            confidence=0.7
        ) 