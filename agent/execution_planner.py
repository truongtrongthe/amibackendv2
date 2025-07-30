"""
Execution Planner Module
========================

Handles generation of detailed multi-step execution plans.
This module contains the execution planning logic extracted from the main agent.py.
"""

import logging
import json
import re
from datetime import datetime
from typing import Dict, Any
from .models import TaskComplexityAnalysis, MultiStepExecutionPlan, ExecutionStep
from .skill_discovery import SkillDiscoveryEngine

logger = logging.getLogger(__name__)


class ExecutionPlanner:
    """Generates detailed multi-step execution plans"""
    
    def __init__(self, available_tools: Dict[str, Any], anthropic_executor, openai_executor):
        self.available_tools = available_tools
        self.anthropic_executor = anthropic_executor
        self.openai_executor = openai_executor
        self.skill_discovery = SkillDiscoveryEngine(available_tools)
    
    async def generate_execution_plan(self, complexity_analysis: TaskComplexityAnalysis, 
                                    user_request: str, agent_config: Dict[str, Any],
                                    llm_provider: str, user_id: str, org_id: str, 
                                    model: str = None) -> MultiStepExecutionPlan:
        """
        Generate detailed multi-step execution plan based on complexity analysis
        WITH discovered skills and knowledge integration
        
        Args:
            complexity_analysis: Task complexity analysis
            user_request: Original user request
            agent_config: Agent configuration
            llm_provider: LLM provider
            user_id: User ID for knowledge access
            org_id: Organization ID for knowledge access
            model: Optional model name
            
        Returns:
            MultiStepExecutionPlan with detailed execution steps enhanced by agent's discovered skills
        """
        
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Re-discover skills for detailed planning (with caching potential)
        discovered_skills = {}
        try:
            discovered_skills_obj = await self.skill_discovery.discover_agent_skills(
                user_request, agent_config, user_id, org_id
            )
            
            # Convert to dict format for backwards compatibility
            discovered_skills = {
                "skills": discovered_skills_obj.skills,
                "methodologies": discovered_skills_obj.methodologies,
                "experience": discovered_skills_obj.experience,
                "frameworks": discovered_skills_obj.frameworks,
                "best_practices": discovered_skills_obj.best_practices,
                "capabilities_enhancement": discovered_skills_obj.capabilities_enhancement
            }
        except Exception as e:
            logger.warning(f"Skill discovery failed during planning: {e}")
            discovered_skills = {"skills": [], "methodologies": [], "experience": [], "capabilities_enhancement": {}}
        
        # Create execution plan generation prompt enhanced with skills
        system_prompt_data = agent_config.get("system_prompt", {})
        available_tools = agent_config.get("tools_list", [])
        knowledge_domains = agent_config.get("knowledge_list", [])
        
        # Build skill-aware planning context
        skills_planning_context = ""
        if discovered_skills.get("skills") or discovered_skills.get("methodologies"):
            skills_planning_context = f"""
AGENT'S DISCOVERED SKILLS & METHODOLOGIES TO INTEGRATE:

SPECIFIC SKILLS:
{chr(10).join(f"- {skill}" for skill in discovered_skills.get('skills', [])[:5])}

METHODOLOGIES & FRAMEWORKS:
{chr(10).join(f"- {method}" for method in discovered_skills.get('methodologies', [])[:5])}

ORGANIZATIONAL EXPERIENCE:
{chr(10).join(f"- {exp}" for exp in discovered_skills.get('experience', [])[:3])}

AVAILABLE FRAMEWORKS:
{chr(10).join(f"- {framework}" for framework in discovered_skills.get('frameworks', [])[:3])}

BEST PRACTICES TO APPLY:
{chr(10).join(f"- {practice}" for practice in discovered_skills.get('best_practices', [])[:3])}

SKILL-BASED ENHANCEMENTS:
- Additional specialized steps: {len(discovered_skills.get('capabilities_enhancement', {}).get('additional_steps', []))}
- Enhanced quality standards: {len(discovered_skills.get('capabilities_enhancement', {}).get('quality_standards', []))}
"""
        
        planning_prompt = f"""
        You are an expert execution planner for AI agents with access to the agent's specific skills and organizational knowledge.
        Create a detailed multi-step execution plan that leverages the agent's discovered capabilities.

        AGENT CONTEXT:
        - Agent: {agent_config.get('name')} ({system_prompt_data.get('agent_type', 'general')})
        - Available Tools: {', '.join(available_tools)}
        - Knowledge Domains: {', '.join(knowledge_domains)}
        - Domain Specialization: {system_prompt_data.get('domain_specialization', ['general'])}

        {skills_planning_context}

        TASK REQUEST: "{user_request}"

        COMPLEXITY ANALYSIS:
        - Score: {complexity_analysis.complexity_score}/10 ({complexity_analysis.complexity_level})
        - Required Steps: {complexity_analysis.required_steps}
        - Task Type: {complexity_analysis.task_type}
        - Key Challenges: {', '.join(complexity_analysis.key_challenges)}
        - Recommended Approach: {complexity_analysis.recommended_approach}

        SKILL-AWARE EXECUTION FRAMEWORK:
        Create {complexity_analysis.required_steps} detailed execution steps following this enhanced pattern:
        1. Context Analysis & Knowledge Activation (activate relevant agent knowledge)
        2. Information Gathering with Skill Application (use discovered methodologies)
        3. Analysis & Processing with Best Practices (apply organizational experience)
        4. Framework-Based Solution Generation (use discovered frameworks)
        5. Experience-Validated Synthesis (validate against organizational lessons learned)
        (+ additional specialized steps based on discovered capabilities)

        IMPORTANT: Incorporate the agent's discovered skills and methodologies into specific steps.
        For example:
        - If agent has "Lean methodology" → include "Apply Lean principles to eliminate waste"
        - If agent has "Risk assessment experience" → include "Conduct risk assessment using organizational experience"
        - If agent has "Quality frameworks" → include "Apply quality validation frameworks"

        Provide your plan in this EXACT JSON format:
        {{
            "execution_steps": [
                {{
                    "step_number": 1,
                    "name": "Context Analysis & Knowledge Activation",
                    "description": "Detailed description incorporating agent's skills",
                    "action": "Specific action that leverages discovered capabilities",
                    "tools_needed": ["tool1", "tool2"],
                    "success_criteria": "How to measure success including skill application",
                    "deliverable": "What this step produces enhanced by agent knowledge",
                    "dependencies": [],
                    "estimated_time": "short|medium|long",
                    "validation_checkpoints": ["checkpoint1", "checkpoint2"],
                    "skills_applied": ["skill1", "methodology1"]
                }}
            ],
            "quality_checkpoints": [
                {{
                    "checkpoint_name": "Knowledge-Based Quality Check",
                    "trigger_after_step": 2,
                    "validation_criteria": "Validate using agent's experience and best practices",
                    "success_threshold": "Success criteria enhanced by discovered knowledge"
                }}
            ],
            "success_metrics": ["metric1", "metric2"],
            "risk_mitigation": ["risk1_mitigation", "risk2_mitigation"],
            "total_estimated_time": "short|medium|long",
            "skills_integration_summary": "How agent's skills enhance the execution plan"
        }}
        """
        
        try:
            # Generate execution plan
            if llm_provider.lower() == "anthropic":
                plan_response = await self.anthropic_executor._analyze_with_anthropic(planning_prompt, model)
            else:
                plan_response = await self.openai_executor._analyze_with_openai(planning_prompt, model)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', plan_response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
                
                # Create ExecutionStep objects with skill information
                execution_steps = []
                for step_data in plan_data.get("execution_steps", []):
                    step = ExecutionStep(
                        step_number=step_data.get("step_number", 1),
                        name=step_data.get("name", "Execution Step"),
                        description=step_data.get("description", ""),
                        action=step_data.get("action", ""),
                        tools_needed=step_data.get("tools_needed", []),
                        success_criteria=step_data.get("success_criteria", ""),
                        deliverable=step_data.get("deliverable", ""),
                        dependencies=step_data.get("dependencies", []),
                        estimated_time=step_data.get("estimated_time", "medium"),
                        validation_checkpoints=step_data.get("validation_checkpoints", [])
                    )
                    execution_steps.append(step)
                
                # Add skill-based additional steps if any
                additional_steps = discovered_skills.get("capabilities_enhancement", {}).get("additional_steps", [])
                for i, additional_step in enumerate(additional_steps):
                    step_number = len(execution_steps) + 1
                    step = ExecutionStep(
                        step_number=step_number,
                        name=additional_step["name"],
                        description=additional_step["description"],
                        action=f"Apply discovered capabilities: {additional_step['description']}",
                        tools_needed=additional_step.get("tools_needed", ["brain_vector"]),
                        success_criteria="Successfully integrate agent's specialized knowledge",
                        deliverable=f"Enhanced output using agent's {additional_step['name'].lower()} capabilities",
                        dependencies=[step_number - 1] if execution_steps else [],
                        estimated_time="medium",
                        validation_checkpoints=["Knowledge integration verified", "Quality standards met"]
                    )
                    execution_steps.append(step)
                
                # Enhance quality checkpoints with skill-based validation
                quality_checkpoints = plan_data.get("quality_checkpoints", [])
                if discovered_skills.get("experience"):
                    quality_checkpoints.append({
                        "checkpoint_name": "Experience-Based Validation",
                        "trigger_after_step": len(execution_steps) - 1,
                        "validation_criteria": "Validate against organizational experience and lessons learned",
                        "success_threshold": "Meets historical success patterns and quality standards"
                    })
                
                # Enhance success metrics with skill-specific measures
                success_metrics = plan_data.get("success_metrics", [])
                if discovered_skills.get("methodologies"):
                    success_metrics.append("Methodology application effectiveness")
                if discovered_skills.get("best_practices"):
                    success_metrics.append("Best practices integration completeness")
                
                return MultiStepExecutionPlan(
                    plan_id=plan_id,
                    task_description=user_request,
                    complexity_analysis=complexity_analysis,
                    execution_steps=execution_steps,
                    quality_checkpoints=quality_checkpoints,
                    success_metrics=success_metrics,
                    risk_mitigation=plan_data.get("risk_mitigation", []),
                    total_estimated_time=plan_data.get("total_estimated_time", "medium"),
                    created_at=datetime.now()
                )
            else:
                # Fallback to skill-enhanced default plan
                return self._create_skill_enhanced_execution_plan(plan_id, user_request, complexity_analysis, discovered_skills)
                
        except Exception as e:
            logger.warning(f"LLM execution planning failed: {e}, using skill-enhanced default plan")
            return self._create_skill_enhanced_execution_plan(plan_id, user_request, complexity_analysis, discovered_skills)
    
    def _create_skill_enhanced_execution_plan(self, plan_id: str, user_request: str, 
                                            complexity_analysis: TaskComplexityAnalysis,
                                            discovered_skills: Dict[str, Any]) -> MultiStepExecutionPlan:
        """Create a default execution plan enhanced with discovered skills"""
        
        # Create basic execution steps based on complexity
        base_steps = [
            ExecutionStep(
                step_number=1,
                name="Context Analysis & Knowledge Activation",
                description="Analyze the request and activate relevant agent knowledge and skills",
                action="Understand task requirements and identify applicable methodologies from knowledge base",
                tools_needed=["context", "brain_vector"],
                success_criteria="Clear understanding of task objectives with relevant skills identified",
                deliverable="Task analysis with activated agent knowledge context",
                dependencies=[],
                estimated_time="short",
                validation_checkpoints=["Requirements clear", "Relevant skills identified", "Context established"]
            ),
            ExecutionStep(
                step_number=2,
                name="Information Gathering with Skill Application",
                description="Gather necessary information using agent's specialized approaches",
                action="Use available tools and apply discovered methodologies to collect relevant information",
                tools_needed=["search_factory", "brain_vector"],
                success_criteria="Sufficient information collected using best practices",
                deliverable="Gathered information enhanced by agent's specialized knowledge",
                dependencies=[1],
                estimated_time="medium",
                validation_checkpoints=["Information quality validated", "Methodology application verified", "Completeness check"]
            ),
            ExecutionStep(
                step_number=3,
                name="Analysis & Processing with Best Practices",
                description="Analyze gathered information using organizational best practices and experience",
                action="Process and analyze collected information applying discovered frameworks",
                tools_needed=["business_logic", "brain_vector"],
                success_criteria="Analysis completed using agent's specialized knowledge",
                deliverable="Processed analysis results enhanced by organizational experience",
                dependencies=[2],
                estimated_time="medium",
                validation_checkpoints=["Analysis accuracy verified", "Best practices applied", "Results validity confirmed"]
            )
        ]
        
        # Add skill-specific steps based on discovered capabilities
        if discovered_skills.get("methodologies"):
            methodology_step = ExecutionStep(
                step_number=len(base_steps) + 1,
                name="Methodology Application",
                description=f"Apply discovered methodologies: {', '.join(discovered_skills['methodologies'][:2])}",
                action="Integrate organizational methodologies and frameworks into solution approach",
                tools_needed=["brain_vector", "business_logic"],
                success_criteria="Methodologies successfully applied to enhance solution quality",
                deliverable="Solution approach enhanced by organizational methodologies",
                dependencies=[len(base_steps)],
                estimated_time="medium",
                validation_checkpoints=["Methodology integration verified", "Framework application confirmed"]
            )
            base_steps.append(methodology_step)
        
        # Add additional steps for higher complexity
        if complexity_analysis.complexity_level in ["standard", "complex"]:
            solution_step = ExecutionStep(
                step_number=len(base_steps) + 1,
                name="Framework-Based Solution Generation",
                description="Generate solutions using agent's discovered frameworks and experience",
                action="Create actionable solutions leveraging organizational knowledge and best practices",
                tools_needed=["business_logic", "brain_vector"],
                success_criteria="Solutions are practical, actionable, and incorporate agent's specialized knowledge",
                deliverable="Generated solutions enhanced by organizational frameworks",
                dependencies=[len(base_steps)],
                estimated_time="medium",
                validation_checkpoints=["Solution feasibility confirmed", "Framework integration verified", "Quality standards met"]
            )
            base_steps.append(solution_step)
        
        if complexity_analysis.complexity_level == "complex":
            validation_step = ExecutionStep(
                step_number=len(base_steps) + 1,
                name="Experience-Based Integration & Validation",
                description="Integrate solutions and validate using organizational experience",
                action="Ensure all components work together effectively using lessons learned",
                tools_needed=["business_logic", "context", "brain_vector"],
                success_criteria="Integrated solution validated against organizational experience",
                deliverable="Validated integrated solution enhanced by historical success patterns",
                dependencies=[len(base_steps)],
                estimated_time="medium",
                validation_checkpoints=["Integration success verified", "Experience validation complete", "Quality assurance passed"]
            )
            base_steps.append(validation_step)
        
        # Final synthesis step enhanced with skills
        final_step_number = len(base_steps) + 1
        synthesis_step = ExecutionStep(
            step_number=final_step_number,
            name="Knowledge-Enhanced Response Synthesis",
            description="Synthesize all results into final response using agent's complete knowledge",
            action="Create comprehensive final response incorporating all discovered capabilities",
            tools_needed=[],
            success_criteria="Complete, coherent response that demonstrates agent's specialized knowledge",
            deliverable="Final comprehensive response enhanced by agent's full knowledge base",
            dependencies=[final_step_number - 1],
            estimated_time="short",
            validation_checkpoints=["Response completeness verified", "Knowledge integration confirmed", "Quality standards exceeded"]
        )
        base_steps.append(synthesis_step)
        
        # Create enhanced quality checkpoints
        quality_checkpoints = [
            {
                "checkpoint_name": "Knowledge Integration Quality Check",
                "trigger_after_step": len(base_steps) // 2,
                "validation_criteria": "Verify effective integration of agent's knowledge and skills",
                "success_threshold": "All discovered capabilities are being effectively applied"
            }
        ]
        
        # Add experience-based checkpoint if available
        if discovered_skills.get("experience"):
            quality_checkpoints.append({
                "checkpoint_name": "Experience-Based Validation",
                "trigger_after_step": len(base_steps) - 1,
                "validation_criteria": "Validate outputs against organizational experience and lessons learned",
                "success_threshold": "Meets or exceeds historical success patterns"
            })
        
        # Enhanced success metrics
        success_metrics = ["Task completion", "Quality standards met", "User satisfaction"]
        if discovered_skills.get("methodologies"):
            success_metrics.append("Methodology application effectiveness")
        if discovered_skills.get("best_practices"):
            success_metrics.append("Best practices integration success")
        if discovered_skills.get("frameworks"):
            success_metrics.append("Framework utilization quality")
        
        # Enhanced risk mitigation
        risk_mitigation = ["Fallback to simpler approach if needed", "Error handling at each step"]
        if discovered_skills.get("experience"):
            risk_mitigation.append("Apply lessons learned to avoid historical pitfalls")
        
        return MultiStepExecutionPlan(
            plan_id=plan_id,
            task_description=user_request,
            complexity_analysis=complexity_analysis,
            execution_steps=base_steps,
            quality_checkpoints=quality_checkpoints,
            success_metrics=success_metrics,
            risk_mitigation=risk_mitigation,
            total_estimated_time=complexity_analysis.estimated_duration,
            created_at=datetime.now()
        ) 