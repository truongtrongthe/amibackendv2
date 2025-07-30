"""
Agent Skill Discovery Module
============================

Handles discovery of agent skills, methodologies, and experience from the knowledge base.
"""

import asyncio
import logging
from typing import Dict, List, Any
from .models import DiscoveredSkills

logger = logging.getLogger(__name__)


class SkillDiscoveryEngine:
    """Engine for discovering agent skills and capabilities from knowledge base"""
    
    def __init__(self, available_tools: Dict[str, Any]):
        self.available_tools = available_tools
        
    async def discover_agent_skills(self, user_request: str, agent_config: Dict[str, Any], 
                                  user_id: str, org_id: str) -> DiscoveredSkills:
        """
        Discover relevant agent skills and experience from knowledge base for the current task
        
        Args:
            user_request: The user's task request
            agent_config: Agent configuration
            user_id: User ID  
            org_id: Organization ID
            
        Returns:
            DiscoveredSkills containing skills, methodologies, and experience relevant to the task
        """
        
        knowledge_list = agent_config.get("knowledge_list", [])
        if not knowledge_list or "brain_vector" not in self.available_tools:
            return DiscoveredSkills(
                skills=[], methodologies=[], experience=[], frameworks=[], 
                best_practices=[], capabilities_enhancement={}
            )
        
        discovered_skills = {
            "skills": [],
            "methodologies": [], 
            "experience": [],
            "frameworks": [],
            "best_practices": [],
            "capabilities_enhancement": {}
        }
        
        try:
            # Query for specific skill types related to the task
            skill_queries = [
                f"{user_request} methodology framework approach",
                f"{user_request} best practices experience lessons learned", 
                f"{user_request} skills expertise capabilities",
                f"{user_request} process workflow procedures",
                f"{user_request} tools techniques methods"
            ]
            
            for query in skill_queries:
                try:
                    # Search across all knowledge domains - call async method directly
                    knowledge_results = await self.available_tools["brain_vector"].query_knowledge(
                        user_id=user_id,
                        org_id=org_id,
                        query=query,
                        limit=10  # More results for skill discovery
                    )
                    
                    if knowledge_results:
                        for result in knowledge_results:
                            content = result.get('content', result.get('raw', ''))
                            
                            # Extract different types of knowledge
                            self._extract_skills_from_content(content, discovered_skills)
                            
                except Exception as e:
                    logger.warning(f"Failed to discover skills with query '{query}': {e}")
                    continue
            
            # Analyze discovered skills for capability enhancement
            discovered_skills["capabilities_enhancement"] = self._analyze_skill_based_capabilities(
                discovered_skills, user_request
            )
            
            logger.info(f"Discovered {len(discovered_skills['skills'])} skills, {len(discovered_skills['methodologies'])} methodologies for task")
            
            return DiscoveredSkills(
                skills=discovered_skills["skills"],
                methodologies=discovered_skills["methodologies"],
                experience=discovered_skills["experience"],
                frameworks=discovered_skills["frameworks"],
                best_practices=discovered_skills["best_practices"],
                capabilities_enhancement=discovered_skills["capabilities_enhancement"]
            )
            
        except Exception as e:
            logger.warning(f"Skill discovery failed: {e}")
            return DiscoveredSkills(
                skills=[], methodologies=[], experience=[], frameworks=[], 
                best_practices=[], capabilities_enhancement={}
            )
    
    def _extract_skills_from_content(self, content: str, discovered_skills: Dict[str, List]):
        """Extract skills, methodologies, and experience from knowledge content"""
        
        content_lower = content.lower()
        
        # Skill indicators
        skill_indicators = [
            'expertise in', 'skilled at', 'proficient in', 'experienced with', 'specializes in',
            'competent in', 'adept at', 'capable of', 'trained in', 'knowledgeable about'
        ]
        
        # Methodology indicators  
        methodology_indicators = [
            'methodology', 'framework', 'approach', 'process', 'procedure', 'workflow',
            'system', 'method', 'technique', 'strategy', 'protocol'
        ]
        
        # Experience indicators
        experience_indicators = [
            'lessons learned', 'best practices', 'experience shows', 'proven approach',
            'successful implementation', 'case study', 'practical experience', 'field experience'
        ]
        
        # Framework indicators
        framework_indicators = [
            'lean', 'agile', 'six sigma', 'kaizen', 'scrum', 'kanban', 'design thinking',
            'waterfall', 'devops', 'continuous improvement', 'quality management'
        ]
        
        # Extract and categorize content
        sentences = content.split('.')
        
        for sentence in sentences[:5]:  # Limit to first 5 sentences to avoid noise
            sentence_lower = sentence.lower().strip()
            
            if any(indicator in sentence_lower for indicator in skill_indicators):
                if len(sentence) < 200:  # Avoid overly long extractions
                    discovered_skills["skills"].append(sentence.strip())
            
            elif any(indicator in sentence_lower for indicator in methodology_indicators):
                if len(sentence) < 200:
                    discovered_skills["methodologies"].append(sentence.strip())
                    
            elif any(indicator in sentence_lower for indicator in experience_indicators):
                if len(sentence) < 200:
                    discovered_skills["experience"].append(sentence.strip())
                    
            elif any(framework in sentence_lower for framework in framework_indicators):
                if len(sentence) < 200:
                    discovered_skills["frameworks"].append(sentence.strip())
        
        # Extract best practices
        if 'best practice' in content_lower or 'recommended approach' in content_lower:
            practice_sentences = [s.strip() for s in sentences if 'best practice' in s.lower() or 'recommended' in s.lower()]
            discovered_skills["best_practices"].extend(practice_sentences[:3])
    
    def _analyze_skill_based_capabilities(self, discovered_skills: Dict[str, List], user_request: str) -> Dict[str, Any]:
        """Analyze discovered skills to enhance agent capabilities"""
        
        capabilities_enhancement = {
            "complexity_boost": 0,
            "additional_steps": [],
            "specialized_approaches": [],
            "quality_standards": [],
            "risk_mitigation": []
        }
        
        # Boost complexity handling if agent has relevant methodologies
        methodology_count = len(discovered_skills.get("methodologies", []))
        framework_count = len(discovered_skills.get("frameworks", []))
        
        if methodology_count >= 2 or framework_count >= 1:
            capabilities_enhancement["complexity_boost"] = min(2, methodology_count)  # Up to +2 complexity points
        
        # Add specialized execution steps based on discovered skills
        if discovered_skills.get("frameworks"):
            capabilities_enhancement["additional_steps"].append({
                "name": "Framework Application",
                "description": "Apply relevant frameworks and methodologies from knowledge base",
                "tools_needed": ["brain_vector", "business_logic"]
            })
        
        if discovered_skills.get("best_practices"):
            capabilities_enhancement["additional_steps"].append({
                "name": "Best Practices Integration", 
                "description": "Integrate organizational best practices and lessons learned",
                "tools_needed": ["brain_vector"]
            })
        
        # Enhance quality standards based on experience
        if discovered_skills.get("experience"):
            capabilities_enhancement["quality_standards"].extend([
                "Apply lessons learned from previous implementations",
                "Validate against organizational experience",
                "Consider historical success patterns"
            ])
        
        return capabilities_enhancement 