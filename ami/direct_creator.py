"""
Ami Direct Creator - Simple Agent Creation
==========================================

Handles direct, one-shot agent creation from user requests.
This is the legacy approach for quick agent creation.
"""

import json
import re
import logging
from typing import Dict, Any, List
from datetime import datetime

from .models import AgentCreationRequest, AgentCreationResult, SimpleAgentConfig

logger = logging.getLogger(__name__)

# Configure direct creation logging
direct_logger = logging.getLogger("ami_direct")
direct_logger.setLevel(logging.INFO)
if not direct_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('⚡ [DIRECT] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    direct_logger.addHandler(handler)


class DirectCreator:
    """
    Handles direct agent creation - simple and fast
    """
    
    def __init__(self, anthropic_executor, openai_executor):
        """Initialize direct creator with LLM executors"""
        self.anthropic_executor = anthropic_executor
        self.openai_executor = openai_executor
        
        # Simple tool mapping based on agent types
        self.available_tools = {
            "document_analysis": ["file_access", "business_logic"],
            "sales": ["search", "context", "business_logic"],
            "support": ["brain_vector", "context", "search"],
            "analyst": ["file_access", "business_logic", "search"],
            "automation": ["file_access", "business_logic"],
            "research": ["search", "brain_vector", "context"],
            "general": ["search", "context"]
        }
        
        direct_logger.info("Direct Creator initialized")
    
    async def create_agent_direct(self, request: AgentCreationRequest) -> AgentCreationResult:
        """
        Main creation method - simple and direct
        
        Args:
            request: Agent creation request containing user requirements
            
        Returns:
            AgentCreationResult with success status and agent details
        """
        direct_logger.info(f"Starting direct agent creation: '{request.user_request[:100]}...'")
        
        try:
            # Step 1: Simple analysis of what kind of agent is needed
            direct_logger.info("Step 1: Analyzing agent requirements...")
            agent_config = await self._analyze_agent_needs(request.user_request, request.llm_provider)
            direct_logger.info(f"Agent analysis complete: {agent_config.name} ({agent_config.agent_type})")
            
            # Step 2: Generate comprehensive system prompt
            direct_logger.info("Step 2: Generating system prompt...")
            system_prompt_data = await self._generate_system_prompt(agent_config, request.llm_provider)
            direct_logger.info("System prompt generated successfully")
            
            # Step 3: Select optimal tools for this agent type
            direct_logger.info("Step 3: Selecting tools...")
            tools_list = self._select_tools(agent_config.agent_type)
            direct_logger.info(f"Selected tools: {tools_list}")
            
            # Step 4: Determine knowledge requirements
            direct_logger.info("Step 4: Setting up knowledge access...")
            knowledge_list = self._determine_knowledge_needs(agent_config)
            direct_logger.info(f"Knowledge domains: {knowledge_list}")
            
            # Step 5: Save to database via org_agent system
            direct_logger.info("Step 5: Saving agent to database...")
            agent_id = await self._save_to_database(
                config={
                    "name": agent_config.name,
                    "description": agent_config.description,
                    "system_prompt": system_prompt_data,
                    "tools_list": tools_list,
                    "knowledge_list": knowledge_list
                },
                org_id=request.org_id,
                user_id=request.user_id
            )
            direct_logger.info(f"Agent saved successfully with ID: {agent_id}")
            
            return AgentCreationResult(
                success=True,
                agent_id=agent_id,
                agent_name=agent_config.name,
                message=f"✅ Created '{agent_config.name}' successfully! The agent is now ready to use.",
                agent_config={
                    "name": agent_config.name,
                    "description": agent_config.description,
                    "agent_type": agent_config.agent_type,
                    "language": agent_config.language,
                    "tools": tools_list,
                    "knowledge": knowledge_list
                }
            )
            
        except Exception as e:
            direct_logger.error(f"Agent creation failed: {str(e)}")
            return AgentCreationResult(
                success=False,
                error=str(e),
                message="❌ Failed to create agent. Please try again or contact support."
            )
    
    async def _analyze_agent_needs(self, user_request: str, provider: str = "anthropic") -> SimpleAgentConfig:
        """
        Simple LLM analysis to understand what kind of agent is needed
        """
        
        analysis_prompt = f"""
        Analyze this agent creation request and respond with JSON only.
        
        User Request: "{user_request}"
        
        Determine:
        1. Professional agent name (descriptive, specific to purpose)
        2. Clear description of what this agent does
        3. Agent type from: sales, support, analyst, document_analysis, research, automation, general
        4. Primary language: english, vietnamese, french, spanish, chinese
        5. Key specialization areas (2-4 specific areas)
        
        Agent Types:
        - sales: Sales assistance, lead qualification, customer communication
        - support: Customer support, troubleshooting, FAQ assistance  
        - analyst: Data analysis, business intelligence, report generation
        - document_analysis: Document processing, content analysis, file management
        - research: Information gathering, market research, competitive analysis
        - automation: Process automation, workflow management, task execution
        - general: Multi-purpose assistant for various tasks
        
        Respond with this exact JSON format:
        {{
            "name": "Specific Agent Name",
            "description": "Clear description of agent's purpose and capabilities",
            "agent_type": "document_analysis",
            "language": "english",
            "specialization": ["area1", "area2", "area3"]
        }}
        
        Examples:
        Request: "I need help with Vietnamese sales documents"
        Response: {{"name": "Vietnamese Sales Document Specialist", "description": "Analyzes Vietnamese sales documents and provides insights for deal progression", "agent_type": "document_analysis", "language": "vietnamese", "specialization": ["sales_document_analysis", "vietnamese_business_context", "deal_assessment"]}}
        
        Make the agent name specific and professional. Focus on the primary task.
        """
        
        try:
            response = await self._call_llm(analysis_prompt, provider)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return SimpleAgentConfig(
                    name=data.get("name", "Custom Agent"),
                    description=data.get("description", "A specialized AI agent"),
                    agent_type=data.get("agent_type", "general"),
                    tools_needed=[],
                    language=data.get("language", "english"),
                    specialization=data.get("specialization", ["general_assistance"])
                )
        except Exception as e:
            direct_logger.warning(f"Agent analysis parsing failed: {e}, using fallback")
        
        # Fallback configuration if parsing fails
        return SimpleAgentConfig(
            name="Custom Agent",
            description="A specialized AI agent tailored to your needs",
            agent_type="general",
            tools_needed=[],
            language="english",
            specialization=["general_assistance"]
        )
    
    async def _generate_system_prompt(self, config: SimpleAgentConfig, provider: str = "anthropic") -> Dict[str, Any]:
        """
        Generate comprehensive system prompt configuration
        """
        
        prompt_generation_request = f"""
        Create a comprehensive system prompt for this AI agent:
        
        Agent Details:
        - Name: {config.name}
        - Type: {config.agent_type}
        - Purpose: {config.description}
        - Language: {config.language}
        - Specialization: {config.specialization}
        
        Create a system prompt that:
        1. Clearly defines the agent's role and expertise
        2. Sets appropriate professional tone and personality
        3. Includes specific language instructions if not English
        4. Mentions specialization areas naturally
        5. Encourages use of available tools when relevant
        6. Is concise but comprehensive (200-400 words)
        
        Make it sound natural and professional. The agent should feel specialized and capable.
        
        Return only the system prompt text, no explanation or formatting.
        """
        
        try:
            system_prompt_text = await self._call_llm(prompt_generation_request, provider)
            
            # Clean up the response
            system_prompt_text = system_prompt_text.strip()
            
            # Create comprehensive prompt data structure
            prompt_data = {
                "base_instruction": system_prompt_text,
                "agent_type": config.agent_type,
                "language": config.language,
                "specialization": config.specialization,
                "personality": {
                    "tone": "professional",
                    "style": "helpful",
                    "approach": "solution-oriented"
                },
                "created_at": datetime.now().isoformat()
            }
            
            return prompt_data
            
        except Exception as e:
            direct_logger.error(f"System prompt generation failed: {e}")
            # Return fallback prompt structure
            fallback_prompt = f"You are {config.name}, a specialized {config.agent_type} agent. Your purpose is to {config.description.lower()}. You are professional, helpful, and focus on providing accurate, relevant assistance."
            
            return {
                "base_instruction": fallback_prompt,
                "agent_type": config.agent_type,
                "language": config.language,
                "specialization": config.specialization,
                "personality": {"tone": "professional", "style": "helpful", "approach": "solution-oriented"},
                "created_at": datetime.now().isoformat()
            }
    
    def _select_tools(self, agent_type: str) -> List[str]:
        """
        Simple tool selection based on agent type
        """
        tools = self.available_tools.get(agent_type, self.available_tools["general"])
        direct_logger.info(f"Selected tools for {agent_type}: {tools}")
        return tools
    
    def _determine_knowledge_needs(self, config: SimpleAgentConfig) -> List[str]:
        """
        Determine what knowledge domains this agent needs access to
        """
        # Start with empty knowledge list - can be populated later
        knowledge_domains = []
        
        # Add basic knowledge based on agent type
        if config.agent_type == "sales":
            knowledge_domains.extend(["sales_techniques", "product_information"])
        elif config.agent_type == "support":
            knowledge_domains.extend(["faq_database", "troubleshooting_guides"])
        elif config.agent_type == "document_analysis":
            knowledge_domains.extend(["document_processing", "business_intelligence"])
        elif config.agent_type == "analyst":
            knowledge_domains.extend(["data_analysis", "business_metrics"])
        
        # Add language-specific knowledge if needed
        if config.language != "english":
            knowledge_domains.append(f"{config.language}_context")
        
        return knowledge_domains
    
    async def _save_to_database(self, config: dict, org_id: str, user_id: str) -> str:
        """
        Save agent configuration to database via org_agent system
        """
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
            
            direct_logger.info(f"Agent saved to database: {agent.name} (ID: {agent.id})")
            return agent.id
            
        except Exception as e:
            direct_logger.error(f"Database save failed: {e}")
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
            direct_logger.error(f"LLM call failed: {e}")
            raise Exception(f"LLM call failed: {str(e)}")