"""
Grading Tool - Comprehensive Agent Capability Analysis and Grading Scenario Generation

This tool provides intelligent grading scenarios by:
1. Comprehensively scanning ALL agent brain vectors (not just search-based)
2. Analyzing capabilities across all knowledge domains
3. Proposing optimal grading scenarios that showcase agent's best capabilities
4. Role-playing as the agent to demonstrate expected performance
5. Executing grading scenarios with human approval
"""

import logging
import asyncio
import json
import traceback
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """Single agent capability extracted from brain vectors"""
    domain: str  # e.g., "financial_analysis", "data_processing"
    skill: str   # e.g., "excel_automation", "due_diligence"
    knowledge_depth: float  # 0.0-1.0 score
    examples: List[str]  # Concrete examples from vectors
    vector_count: int   # How many vectors support this capability
    confidence: float   # How confident we are in this capability

@dataclass
class GradingScenario:
    """Proposed grading scenario for agent testing with diagram support"""
    scenario_name: str
    description: str
    agent_role_play: str  # How agent would introduce itself
    test_inputs: List[Dict[str, Any]]  # Sample inputs for testing
    expected_outputs: List[Dict[str, Any]]  # Expected results
    showcased_capabilities: List[str]  # Which capabilities this tests
    difficulty_level: str  # "basic", "intermediate", "advanced"
    estimated_time: str  # "5 minutes", "15 minutes"
    success_criteria: List[str]  # How to judge if agent performed well
    # NEW: Diagram support
    scenario_diagram: Optional[str] = None  # Mermaid diagram showing workflow
    process_diagrams: List[Dict[str, str]] = None  # Step-by-step process diagrams
    capability_map: Optional[str] = None  # Visual capability mapping diagram

class GradingTool:
    """Comprehensive agent grading and capability showcase tool"""
    
    def __init__(self):
        """Initialize the grading tool"""
        self.brain_vector_tool = None
        self.pccontroller = None
        self._initialize_brain_access()
    
    def _initialize_brain_access(self):
        """Initialize connections to brain vector systems"""
        try:
            # Get comprehensive brain vector access
            from brain_vector_tool import BrainVectorTool
            from pccontroller import query_knowledge, get_production_index
            
            self.brain_vector_tool = BrainVectorTool()
            self.pccontroller = {
                'query_knowledge': query_knowledge,
                'get_production_index': get_production_index
            }
            logger.info("Grading tool brain access initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize brain access: {e}")
    
    async def get_comprehensive_agent_capabilities(
        self, 
        user_id: str, 
        org_id: str,
        max_vectors: int = 500
    ) -> Dict[str, Any]:
        """
        Get comprehensive view of agent capabilities by loading ALL brain vectors
        
        This method loads all available vectors for the organization without keyword filtering
        to get the most complete picture of agent capabilities.
        
        Args:
            user_id: User identifier
            org_id: Organization identifier  
            max_vectors: Maximum vectors to load (default 500 for comprehensive analysis)
            
        Returns:
            Comprehensive capability analysis based on ALL available vectors
        """
        try:
            logger.info(f"Loading ALL brain vectors for comprehensive analysis - org: {org_id}")
            
            all_vectors = []
            
            # Strategy: Load ALL vectors using very broad queries
            try:
                if self.pccontroller and self.pccontroller['query_knowledge']:
                    # Use strategic broad queries that work better with semantic search
                    # Each query targets meaningful concepts that appear across domains
                    broad_queries = [
                        "how to do work process",              # Process/workflow content
                        "information data analysis report",    # Data and reporting content  
                        "business company organization",       # Business content
                        "customer service support help",       # Customer service content
                        "system tool software application",    # Technical content
                        "manage project task activity",        # Management content
                        "document file create generate",       # Document management
                        "decision strategy planning goal",     # Strategic content
                        "problem solution answer resolve",     # Problem solving content
                        "knowledge learn understand know",     # Learning content
                    ]
                    
                    all_vectors = []
                    unique_vector_ids = set()
                    
                    for query in broad_queries:
                        try:
                            query_response = await self.pccontroller['query_knowledge'](
                                query=query,
                                org_id=org_id,
                                user_id=user_id,
                                top_k=max_vectors // len(broad_queries) + 50,  # Distribute across queries with overlap
                                min_similarity=0.0  # Accept everything - no similarity filtering
                            )
                            
                            if query_response:
                                # Deduplicate vectors by ID
                                for vector in query_response:
                                    vector_id = vector.get('id')
                                    if vector_id and vector_id not in unique_vector_ids:
                                        unique_vector_ids.add(vector_id)
                                        all_vectors.append(vector)
                                        
                                logger.info(f"Query '{query[:30]}...' returned {len(query_response)} vectors, {len(all_vectors)} total unique")
                            
                            # Stop if we've reached our target
                            if len(all_vectors) >= max_vectors:
                                break
                                
                        except Exception as e:
                            logger.warning(f"Query '{query[:30]}...' failed: {e}")
                            continue
                    
                    logger.info(f"Successfully loaded {len(all_vectors)} unique vectors using broad queries")
                    
                    # Fallback: If we didn't get enough vectors, try the original keyword approach
                    if len(all_vectors) < 10:  # If less than 10 vectors, try original approach
                        logger.warning(f"Only found {len(all_vectors)} vectors with broad queries, trying keyword approach as fallback")
                        
                        original_keywords = [
                            "business processes and workflows",
                            "data analysis and processing", 
                            "financial analysis and reporting",
                            "customer service and communication",
                            "technical integration and automation",
                            "decision making and strategy",
                            "document processing and management",
                            "project management and planning"
                        ]
                        
                        for keyword in original_keywords:
                            try:
                                keyword_response = await self.pccontroller['query_knowledge'](
                                    query=keyword,
                                    org_id=org_id,
                                    user_id=user_id,
                                    top_k=50,
                                    min_similarity=0.2
                                )
                                
                                if keyword_response:
                                    # Deduplicate with existing vectors
                                    for vector in keyword_response:
                                        vector_id = vector.get('id')
                                        if vector_id and vector_id not in unique_vector_ids:
                                            unique_vector_ids.add(vector_id)
                                            vector['source_domain'] = keyword
                                            all_vectors.append(vector)
                                            
                                    logger.info(f"Keyword '{keyword}' added {len(keyword_response)} more vectors, total: {len(all_vectors)}")
                                    
                            except Exception as e:
                                logger.warning(f"Keyword '{keyword}' failed: {e}")
                                continue
                        
                        logger.info(f"Combined approach loaded {len(all_vectors)} total vectors")
                        
            except Exception as e:
                logger.error(f"Failed to load all vectors: {e}")
                # If direct loading fails, we have no vectors to analyze
                return {
                    "success": False,
                    "error": f"Could not load brain vectors: {str(e)}",
                    "capabilities": []
                }
            
            if not all_vectors:
                logger.warning(f"No brain vectors found for org: {org_id}")
                return {
                    "success": True,
                    "total_vectors_analyzed": 0,
                    "unique_domains_covered": 0,
                    "capabilities": [],
                    "analysis_timestamp": datetime.now().isoformat(),
                    "message": "No brain vectors found - agent may not have learned any knowledge yet"
                }
            
            # Add source tracking for all vectors
            for vector in all_vectors:
                vector['source_domain'] = 'comprehensive_scan'
            
            logger.info(f"Loaded {len(all_vectors)} total vectors for comprehensive capability analysis")
            
            # Analyze capabilities from ALL collected vectors
            capabilities = await self._analyze_vectors_for_capabilities(all_vectors)
            
            return {
                "success": True,
                "total_vectors_analyzed": len(all_vectors),
                "unique_domains_covered": len(set(self._extract_domain_from_content(v.get('raw', '')) for v in all_vectors)),
                "capabilities": capabilities,
                "analysis_timestamp": datetime.now().isoformat(),
                "loading_strategy": "comprehensive_all_vectors"
            }
            
        except Exception as e:
            logger.error(f"Comprehensive capability analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "capabilities": []
            }
    
    async def _analyze_vectors_for_capabilities(self, vectors: List[Dict]) -> List[AgentCapability]:
        """Analyze ALL vectors to extract discrete agent capabilities"""
        
        try:
            logger.info(f"Analyzing {len(vectors)} vectors for capability extraction")
            
            # Group vectors by content similarity and domain
            capability_groups = {}
            domain_stats = {}
            
            for vector in vectors:
                content = vector.get('raw', '')
                if not content:
                    continue
                    
                domain = self._extract_domain_from_content(content)
                
                if domain not in capability_groups:
                    capability_groups[domain] = []
                    domain_stats[domain] = 0
                    
                capability_groups[domain].append(vector)
                domain_stats[domain] += 1
            
            logger.info(f"Grouped vectors into {len(capability_groups)} domains: {dict(domain_stats)}")
            
            # Extract capabilities from each domain group
            capabilities = []
            for domain, domain_vectors in capability_groups.items():
                logger.info(f"Extracting capabilities from {domain} domain ({len(domain_vectors)} vectors)")
                domain_capabilities = await self._extract_domain_capabilities(domain, domain_vectors)
                capabilities.extend(domain_capabilities)
                logger.info(f"Extracted {len(domain_capabilities)} capabilities from {domain}")
            
            # Sort by confidence and knowledge depth
            capabilities.sort(key=lambda x: (x.confidence * x.knowledge_depth), reverse=True)
            
            logger.info(f"Total capabilities extracted: {len(capabilities)}")
            if capabilities:
                top_capabilities = capabilities[:5]
                logger.info(f"Top 5 capabilities: {[f'{cap.skill} (confidence: {cap.confidence:.2f})' for cap in top_capabilities]}")
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Vector capability analysis failed: {e}")
            return []
    
    def _generate_scenario_workflow_diagram(self, scenario_name: str, capabilities: List[str], test_inputs: List[Dict], expected_outputs: List[Dict]) -> str:
        """Generate Mermaid diagram showing the grading scenario workflow"""
        
        try:
            # Create a flowchart showing the grading process
            diagram_content = f"""
flowchart TD
    A["{scenario_name}<br/>Grading Scenario"] --> B["🧠 Agent Analysis"]
    B --> C["📋 Input Processing"]
    
"""
            
            # Add input processing nodes
            for i, test_input in enumerate(test_inputs):
                input_node = f"I{i+1}"
                input_desc = test_input.get('description', 'Input').replace('"', "'")[:40]
                diagram_content += f'    C --> {input_node}["{input_desc}"]\n'
                
            # Add capability processing nodes
            for i, capability in enumerate(capabilities[:3]):  # Limit to 3 for readability
                cap_node = f"P{i+1}"
                cap_name = capability.replace('_', ' ').title()[:30]
                diagram_content += f'    {input_node} --> {cap_node}["{cap_name}<br/>Processing"]\n'
                
            # Add output nodes  
            for i, output in enumerate(expected_outputs):
                output_node = f"O{i+1}"
                output_desc = output.get('description', 'Output').replace('"', "'")[:40]
                diagram_content += f'    {cap_node} --> {output_node}["{output_desc}"]\n'
                
            # Final assessment
            diagram_content += f'    {output_node} --> E["📊 Performance<br/>Assessment"]\n'
            diagram_content += '    E --> F["🎯 Results &<br/>Recommendations"]\n'
            
            # Add styling
            diagram_content += """
    classDef inputNode fill:#e1f5fe
    classDef processNode fill:#f3e5f5
    classDef outputNode fill:#e8f5e8
    classDef assessNode fill:#fff3e0
    
    class A,B assessNode
    class C processNode
"""
            
            # Apply classes to nodes
            for i in range(len(test_inputs)):
                diagram_content += f"    class I{i+1} inputNode\n"
            for i in range(len(capabilities[:3])):
                diagram_content += f"    class P{i+1} processNode\n" 
            for i in range(len(expected_outputs)):
                diagram_content += f"    class O{i+1} outputNode\n"
            diagram_content += "    class E,F assessNode\n"
            
            return diagram_content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate scenario workflow diagram: {e}")
            return self._create_fallback_workflow_diagram(scenario_name)
    
    def _create_fallback_workflow_diagram(self, scenario_name: str) -> str:
        """Create a simple fallback workflow diagram"""
        return f"""
flowchart TD
    A["{scenario_name}"] --> B["📥 Input Analysis"]
    B --> C["🧠 Knowledge Processing"]
    C --> D["⚡ Response Generation"]
    D --> E["📤 Output Delivery"]
    E --> F["📊 Performance Assessment"]
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px
    classDef highlight fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    
    class A,F highlight
"""
    
    def _generate_capability_map_diagram(self, capabilities: List[AgentCapability]) -> str:
        """Generate a visual map of agent capabilities"""
        
        try:
            if not capabilities:
                return self._create_empty_capability_map()
            
            # Create a mindmap-style diagram of capabilities
            diagram_content = """
mindmap
  root)🤖 Agent Capabilities(
"""
            
            # Group capabilities by domain
            domain_groups = {}
            for cap in capabilities:
                domain = cap.domain
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(cap)
            
            # Add domain branches
            for domain, domain_caps in domain_groups.items():
                domain_name = domain.replace('_', ' ').title()
                diagram_content += f"    {domain_name}\n"
                
                # Add capabilities under each domain
                for cap in domain_caps[:4]:  # Limit to 4 per domain for readability
                    skill_name = cap.skill.replace('_', ' ').title()
                    confidence_indicator = "🟢" if cap.confidence > 0.8 else "🟡" if cap.confidence > 0.6 else "🔴"
                    diagram_content += f"      {confidence_indicator} {skill_name}\n"
            
            return diagram_content
            
        except Exception as e:
            logger.error(f"Failed to generate capability map: {e}")
            return self._create_empty_capability_map()
    
    def _create_empty_capability_map(self) -> str:
        """Create empty capability map"""
        return """
mindmap
  root)🤖 Agent Capabilities(
    General
      📋 Problem Solving
      💬 Communication
    Learning
      📚 Knowledge Processing
      🔄 Adaptation
"""

    def _generate_process_diagrams(self, scenario_name: str, test_inputs: List[Dict], capabilities: List[str]) -> List[Dict[str, str]]:
        """Generate step-by-step process diagrams for the scenario"""
        
        diagrams = []
        
        try:
            # Diagram 1: Input Processing Flow
            input_diagram = """
sequenceDiagram
    participant U as User
    participant A as Agent
    participant K as Knowledge Base
    participant P as Processor
    
    U->>A: Provides Test Input
    A->>K: Queries Relevant Knowledge
    K-->>A: Returns Domain Expertise
    A->>P: Applies Processing Logic
    P-->>A: Generated Analysis
    A-->>U: Delivers Results
"""
            diagrams.append({
                "title": "Input Processing Flow",
                "description": "How the agent processes test inputs using its knowledge",
                "diagram": input_diagram
            })
            
            # Diagram 2: Decision Making Process
            decision_diagram = f"""
flowchart TD
    A["📥 Receive Input"] --> B{{"🤔 Analyze Input Type"}}
    B -->|"Data Analysis"| C["📊 Apply Analytics"]
    B -->|"Process Query"| D["⚙️ Execute Workflow"]
    B -->|"Knowledge Query"| E["🧠 Search Knowledge"]
    
    C --> F["📈 Generate Insights"]
    D --> G["✅ Complete Process"]
    E --> H["💡 Provide Information"]
    
    F --> I["📋 Format Results"]
    G --> I
    H --> I
    
    I --> J["🎯 Validate Quality"]
    J --> K["📤 Deliver Output"]
    
    classDef inputNode fill:#e3f2fd
    classDef processNode fill:#f3e5f5  
    classDef outputNode fill:#e8f5e8
    
    class A inputNode
    class B,C,D,E,F,G,H processNode
    class I,J,K outputNode
"""
            diagrams.append({
                "title": "Agent Decision Making Process",
                "description": f"Decision tree for {scenario_name} scenario",
                "diagram": decision_diagram
            })
            
            return diagrams
            
        except Exception as e:
            logger.error(f"Failed to generate process diagrams: {e}")
            return [{
                "title": "Basic Process Flow",
                "description": "Agent processing workflow",
                "diagram": """
flowchart TD
    A[Input] --> B[Processing]
    B --> C[Output]
    
    classDef default fill:#f9f9f9
"""
            }]
    
    def _extract_domain_from_content(self, content: str) -> str:
        """Extract domain from vector content using keyword analysis"""
        
        content_lower = content.lower()
        
        # Define domain keywords
        domain_keywords = {
            "financial_analysis": ["financial", "revenue", "profit", "budget", "analysis", "excel", "spreadsheet"],
            "customer_service": ["customer", "service", "support", "response", "communication", "inquiry"],
            "data_processing": ["data", "process", "file", "automation", "workflow", "integration"],
            "business_strategy": ["strategy", "decision", "planning", "management", "objective", "goal"],
            "technical_integration": ["api", "integration", "system", "automation", "technical", "setup"],
            "document_management": ["document", "file", "report", "template", "generation", "format"]
        }
        
        # Score each domain based on keyword matches
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score, or 'general' if no matches
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return "general"
    
    async def _extract_domain_capabilities(self, domain: str, vectors: List[Dict]) -> List[AgentCapability]:
        """Extract specific capabilities from a domain's vectors"""
        
        # Combine all content from domain vectors
        combined_content = "\n".join([v.get('raw', '') for v in vectors])
        
        # Use LLM to analyze and extract capabilities
        analysis_prompt = f"""
        Analyze this agent's knowledge in the {domain} domain and extract discrete capabilities.
        
        Agent Knowledge Content:
        {combined_content[:3000]}...  # Limit to prevent token overflow
        
        Total vectors in this domain: {len(vectors)}
        
        Extract specific capabilities in this JSON format:
        [
            {{
                "skill": "specific_skill_name",
                "description": "what the agent can do",
                "examples": ["concrete example 1", "concrete example 2"],
                "knowledge_depth": 0.8,
                "confidence": 0.9
            }}
        ]
        
        Focus on ACTIONABLE capabilities the agent can demonstrate, not just knowledge areas.
        """
        
        try:
            # This would call an LLM to analyze - for now return structured result
            # In full implementation, would use OpenAI/Anthropic API here
            
            # Mock capability extraction for demonstration
            capabilities = []
            
            # Simple keyword-based capability extraction as fallback
            if "financial" in combined_content.lower():
                capabilities.append(AgentCapability(
                    domain=domain,
                    skill="financial_analysis",
                    knowledge_depth=0.8,
                    examples=["Analyze P&L statements", "Calculate financial ratios"],
                    vector_count=len(vectors),
                    confidence=0.7
                ))
            
            if "excel" in combined_content.lower():
                capabilities.append(AgentCapability(
                    domain=domain,
                    skill="excel_automation",
                    knowledge_depth=0.7,
                    examples=["Process Excel files", "Extract data from spreadsheets"],
                    vector_count=len(vectors),
                    confidence=0.8
                ))
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Domain capability extraction failed for {domain}: {e}")
            return []
    
    async def generate_optimal_grading_scenario(
        self, 
        capabilities: List[AgentCapability],
        agent_name: str = "Your Agent"
    ) -> GradingScenario:
        """
        Generate the optimal grading scenario based on agent capabilities
        With enhanced diagram generation for visual representation
        
        Args:
            capabilities: List of agent capabilities
            agent_name: Name of the agent being tested
            
        Returns:
            Optimal grading scenario that showcases agent's best capabilities
        """
        
        try:
            # Find the strongest capabilities (highest confidence * knowledge_depth)
            if not capabilities:
                return self._create_fallback_scenario(agent_name)
            
            # Sort capabilities by strength
            strong_capabilities = sorted(
                capabilities, 
                key=lambda x: x.confidence * x.knowledge_depth, 
                reverse=True
            )[:3]  # Top 3 capabilities
            
            # Create scenario that combines multiple strong capabilities
            primary_capability = strong_capabilities[0]
            
            # Generate scenario based on primary capability
            scenario = await self._create_scenario_for_capability(primary_capability, strong_capabilities, agent_name)
            
            return scenario
            
        except Exception as e:
            logger.error(f"Grading scenario generation failed: {e}")
            return self._create_fallback_scenario(agent_name)
    
    async def _create_scenario_for_capability(
        self, 
        primary_capability: AgentCapability, 
        all_strong_capabilities: List[AgentCapability],
        agent_name: str
    ) -> GradingScenario:
        """Create a grading scenario for a specific capability"""
        
        capability_scenarios = {
            "financial_analysis": {
                "name": "Financial Due Diligence Analysis",
                "description": "Analyze a company's financial statements and provide investment recommendations",
                "role_play": f"As {agent_name}, I specialize in financial analysis and due diligence. I can read financial statements, calculate key ratios, identify trends, and assess investment risks. Let me show you how I analyze a company's financial health.",
                "test_inputs": [
                    {"type": "excel_file", "description": "Company financial statements (P&L, Balance Sheet)"},
                    {"type": "requirements", "description": "Analysis focus: profitability, growth trends, risk factors"}
                ],
                "expected_outputs": [
                    {"type": "financial_summary", "description": "Key financial metrics and ratios"},
                    {"type": "risk_assessment", "description": "Identified financial risks and concerns"},
                    {"type": "recommendation", "description": "Investment recommendation with justification"}
                ]
            },
            "excel_automation": {
                "name": "Data Processing Automation",
                "description": "Process and analyze data from Excel files with automated insights",
                "role_play": f"As {agent_name}, I excel at automating data processing workflows. I can read Excel files, extract key information, perform calculations, and generate insights. Watch me process complex data automatically.",
                "test_inputs": [
                    {"type": "excel_file", "description": "Raw data spreadsheet with multiple sheets"},
                    {"type": "processing_rules", "description": "Data validation and calculation requirements"}
                ],
                "expected_outputs": [
                    {"type": "processed_data", "description": "Cleaned and validated dataset"},
                    {"type": "calculations", "description": "Automated calculations and derived metrics"},
                    {"type": "insights", "description": "Data trends and anomalies identified"}
                ]
            },
            "customer_service": {
                "name": "Customer Inquiry Management",
                "description": "Handle customer inquiries with appropriate responses and escalation",
                "role_play": f"As {agent_name}, I handle customer service inquiries efficiently. I can categorize requests, provide appropriate responses, and escalate complex issues. Let me demonstrate my customer service capabilities.",
                "test_inputs": [
                    {"type": "customer_inquiries", "description": "Various customer service requests"},
                    {"type": "service_policies", "description": "Company policies and response guidelines"}
                ],
                "expected_outputs": [
                    {"type": "inquiry_classification", "description": "Categorized customer requests"},
                    {"type": "responses", "description": "Appropriate responses to each inquiry"},
                    {"type": "escalations", "description": "Complex cases escalated with reasoning"}
                ]
            }
        }
        
        # Use primary capability to select scenario template
        template_key = primary_capability.skill
        if template_key not in capability_scenarios:
            template_key = list(capability_scenarios.keys())[0]  # Fallback
        
        template = capability_scenarios[template_key]
        
        # Customize scenario with agent's specific capabilities
        showcased_capabilities = [cap.skill for cap in all_strong_capabilities]
        
        # Generate diagrams for the scenario
        scenario_diagram = self._generate_scenario_workflow_diagram(
            template["name"], 
            showcased_capabilities, 
            template["test_inputs"], 
            template["expected_outputs"]
        )
        
        capability_map = self._generate_capability_map_diagram(all_strong_capabilities)
        
        process_diagrams = self._generate_process_diagrams(
            template["name"],
            template["test_inputs"], 
            showcased_capabilities
        )
        
        return GradingScenario(
            scenario_name=template["name"],
            description=template["description"],
            agent_role_play=template["role_play"],
            test_inputs=template["test_inputs"],
            expected_outputs=template["expected_outputs"],
            showcased_capabilities=showcased_capabilities,
            difficulty_level="intermediate",
            estimated_time="15-20 minutes",
            success_criteria=[
                "Demonstrates understanding of domain knowledge",
                "Shows ability to process inputs correctly",
                "Provides accurate and relevant outputs",
                "Explains reasoning clearly"
            ],
            # NEW: Enhanced with diagrams
            scenario_diagram=scenario_diagram,
            capability_map=capability_map,
            process_diagrams=process_diagrams
        )
    
    def _create_fallback_scenario(self, agent_name: str) -> GradingScenario:
        """Create a fallback scenario when capability analysis fails"""
        
        # Define fallback scenario components
        test_inputs = [
            {"type": "sample_query", "description": "A question or problem in your domain"}
        ]
        expected_outputs = [
            {"type": "analysis", "description": "Thoughtful analysis of the problem"},
            {"type": "solution", "description": "Proposed solution or recommendation"}
        ]
        showcased_capabilities = ["general_problem_solving"]
        scenario_name = "General Capability Demonstration"
        
        # Generate diagrams even for fallback scenario
        scenario_diagram = self._generate_scenario_workflow_diagram(
            scenario_name, 
            showcased_capabilities, 
            test_inputs, 
            expected_outputs
        )
        
        # Create mock capabilities for diagram generation
        mock_capabilities = [
            AgentCapability(
                domain="general",
                skill="problem_solving",
                knowledge_depth=0.7,
                examples=["Analyze problems", "Provide solutions"],
                vector_count=1,
                confidence=0.8
            ),
            AgentCapability(
                domain="general", 
                skill="communication",
                knowledge_depth=0.6,
                examples=["Clear explanations", "Structured responses"],
                vector_count=1,
                confidence=0.7
            )
        ]
        
        capability_map = self._generate_capability_map_diagram(mock_capabilities)
        
        process_diagrams = self._generate_process_diagrams(
            scenario_name,
            test_inputs, 
            showcased_capabilities
        )
        
        return GradingScenario(
            scenario_name=scenario_name,
            description="Demonstrate your agent's general knowledge and problem-solving abilities",
            agent_role_play=f"As {agent_name}, I'm ready to demonstrate my capabilities. I can process information, analyze problems, and provide helpful responses based on my knowledge.",
            test_inputs=test_inputs,
            expected_outputs=expected_outputs,
            showcased_capabilities=showcased_capabilities,
            difficulty_level="basic",
            estimated_time="10 minutes",
            success_criteria=[
                "Shows understanding of the problem",
                "Provides relevant and helpful response"
            ],
            # NEW: Add diagrams to fallback scenario
            scenario_diagram=scenario_diagram,
            capability_map=capability_map,
            process_diagrams=process_diagrams
        )
    
    async def execute_grading_scenario(
        self, 
        scenario: GradingScenario,
        test_inputs: Dict[str, Any],
        user_id: str,
        org_id: str
    ) -> Dict[str, Any]:
        """
        Execute the grading scenario by role-playing as the agent
        
        Args:
            scenario: The grading scenario to execute
            test_inputs: Actual test inputs provided by human
            user_id: User identifier
            org_id: Organization identifier
            
        Returns:
            Execution results showing agent performance
        """
        
        try:
            logger.info(f"Executing grading scenario: {scenario.scenario_name}")
            
            # Get agent's relevant knowledge for this scenario
            relevant_knowledge = await self._get_scenario_relevant_knowledge(
                scenario, user_id, org_id
            )
            
            # Execute scenario steps
            execution_results = {
                "scenario_name": scenario.scenario_name,
                "agent_introduction": scenario.agent_role_play,
                "execution_steps": [],
                "final_assessment": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 1: Agent introduction
            execution_results["execution_steps"].append({
                "step": "agent_introduction",
                "content": scenario.agent_role_play,
                "status": "completed"
            })
            
            # Step 2: Process each test input
            for i, test_input in enumerate(test_inputs.get("inputs", [])):
                step_result = await self._process_test_input(
                    test_input, relevant_knowledge, scenario
                )
                execution_results["execution_steps"].append({
                    "step": f"input_processing_{i+1}",
                    "input": test_input,
                    "output": step_result,
                    "status": "completed"
                })
            
            # Step 3: Generate final assessment
            assessment = self._assess_scenario_performance(execution_results, scenario)
            execution_results["final_assessment"] = assessment
            
            return {
                "success": True,
                "results": execution_results
            }
            
        except Exception as e:
            logger.error(f"Grading scenario execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "scenario_name": scenario.scenario_name
            }
    
    async def _get_scenario_relevant_knowledge(
        self, 
        scenario: GradingScenario, 
        user_id: str, 
        org_id: str
    ) -> List[Dict]:
        """Get agent knowledge relevant to the grading scenario"""
        
        try:
            # Create search query from scenario capabilities
            query = " ".join(scenario.showcased_capabilities)
            
            if self.pccontroller and self.pccontroller['query_knowledge']:
                knowledge = await self.pccontroller['query_knowledge'](
                    query=query,
                    org_id=org_id,
                    user_id=user_id,
                    top_k=20,
                    min_similarity=0.3
                )
                return knowledge
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get scenario relevant knowledge: {e}")
            return []
    
    async def _process_test_input(
        self, 
        test_input: Dict[str, Any], 
        knowledge: List[Dict], 
        scenario: GradingScenario
    ) -> Dict[str, Any]:
        """Process a single test input using agent's knowledge"""
        
        # This would use LLM to process the input as the agent
        # For now, return structured mock response
        
        return {
            "analysis": f"Processing {test_input.get('type', 'input')} using my knowledge of {', '.join(scenario.showcased_capabilities)}",
            "steps_taken": [
                "Analyzed the input requirements",
                "Applied relevant knowledge from my training",
                "Generated appropriate response"
            ],
            "output": f"Based on my analysis, here's how I would handle this {test_input.get('type', 'input')}...",
            "confidence": 0.85
        }
    
    def _assess_scenario_performance(
        self, 
        execution_results: Dict[str, Any], 
        scenario: GradingScenario
    ) -> Dict[str, Any]:
        """Assess how well the agent performed in the scenario"""
        
        return {
            "overall_score": 0.82,  # Mock score
            "criteria_met": len(scenario.success_criteria),
            "total_criteria": len(scenario.success_criteria),
            "strengths": [
                "Demonstrated clear understanding of domain",
                "Provided structured and logical responses",
                "Showed appropriate use of knowledge"
            ],
            "areas_for_improvement": [
                "Could provide more specific examples",
                "Response time could be optimized"
            ],
            "recommendation": "Strong performance - agent shows excellent capabilities in this domain"
        }
    
    def _generate_assessment_diagram(self, assessment: Dict[str, Any], scenario: GradingScenario) -> str:
        """Generate a visual diagram for assessment results"""
        
        try:
            overall_score = assessment.get('overall_score', 0.5)
            criteria_met = assessment.get('criteria_met', 0)
            total_criteria = assessment.get('total_criteria', 1)
            
            # Create a comprehensive assessment visualization
            performance_level = "Excellent" if overall_score > 0.8 else "Good" if overall_score > 0.6 else "Needs Improvement"
            
            assessment_diagram = f"""
flowchart TD
    A["🎯 {scenario.scenario_name}<br/>Assessment Results"] --> B["📊 Overall Score<br/>{overall_score:.1%}"]
    B --> C{{"Performance Level"}}
    
    C -->|"Score > 80%"| D["🟢 Excellent<br/>Performance"]
    C -->|"Score 60-80%"| E["🟡 Good<br/>Performance"] 
    C -->|"Score < 60%"| F["🔴 Needs<br/>Improvement"]
    
    B --> G["📋 Criteria Analysis<br/>{criteria_met}/{total_criteria} Met"]
    
    G --> H["✅ Strengths"]
    G --> I["🔄 Improvements"]
    
    H --> J["🚀 Recommendations"]
    I --> J
    
    classDef excellent fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    classDef good fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef improvement fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef neutral fill:#f5f5f5,stroke:#666,stroke-width:2px
    
    class D excellent
    class E good
    class F improvement
    class A,B,G,H,I,J neutral
"""
            
            return assessment_diagram
            
        except Exception as e:
            logger.error(f"Failed to generate assessment diagram: {e}")
            return """
flowchart TD
    A[Assessment Complete] --> B[Results Available]
    B --> C[Review Recommendations]
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px
"""

# Tool interface functions for LLM integration
async def analyze_agent_capabilities(user_id: str, org_id: str) -> str:
    """
    Analyze agent's comprehensive capabilities for grading
    
    Args:
        user_id: User identifier
        org_id: Organization identifier
        
    Returns:
        Formatted capability analysis
    """
    try:
        grading_tool = GradingTool()
        
        # Get comprehensive capabilities
        analysis = await grading_tool.get_comprehensive_agent_capabilities(
            user_id=user_id,
            org_id=org_id,
            max_vectors=200
        )
        
        if analysis["success"]:
            capabilities = analysis["capabilities"]
            return f"""
🧠 **Agent Capability Analysis Complete**

📊 **Analysis Summary:**
- Total vectors analyzed: {analysis['total_vectors_analyzed']}
- Domains covered: {analysis['unique_domains_covered']}
- Capabilities identified: {len(capabilities)}

🎯 **Top Capabilities:**
{chr(10).join([f"• {cap.skill} (confidence: {cap.confidence:.1%})" for cap in capabilities[:5]])}

Ready to generate optimal grading scenario based on these capabilities.
"""
        else:
            return f"❌ Capability analysis failed: {analysis.get('error', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"Agent capability analysis failed: {e}")
        return f"❌ Error analyzing agent capabilities: {str(e)}"

async def generate_grading_scenario(user_id: str, org_id: str, agent_name: str = "Your Agent") -> str:
    """
    Generate optimal grading scenario for the agent
    
    Args:
        user_id: User identifier
        org_id: Organization identifier
        agent_name: Name of the agent
        
    Returns:
        Formatted grading scenario proposal
    """
    try:
        grading_tool = GradingTool()
        
        # Get capabilities
        analysis = await grading_tool.get_comprehensive_agent_capabilities(
            user_id=user_id,
            org_id=org_id,
            max_vectors=200
        )
        
        if not analysis["success"]:
            return f"❌ Failed to analyze capabilities: {analysis.get('error')}"
        
        # Generate optimal scenario
        scenario = await grading_tool.generate_optimal_grading_scenario(
            analysis["capabilities"], 
            agent_name
        )
        
        return f"""
🎯 **Optimal Grading Scenario Generated**

**Scenario:** {scenario.scenario_name}
**Description:** {scenario.description}
**Estimated Time:** {scenario.estimated_time}
**Difficulty:** {scenario.difficulty_level}

🤖 **Agent Role-Play Introduction:**
"{scenario.agent_role_play}"

📋 **Test Components:**
{chr(10).join([f"• {inp['description']}" for inp in scenario.test_inputs])}

✅ **Expected Demonstrations:**
{chr(10).join([f"• {out['description']}" for out in scenario.expected_outputs])}

🎖️ **Capabilities Showcased:**
{chr(10).join([f"• {cap}" for cap in scenario.showcased_capabilities])}

**Approve this grading scenario?** [Yes] [Generate Alternative]
"""
        
    except Exception as e:
        logger.error(f"Grading scenario generation failed: {e}")
        return f"❌ Error generating grading scenario: {str(e)}" 