"""
Business Logic Tool - Domain-specific business operations
Handles sales analysis, reporting, and other business intelligence tasks
Uses vector knowledge retrieval for dynamic analysis instructions
"""

import logging
from typing import Dict, Any, List
import json
import re
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

# Configure business logic specific logging
business_logger = logging.getLogger("business_logic")
business_logger.setLevel(logging.INFO)
if not business_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('📊 [BUSINESS] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    business_logger.addHandler(handler)


class BusinessLogicTool:
    """Tool for business-specific operations and analysis using vector knowledge"""
    
    def __init__(self):
        self.name = "business_logic_tool"
        self._initialize_vector_system()
    
    def _initialize_vector_system(self):
        """Initialize connection to vector knowledge system"""
        try:
            # Import vector knowledge system
            from pccontroller import query_knowledge
            self.query_knowledge = query_knowledge
            business_logger.info("Vector knowledge system initialized successfully")
        except Exception as e:
            business_logger.error(f"Failed to initialize vector knowledge system: {e}")
            self.query_knowledge = None
    
    def analyze_document(self, document_content: str, analysis_type: str = "comprehensive") -> str:
        """
        Analyze business documents using dynamic instructions from vector knowledge
        
        Args:
            document_content: The document content to analyze
            analysis_type: Type of analysis to perform (e.g., "sales", "financial", "strategy", "comprehensive")
            
        Returns:
            Dynamic analysis based on vector knowledge instructions
        """
        business_logger.info(f"ANALYZE_DOCUMENT - Starting analysis")
        business_logger.info(f"ANALYZE_DOCUMENT - Document length: {len(document_content)} chars")
        business_logger.info(f"ANALYZE_DOCUMENT - Analysis type: {analysis_type}")
        
        try:
            # Generate knowledge query based on analysis type
            knowledge_query = f"how to analyze {analysis_type} documents business analysis framework"
            
            # Get analysis instructions from vector knowledge
            analysis_instructions = self._get_analysis_instructions(knowledge_query, "unknown", "unknown")
            
            # Apply instructions to document content
            analysis_result = self._apply_analysis_instructions(document_content, analysis_instructions, analysis_type)
            
            return analysis_result
            
        except Exception as e:
            business_logger.error(f"ANALYZE_DOCUMENT - Error: {str(e)}")
            return f"Error analyzing document: {str(e)}"
    
    def process_with_knowledge(self, document_content: str, knowledge_query: str, user_id: str = "unknown", org_id: str = "unknown") -> str:
        """
        Process document using specific knowledge retrieved from vectors
        
        Args:
            document_content: The document content to analyze
            knowledge_query: Query to retrieve relevant knowledge from vectors
            user_id: User identifier for vector search
            org_id: Organization identifier for vector search
            
        Returns:
            Analysis using retrieved knowledge instructions
        """
        business_logger.info(f"PROCESS_WITH_KNOWLEDGE - Starting knowledge-based analysis")
        business_logger.info(f"PROCESS_WITH_KNOWLEDGE - Knowledge query: {knowledge_query}")
        business_logger.info(f"PROCESS_WITH_KNOWLEDGE - User: {user_id}, Org: {org_id}")
        
        try:
            # Retrieve analysis instructions from vector knowledge
            analysis_instructions = self._get_analysis_instructions(knowledge_query, user_id, org_id)
            
            # Apply the instructions to the document
            analysis_result = self._apply_analysis_instructions(document_content, analysis_instructions, "knowledge_based")
            
            return analysis_result
            
        except Exception as e:
            business_logger.error(f"PROCESS_WITH_KNOWLEDGE - Error: {str(e)}")
            return f"Error in knowledge-based processing: {str(e)}"
    
    def _get_analysis_instructions(self, query: str, user_id: str, org_id: str) -> List[Dict]:
        """Retrieve analysis instructions from vector knowledge system"""
        try:
            if not self.query_knowledge:
                business_logger.warning("Vector knowledge system not available")
                return []
            
            # Run async query_knowledge synchronously
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, use a different approach
                future = asyncio.ensure_future(
                    self.query_knowledge(
                        query=query,
                        org_id=org_id,
                        user_id=user_id,
                        top_k=5,
                        min_similarity=0.3
                    )
                )
                # Wait for completion
                while not future.done():
                    continue
                knowledge_results = future.result()
            except RuntimeError:
                # No event loop running, create a new one
                knowledge_results = asyncio.run(
                    self.query_knowledge(
                        query=query,
                        org_id=org_id,
                        user_id=user_id,
                        top_k=5,
                        min_similarity=0.3
                    )
                )
            
            business_logger.info(f"Retrieved {len(knowledge_results)} knowledge entries for analysis instructions")
            return knowledge_results
            
        except Exception as e:
            business_logger.error(f"Error retrieving analysis instructions: {e}")
            return []
    
    def _apply_analysis_instructions(self, document_content: str, instructions: List[Dict], analysis_type: str) -> str:
        """Apply retrieved instructions to analyze the document"""
        try:
            analysis_parts = []
            analysis_parts.append("🧠 KNOWLEDGE-BASED DOCUMENT ANALYSIS")
            analysis_parts.append("=" * 60)
            analysis_parts.append(f"Analysis Type: {analysis_type}")
            analysis_parts.append(f"Document Size: {len(document_content):,} characters")
            analysis_parts.append(f"Knowledge Sources: {len(instructions)} instructions retrieved")
            analysis_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            analysis_parts.append("")
            
            if instructions:
                analysis_parts.append("📚 APPLIED KNOWLEDGE INSTRUCTIONS:")
                analysis_parts.append("-" * 40)
                
                for i, instruction in enumerate(instructions[:3], 1):  # Use top 3 instructions
                    content = instruction.get("raw", "")[:300] + "..." if len(instruction.get("raw", "")) > 300 else instruction.get("raw", "")
                    score = instruction.get("score", 0)
                    confidence = instruction.get("confidence", 0)
                    
                    analysis_parts.append(f"{i}. [Score: {score:.3f}, Confidence: {confidence:.3f}]")
                    analysis_parts.append(f"   {content}")
                    analysis_parts.append("")
                
                analysis_parts.append("📋 DOCUMENT ANALYSIS RESULTS:")
                analysis_parts.append("-" * 40)
                
                # Apply the knowledge to analyze the document
                analysis_insights = self._generate_insights_from_knowledge(document_content, instructions)
                analysis_parts.extend(analysis_insights)
                
            else:
                analysis_parts.append("⚠️ No specific analysis instructions found")
                analysis_parts.append("• Using general document analysis approach")
                analysis_parts.append("• Consider adding analysis frameworks to knowledge base")
                analysis_parts.append("")
                analysis_parts.append("📋 BASIC DOCUMENT PREVIEW:")
                analysis_parts.append("-" * 40)
                preview = document_content[:500] + "..." if len(document_content) > 500 else document_content
                analysis_parts.append(preview)
            
            final_analysis = "\n".join(analysis_parts)
            business_logger.info(f"Generated knowledge-based analysis ({len(final_analysis)} chars)")
            
            return final_analysis
            
        except Exception as e:
            business_logger.error(f"Error applying analysis instructions: {e}")
            return f"Error in analysis application: {str(e)}"
    
    def _generate_insights_from_knowledge(self, document_content: str, instructions: List[Dict]) -> List[str]:
        """Generate insights by applying knowledge instructions to document content"""
        insights = []
        
        try:
            # Extract key themes from knowledge instructions
            knowledge_themes = []
            for instruction in instructions:
                content = instruction.get("raw", "").lower()
                if "business plan" in content or "strategy" in content:
                    knowledge_themes.append("business_strategy")
                if "financial" in content or "revenue" in content or "profit" in content:
                    knowledge_themes.append("financial_analysis")
                if "market" in content or "competition" in content:
                    knowledge_themes.append("market_analysis")
                if "risk" in content or "challenge" in content:
                    knowledge_themes.append("risk_assessment")
            
            # Generate insights based on themes
            if "business_strategy" in knowledge_themes:
                insights.extend(self._analyze_business_strategy(document_content))
            
            if "financial_analysis" in knowledge_themes:
                insights.extend(self._analyze_financial_aspects(document_content))
            
            if "market_analysis" in knowledge_themes:
                insights.extend(self._analyze_market_aspects(document_content))
            
            if "risk_assessment" in knowledge_themes:
                insights.extend(self._analyze_risks(document_content))
            
            # If no specific themes, provide general analysis
            if not insights:
                insights.extend(self._general_document_analysis(document_content))
            
        except Exception as e:
            business_logger.error(f"Error generating insights: {e}")
            insights.append(f"Error in insight generation: {str(e)}")
        
        return insights
    
    def _analyze_business_strategy(self, content: str) -> List[str]:
        """Analyze business strategy aspects"""
        insights = []
        insights.append("🎯 BUSINESS STRATEGY ANALYSIS:")
        insights.append("• Strategy Focus: " + ("Identified" if any(word in content.lower() for word in ["strategy", "plan", "approach"]) else "Not clearly defined"))
        insights.append("• Business Model: " + ("Present" if any(word in content.lower() for word in ["model", "revenue", "service", "product"]) else "Not detailed"))
        insights.append("• Competitive Advantage: " + ("Mentioned" if any(word in content.lower() for word in ["competitive", "advantage", "unique", "differentiation"]) else "Not addressed"))
        return insights
    
    def _analyze_financial_aspects(self, content: str) -> List[str]:
        """Analyze financial aspects"""
        insights = []
        insights.append("💰 FINANCIAL ANALYSIS:")
        insights.append("• Revenue Model: " + ("Detailed" if any(word in content.lower() for word in ["revenue", "income", "sales", "pricing"]) else "Basic"))
        insights.append("• Financial Projections: " + ("Present" if any(word in content.lower() for word in ["projection", "forecast", "budget", "financial"]) else "Missing"))
        insights.append("• Funding Requirements: " + ("Specified" if any(word in content.lower() for word in ["funding", "investment", "capital", "financing"]) else "Not detailed"))
        return insights
    
    def _analyze_market_aspects(self, content: str) -> List[str]:
        """Analyze market aspects"""
        insights = []
        insights.append("🌍 MARKET ANALYSIS:")
        insights.append("• Target Market: " + ("Defined" if any(word in content.lower() for word in ["market", "customer", "target", "audience"]) else "Not specified"))
        insights.append("• Competition: " + ("Analyzed" if any(word in content.lower() for word in ["competition", "competitor", "competitive"]) else "Not addressed"))
        insights.append("• Market Size: " + ("Estimated" if any(word in content.lower() for word in ["size", "market", "demand", "opportunity"]) else "Not quantified"))
        return insights
    
    def _analyze_risks(self, content: str) -> List[str]:
        """Analyze risk factors"""
        insights = []
        insights.append("⚠️ RISK ASSESSMENT:")
        insights.append("• Risk Factors: " + ("Identified" if any(word in content.lower() for word in ["risk", "challenge", "threat", "concern"]) else "Not addressed"))
        insights.append("• Mitigation Strategies: " + ("Present" if any(word in content.lower() for word in ["mitigation", "strategy", "solution", "approach"]) else "Not detailed"))
        return insights
    
    def _general_document_analysis(self, content: str) -> List[str]:
        """General document analysis when no specific themes are found"""
        insights = []
        insights.append("📋 GENERAL DOCUMENT ANALYSIS:")
        insights.append(f"• Document Type: Business document ({len(content)} characters)")
        insights.append("• Key Sections: " + str(len([line for line in content.split('\n') if line.strip()])))
        insights.append("• Content Preview: " + content[:200] + "..." if len(content) > 200 else content)
        return insights
    
    def get_tool_description(self):
        """Return tool descriptions for LLM function calling"""
        return [
            {
                "name": "analyze_document",
                "description": "CRITICAL: Analyze business documents dynamically using vector knowledge. Use this for comprehensive document analysis with customizable instructions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_content": {
                            "type": "string",
                            "description": "The document content to analyze (from file reading tools)"
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis to perform (e.g., 'sales', 'financial', 'strategy', 'comprehensive')",
                            "default": "comprehensive"
                        }
                    },
                    "required": ["document_content"]
                }
            },
            {
                "name": "process_with_knowledge",
                "description": "CRITICAL: Process documents using specific knowledge retrieved from vector search. Use this when you need specialized analysis instructions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_content": {
                            "type": "string",
                            "description": "The document content to analyze"
                        },
                        "knowledge_query": {
                            "type": "string",
                            "description": "Query to retrieve relevant analysis knowledge from vectors (e.g., 'how to analyze business plans', 'financial document analysis')"
                        },
                        "user_id": {
                            "type": "string",
                            "description": "User identifier for vector search context",
                            "default": "unknown"
                        },
                        "org_id": {
                            "type": "string",
                            "description": "Organization identifier for vector search context",
                            "default": "unknown"
                        }
                    },
                    "required": ["document_content", "knowledge_query"]
                }
            }
        ] 