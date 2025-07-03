import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from cachetools import TTLCache
import re
import pytz

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tool6support import emit_analysis_event  
from pccontroller import query_knowledge_from_graph

from utilities import logger

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model to a JSON-serializable dict."""
    return json.loads(json.dumps(model.model_dump(), cls=DateTimeEncoder))

# Initialize LLM
LLM = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.01)

# Pydantic model for UserProfile
class UserProfileModel(BaseModel):
    user_id: str
    skills: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    classification: str = "unknown"
    business_goal: str = ""
    other_aspects: Dict[str, Any] = Field(default_factory=dict)
    analysis_summary: Dict[str, Any] = Field(default_factory=lambda: {
        "key_findings": [],
        "user_needs": [],
        "potential_challenges": [],
        "narrative": "No analysis available."
    })
    action_plan: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    analysis_queries: List[str] = Field(default_factory=list)

class CoTProcessor:
    def __init__(self):
        self.user_profiles = TTLCache(maxsize=1000, ttl=3600)
        # Don't set a default graph_version_id here, will be set in initialize
        self.graph_version_id = None
        # Initialize profiling_skills with empty dict, will be loaded asynchronously
        self.profiling_skills = {"knowledge_context": "Loading...", "metadata": {}}
        self.ai_business_objectives = {"knowledge_context": "Loading...", "metadata": {}}
        self.communication_skills = {"knowledge_context": "Loading...", "metadata": {}}
        self.comprehensive_profiling_skills = {"knowledge_context": "Loading...", "metadata": {}}
        self.comprehensive_communication_skills = {"knowledge_context": "Loading...", "metadata": {}}
        self.comprehensive_business_objectives = {"knowledge_context": "Loading...", "metadata": {}}

    async def initialize(self, graph_version_id=None):
        """Initialize the processor asynchronously"""
        # Use provided graph_version_id or fallback to current or generate a new one
        self.graph_version_id = graph_version_id or "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"
        logger.info(f"Initializing CoTProcessor with graph_version_id: {self.graph_version_id}")
        
        # First load the basic knowledge bases
        self.profiling_skills = await self._load_profiling_skills()
        self.ai_business_objectives = await self._load_business_objectives_awareness()
        self.communication_skills = await self._load_communication_skills()
        
        # Then compile the comprehensive versions
        compilation_result = await self._compile_self_awareness()
        #logger.info(f"Self-awareness compilation completed: {compilation_result.get('status', 'unknown')}")
        #logger.info(f"Profiling knowledge as Instinct: {self.comprehensive_profiling_skills.get('knowledge_context', '')}")
        #logger.info(f"Communication knowledge as Instinct: {self.comprehensive_communication_skills.get('knowledge_context', '')}")
        #logger.info(f"Business objectives knowledge as Instinct: {self.comprehensive_business_objectives.get('knowledge_context', '')}")
        
        return self

    async def _compile_self_awareness(self) -> Dict[str, Any]:
        """Compile self awareness knowledge base.
        This function compiles comprehensive knowledge bases by synthesizing information from
        profiling_skills, communication_skills, and ai_business_objectives using LLM.
        """
        logger.info("Compiling self awareness knowledge bases")
        
        # Extract knowledge contexts from the source variables
        profiling_context = self.profiling_skills.get("knowledge_context", "")
        communication_context = self.communication_skills.get("knowledge_context", "")
        business_context = self.ai_business_objectives.get("knowledge_context", "")
        
        # Create prompts for each comprehensive knowledge base with language preservation instructions
        profiling_prompt = f"""
        TASK: Create a comprehensive, well-structured INSTRUCTION MANUAL for user profiling techniques.
        
        SOURCE KNOWLEDGE:
        {profiling_context}
        
        CRITICAL INSTRUCTIONS:
        1. PRESERVE THE ORIGINAL LANGUAGE - maintain the same language as the source (Vietnamese/English)
        2. DO NOT translate Vietnamese terms or phrases into English or vice versa
        3. KEEP ALL ORIGINAL EXAMPLES exactly as they appear in the source
        4. Maintain cultural context and specific terminology from the original
        5. CREATE EXECUTABLE INSTRUCTION MANUAL - transform all knowledge into direct, actionable commands
        6. USE IMPERATIVE LANGUAGE - write "DO this" not "you can do this" or "this is suitable for"
        7. ORGANIZE AS DECISION TREES - create IF-THEN-ELSE command structures
        8. Remove redundant information while preserving all unique content
        9. Use the EXACT SAME PRONOUNS as the original (e.g., "em"/"anh" in Vietnamese)
        
        CONFLICT RESOLUTION INSTRUCTIONS:
        IMPORTANT: When you encounter contradictory instructions in the source knowledge, apply TEMPORAL PRECEDENCE - prioritize the instruction that appears LATER in the source sequence, as it represents the most recent update or override.
        
        9. CONVERT CONFLICTS TO DIRECT INSTRUCTIONS: When you find contradictory profiling approaches:
           - Transform conflicts into clear decision rules using source language (e.g., "HƯỚNG DẪN QUYẾT ĐỊNH:" for Vietnamese, "DECISION GUIDE:" for English)
           - Write direct commands: "Use Method A when..." instead of "Method A is suitable for..."
           - Create IF-THEN instruction blocks for different scenarios
        10. METHODOLOGY SELECTION RULES: For different profiling methodologies:
           - Write as executable instructions: "EXECUTE this approach IF user shows X characteristics"
           - Use imperative language in original language (e.g., "HÃY SỬ DỤNG:" for Vietnamese, "USE:" for English)
           - Provide step-by-step decision trees rather than comparisons
        11. CLASSIFICATION DECISION FRAMEWORK: When sources disagree on user types:
           - Create classification flowcharts in text form
           - Write as commands: "CLASSIFY as Type A IF conditions X, Y, Z are met"
           - Use action-oriented language in source language (e.g., "PHÂN LOẠI:" for Vietnamese, "CLASSIFY:" for English)
        12. ACTIONABLE GUIDANCE: Transform all expert disagreements into practical decision-making instructions using source language
        
        FORMAT: Respond with only the INSTRUCTION MANUAL text in the original language. Structure as executable commands, not descriptive analysis.
        """
        
        communication_prompt = f"""
        TASK: Create a comprehensive, well-structured INSTRUCTION MANUAL for effective communication techniques.
        
        SOURCE KNOWLEDGE:
        {communication_context}
        
        CRITICAL INSTRUCTIONS:
        1. PRESERVE THE ORIGINAL LANGUAGE - maintain the same language as the source (Vietnamese/English)
        2. DO NOT translate Vietnamese terms or phrases into English or vice versa
        3. KEEP ALL ORIGINAL EXAMPLES exactly as they appear in the source
        4. Maintain cultural context and specific terminology from the original
        5. CREATE EXECUTABLE INSTRUCTION MANUAL - transform all knowledge into direct, actionable commands
        6. USE IMPERATIVE LANGUAGE - write "DO this" not "you can do this" or "this is suitable for"
        7. ORGANIZE AS DECISION TREES - create IF-THEN-ELSE command structures
        8. Remove redundant information while preserving all unique content
        9. Use the EXACT SAME PRONOUNS as the original (e.g., "em"/"anh" in Vietnamese)
        10. Preserve colloquial phrases if they appear in the original
        
        CONFLICT RESOLUTION INSTRUCTIONS:
        IMPORTANT: When you encounter contradictory instructions in the source knowledge, apply TEMPORAL PRECEDENCE - prioritize the instruction that appears LATER in the source sequence, as it represents the most recent update or override.
        
        10. COMMUNICATION EXECUTION RULES: When sources suggest different communication approaches:
           - Create direct execution commands using source language (e.g., "THỰC HIỆN GIAO TIẾP:" for Vietnamese, "EXECUTE COMMUNICATION:" for English)
           - Write specific instructions: "SPEAK formally IF user is [classification]" instead of describing style variations
           - Use command format: "DO this, NOT that" for clear guidance
        11. TONE COMMAND FRAMEWORK: For contradictory advice on formality, friendliness, etc.:
           - Write as direct tone commands using original language (e.g., "SỬ DỤNG GIỌNG ĐIỆU:" for Vietnamese)
           - Create conditional instructions: "BE formal IF situation X, BE casual IF situation Y"
           - Use imperative verbs in source language for immediate action
        12. LANGUAGE SELECTION COMMANDS: When sources disagree on language choice or expressions:
           - Write as selection algorithms using source language (e.g., "CHỌN NGÔN NGỮ:" for Vietnamese)
           - Create decision trees: "IF audience = X, THEN use expression Y"
           - Provide exact phrases and expressions to use, not just options
        13. TECHNIQUE EXECUTION GUIDE: For contradictory communication techniques:
           - Transform into step-by-step execution guides in source language (e.g., "THỰC HIỆN KỸ THUẬT:" for Vietnamese)
           - Write as procedures: "Step 1: Do X, Step 2: If response Y, then do Z"
           - Focus on WHEN and HOW to execute, not just what techniques exist
        14. DIRECT CULTURAL COMMANDS: Transform cultural considerations into specific behavioral instructions using source language
        
        FORMAT: Respond with only the INSTRUCTION MANUAL text in the original language. Structure as executable commands, not descriptive analysis.
        """
        
        business_prompt = f"""
        TASK: Create a comprehensive, well-structured INSTRUCTION MANUAL for business objectives.
        
        SOURCE KNOWLEDGE:
        {business_context}
        
        CRITICAL INSTRUCTIONS:
        1. PRESERVE THE ORIGINAL LANGUAGE - maintain the same language as the source (Vietnamese/English)
        2. DO NOT translate Vietnamese terms or phrases into English or vice versa
        3. KEEP ALL ORIGINAL EXAMPLES exactly as they appear in the source
        4. Maintain cultural context and specific terminology from the original
        5. CREATE EXECUTABLE INSTRUCTION MANUAL - transform all knowledge into direct, actionable commands
        6. USE IMPERATIVE LANGUAGE - write "DO this" not "you can do this" or "this is suitable for"
        7. ORGANIZE AS DECISION TREES - create IF-THEN-ELSE command structures
        8. Remove redundant information while preserving all unique content
        9. Preserve all references to specific services, courses, or products
        
        CONFLICT RESOLUTION INSTRUCTIONS:
        IMPORTANT: When you encounter contradictory instructions in the source knowledge, apply TEMPORAL PRECEDENCE - prioritize the instruction that appears LATER in the source sequence, as it represents the most recent update or override.
        
        9. PRIORITY EXECUTION COMMANDS: When sources prioritize different business objectives:
           - Create priority decision algorithms using source language (e.g., "THỰC HIỆN ƯU TIÊN:" for Vietnamese, "EXECUTE PRIORITY:" for English)
           - Write as conditional commands: "PRIORITIZE objective A IF business context X, PRIORITIZE objective B IF context Y"
           - Use directive language: "FOCUS on X first, THEN move to Y" instead of describing variations
        10. STRATEGY SELECTION RULES: For contradictory business strategies or approaches:
           - Transform into strategy execution commands using source language (e.g., "THỰC HIỆN CHIẾN LƯỢC:" for Vietnamese)
           - Write as decision algorithms: "APPLY strategy A IF conditions X, Y, Z are met"
           - Create step-by-step implementation guides rather than option descriptions
        11. MEASUREMENT EXECUTION FRAMEWORK: When sources disagree on success measurements:
           - Write as measurement commands using source language (e.g., "ĐO LƯỜNG BẰNG CÁCH:" for Vietnamese)
           - Create specific measurement protocols: "MEASURE X using method Y IF goal is Z"
           - Provide exact metrics and calculation methods, not just approaches
        12. GOAL ACHIEVEMENT COMMANDS: For contradictory business goals or outcomes:
           - Transform into goal execution instructions using source language (e.g., "ĐẠT MỤC TIÊU:" for Vietnamese)
           - Write as achievement pathways: "TO achieve goal X, EXECUTE steps A, B, C in sequence"
           - Focus on HOW to achieve goals, not just what goals exist
        13. IMPLEMENTATION EXECUTION GUIDE: For different execution approaches:
           - Create step-by-step implementation commands using source language (e.g., "TRIỂN KHAI THEO BƯỚC:" for Vietnamese)
           - Write as procedural instructions: "Step 1: ASSESS resources, Step 2: IF sufficient, THEN proceed with method A"
           - Provide specific action sequences rather than approach comparisons
        14. BUSINESS COMMAND FRAMEWORK: Transform all business context considerations into specific operational instructions using source language
        
        FORMAT: Respond with only the INSTRUCTION MANUAL text in the original language. Structure as executable commands, not descriptive analysis.
        """
        
        try:
            # Process all three knowledge bases in parallel
            responses = await asyncio.gather(
                LLM.ainvoke(profiling_prompt),
                LLM.ainvoke(communication_prompt),
                LLM.ainvoke(business_prompt),
                return_exceptions=True
            )
            
            # Process the responses and update the comprehensive variables
            timestamp = datetime.now().isoformat()
            
            # Process profiling response
            if isinstance(responses[0], Exception):
                logger.error(f"Error compiling comprehensive profiling skills: {str(responses[0])}")
                profiling_result = {"knowledge_context": "Compilation failed.", "metadata": {"error": str(responses[0]), "timestamp": timestamp}}
            else:
                profiling_content = responses[0].content.strip()
                profiling_result = {
                    "knowledge_context": profiling_content,
                    "metadata": {
                        "source": "synthesized from profiling_skills",
                        "timestamp": timestamp,
                        "char_count": len(profiling_content),
                        "language_preserved": True
                    }
                }
            self.comprehensive_profiling_skills = profiling_result
            
            # Process communication response
            if isinstance(responses[1], Exception):
                logger.error(f"Error compiling comprehensive communication skills: {str(responses[1])}")
                communication_result = {"knowledge_context": "Compilation failed.", "metadata": {"error": str(responses[1]), "timestamp": timestamp}}
            else:
                communication_content = responses[1].content.strip()
                communication_result = {
                    "knowledge_context": communication_content,
                    "metadata": {
                        "source": "synthesized from communication_skills",
                        "timestamp": timestamp,
                        "char_count": len(communication_content),
                        "language_preserved": True
                    }
                }
            self.comprehensive_communication_skills = communication_result
            
            # Process business objectives response
            if isinstance(responses[2], Exception):
                logger.error(f"Error compiling comprehensive business objectives: {str(responses[2])}")
                business_result = {"knowledge_context": "Compilation failed.", "metadata": {"error": str(responses[2]), "timestamp": timestamp}}
            else:
                business_content = responses[2].content.strip()
                business_result = {
                    "knowledge_context": business_content,
                    "metadata": {
                        "source": "synthesized from ai_business_objectives",
                        "timestamp": timestamp,
                        "char_count": len(business_content),
                        "language_preserved": True
                    }
                }
            self.comprehensive_business_objectives = business_result
            
            logger.info(f"Successfully compiled all three comprehensive knowledge bases with language preservation")
            
            # Return summary of the compilation
            return {
                "status": "success",
                "profiling_chars": len(profiling_result.get("knowledge_context", "")),
                "communication_chars": len(communication_result.get("knowledge_context", "")),
                "business_chars": len(business_result.get("knowledge_context", "")),
                "language_preserved": True,
                "timestamp": timestamp
            }
        
        except Exception as e:
            logger.error(f"Error in _compile_self_awareness: {str(e)}")
            return {"status": "error", "message": str(e), "timestamp": datetime.now().isoformat()}
        
    async def _load_profiling_skills(self) -> Dict[str, Any]:
        """Load knowledge base concurrently using fetch_knowledge."""
        logger.info(f"Loading knowledge base with graph_version_id: {self.graph_version_id}")
        queries = [
            "Cách phân nhóm người dùng",
            "Cách xây dựng hồ sơ người dùng",
            "Cách phân tích hồ sơ người dùng",
            "Cần có những gì trong hồ sơ người dùng"
        ]
        
        try:
            results = await asyncio.gather(
                *[query_knowledge_from_graph(query, self.graph_version_id,top_k=100,min_similarity=0.35) for query in queries],
                return_exceptions=True
            )
            profiling_skills = {}
            for query, result in zip(queries, results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching knowledge for query {query}: {str(result)}")
                    profiling_skills[query] = {}
                else:
                    # Log raw result structure for debugging
                    if isinstance(result, list) and result:
                        logger.info(f"Query '{query}' returned list result with {len(result)} items")
                        logger.info(f"First result keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'not a dict'}")
                    elif isinstance(result, dict):
                        logger.info(f"Query '{query}' returned dict result with keys: {list(result.keys())}")
                    else:
                        logger.info(f"Query '{query}' returned result of type: {type(result)}")
                    
                    try:
                        profiling_skills[query] = json.loads(result) if isinstance(result, str) else result
                    except json.JSONDecodeError:
                        profiling_skills[query] = {"content": result}

            # Process the results
            processed = self._process_profiling_skills(profiling_skills)
            
            # Add additional logging
            #logger.info(f"Profiling knowledge processed: {processed.get('metadata', {}).get('entry_count', 0)} entries")
            if processed.get('knowledge_context', '') == "No knowledge available.":
                logger.warning("No profiling knowledge entries were extracted from the query results")
                
            return processed
            
        except Exception as e:
            logger.error(f"Error loading profiling skills: {str(e)}")
            return {"knowledge_context": "No knowledge available.", "metadata": {"error": str(e)}}
        
    async def _load_communication_skills(self) -> Dict[str, Any]:
        """Load communication knowledge base concurrently using fetch_knowledge."""
        logger.info(f"Loading communication skills with graph_version_id: {self.graph_version_id}")
        queries = [
            "Cách giao tiếp và nói năng với khác"
        ]
        
        try:
            results = await asyncio.gather(
                *[query_knowledge_from_graph(query, self.graph_version_id,top_k=100,min_similarity=0.4) for query in queries],
                return_exceptions=True
            )
            communication_skills = {}
            for query, result in zip(queries, results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching knowledge for query {query}: {str(result)}")
                    communication_skills[query] = {}
                else:     
                    try:
                        communication_skills[query] = json.loads(result) if isinstance(result, str) else result
                    except json.JSONDecodeError:
                        communication_skills[query] = {"content": result}
            
            # Process the results
            processed = self._process_communication_skills(communication_skills)
            
            if processed.get('knowledge_context', '') == "No knowledge available.":
                logger.warning("No communication knowledge entries were extracted from the query results")
            
            return processed
        except Exception as e:
            logger.error(f"Error loading communication skills: {str(e)}")
            return {"knowledge_context": "No knowledge available.", "metadata": {"error": str(e)}}
        
    async def _load_business_objectives_awareness(self) -> Dict[str, Any]:
        """Load knowledge base concurrently using fetch_knowledge."""
        logger.info(f"Loading business objectives knowledge with graph_version_id: {self.graph_version_id}")
        queries = [
            "Các mục mục têu công việc"
        ]
        
        try:
            results = await asyncio.gather(
                *[query_knowledge_from_graph(query, self.graph_version_id,top_k=100,min_similarity=0.25) for query in queries],
                return_exceptions=True
            )
            business_objectives = {}
            for query, result in zip(queries, results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching knowledge for query {query}: {str(result)}")
                    business_objectives[query] = {}
                else:
                    # Log raw result structure for debugging
                    if isinstance(result, list) and result:
                        logger.info(f"Query '{query}' returned list result with {len(result)} items")
                        logger.info(f"First result keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'not a dict'}")
                    elif isinstance(result, dict):
                        logger.info(f"Query '{query}' returned dict result with keys: {list(result.keys())}")
                    else:
                        logger.info(f"Query '{query}' returned result of type: {type(result)}")
                    
                    try:
                        business_objectives[query] = json.loads(result) if isinstance(result, str) else result
                    except json.JSONDecodeError:
                        business_objectives[query] = {"content": result}
                        
            # Process the results
            logger.info(f"Business objectives knowledge RAW: {business_objectives}")    
            processed = self._process_business_objectives_knowledge(business_objectives)
            
            # Add additional logging
            #logger.info(f"Business objectives knowledge processed: {processed.get('metadata', {}).get('entry_count', 0)} entries")
            if processed.get('knowledge_context', '') == "No knowledge available.":
                logger.warning("No business objectives knowledge entries were extracted from the query results")
                
            return processed
            
        except Exception as e:
            logger.error(f"Error loading profiling skills: {str(e)}")
            return {"knowledge_context": "No knowledge available.", "metadata": {"error": str(e)}}

    def _extract_knowledge_content(self, knowledge_data, extract_user_part=True):
        """Extract knowledge content from various data types, handling AI synthesis content.
        
        Args:
            knowledge_data: The knowledge data (string, dict, list)
            extract_user_part: Whether to extract the User part (True) or AI part (False) - now only used for backwards compatibility
            
        Returns:
            str: The extracted knowledge content with "AI:" prefix stripped
        """
        knowledge_content = ""
        #logger.info(f"KNOWLEDGE data: {knowledge_data}")
        
        # Handle string data
        if isinstance(knowledge_data, str):
            knowledge_content = knowledge_data
            # Strip "AI:" prefix if it exists
            if knowledge_content.startswith("AI:"):
                knowledge_content = knowledge_content[3:].strip()
                logger.info(f"Stripped AI: prefix from string knowledge")
        
        # Handle dict data
        elif isinstance(knowledge_data, dict):
            if "raw" in knowledge_data:
                raw_content = knowledge_data["raw"]
                # Strip "AI:" prefix if it exists
                if raw_content.startswith("AI:"):
                    knowledge_content = raw_content[3:].strip()
                    logger.info(f"Stripped AI: prefix from dict knowledge")
                else:
                    knowledge_content = raw_content
            elif "content" in knowledge_data:
                knowledge_content = knowledge_data["content"]
                # Strip "AI:" prefix if it exists
                if knowledge_content.startswith("AI:"):
                    knowledge_content = knowledge_content[3:].strip()
            else:
                # Serialize the dictionary as a fallback
                knowledge_content = json.dumps(knowledge_data)
        
        # Handle list data (process all items)
        elif isinstance(knowledge_data, list) and knowledge_data:
            # Process all items instead of just the first one
            knowledge_items = []
            for item in knowledge_data:
                # Recursively process each item
                processed_content = self._extract_knowledge_content(item, extract_user_part)
                if processed_content:
                    # Normalize whitespace before adding
                    processed_content = re.sub(r'\n{3,}', '\n\n', processed_content)
                    processed_content = re.sub(r' +', ' ', processed_content)
                    processed_content = processed_content.strip()
                    knowledge_items.append(processed_content)
            
            # Join all knowledge items with a cleaner separator
            knowledge_content = "\n---\n".join(knowledge_items) if knowledge_items else ""
        
        # Handle other types
        else:
            knowledge_content = str(knowledge_data)
            # Strip "AI:" prefix if it exists
            if knowledge_content.startswith("AI:"):
                knowledge_content = knowledge_content[3:].strip()
            
        # Filter out content starting with "AI Synthesis:" directly (keep this as additional safety)
        if knowledge_content.startswith("AI Synthesis:"):
            logger.info(f"Filtered out AI Synthesis content")
            return ""
        
        # Filter out AI Synthesis sections from combined content
        if "\n---\n" in knowledge_content:
            sections = knowledge_content.split("\n---\n")
            filtered_sections = [section for section in sections if not section.strip().startswith("AI Synthesis:")]
            
            if len(filtered_sections) < len(sections):
                logger.info(f"Filtered out {len(sections) - len(filtered_sections)} AI Synthesis sections")
            
            knowledge_content = "\n---\n".join(filtered_sections)
        
        # Final whitespace normalization
        knowledge_content = re.sub(r'\n{3,}', '\n\n', knowledge_content)
        knowledge_content = re.sub(r' +', ' ', knowledge_content)
        knowledge_content = knowledge_content.strip()
        #logger.info(f"KNOWLEDGE content after extract knowledge content: {knowledge_content}")
        return knowledge_content
    
    
    def _clean_knowledge_content(self, knowledge_content: str) -> str:
        """Clean the knowledge content."""
        # Input validation
        if not knowledge_content or not isinstance(knowledge_content, str):
            return ""
        
        # Remove excessive newlines and spaces
        knowledge_content = re.sub(r'\n{3,}', '\n\n', knowledge_content)
        knowledge_content = re.sub(r' +', ' ', knowledge_content)
        knowledge_content = knowledge_content.strip()
        return knowledge_content
    
    def _process_profiling_skills(self, profiling_skills: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the fetched knowledge."""
        # Extract and combine all knowledge entries
        knowledge_context = []
        
        for query, knowledge in profiling_skills.items():
            # Process knowledge using the common utility function
            processed_content = self._extract_knowledge_content(knowledge)
            if processed_content:
                knowledge_context.append(processed_content)

        # Clean up the knowledge context using the helper function
        cleaned_context = []
        for entry in knowledge_context:
            cleaned = self._clean_knowledge_content(entry)
            if cleaned:
                cleaned_context.append(cleaned)

        # Combine all knowledge into a single context
        full_knowledge = "\n\n".join(cleaned_context) if cleaned_context else "No knowledge available."

        #logger.info(f"Raw Profiling knowledge: {full_knowledge}")
        
        return {
            "knowledge_context": full_knowledge,
            "metadata": {
                "source_queries": list(profiling_skills.keys()),
                "entry_count": len(cleaned_context),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _process_communication_skills(self, communication_skills: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the fetched communication knowledge."""
        # Extract and combine all knowledge entries
        knowledge_context = []
        
        for query, knowledge in communication_skills.items():
            # Process knowledge using the common utility function
            processed_content = self._extract_knowledge_content(knowledge)
            if processed_content:
                knowledge_context.append(processed_content)

        # Clean up the knowledge context using the helper function
        cleaned_context = []
        for entry in knowledge_context:
            cleaned = self._clean_knowledge_content(entry)
            if cleaned:
                cleaned_context.append(cleaned)

        # Combine all knowledge into a single context
        full_knowledge = "\n\n".join(cleaned_context) if cleaned_context else "No knowledge available."
        
        logger.info(f"Raw Communication knowledge: {full_knowledge}")
        
        return {
            "knowledge_context": full_knowledge,
            "metadata": {
                "source_queries": list(communication_skills.keys()),
                "entry_count": len(cleaned_context),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _process_business_objectives_knowledge(self, business_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the fetched knowledge."""
        # Extract and combine all knowledge entries
        knowledge_context = []
        
        for query, knowledge in business_objectives.items():
            # Process knowledge using the common utility function
            processed_content = self._extract_knowledge_content(knowledge)
            if processed_content:
                knowledge_context.append(processed_content)
        #logger.info(f"Business objectives knowledge context: {knowledge_context}")
        
        # Clean up the knowledge context using the helper function
        cleaned_context = []
        for entry in knowledge_context:
            cleaned = self._clean_knowledge_content(entry)
            if cleaned:
                cleaned_context.append(cleaned)

        # Combine all knowledge into a single context - use a single newline for better readability
        full_knowledge = "\n\n".join(cleaned_context) if cleaned_context else "No knowledge available."
        
        # One final cleanup pass using the helper function
        full_knowledge = self._clean_knowledge_content(full_knowledge)
        
        logger.info(f"Raw Business objectives knowledge: {full_knowledge}")
        
        return {
            "knowledge_context": full_knowledge,
            "metadata": {
                "source_queries": list(business_objectives.keys()),
                "entry_count": len(cleaned_context),
                "timestamp": datetime.now().isoformat()
            }
        }

    async def process_incoming_message(self, message: str, conversation_context: str, user_id: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming message with CoT flow.
        This function is the main function that processes the incoming message.
        It will get or build user profile, search analysis knowledge, propose an action plan, and execute the action plan.  
        """
        logger.info(f"Processing message from user {user_id}")
        try:
            # Ensure profiling skills and communication skills are loaded
            if "knowledge_context" in self.profiling_skills and self.profiling_skills["knowledge_context"] == "Loading...":
                logger.info("Profiling skills not loaded yet, loading now...")
                self.profiling_skills = await self._load_profiling_skills()
            
            # Ensure communication skills are loaded
            if "knowledge_context" in self.communication_skills and self.communication_skills["knowledge_context"] == "Loading...":
                logger.info("Communication skills not loaded yet, loading now...")
                self.communication_skills = await self._load_communication_skills()
            
            # Emit initial analysis event
            if thread_id:
                await emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": "Starting analysis",
                    "complete": False,
                    "status": "starting"
                })

            logger.info(f"Now I am processing message from user {user_id}")
            user_profile = await self._get_or_build_user_profile(user_id, message, conversation_context)
            
            # Emit user profile analysis event
            if thread_id:
                profile_data = serialize_model(user_profile)
                logger.info(f"Emitting profile_complete event with data: {json.dumps(profile_data, indent=2)}")
                await emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": "User profile analysis complete",
                    "complete": False,
                    "status": "profile_complete",
                    "user_profile": profile_data
                })
            
            logger.info("I've Built User Profile:")
            logger.info(f"Classification: {user_profile.classification}")
            logger.info(f"Portrait: {user_profile.other_aspects.get('portrait_paragraph', 'Not available')}")
            logger.info(f"Why I classified like this: {user_profile.other_aspects.get('classification_criteria', 'Not available')}")
            logger.info(f"I propose these queries to analyze user: {user_profile.analysis_queries}")
            
            # Emit knowledge search event
            if thread_id:
                await emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": "Searching for analysis knowledge...",
                    "complete": False,
                    "status": "searching_knowledge"
                })
            
            logger.info(f"Now I am searching knowledge follow the queries ...")
            analysis_knowledge = await self._search_analysis_knowledge(user_profile.classification, user_profile, message)
            
            # Emit knowledge found event
            if thread_id:
                await emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": "Analysis knowledge found.Analyzing it and proposing an action plan.",
                    "complete": False,
                    "status": "knowledge_found",
                    "analysis_knowledge": analysis_knowledge
                })
            
            logger.info(f"Now I am proposing an action plan for user {user_id} ...")
            action_plan = await self._decide_action_plan_with_llm(user_profile, analysis_knowledge)
            user_profile.action_plan = action_plan
            user_profile.analysis_summary = action_plan.get("analysis_summary", "Analysis completed.")
            
            # Emit action plan event
            if thread_id:
                await emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": "Action plan created",
                    "complete": False,
                    "status": "action_plan_created",
                    "action_plan": action_plan
                })
            
            # Log detailed analysis results
            logger.info("Here is the analysis results:")
            analysis_summary = action_plan.get("analysis_summary", {})
            logger.info(f"Key Findings: {analysis_summary.get('key_findings', [])}")
            logger.info(f"User Needs: {analysis_summary.get('user_needs', [])}")
            logger.info(f"Analysis Narrative: {analysis_summary.get('narrative', 'No narrative available')}")
            
            # Log action plan details
            next_actions = action_plan.get("next_actions", [])
            logger.info("Here is the action plan:")
            for i, action in enumerate(next_actions, 1):
                logger.info(f"Action {i}:")
                logger.info(f"  - What: {action.get('action', 'No action specified')}")
                logger.info(f"  - Priority: {action.get('priority', 'Not specified')}")
                logger.info(f"  - Why: {action.get('reasoning', 'No reasoning provided')}")
                logger.info(f"  - Expected: {action.get('expected_outcome', 'No expected outcome specified')}")
            
            logger.info(f"Now I am executing the action plan for user {user_id} ...")
            response = await self._execute_action_plan(user_profile, message, conversation_context)
            logger.info(f"Now I have response: {response}")
            self.user_profiles[user_id] = user_profile
            logger.info(f"I have updated user profile!")

            

            if isinstance(response, str):
                response = {"status": "success", "message": response, "user_profile": serialize_model(user_profile)}
            else:
                response.setdefault("status", "success")
                response.setdefault("user_profile", serialize_model(user_profile))
            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            # Emit error event
            if thread_id:
                await emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": str(e),
                    "complete": True,
                    "status": "error"
                })
            return {"status": "error", "message": f"Error: {str(e)}"}

    async def _get_or_build_user_profile(self, user_id: str, message: str, conversation_context: str) -> UserProfileModel:
        """Get cached profile or build new one."""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            if (datetime.now() - profile.timestamp).total_seconds() < self.user_profiles.ttl:
                logger.info(f"Using cached profile for user {user_id}")
                return profile
        return await self._build_user_portrait_with_llm(user_id, message,conversation_context=conversation_context)

    async def _build_user_portrait_with_llm(self, user_id: str, message: str, conversation_context: str) -> UserProfileModel:
        """Build user profile using LLM."""
        
        # Get the knowledge context
        knowledge_context = self.profiling_skills.get("knowledge_context", "")
        business_objectives = self.comprehensive_business_objectives.get("knowledge_context", "")
        
        # Add temporal context for resolving relative time references
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        
        # Create a temporal context string
        temporal_context = f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."
        
        # Check if message contains temporal references that need resolution
        temporal_keywords = [
            "today", "tomorrow", "yesterday", 
            "next week", "last week", "this week",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "morning", "afternoon", "evening", "night",
            "ngày mai", "hôm nay", "hôm qua", "tuần tới", "tuần sau", "tuần trước", 
            "thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"
        ]
        
        needs_temporal_resolution = any(keyword in message.lower() for keyword in temporal_keywords)
        
        # Add temporal resolution instruction if needed
        temporal_resolution_instruction = ""
        if needs_temporal_resolution:
            temporal_resolution_instruction = """
            "temporal_references": {
                "original_mentions": ["array of temporal expressions found in message"],
                "resolved_dates": ["array of resolved absolute dates in YYYY-MM-DD format"],
                "resolved_times": ["array of resolved times in HH:MM format if applicable"]
            },
            """
            logger.info(f"Detected temporal references in message. Adding temporal resolution instructions.")
        
        # Log conversation context length to help with debugging
        conversation_lines = conversation_context.count('\n') if conversation_context else 0
        
        prompt = f"""Build a comprehensive user profile by THOROUGHLY ANALYZING the ENTIRE conversation history:

                ===== CRITICAL LANGUAGE INSTRUCTION =====
                YOU MUST RESPOND ENTIRELY IN THE SAME LANGUAGE AS THE USER'S MESSAGE:
                - If the user wrote in Vietnamese → RESPOND 100% IN VIETNAMESE 
                - If the user wrote in English → RESPOND 100% IN ENGLISH
                - DO NOT mix languages in your response
                - EVERY part of your JSON output must be in the user's language
                - This includes ALL fields, descriptions, queries, and analysis
                ============================================

                KNOWLEDGE: {knowledge_context}
                CONVERSATION HISTORY: {conversation_context}
                CURRENT MESSAGE: {message}
                TIME CONTEXT: {temporal_context}
                BUSINESS OBJECTIVES: {business_objectives}

                DEEP CONVERSATION ANALYSIS INSTRUCTIONS:
                1. THOROUGHLY EXAMINE the COMPLETE conversation history - NOT just the current message
                2. Identify patterns, topics, and key information across ALL messages
                3. Track changes in user's tone, needs, and requirements throughout the conversation
                4. Look for implicit information and context clues across the ENTIRE conversation
                5. Pay special attention to consistency/inconsistency in user statements over time
                6. Consider how earlier messages provide context for later statements
                7. Look for previous questions that may not have been fully answered

                Analyze the conversation using classification techniques from the knowledge context.
                
                CONVERSATION STAGE AWARENESS:
                Identify the current stage (first contact, information gathering, problem identification, etc.) and adapt your approach accordingly. Consider whether the conversation is progressing naturally or needs redirection.

                KEY FOCUS AREAS:
                • Apply classification techniques from knowledge context strictly. Identify if the user profile is completed.
                • Identify behavioral patterns and hidden needs beyond stated requirements
                • Determine information gaps and plan appropriate next steps
                • Analyze alignment with business objectives
                • Identify recurring themes or topics throughout the conversation
                • Track any evolution in the user's needs or priorities across messages
                {f"• Resolve temporal references (like 'Sunday', 'tomorrow', 'next week') to specific dates" if needs_temporal_resolution else ""}

                BUSINESS_GOAL SELECTION (CRITICAL):
                1. ANALYZE EACH BUSINESS OBJECTIVE CAREFULLY To UNDERSTAND when you are allowed to use each objective.
                2. SELECT the best business objective that is allowed to use against the user portrait.
                3. JUSTIFY YOUR CHOICE:
                   - Explain why the selected objective is allowed to use now
                   - Show how it addresses user needs
                   - Bold the final business_goal

                GENERATE 3-5 ANALYSIS_QUERIES:
                [OBJECTIVE-SPECIFIC QUERIES - generate your own based on context]
                - Generate a query about the definition of the selected business objective
                - Generate a query about implementation strategies for the selected objective

                [USER-SPECIFIC QUERIES - generate your own based on context]
                - Generate a query to analyze the user with their identified classification
                - Generate a query about solutions for the specific user need/problem identified
                - Generate a query about appropriate reactions with the user's behavior patterns
                - Optionally add one more context-specific query if needed

                RESPOND WITH JSON (following the language instruction at the top of this prompt):
                {{
                    "classification": "string (from knowledge context)",
                    "skills": ["identified skills/capabilities"],
                    "requirements": ["specific needs/requirements"],
                    "analysis_queries": ["3-5 precise queries as instructed above"],
                    "business_goal": "**SELECTED OBJECTIVE: clearly formatted SMART goal statement**",
                    "other_aspects": {{
                        "conversation_stage": "identified stage of the conversation",
                        "behavioral_patterns": "observed behavior description",
                        "hidden_needs": "underlying needs analysis",
                        "required_info": ["missing information needed from the user"],
                        "next_steps": ["engagement steps appropriate for conversation stage"],
                        "classification_criteria": "classification rationale",
                        "conversation_history_insights": "Key insights extracted from analyzing the COMPLETE conversation history",
                        "business_objective_justification": "DETAILED explanation of why this business objective is most appropriate"
                        {temporal_resolution_instruction if needs_temporal_resolution else ""}
                    }},
                    "portrait_paragraph": "Comprehensive profile including classification, needs, behavior, and business objective alignment, incorporating insights from the ENTIRE conversation history."
                }}
                """
        
        try:
            response = await LLM.ainvoke(prompt)
            content = response.content.strip()
            logger.info(f"RAW PORTRAIT RESPONSE: {content}")
            
            # Try to extract JSON if there's any surrounding text
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group(0)
            try:
                llm_response = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                logger.error(f"Raw content: {content}")
                # Create a default response
                llm_response = {
                    "classification": "unknown",
                    "skills": [],
                    "requirements": [],
                    "analysis_queries": [],
                    "business_goal": "Keep user stayed engaged",
                    "other_aspects": {
                        "error": "Failed to parse LLM response",
                        "raw_content": content[:100] + "..." if len(content) > 100 else content
                    },
                    "portrait_paragraph": "Unable to generate portrait at this time."
                }
            
            # Extract portrait_paragraph from the response
            portrait_paragraph = llm_response.get("portrait_paragraph", "")
            
            # Create user profile with all fields
            user_profile = UserProfileModel(
                user_id=user_id,
                skills=llm_response.get("skills", []),
                requirements=llm_response.get("requirements", []),
                classification=llm_response.get("classification", "unknown"),
                analysis_queries=llm_response.get("analysis_queries", []),
                business_goal=llm_response.get("business_goal", ""),
                other_aspects={
                    **llm_response.get("other_aspects", {}),
                    "portrait_paragraph": portrait_paragraph
                }
            )        
            # If we resolved temporal references, log them
            if needs_temporal_resolution and "temporal_references" in user_profile.other_aspects:
                temporal_refs = user_profile.other_aspects.get("temporal_references", {})
                logger.info(f"Resolved temporal references: {json.dumps(temporal_refs, indent=2)}")
            
            self.user_profiles[user_id] = user_profile
            return user_profile
        except Exception as e:
            logger.error(f"Error building user portrait: {str(e)}")
            return UserProfileModel(
                user_id=user_id,
                business_goal_query="How to identify business objectives for error case",
                other_aspects={
                    "error": str(e),
                    "portrait_paragraph": "Error occurred while generating portrait."
                }
            )

    async def _search_analysis_knowledge(self, classification: str, user_profile: Optional[UserProfileModel], message: str) -> Dict[str, Any]:
        """Search for analysis methods based on user profile's analysis queries."""
        if not user_profile or not user_profile.analysis_queries:
            logger.warning("No analysis queries available for user profile")
            return {
                "analysis_methods": [],
                "focus_areas": [],
                "additional_context_needed": [],
                "rationale": "No analysis queries available"
            }

        # Fetch knowledge for each analysis query
        knowledge_entries = []
        logger.info(f"Searching for analysis knowledge: {user_profile.analysis_queries} using graph_version_id: {self.graph_version_id}")
        for query in user_profile.analysis_queries:
            try:
                # Use the query tool to fetch specific knowledge
                knowledge = await query_knowledge_from_graph(query, self.graph_version_id, include_categories=["ai_synthesis"])
                if knowledge:
                    # Handle different types of knowledge data
                    if isinstance(knowledge, dict) and "status" in knowledge and knowledge["status"] == "error":
                        logger.warning(f"Error in knowledge for query '{query}': {knowledge.get('message', 'Unknown error')}")
                        continue
                    
                    # Process the knowledge content based on its type
                    knowledge_content = self._extract_knowledge_content(knowledge)
                    
                    knowledge_entries.append({
                        "query": query,
                        "knowledge": knowledge_content
                    })
            except Exception as e:
                logger.error(f"Error fetching knowledge for query '{query}': {str(e)}")

        # Combine all knowledge entries
        combined_knowledge = "\n\n".join([
            f"Query: {entry['query']}\nKnowledge: {entry['knowledge']}"
            for entry in knowledge_entries
        ])

        return {
            "analysis_methods": [entry["query"] for entry in knowledge_entries],
            "knowledge_context": combined_knowledge,
            "query_count": len(knowledge_entries),
            "rationale": f"Analyzed {len(knowledge_entries)} queries for user profile"
        }

    
    
    def _format_analysis_summary(self, analysis_summary: Dict[str, Any]) -> str:
        """Convert analysis summary JSON to readable text format."""
        if not analysis_summary:
            return "No analysis available."
        
        key_findings = analysis_summary.get('key_findings', [])
        user_needs = analysis_summary.get('user_needs', [])
        potential_challenges = analysis_summary.get('potential_challenges', [])
        narrative = analysis_summary.get('narrative', '')
        
        formatted = "ANALYSIS SUMMARY:\n"
        
        if key_findings:
            formatted += "Key Findings:\n"
            for i, finding in enumerate(key_findings, 1):
                formatted += f"  {i}. {finding}\n"
        
        if user_needs:
            formatted += "User Needs:\n"
            for i, need in enumerate(user_needs, 1):
                formatted += f"  {i}. {need}\n"
        
        if potential_challenges:
            formatted += "Potential Challenges:\n"
            for i, challenge in enumerate(potential_challenges, 1):
                formatted += f"  {i}. {challenge}\n"
        
        if narrative:
            formatted += f"Overall Assessment: {narrative}"
        
        return formatted.strip()
    
    def _format_action_plan(self, next_actions: List[Dict[str, Any]]) -> str:
        """Convert action plan JSON to readable text format."""
        if not next_actions:
            return "No actions planned."
        
        formatted = "ACTION PLAN (Execute in this order):\n"
        
        for i, action in enumerate(next_actions, 1):
            action_text = action.get('action', 'Unknown action')
            priority = action.get('priority', 'normal')
            reasoning = action.get('reasoning', '')
            expected_outcome = action.get('expected_outcome', '')
            
            formatted += f"\n{i}. [{priority.upper()} PRIORITY] {action_text}\n"
            if reasoning:
                formatted += f"   Why: {reasoning}\n"
            if expected_outcome:
                formatted += f"   Expected Result: {expected_outcome}\n"
        
        return formatted.strip()
    
    def _format_important_notes(self, important_notes: Dict[str, Any]) -> str:
        """Convert important notes JSON to readable text format."""
        if not important_notes:
            return "No special notes."
        
        immediate_concerns = important_notes.get('immediate_concerns', [])
        long_term_considerations = important_notes.get('long_term_considerations', [])
        additional_context_needed = important_notes.get('additional_context_needed', [])
        narrative = important_notes.get('narrative', '')
        
        formatted = "IMPORTANT CONSIDERATIONS:\n"
        
        if immediate_concerns:
            formatted += "Immediate Concerns:\n"
            for concern in immediate_concerns:
                formatted += f"  • {concern}\n"
        
        if long_term_considerations:
            formatted += "Long-term Considerations:\n"
            for consideration in long_term_considerations:
                formatted += f"  • {consideration}\n"
        
        if additional_context_needed:
            formatted += "Additional Context Needed:\n"
            for context in additional_context_needed:
                formatted += f"  • {context}\n"
        
        if narrative:
            formatted += f"\nSummary: {narrative}"
        
        return formatted.strip()

    async def _synthesize_knowledge_for_query(self, query: str, raw_knowledge: str) -> str:
        """Synthesize raw knowledge to specifically answer the given query."""
        if not raw_knowledge or not query:
            return "No relevant knowledge available."
        
        synthesis_prompt = f"""You are a knowledge synthesizer. Your task is to analyze the provided knowledge and create a focused, actionable summary that specifically answers the given query.

            QUERY TO ANSWER: {query}

            RAW KNOWLEDGE TO SYNTHESIZE:
            {raw_knowledge}

            SYNTHESIS REQUIREMENTS:
            1. Focus ONLY on information that directly answers the query
            2. Organize the information in a clear, logical structure
            3. Remove redundant or irrelevant information
            4. Preserve specific examples, techniques, and actionable advice
            5. Maintain the original language (Vietnamese/English) from the source
            6. Keep cultural context and specific terminology intact
            7. Present the synthesized knowledge as practical guidance

            IMPORTANT LANGUAGE PRESERVATION:
            - If the knowledge is in Vietnamese, respond COMPLETELY in Vietnamese
            - If the knowledge is in English, respond COMPLETELY in English
            - DO NOT translate or mix languages
            - Preserve original expressions, pronouns, and cultural references exactly

            FORMAT: Provide a concise, well-structured synthesis that directly addresses the query. Focus on actionable insights and specific guidance."""

        try:
            response = await LLM.ainvoke(synthesis_prompt)
            synthesized = response.content.strip()
            
            # Log the synthesis for debugging
            logger.info(f"Synthesized knowledge for query '{query}': {len(synthesized)} chars")
            
            return synthesized
        except Exception as e:
            logger.error(f"Error synthesizing knowledge for query '{query}': {str(e)}")
            # Fallback to original knowledge if synthesis fails
            return raw_knowledge

    async def _user_story(self, user_profile: UserProfileModel) -> str:
        """Transform user profile into a compact narrative format."""
        # Extract key elements
        classification = user_profile.classification
        skills = ', '.join(user_profile.skills) if user_profile.skills else 'None'
        requirements = ', '.join(user_profile.requirements) if user_profile.requirements else 'None'
        
        # Extract other aspects
        portrait = user_profile.other_aspects.get("portrait_paragraph", "")
        behavioral = user_profile.other_aspects.get("behavioral_patterns", "")
        hidden_needs = user_profile.other_aspects.get("hidden_needs", "")
        criteria = user_profile.other_aspects.get("classification_criteria", "")
        required_info = ', '.join(user_profile.other_aspects.get("required_info", [])) or "None"
        next_steps = ', '.join(user_profile.other_aspects.get("next_steps", [])) or "None"
        business_goal = user_profile.business_goal
        
        # Extract conversation history insights if available
        conversation_insights = user_profile.other_aspects.get("conversation_history_insights", "")
        conversation_stage = user_profile.other_aspects.get("conversation_stage", "")
        
        # Construct compact narrative with conversation history insights
        return f"""This user is {classification} who {criteria}. {portrait} Their behavior shows {behavioral}, 
        suggesting {hidden_needs} as underlying needs. With {skills} skills, they're looking to {requirements}. 
        The conversation is currently in the {conversation_stage} stage. {conversation_insights if conversation_insights else ""}
        Missing information we still need: {required_info}. Next best steps: {next_steps}. AI business goal: {business_goal}"""

    async def _decide_action_plan_with_llm(self, user_profile: UserProfileModel, analysis_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Create action plan using LLM based on user profile and analysis knowledge."""
        

        # Get the user story
        user_story = await self._user_story(user_profile)
        
        # Extract temporal references if available
        temporal_info = ""
        if "temporal_references" in user_profile.other_aspects:
            temporal_references = user_profile.other_aspects.get("temporal_references", {})
            original_mentions = temporal_references.get("original_mentions", [])
            resolved_dates = temporal_references.get("resolved_dates", [])
            resolved_times = temporal_references.get("resolved_times", [])
            
            if original_mentions and resolved_dates:
                temporal_info = "TEMPORAL REFERENCES:\n"
                for i, mention in enumerate(original_mentions):
                    resolved_date = resolved_dates[i] if i < len(resolved_dates) else "unknown"
                    resolved_time = resolved_times[i] if resolved_times and i < len(resolved_times) else ""
                    
                    if resolved_time:
                        temporal_info += f"When user mentioned '{mention}', they meant: {resolved_date} at {resolved_time}\n"
                    else:
                        temporal_info += f"When user mentioned '{mention}', they meant: {resolved_date}\n"
                
                logger.info(f"Including temporal references in action planning: {json.dumps(temporal_references, indent=2)}")
        
        # Add current date/time information for temporal context
        from datetime import datetime
        import pytz
        
        # Use Vietnam timezone by default
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        
        #logger.info(f"ANALYSIS KNOWLEDGE: {analysis_knowledge.get('knowledge_context', '')}")
        # Create a temporal context string
        temporal_context = f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."
        
        prompt = f"""
        Create a comprehensive action plan based on the user's profile and the analysis:

        USER PROFILE NARRATIVE:
        {user_story}

        ANALYSIS KNOWLEDGE:
        {analysis_knowledge.get('knowledge_context', '')}

        ANALYSIS QUERIES USED:
        {', '.join(analysis_knowledge.get('analysis_methods', []))}

        TIME CONTEXT:
        {temporal_context}
        {temporal_info}

        LANGUAGE REQUIREMENTS:
        1. STRICTLY use the SAME LANGUAGE as the user's message
        2. If the message is in Vietnamese, respond COMPLETELY in Vietnamese
        3. If the message is in English, respond COMPLETELY in English
        4. DO NOT mix languages in your response
        5. Maintain consistent language throughout all sections
        6. Use natural expressions and terminology from the user's language
        7. Follow cultural communication patterns of the user's language

        YOUR GOAL:
        The business goal identified in the USER PROFILE NARRATIVE is YOUR goal for this interaction. Every action you plan must contribute to achieving this goal.

        Your task has two parts:
        
        PART 1: CREATE ACTION PLAN
        First, create a detailed action plan that addresses the user's needs and requirements while ensuring it contributes to achieving YOUR goal identified in the USER PROFILE NARRATIVE.
        
        IMPORTANT: You MUST incorporate the "Next best steps" identified in the USER PROFILE NARRATIVE into your action plan. These steps were specifically identified for this user and should be prioritized in your planning.
        
        When creating the action plan:
        1. First, review the "Next best steps" from the user profile
        2. Consider how these steps contribute to achieving YOUR goal identified in the USER PROFILE NARRATIVE
        3. Convert these steps into specific, actionable items
        4. Maintain their priority and order as identified in the profile
        5. Add any additional actions needed to achieve YOUR goal
        6. Ensure all actions have clear reasoning and expected outcomes
        7. Write all narrative sections in the same language style as the user's message
        8. If temporal references were detected, ensure the action plan explicitly addresses timing requirements

        PART 2: IDENTIFY KNOWLEDGE GAPS
        After creating the action plan, identify 3-5 specific knowledge queries we should run to get additional information needed to execute the plan effectively.
        
        CRITICAL LANGUAGE REQUIREMENT FOR KNOWLEDGE QUERIES:
        - ALL knowledge queries MUST be written in the EXACT SAME LANGUAGE as the user's message
        - If user wrote in Vietnamese → ALL queries must be in Vietnamese
        - If user wrote in English → ALL queries must be in English
        - DO NOT translate or mix languages in the queries
        - Use natural expressions and terminology from the user's language for the queries
        
        When identifying knowledge gaps:
        1. Review each action in your plan
        2. For each action, identify what additional knowledge would help finding how to execute.
        3. Create specific queries to gather this knowledge by adding "How to" or "What is" to the action
        4. ENSURE ALL QUERIES ARE IN THE SAME LANGUAGE AS THE USER'S MESSAGE
        5. Use culturally appropriate query formulations for the user's language

        IMPORTANT: You MUST respond with a valid JSON object only. Do not include any other text or explanation.
        The JSON must follow this exact structure:
        {{
            "next_actions": [
                {{
                    "action": "string (specific action to take)",
                    "priority": "high|medium|low",
                    "reasoning": "string (why this action is needed)",
                    "expected_outcome": "string (what this action should achieve)",
                    "source": "string (whether this action came from user profile next steps or was added based on current context)",
                    "business_goal_alignment": "string (how this action helps achieve YOUR goal identified in the user profile)"
                }}
            ],
            "knowledge_queries": [
                {{
                    "query": "string (specific query to get more information)",
                    "purpose": "string (which action this query supports)",
                    "expected_knowledge": "string (what information we expect to get)"
                }}
            ],
            "analysis_summary": {{
                "key_findings": ["array of key findings"],
                "user_needs": ["array of identified needs"],
                "potential_challenges": ["array of potential challenges"],
                "narrative": "A detailed paragraph summarizing the analysis, written in the user's language style"
            }},
            "important_notes": {{
                "immediate_concerns": ["array of immediate concerns"],
                "long_term_considerations": ["array of long-term considerations"],
                "additional_context_needed": ["array of additional information needed"],
                "narrative": "A paragraph explaining the important considerations, written in the user's language style"
            }}
        }}
        """
        
        try:
            response = await LLM.ainvoke(prompt)
            content = response.content.strip()
            #logger.info(f"RAW ACTION PLAN RESPONSE: {content}")
            
            # Try to extract JSON if there's any surrounding text
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group(0)
            
            try:
                action_plan = json.loads(content)
                # Add metadata
                action_plan["metadata"] = {
                    "timestamp": datetime.now().isoformat(),
                    "analysis_query_count": len(analysis_knowledge.get('analysis_methods', [])),
                    "user_classification": user_profile.classification
                }
                
                # Add temporal information to metadata if available
                if "temporal_references" in user_profile.other_aspects:
                    action_plan["metadata"]["temporal_references"] = user_profile.other_aspects.get("temporal_references", {})
                
                return action_plan
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM action plan: {str(e)}")
                logger.error(f"Raw content: {content}")
                return {
                    "next_actions": [],
                    "knowledge_queries": [],
                    "analysis_summary": {
                        "key_findings": [],
                        "user_needs": [],
                        "potential_challenges": [],
                        "narrative": "Unable to generate analysis summary at this time."
                    },
                    "important_notes": {
                        "immediate_concerns": [],
                        "long_term_considerations": [],
                        "additional_context_needed": [],
                        "narrative": "Unable to generate important notes at this time."
                    },
                    "metadata": {
                        "error": "Failed to parse action plan",
                        "timestamp": datetime.now().isoformat()
                    }
                }
        except Exception as e:
            logger.error(f"Error in action plan: {str(e)}")
            return {
                "next_actions": [],
                "knowledge_queries": [],
                "analysis_summary": {
                    "key_findings": [],
                    "user_needs": [],
                    "potential_challenges": [],
                    "narrative": "Error occurred while generating analysis summary."
                },
                "important_notes": {
                    "immediate_concerns": [],
                    "long_term_considerations": [],
                    "additional_context_needed": [],
                    "narrative": "Error occurred while generating important notes."
                },
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }

    async def _execute_action_plan(self, user_profile: UserProfileModel, message: str, conversation_context: str = "") -> Dict[str, Any]:
        """Generate user-friendly response based on the action plan."""
        logger.info("Executing action plan")
        
        if not user_profile.action_plan:
            return {
                "status": "error",
                "message": "No action plan available"
            }

        # Get the action plan details
        action_plan = user_profile.action_plan
        analysis_summary = action_plan.get("analysis_summary", {})
        important_notes = action_plan.get("important_notes", {})
        next_actions = action_plan.get("next_actions", [])
        knowledge_queries = action_plan.get("knowledge_queries", [])
        
        logger.info(f"Analysis summary: {analysis_summary}")
        logger.info(f"Important notes: {important_notes}")
        logger.info(f"Next actions: {next_actions}")
        logger.info(f"Knowledge queries: {knowledge_queries}")

        # Extract resolved temporal references if available
        temporal_references = {}
        if "temporal_references" in user_profile.other_aspects:
            temporal_references = user_profile.other_aspects.get("temporal_references", {})
            logger.info(f"Including temporal references in response generation: {json.dumps(temporal_references, indent=2)}")
        
        # Fetch additional knowledge if queries are provided
        additional_knowledge = ""
        if knowledge_queries:
            logger.info(f"Fetching and synthesizing additional knowledge for {len(knowledge_queries)} queries")
            try:
                knowledge_results = []
                for query_info in knowledge_queries[:3]:  # Limit to top 3 queries
                    try:
                        # Extract just the query string from the query info dictionary
                        query = query_info.get('query', '') if isinstance(query_info, dict) else query_info
                        if query:
                            # Step 1: Fetch raw knowledge from graph
                            knowledge = await query_knowledge_from_graph(query, self.graph_version_id, include_categories=["ai_synthesis"])
                            if knowledge:
                                # Step 2: Extract knowledge content based on its type
                                raw_knowledge_content = self._extract_knowledge_content(knowledge)
                                
                                # Step 3: Synthesize the knowledge to specifically answer the query
                                synthesized_knowledge = await self._synthesize_knowledge_for_query(query, raw_knowledge_content)
                                
                                knowledge_results.append(f"Query: {query}\nSynthesized Answer: {synthesized_knowledge}")
                    except Exception as e:
                        logger.error(f"Error processing knowledge for query '{query}': {str(e)}")
                
                additional_knowledge = "\n\n".join(knowledge_results)
                logger.info(f"Synthesized additional knowledge: {len(additional_knowledge)} chars")
            except Exception as e:
                logger.error(f"Error fetching additional knowledge: {str(e)}")

        # Include current date/time information for temporal context
        from datetime import datetime
        import pytz
        
        # Use Vietnam timezone by default (can be customized based on user preferences later)
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        
        # Create a temporal context string
        temporal_context = f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."
        
        # Format temporal references for inclusion in the prompt if available
        temporal_info = ""
        if temporal_references:
            original_mentions = temporal_references.get("original_mentions", [])
            resolved_dates = temporal_references.get("resolved_dates", [])
            resolved_times = temporal_references.get("resolved_times", [])
            
            temporal_info = "TEMPORAL REFERENCES:\n"
            if original_mentions and resolved_dates:
                for i, mention in enumerate(original_mentions):
                    resolved_date = resolved_dates[i] if i < len(resolved_dates) else "unknown"
                    resolved_time = resolved_times[i] if resolved_times and i < len(resolved_times) else ""
                    
                    if resolved_time:
                        temporal_info += f"When user mentioned '{mention}', they meant: {resolved_date} at {resolved_time}\n"
                    else:
                        temporal_info += f"When user mentioned '{mention}', they meant: {resolved_date}\n"
        
        # Get communication skills for use in prompt
        communication_knowledge = self.comprehensive_communication_skills.get("knowledge_context", "")
        
        #logger.info(f"Communication knowledge: {communication_knowledge}")
        #logger.info(f"Addtional info: {additional_knowledge}")
        #logger.info(f"ACTION PLAN: {json.dumps(next_actions, indent=2) if next_actions else "N/A"}")
        #logger.info(f"Conversation context: {conversation_context}")

        
        # Format structured data into readable text
        formatted_analysis = self._format_analysis_summary(analysis_summary)
        formatted_action_plan = self._format_action_plan(next_actions)
        formatted_important_notes = self._format_important_notes(important_notes)
        
        # Build prompt for LLM to generate response
        prompt = f"""Generate a response to the user that STRICTLY FOLLOWS the action plan and communication guidelines:

        CURRENT MESSAGE: {message}
        CONVERSATION: {conversation_context}
        TIME CONTEXT: {temporal_context}
        {temporal_info if temporal_info else ""}
        
        {formatted_analysis}
        
        {formatted_action_plan}
        
        {formatted_important_notes}
        
        USER CONTEXT: Classification={user_profile.classification}, Business Goal={user_profile.business_goal}
        ADDITIONAL INFO: {additional_knowledge}
        COMMUNICATION KNOWLEDGE: {communication_knowledge}

        PRIORITY ORDER (Follow in this exact sequence):
        1. Execute ACTION PLAN actions exactly as specified - nothing more, nothing less
        2. Apply COMMUNICATION STYLE from knowledge base for user classification: {user_profile.classification}
        3. Maintain user's language (Vietnamese/English) consistently throughout
        4. Follow FORMAT GUIDELINES for professional delivery

        STRICT EXECUTION REQUIREMENTS:
        • ONLY implement the actions listed in the ACTION PLAN
        • DO NOT introduce any external services, specialists, websites, or social media not mentioned in the plan
        • DO NOT mention or suggest Facebook, messengers, or any external contacts unless explicitly in the plan
        • Follow the exact priority order of the actions as they appear in the plan

        LANGUAGE REQUIREMENTS:
        • Detect user language: Vietnamese → respond 100% in Vietnamese, English → respond 100% in English
        • Use pronouns as specified in communication knowledge (e.g., "em"/"anh" for Vietnamese users)
        • Maintain cultural communication patterns and expressions (e.g., "kiểu như là í" in Vietnamese)
        • Keep original language tone and formality level

        COMMUNICATION STYLE SPECIFICS:
        • Use the EXACT COMMUNICATION from COMMUNICATION KNOWLEDGE that matches this user's classification ({user_profile.classification})
        • READ, UNDERSTAND then FOLLOW the INSTRUCTIONS in the COMMUNICATION KNOWLEDGE to generate the response
        • Avoid repetition by using varied expressions as instructed in the communication guidelines
        • Adapt your tone to build trust with this specific user classification

        CONFLICT RESOLUTION:
        If communication knowledge suggests actions NOT in ACTION PLAN:
        • Prioritize ACTION PLAN actions over communication suggestions
        • Apply communication style and tone WITHOUT adding new actions
        • Use communication knowledge for HOW to say things, not WHAT to say

        FORMAT GUIDELINES:
        • Be concise and professional
        • Match the user's language and cultural context
        • Skip formulaic greetings and focus on delivering the actions
        • Adapt to the specific user needs indicated in the analysis
        • Follow communication patterns appropriate for the user's classification

        CRITICAL: Your response must implement the actions from the ACTION PLAN while applying the communication style specified above. Prioritize both action accuracy AND communication style."""
        
        # Check prompt complexity and use chunking if needed
        prompt_length = len(prompt)
        logger.info(f"Prompt length: {prompt_length} characters")
                
        logger.info(f"Prompt Before Taking Action: {prompt}")
        try:
            response = await LLM.ainvoke(prompt)
            return {
                "status": "success",
                "message": response.content.strip(),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_profile.user_id,
                    "knowledge_queries_used": knowledge_queries,
                    "additional_knowledge_found": bool(additional_knowledge),
                    "temporal_references": temporal_references if temporal_references else None
                }
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "status": "error",
                "message": "I apologize, but I encountered an error while processing your request. Please try again."
            }

async def process_llm_with_tools(
    user_message: str,
    conversation_history: List[Dict],
    state: Dict,
    graph_version_id: str,
    thread_id: Optional[str] = None
) -> AsyncGenerator[Union[str, Dict], None]:
    """Process user message with tools."""
    logger.info(f"Processing message for user {state.get('user_id', 'unknown')}")
    # Initialize conversation context with current message
    conversation_context = f"User: {user_message}\n"
    
    # Include previous messages for context (increased from 30 to 50 for more comprehensive history)
    if conversation_history:
        # Extract the messages excluding the current message
        recent_messages = []
        message_count = 0
        max_messages = 50  # Increased from 30 to 50 messages
        
        # Variables to track last message for deduplication
        last_role = None
        last_content = None
        
        for msg in reversed(conversation_history[:-1]):  # Skip the current message
            try:
                role = msg.get("role", "").lower() 
                content = msg.get("content", "")
                
                if role and content:
                    # Skip duplicate consecutive messages from the same role
                    if role == last_role and content == last_content:
                        logger.info(f"Skipping duplicate message from {role}")
                        continue
                            
                    # Format based on role with clear separation between messages
                    if role in ["assistant", "ai"]:
                        recent_messages.append(f"AI: {content.strip()}")
                        message_count += 1
                    elif role in ["user", "human"]:
                        recent_messages.append(f"User: {content.strip()}")
                        message_count += 1
                    # All other roles are now explicitly skipped
                    
                    # Update last message tracking for deduplication
                    last_role = role
                    last_content = content
                    
                if message_count >= max_messages:
                    # We've reached our limit, but add a note about truncation
                    if len(conversation_history) > max_messages + 1:  # +1 accounts for current message
                        recent_messages.append(f"[Note: Conversation history truncated. Total messages: {len(conversation_history)}]")
                    break
            except Exception as e:
                logger.warning(f"Error processing message in conversation history: {e}")
                continue
        
        # Add messages in chronological order with clear formatting
        if recent_messages:
            # Add a header to highlight the importance of the conversation history
            header = "==== CONVERSATION HISTORY (CHRONOLOGICAL ORDER) ====\n"
            # Reverse the list to get chronological order and join with double newlines for better separation
            conversation_history_text = "\n\n".join(reversed(recent_messages))
            # Add a separator between history and current message
            separator = "\n==== CURRENT INTERACTION ====\n"
            
            conversation_context = f"{header}{conversation_history_text}{separator}\nUser: {user_message}\n"
            
            logger.info(f"Added {len(recent_messages)} messages from conversation history")
        else:
            logger.warning("No usable messages found in conversation history")
    
    # Store the graph_version_id in state for future reference
    state['graph_version_id'] = graph_version_id
    
    try:
        # Get or create a CoTProcessor instance
        if 'cot_processor' not in state:
            cot_processor = CoTProcessor()
            # Initialize properly with the provided graph_version_id
            await cot_processor.initialize(graph_version_id)
            state['cot_processor'] = cot_processor
        else:
            cot_processor = state['cot_processor']
            # Update the graph_version_id if it has changed
            if cot_processor.graph_version_id != graph_version_id:
                logger.info(f"Updating CoTProcessor graph_version_id from {cot_processor.graph_version_id} to {graph_version_id}")
                cot_processor.graph_version_id = graph_version_id
                
        # Process the message - events will be emitted from process_incoming_message
        response = await cot_processor.process_incoming_message(
            user_message, 
            conversation_context, 
            state.get('user_id', 'unknown'),
            thread_id
        )
        
        yield response
        state.setdefault("messages", []).append({"role": "assistant", "content": response.get("message", "")})
        state["prompt_str"] = response.get("message", "")
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in CoT processing: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        error_response = {"status": "error", "message": f"Error: {str(e)}", "traceback": error_traceback}
        yield error_response
    yield {"state": state}

async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool."""
    logger.info(f"Executing tool: {tool_name}")
    try:
        if tool_name == "knowledge_query":
            return await knowledge_query_helper(
                parameters.get("query", ""),
                parameters.get("context", ""),
                parameters.get("graph_version_id", "")
            )
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return {"status": "error", "message": str(e)}

async def knowledge_query_helper(query: str, context: str, graph_version_id: str) -> Dict[str, Any]:
    """Query knowledge base."""
    logger.info(f"Querying knowledge: {query} using graph_version_id: {graph_version_id}")
    try:
        # Create temporary processor to use its _extract_knowledge_content method
        processor = CoTProcessor()
        
        knowledge_data = await query_knowledge_from_graph(query, graph_version_id, include_categories=["ai_synthesis"])
        
        # Handle different return types from fetch_knowledge
        if isinstance(knowledge_data, dict) and "status" in knowledge_data and knowledge_data["status"] == "error":
            # If fetch_knowledge returned an error dictionary
            return knowledge_data
        
        # Process knowledge data for both primary types
        if isinstance(knowledge_data, list) and knowledge_data:
            # Extract first item's content using the utility function
            first_item = knowledge_data[0]
            processed_content = processor._extract_knowledge_content(first_item)
            result_data = {"raw": processed_content, "content": processed_content}
        else:
            # Process direct knowledge content
            processed_content = processor._extract_knowledge_content(knowledge_data)
            
            # Filter out vector data if present
            if processed_content and ("[" in processed_content and "]" in processed_content):
                if any(pattern in processed_content for pattern in ["-0.", "0.", "1.", "2.", "..."]):
                    # Filter out the vector data sections
                    lines = processed_content.split("\n")
                    filtered_lines = []
                    skip_until_next_entry = False
                    
                    for line in lines:
                        # If line contains vector data pattern, skip it
                        if ("[" in line and "]" in line and 
                            any(pattern in line for pattern in ["-0.", "0.", "1.", "2.", "..."])):
                            skip_until_next_entry = True
                            filtered_lines.append("Content unavailable (vector data)")
                        # Reset skip flag when reaching entry separator
                        elif "----" in line:
                            skip_until_next_entry = False
                            filtered_lines.append(line)
                        # Include line if not in skip mode
                        elif not skip_until_next_entry:
                            filtered_lines.append(line)
                    
                    # Replace knowledge data with filtered version
                    processed_content = "\n".join(filtered_lines)
                    logger.info("Filtered out vector data from knowledge response")
            
            # Try to parse as JSON, but use raw text if it fails
            try:
                result_data = json.loads(processed_content)
            except (json.JSONDecodeError, TypeError):
                result_data = {"raw": processed_content, "content": processed_content}
        
        return {
            "status": "success",
            "message": f"Knowledge queried for {query}",
            "data": result_data
        }
    except Exception as e:
        logger.error(f"Error fetching knowledge: {str(e)}")
        return {"status": "error", "message": f"Failed to fetch knowledge: {str(e)}"}
