"""
Prompt Builder Module
====================

Handles dynamic system prompt generation based on agent configuration and mode.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import language detection (shared with Ami)
try:
    from language import detect_language_with_llm, LanguageDetector
    LANGUAGE_DETECTION_AVAILABLE = True
    logger.info("Language detection imported successfully from language.py")
except Exception as e:
    logger.warning(f"Failed to import language detection from language.py: {e}")
    LANGUAGE_DETECTION_AVAILABLE = False


class PromptBuilder:
    """Builds dynamic system prompts for different agent modes and contexts"""
    
    def __init__(self):
        pass
    
    def build_dynamic_system_prompt(self, agent_config: Dict[str, Any], 
                                  user_request: str, agent_mode: str = "execute") -> str:
        """
        Build dynamic system prompt from agent configuration with mode support
        
        PRIORITY: Uses compiled system prompt if available, falls back to template building
        
        Args:
            agent_config: Loaded agent configuration
            user_request: User's request for context
            agent_mode: "collaborate" or "execute" mode
            
        Returns:
            Mode-specific system prompt string
        """
        try:
            # PRIORITY: Use compiled system prompt if available
            if (agent_config.get("compiled_system_prompt") and 
                agent_config.get("compilation_status") == "compiled"):
                
                logger.info(f"Using compiled system prompt for agent {agent_config.get('name')}")
                base_prompt = agent_config["compiled_system_prompt"]
                
                # Add mode-specific enhancements to compiled prompt
                if agent_mode.lower() == "collaborate":
                    return self._enhance_compiled_for_collaborate_mode(base_prompt, user_request, agent_config)
                else:
                    return self._enhance_compiled_for_execute_mode(base_prompt, user_request, agent_config)
            
            # FALLBACK: Use template building for uncompiled blueprints
            else:
                logger.warning(f"Using fallback prompt building for agent {agent_config.get('name')} - blueprint not compiled (status: {agent_config.get('compilation_status', 'unknown')})")
                if agent_mode.lower() == "collaborate":
                    return self._build_collaborate_prompt(agent_config, user_request)
                else:
                    return self._build_execute_prompt(agent_config, user_request)
                
        except Exception as e:
            logger.error(f"Failed to build dynamic system prompt: {e}")
            # Fallback to basic prompt
            return f"You are {agent_config.get('name', 'AI Agent')}, a specialized AI assistant. {agent_config.get('description', 'I help with various tasks.')} Use your available tools to provide helpful assistance."
    
    def _enhance_compiled_for_collaborate_mode(self, compiled_prompt: str, user_request: str, agent_config: Dict[str, Any]) -> str:
        """
        Enhance compiled system prompt for COLLABORATE mode
        
        Args:
            compiled_prompt: The fully compiled system prompt from blueprint
            user_request: User's request for context
            agent_config: Agent configuration
            
        Returns:
            Enhanced prompt for collaboration mode
        """
        collaborate_enhancement = f"""

# COLLABORATION MODE ACTIVE
You are now operating in COLLABORATE mode - focus on interactive discussion and guidance.

COLLABORATION APPROACH:
- Engage in thoughtful dialogue about the request
- Ask clarifying questions when needed
- Provide step-by-step guidance and explanations
- Share your reasoning process openly
- Encourage user participation in problem-solving
- Be patient and educational in your responses

CURRENT USER REQUEST: "{user_request}"

Remember: You have access to all your configured integrations, tools, and domain knowledge as specified above. Use them to provide comprehensive collaborative assistance.
"""
        return compiled_prompt + collaborate_enhancement
    
    def _enhance_compiled_for_execute_mode(self, compiled_prompt: str, user_request: str, agent_config: Dict[str, Any]) -> str:
        """
        Enhance compiled system prompt for EXECUTE mode
        
        Args:
            compiled_prompt: The fully compiled system prompt from blueprint
            user_request: User's request for context
            agent_config: Agent configuration
            
        Returns:
            Enhanced prompt for execution mode
        """
        execute_enhancement = f"""

# EXECUTION MODE ACTIVE
You are now operating in EXECUTE mode - focus on efficient task completion.

EXECUTION APPROACH:
⚡ EXECUTION PRINCIPLES:
- Focus on completing the requested task efficiently
- Use your configured tools and integrations directly when needed
- Provide thorough, actionable results
- Minimize unnecessary back-and-forth questions
- Be direct and solution-focused
- Deliver comprehensive outputs that address the full request
- Take initiative to gather needed information using your available tools

CURRENT USER REQUEST: "{user_request}"

CRITICAL REMINDERS:
- PRIORITY: If the user provides a Google Drive link (docs.google.com), ALWAYS use the appropriate read tool FIRST
- Use your configured integrations and credentials as specified in your integration configurations above
- Follow your workflow steps and business context as defined in your blueprint
- Apply your domain knowledge and tool usage instructions as configured

Remember: All your credentials, API keys, and integration details have been securely configured and are ready to use.
"""
        return compiled_prompt + execute_enhancement
    
    def _build_collaborate_prompt(self, agent_config: Dict[str, Any], user_request: str) -> str:
        """
        Build system prompt for COLLABORATE mode - interactive, discussion-focused
        
        Args:
            agent_config: Loaded agent configuration
            user_request: User's request for context
            
        Returns:
            Collaborate mode system prompt
        """
        try:
            system_prompt_data = agent_config.get("system_prompt", {})
            
            # Extract prompt components
            base_instruction = system_prompt_data.get("base_instruction", "")
            agent_type = system_prompt_data.get("agent_type", "general")
            language = system_prompt_data.get("language", "english")
            specialization = system_prompt_data.get("specialization", [])
            personality = system_prompt_data.get("personality", {})
            
            # Build collaborate-focused prompt
            collaborate_prompt = f"""You are {agent_config['name']}, operating in COLLABORATE mode.

{base_instruction}

AGENT PROFILE:
- Type: {agent_type.replace('_', ' ').title()} Agent
- Specialization: {', '.join(specialization) if specialization else 'General assistance'}
- Language: {language.title()}
- Personality: {personality.get('tone', 'professional')}, {personality.get('style', 'helpful')}, {personality.get('approach', 'solution-oriented')}

COLLABORATION APPROACH:
You are designed to work WITH the user in an interactive, collaborative manner:

🤝 COLLABORATION PRINCIPLES:
- Ask clarifying questions to better understand their needs
- Discuss options and alternatives before proceeding
- Explain your reasoning and approach
- Seek feedback and confirmation before taking major steps
- Be conversational and engage in back-and-forth dialogue
- Offer suggestions and recommendations, not just direct answers
- Help users think through problems step by step

AVAILABLE TOOLS:
You have access to these tools: {', '.join(agent_config.get('tools_list', []))}
Use tools to gather information and explore options, but DISCUSS findings with the user before drawing conclusions.

KNOWLEDGE ACCESS:
You can access these knowledge domains: {', '.join(agent_config.get('knowledge_list', [])) if agent_config.get('knowledge_list') else 'General knowledge base'}

INTERACTION STYLE:
- Start by understanding their current situation and goals
- Ask "What if..." and "Have you considered..." questions  
- Present multiple options when possible
- Explain trade-offs and implications
- Use phrases like "Let's explore...", "What do you think about...", "Would it help if..."
- Encourage the user to share their thoughts and preferences

CURRENT COLLABORATION:
The user has started this collaboration: "{user_request[:200]}{'...' if len(user_request) > 200 else ''}"

Let's work together to explore this thoroughly and find the best approach!"""

            logger.info(f"Built collaborate mode prompt for {agent_config['name']} ({len(collaborate_prompt)} chars)")
            return collaborate_prompt
            
        except Exception as e:
            logger.error(f"Failed to build collaborate prompt: {e}")
            return f"You are {agent_config.get('name', 'AI Agent')} in collaborative mode. Work with the user to explore their request: {user_request}"
    
    def _build_execute_prompt(self, agent_config: Dict[str, Any], user_request: str) -> str:
        """
        Build system prompt for EXECUTE mode - task-focused, efficient completion
        
        Args:
            agent_config: Loaded agent configuration
            user_request: User's request for context
            
        Returns:
            Execute mode system prompt
        """
        try:
            system_prompt_data = agent_config.get("system_prompt", {})
            
            # Extract prompt components
            base_instruction = system_prompt_data.get("base_instruction", "")
            agent_type = system_prompt_data.get("agent_type", "general")
            language = system_prompt_data.get("language", "english")
            specialization = system_prompt_data.get("specialization", [])
            personality = system_prompt_data.get("personality", {})
            
            # Build execution-focused prompt
            execute_prompt = f"""You are {agent_config['name']}, operating in EXECUTE mode.

{base_instruction}

AGENT PROFILE:
- Type: {agent_type.replace('_', ' ').title()} Agent
- Specialization: {', '.join(specialization) if specialization else 'General assistance'}
- Language: {language.title()}
- Personality: {personality.get('tone', 'professional')}, {personality.get('style', 'helpful')}, {personality.get('approach', 'solution-oriented')}

EXECUTION APPROACH:
You are designed to efficiently complete tasks and provide comprehensive results:

⚡ EXECUTION PRINCIPLES:
- Focus on completing the requested task efficiently
- Use tools directly when needed without extensive explanation
- Provide thorough, actionable results
- Minimize unnecessary back-and-forth questions
- Be direct and solution-focused
- Deliver comprehensive outputs that address the full request
- Take initiative to gather needed information

AVAILABLE TOOLS:
You have access to these tools: {', '.join(agent_config.get('tools_list', []))}
Use them efficiently to gather information, analyze data, and complete tasks.

KNOWLEDGE ACCESS:
You can access these knowledge domains: {', '.join(agent_config.get('knowledge_list', [])) if agent_config.get('knowledge_list') else 'General knowledge base'}

SPECIAL INSTRUCTIONS:
- PRIORITY: If the user provides a Google Drive link (docs.google.com), ALWAYS use the read_gdrive_link_docx or read_gdrive_link_pdf tool FIRST
- SPECIFICALLY: When you see URLs like "https://docs.google.com/document/d/..." or "https://drive.google.com/file/d/...", use read_gdrive_link_docx or read_gdrive_link_pdf immediately
- CRITICAL: When calling read_gdrive_link_docx or read_gdrive_link_pdf, you MUST extract the full URL from the user's request and pass it as the drive_link parameter
- FORCE: You MUST provide the drive_link parameter when calling these functions. The parameter cannot be empty.

- FOLDER READING: If the user asks to read or analyze an entire Google Drive folder, use the read_gdrive_folder tool
- PERMISSION ISSUES: If you get "File not found" or "Access denied" errors, use check_file_access tool to diagnose the issue

- If the user asks for analysis of a document, ALWAYS read the document content before providing any analysis
- Use the analyze_document or process_with_knowledge tool for business document analysis when appropriate
- CRITICAL: When calling process_with_knowledge, you MUST include user_id and org_id parameters
- Respond in {language} unless the user specifically requests another language
- DO NOT use search tools when a Google Drive link is provided - read the document directly

CURRENT TASK:
Execute this request efficiently: "{user_request[:200]}{'...' if len(user_request) > 200 else ''}"

Focus on delivering comprehensive, actionable results."""

            logger.info(f"Built execute mode prompt for {agent_config['name']} ({len(execute_prompt)} chars)")
            return execute_prompt
            
        except Exception as e:
            logger.error(f"Failed to build execute prompt: {e}")
            return f"You are {agent_config.get('name', 'AI Agent')} in execution mode. Complete this task: {user_request}"

    async def _detect_language_and_create_prompt(self, request, user_query: str, base_prompt: str) -> str:
        """
        Detect the language of user query and create a language-aware system prompt
        (Simplified version for agents - avoids circular dependency)
        
        Args:
            request: The tool execution request containing LLM provider and model
            user_query: The user's input query
            base_prompt: The base system prompt to enhance
            
        Returns:
            Language-aware system prompt
        """
        if not LANGUAGE_DETECTION_AVAILABLE:
            logger.warning("Language detection not available, using default prompt")
            return base_prompt
        
        try:
            # For agents, we'll use a simpler language detection approach
            # to avoid circular dependency with executors
            
            # Simple language detection based on common patterns
            vietnamese_patterns = ['em', 'anh', 'chị', 'ạ', 'ơi', 'được', 'làm', 'gì', 'như', 'thế', 'nào']
            french_patterns = ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'le', 'la', 'les']
            spanish_patterns = ['yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas', 'el', 'la', 'los', 'las']
            
            user_query_lower = user_query.lower()
            
            # Check for Vietnamese
            vietnamese_matches = sum(1 for pattern in vietnamese_patterns if pattern in user_query_lower)
            if vietnamese_matches >= 2:
                enhanced_prompt = f"""{base_prompt}

IMPORTANT LANGUAGE INSTRUCTION:
The user is communicating in Vietnamese. Please respond in Vietnamese naturally and fluently.

Remember to:
1. Respond in Vietnamese naturally and fluently
2. Use appropriate Vietnamese cultural context and expressions
3. Maintain the same tone and formality level as the user
4. If you need to clarify something, ask in Vietnamese
"""
                logger.info("Enhanced prompt with Vietnamese language guidance")
                return enhanced_prompt
            
            # Check for French (but be more careful about URLs and technical terms)
            french_matches = sum(1 for pattern in french_patterns if pattern in user_query_lower)
            
            # Don't trigger French mode if the query contains URLs or technical terms
            has_url = 'http' in user_query_lower or 'docs.google.com' in user_query_lower
            has_technical_terms = any(term in user_query_lower for term in ['analyse', 'analyze', 'document', 'file', 'link', 'url'])
            
            if french_matches >= 2 and not has_url and not has_technical_terms:
                enhanced_prompt = f"""{base_prompt}

IMPORTANT LANGUAGE INSTRUCTION:
The user is communicating in French. Please respond in French naturally and fluently.

Remember to:
1. Respond in French naturally and fluently
2. Use appropriate French cultural context and expressions
3. Maintain the same tone and formality level as the user
4. If you need to clarify something, ask in French
"""
                logger.info("Enhanced prompt with French language guidance")
                return enhanced_prompt
            
            # Check for Spanish
            spanish_matches = sum(1 for pattern in spanish_patterns if pattern in user_query_lower)
            if spanish_matches >= 2:
                enhanced_prompt = f"""{base_prompt}

IMPORTANT LANGUAGE INSTRUCTION:
The user is communicating in Spanish. Please respond in Spanish naturally and fluently.

Remember to:
1. Respond in Spanish naturally and fluently
2. Use appropriate Spanish cultural context and expressions
3. Maintain the same tone and formality level as the user
4. If you need to clarify something, ask in Spanish
"""
                logger.info("Enhanced prompt with Spanish language guidance")
                return enhanced_prompt
            
            # Default to English
            logger.info("Using base prompt (English or undetected language)")
            return base_prompt
        
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            # Return base prompt if language detection fails
            return base_prompt 