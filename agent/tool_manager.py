"""
Tool Manager Module
==================

Handles tool selection and management for agent execution.
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class ToolManager:
    """Manages tool selection and configuration for agent tasks"""
    
    def __init__(self, available_tools: Dict[str, Any]):
        self.available_tools = available_tools
    
    def determine_dynamic_tools(self, agent_config: Dict[str, Any], 
                              user_request: str) -> Tuple[List[str], bool]:
        """
        Determine tools and force_tools setting based on agent config and request
        
        PRIORITY: Uses blueprint-configured tools if available, falls back to generic detection
        
        Args:
            agent_config: Loaded agent configuration
            user_request: User's request
            
        Returns:
            Tuple of (tools_whitelist, force_tools)
        """
        # PRIORITY: Use blueprint-configured tools if available
        if agent_config.get("compilation_status") == "compiled" and agent_config.get("blueprint_tool_configs"):
            return self._determine_blueprint_tools(agent_config, user_request)
        
        # FALLBACK: Use generic tool detection for uncompiled agents
        return self._determine_generic_tools(agent_config, user_request)
    
    def _determine_blueprint_tools(self, agent_config: Dict[str, Any], user_request: str) -> Tuple[List[str], bool]:
        """
        Determine tools based on blueprint configurations and integrations
        
        Args:
            agent_config: Agent configuration with blueprint tool configs
            user_request: User's request
            
        Returns:
            Tuple of (tools_whitelist, force_tools)
        """
        blueprint_configs = agent_config.get("blueprint_tool_configs", {})
        tool_mappings = blueprint_configs.get("tool_mappings", {})
        integrations = blueprint_configs.get("integrations", {})
        
        # Start with blueprint-defined tool categories
        tools_list = list(tool_mappings.keys())
        
        # Analyze request to determine which blueprint tools are needed
        user_request_lower = user_request.lower()
        
        # Google Drive / File access requests
        if any(keyword in user_request_lower for keyword in [
            'docs.google.com', 'drive.google.com', 'google drive', 'gdrive', 
            'folder', 'folders', 'directory', 'documents in', 'files in', 'read', 'analyze'
        ]):
            if "file_access" in tool_mappings:
                if "file_access" not in tools_list:
                    tools_list.append("file_access")
                # Add business logic for analysis
                if "business_logic" not in tools_list:
                    tools_list.append("business_logic")
                logger.info(f"Blueprint Google Drive integration detected - using configured tools: {tools_list}")
        
        # Slack / Communication requests
        if any(keyword in user_request_lower for keyword in ['slack', 'channel', 'message', 'notify', 'send']):
            if "communication" in tool_mappings:
                if "communication" not in tools_list:
                    tools_list.append("communication")
                logger.info(f"Blueprint communication integration detected - using configured tools: {tools_list}")
        
        # Email requests
        if any(keyword in user_request_lower for keyword in ['email', 'mail', 'send email']):
            if "email" in tool_mappings:
                if "email" not in tools_list:
                    tools_list.append("email")
                logger.info(f"Blueprint email integration detected - using configured tools: {tools_list}")
        
        # Database / CRM requests
        if any(keyword in user_request_lower for keyword in ['database', 'crm', 'customer', 'sales data']):
            if "business_logic" in tool_mappings:
                if "business_logic" not in tools_list:
                    tools_list.append("business_logic")
                logger.info(f"Blueprint business logic integration detected - using configured tools: {tools_list}")
        
        # Always include search and context tools for compiled agents
        if "search_factory" not in tools_list:
            tools_list.append("search_factory")
        if "context" not in tools_list:
            tools_list.append("context")
        
        # Force tools if we have blueprint integrations configured
        force_tools = len(integrations) > 0
        
        logger.info(f"Blueprint-based tools determined: {tools_list} (force: {force_tools})")
        return tools_list, force_tools
    
    def _determine_generic_tools(self, agent_config: Dict[str, Any], user_request: str) -> Tuple[List[str], bool]:
        """
        Fallback generic tool determination for uncompiled agents
        
        Args:
            agent_config: Agent configuration
            user_request: User's request
            
        Returns:
            Tuple of (tools_whitelist, force_tools)
        """
        tools_list = agent_config.get("tools_list", [])
        
        # Check if this is a Google Drive request
        user_request_lower = user_request.lower()
        is_gdrive_request = (
            'docs.google.com' in user_request_lower or 
            'drive.google.com' in user_request_lower or
            'google drive' in user_request_lower or
            'gdrive' in user_request_lower or
            any(keyword in user_request_lower for keyword in ['folder', 'folders', 'directory', 'documents in', 'files in'])
        )
        
        # For Google Drive requests, ensure file_access tools are available
        if is_gdrive_request:
            if 'file_access' not in tools_list:
                tools_list = tools_list + ['file_access']
            if 'business_logic' not in tools_list:
                tools_list = tools_list + ['business_logic']
            force_tools = True
            logger.info(f"Generic Google Drive request detected - enhanced tools: {tools_list}")
        else:
            force_tools = False
        
        return tools_list, force_tools
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.available_tools.keys())
    
    def get_blueprint_tool_configurations(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get blueprint tool configurations for credential and parameter injection
        
        Args:
            agent_config: Agent configuration with blueprint tool configs
            
        Returns:
            Dict with tool-specific configurations
        """
        if not agent_config.get("blueprint_tool_configs"):
            return {}
        
        blueprint_configs = agent_config["blueprint_tool_configs"]
        
        # Organize configurations by tool category
        tool_configs = {}
        
        # Process integrations
        for todo_id, integration_config in blueprint_configs.get("integrations", {}).items():
            if integration_config:
                service_name = integration_config.get("service_name", todo_id)
                
                # Map to tool configuration format
                if "google drive" in service_name.lower():
                    tool_configs["file_access"] = {
                        "service": "google_drive",
                        "folder_path": integration_config.get("folder_name", "/"),
                        "access_level": integration_config.get("access_level", "read"),
                        "file_types": integration_config.get("file_types", "all"),
                        "credentials_configured": True
                    }
                elif "slack" in service_name.lower():
                    tool_configs["communication"] = {
                        "service": "slack",
                        "channel": integration_config.get("channel_name", "#general"),
                        "bot_token": "[CONFIGURED]",
                        "credentials_configured": True
                    }
                elif "email" in service_name.lower():
                    tool_configs["email"] = {
                        "service": "email",
                        "smtp_server": integration_config.get("smtp_server", ""),
                        "email_address": integration_config.get("email_address", ""),
                        "credentials_configured": True
                    }
        
        # Process tool-specific configurations
        for todo_id, tool_config in blueprint_configs.get("tools", {}).items():
            if tool_config:
                tool_name = tool_config.get("tool_name", todo_id)
                tool_configs[tool_name] = tool_config
        
        logger.info(f"Blueprint tool configurations prepared: {list(tool_configs.keys())}")
        return tool_configs 