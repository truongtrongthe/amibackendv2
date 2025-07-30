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
        
        Args:
            agent_config: Loaded agent configuration
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
            logger.info(f"Google Drive request detected - enhanced tools: {tools_list}")
        else:
            force_tools = False
        
        return tools_list, force_tools
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.available_tools.keys()) 