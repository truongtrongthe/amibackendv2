"""
Search Tool implementations for different LLM providers
"""

import os
import json
from typing import Dict, List, Any
from serpapi import Client

class SearchTool:
    """Google Search Tool implementation using SERPAPI (for OpenAI)"""
    
    def __init__(self):
        """Initialize Google Search tool"""
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY environment variable is required")
        
        # Initialize SerpAPI client
        self.client = Client(api_key=self.api_key)
        self.name = "serpapi_search"
    
    def search(self, query: str, num_results: int = 5) -> str:
        """
        Search Google for the given query using SERPAPI
        
        Args:
            query: The search query
            num_results: Number of results to return (default: 5)
            
        Returns:
            Formatted search results as a string
        """
        
        try:
            # Setup search parameters
            params = {
                "q": query,
                "hl": "en",
                "gl": "us",
                "google_domain": "google.com",
                "num": num_results
            }
            
            # Perform search using the new API
            results = self.client.search(params)
            
            # Convert to dictionary if needed
            if hasattr(results, 'data'):
                results_dict = results.data
            else:
                results_dict = results
            
            # Format results
            return self._format_results(results_dict, query)
            
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    def _format_results(self, results: Dict[str, Any], query: str) -> str:
        """Format search results for LLM consumption"""
        
        if not results:
            return f"No search results found for query: {query}"
        
        formatted_results = []
        formatted_results.append(f"Search Results for: {query}")
        formatted_results.append("=" * 50)
        
        # Process organic results
        organic_results = results.get("organic_results", [])
        if organic_results:
            for i, result in enumerate(organic_results, 1):
                title = result.get("title", "No title")
                link = result.get("link", "No link")
                snippet = result.get("snippet", "No description available")
                
                formatted_results.append(f"{i}. {title}")
                formatted_results.append(f"   URL: {link}")
                formatted_results.append(f"   {snippet}")
                formatted_results.append("")
        
        # Process answer box if available
        answer_box = results.get("answer_box", {})
        if answer_box:
            formatted_results.append("Quick Answer:")
            formatted_results.append("-" * 20)
            
            answer = answer_box.get("answer", "")
            if answer:
                formatted_results.append(f"Answer: {answer}")
            
            snippet = answer_box.get("snippet", "")
            if snippet:
                formatted_results.append(f"Details: {snippet}")
            
            formatted_results.append("")
        
        # Process knowledge graph if available
        knowledge_graph = results.get("knowledge_graph", {})
        if knowledge_graph:
            formatted_results.append("Knowledge Graph:")
            formatted_results.append("-" * 20)
            
            title = knowledge_graph.get("title", "")
            if title:
                formatted_results.append(f"Title: {title}")
            
            description = knowledge_graph.get("description", "")
            if description:
                formatted_results.append(f"Description: {description}")
            
            formatted_results.append("")
        
        return "\n".join(formatted_results)
    
    def get_tool_description(self) -> Dict[str, Any]:
        """
        Get tool description for LLM function calling
        
        Returns:
            Tool description dictionary
        """
        return {
            "name": "search_google",
            "description": "Search Google for information on any topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }


class AnthropicSearchTool:
    """Native web search tool for Anthropic Claude"""
    
    def __init__(self):
        """Initialize Anthropic native search tool"""
        self.name = "anthropic_native_search"
        # No API key needed - uses Claude's native web search
    
    def supports_native_search(self, model: str) -> bool:
        """Check if the model supports native web search"""
        supported_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620", 
            "claude-3-7-sonnet-20240509",
            "claude-3-5-haiku-20241022"
        ]
        return model in supported_models
    
    def get_tool_description(self) -> Dict[str, Any]:
        """
        Get tool description for Anthropic's native web search
        
        Returns:
            Tool description dictionary for Claude's native web search
        """
        return {
            "name": "web_search",
            "description": "Search the web for current information on any topic",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up current information"
                    }
                },
                "required": ["query"]
            }
        }
    
    def search(self, query: str) -> str:
        """
        This method exists for compatibility but native web search
        is handled directly by Claude API - no external implementation needed
        """
        return f"Native web search for: {query} (handled by Claude API)"


def create_search_tool(llm_provider: str, model: str = None) -> Any:
    """
    Factory function to create the appropriate search tool based on LLM provider
    
    Args:
        llm_provider: Either 'anthropic' or 'openai'
        model: Model name (used for Anthropic to check web search support)
        
    Returns:
        Appropriate search tool instance
    """
    if llm_provider.lower() == "anthropic":
        # Use native web search for Anthropic Claude
        return AnthropicSearchTool()
    elif llm_provider.lower() == "openai":
        # Use SERPAPI for OpenAI
        return SearchTool()
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}") 