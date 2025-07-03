"""
Google Search Tool implementation
"""

import os
import json
from typing import Dict, List, Any
from serpapi import Client

class SearchTool:
    def __init__(self):
        """Initialize Google Search tool"""
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY environment variable is required")
        
        # Initialize SerpAPI client
        self.client = Client(api_key=self.api_key)
    
    def search(self, query: str, num_results: int = 5) -> str:
        """
        Search Google for the given query
        
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
        """
        Format search results into a readable string
        
        Args:
            results: Raw search results from SerpAPI
            query: The original search query
            
        Returns:
            Formatted search results
        """
        
        if "organic_results" not in results:
            return f"No search results found for query: '{query}'"
        
        formatted_results = f"🔍 Search Results for: '{query}'\n"
        formatted_results += "=" * 50 + "\n\n"
        
        organic_results = results["organic_results"]
        
        for i, result in enumerate(organic_results[:5], 1):
            title = result.get("title", "No title")
            link = result.get("link", "No link")
            snippet = result.get("snippet", "No description available")
            
            formatted_results += f"{i}. {title}\n"
            formatted_results += f"   URL: {link}\n"
            formatted_results += f"   Description: {snippet}\n\n"
        
        # Add answer box if available
        if "answer_box" in results:
            answer_box = results["answer_box"]
            if "answer" in answer_box:
                formatted_results += f"💡 Quick Answer: {answer_box['answer']}\n\n"
            elif "snippet" in answer_box:
                formatted_results += f"💡 Quick Answer: {answer_box['snippet']}\n\n"
        
        # Add knowledge graph if available
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            if "description" in kg:
                formatted_results += f"📚 Knowledge Graph: {kg['description']}\n\n"
        
        return formatted_results
    
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