"""
Web Research Module for Todo Generation
======================================

This module provides web search capabilities to research unknown tools
and services mentioned in agent blueprints.
"""

import logging
import requests
import json
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def search_tool_requirements(tool_name: str) -> Optional[str]:
    """
    Search for technical requirements and integration information for a specific tool
    
    Args:
        tool_name: Name of the tool/service to research
        
    Returns:
        Formatted research information or None if search fails
    """
    try:
        # Construct search query for technical information
        search_query = f"{tool_name} API integration documentation technical requirements"
        
        # Perform web search
        search_results = perform_web_search(search_query)
        
        if not search_results:
            logger.warning(f"No search results found for: {tool_name}")
            return None
            
        # Process and format the results
        formatted_info = format_research_results(tool_name, search_results)
        
        logger.info(f"Successfully researched: {tool_name}")
        return formatted_info
        
    except Exception as e:
        logger.error(f"Failed to research tool {tool_name}: {str(e)}")
        return None

def perform_web_search(query: str) -> List[Dict[str, Any]]:
    """
    Perform web search using available search APIs
    
    Args:
        query: Search query string
        
    Returns:
        List of search results with title, snippet, and URL
    """
    try:
        # Option 1: Try Google Custom Search API (if configured)
        results = _try_google_search(query)
        if results:
            return results
            
        # Option 2: Try Bing Search API (if configured)
        results = _try_bing_search(query)
        if results:
            return results
            
        # Option 3: Try SerpAPI (if configured)
        results = _try_serp_api(query)
        if results:
            return results
            
        # Option 4: Try DuckDuckGo API (if available)
        results = _try_duckduckgo_search(query)
        if results:
            return results
            
        logger.warning(f"No search APIs available for query: {query}")
        return []
        
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        return []

def _try_google_search(query: str) -> Optional[List[Dict[str, Any]]]:
    """Try Google Custom Search API"""
    try:
        import os
        
        api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        if not api_key or not search_engine_id:
            return None
            
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': query,
            'num': 5  # Top 5 results
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get('items', []):
            results.append({
                'title': item.get('title', ''),
                'snippet': item.get('snippet', ''),
                'url': item.get('link', '')
            })
            
        return results
        
    except Exception as e:
        logger.debug(f"Google search failed: {str(e)}")
        return None

def _try_bing_search(query: str) -> Optional[List[Dict[str, Any]]]:
    """Try Bing Search API"""
    try:
        import os
        
        api_key = os.getenv('BING_SEARCH_API_KEY')
        if not api_key:
            return None
            
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {'Ocp-Apim-Subscription-Key': api_key}
        params = {'q': query, 'count': 5}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get('webPages', {}).get('value', []):
            results.append({
                'title': item.get('name', ''),
                'snippet': item.get('snippet', ''),
                'url': item.get('url', '')
            })
            
        return results
        
    except Exception as e:
        logger.debug(f"Bing search failed: {str(e)}")
        return None

def _try_serp_api(query: str) -> Optional[List[Dict[str, Any]]]:
    """Try SerpAPI"""
    try:
        import os
        
        api_key = os.getenv('SERPAPI_KEY')
        if not api_key:
            return None
            
        url = "https://serpapi.com/search"
        params = {
            'api_key': api_key,
            'q': query,
            'engine': 'google',
            'num': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get('organic_results', []):
            results.append({
                'title': item.get('title', ''),
                'snippet': item.get('snippet', ''),
                'url': item.get('link', '')
            })
            
        return results
        
    except Exception as e:
        logger.debug(f"SerpAPI search failed: {str(e)}")
        return None

def _try_duckduckgo_search(query: str) -> Optional[List[Dict[str, Any]]]:
    """Try DuckDuckGo API (free but limited)"""
    try:
        # DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_redirect': '1',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        # Check for abstract (summary info)
        if data.get('Abstract'):
            results.append({
                'title': data.get('Heading', query),
                'snippet': data.get('Abstract'),
                'url': data.get('AbstractURL', '')
            })
            
        # Check for related topics
        for topic in data.get('RelatedTopics', [])[:3]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append({
                    'title': topic.get('Text', '')[:100] + '...',
                    'snippet': topic.get('Text', ''),
                    'url': topic.get('FirstURL', '')
                })
                
        return results if results else None
        
    except Exception as e:
        logger.debug(f"DuckDuckGo search failed: {str(e)}")
        return None

def format_research_results(tool_name: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Format search results into useful research information
    
    Args:
        tool_name: Name of the tool being researched
        search_results: List of search result dictionaries
        
    Returns:
        Formatted research information string
    """
    try:
        if not search_results:
            return _generate_fallback_info(tool_name)
            
        info_parts = []
        
        # Extract key information from search results
        for result in search_results[:3]:  # Top 3 results
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            url = result.get('url', '')
            
            if snippet:
                # Clean and format the snippet
                clean_snippet = snippet.replace('\n', ' ').strip()
                if len(clean_snippet) > 200:
                    clean_snippet = clean_snippet[:200] + '...'
                    
                info_parts.append(f"- {title}: {clean_snippet}")
                
        if info_parts:
            formatted_info = f"""
Based on web research for {tool_name.title()}:

{chr(10).join(info_parts)}

Common Integration Requirements:
- API Documentation: Check official docs for endpoints and authentication
- Credentials: Likely needs API keys, tokens, or authentication credentials  
- Base URL: Verify the correct API base URL from official documentation
- Rate Limits: Review API usage limits and throttling policies
- Data Format: Most modern APIs use JSON, some legacy systems use XML
- SDKs: Look for official libraries or SDKs for easier integration
"""
            return formatted_info.strip()
        else:
            return _generate_fallback_info(tool_name)
            
    except Exception as e:
        logger.error(f"Failed to format research results: {str(e)}")
        return _generate_fallback_info(tool_name)

def _generate_fallback_info(tool_name: str) -> str:
    """Generate fallback information when search fails"""
    return f"""
Research for {tool_name.title()} (Limited Information Available):

- API Integration: Most enterprise software provides REST API or SOAP API
- Authentication: Typically requires API keys, OAuth tokens, or basic auth
- Documentation: Search for official API docs or developer guides  
- Common Requirements: Base URL, authentication credentials, permission scopes
- Data Formats: Usually JSON for modern APIs, XML for legacy systems
- Rate Limits: Check documentation for API call limits
- SDKs: Look for official libraries in Python, JavaScript, etc.
- Support: Contact vendor support for specific integration requirements

Note: Limited information available online. Recommend contacting the vendor directly for detailed integration requirements.
"""
