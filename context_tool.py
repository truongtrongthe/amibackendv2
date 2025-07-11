"""
Context Tool - Retrieves relevant context/knowledge for LLM queries

This tool allows LLMs to explicitly request context from knowledge bases,
user profiles, system status, or other relevant sources.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio

logger = logging.getLogger(__name__)


class ContextTool:
    """Tool for retrieving contextual information and knowledge"""
    
    def __init__(self, knowledge_sources: Optional[Dict[str, Any]] = None):
        """
        Initialize the context tool
        
        Args:
            knowledge_sources: Dictionary of available knowledge sources
        """
        self.knowledge_sources = knowledge_sources or {}
        self.setup_default_sources()
    
    def setup_default_sources(self):
        """Setup default knowledge sources"""
        if not self.knowledge_sources:
            self.knowledge_sources = {
                "user_profile": self._get_user_profile,
                "system_status": self._get_system_status,
                "organization_info": self._get_organization_info,
                "knowledge_base": self._get_knowledge_base,
                "recent_activity": self._get_recent_activity
            }
    
    def get_context(self, query: str, source_types: List[str] = None, user_id: str = "unknown", org_id: str = "unknown") -> str:
        """
        Retrieve context based on query and source types
        
        Args:
            query: The context query/topic
            source_types: List of source types to query (default: all)
            user_id: User identifier for personalized context
            org_id: Organization identifier
            
        Returns:
            Formatted context string
        """
        try:
            logger.info(f"ContextTool: Retrieving context for query: '{query}' (user: {user_id}, org: {org_id})")
            
            if source_types is None:
                source_types = list(self.knowledge_sources.keys())
            
            context_results = []
            
            for source_type in source_types:
                if source_type in self.knowledge_sources:
                    try:
                        source_func = self.knowledge_sources[source_type]
                        result = source_func(query, user_id, org_id)
                        if result:
                            context_results.append(f"=== {source_type.upper().replace('_', ' ')} ===\n{result}")
                    except Exception as e:
                        logger.error(f"Error retrieving context from {source_type}: {e}")
                        context_results.append(f"=== {source_type.upper().replace('_', ' ')} ===\n[Error retrieving information]")
            
            if not context_results:
                return "No relevant context found for the query."
            
            full_context = "\n\n".join(context_results)
            logger.info(f"ContextTool: Retrieved {len(context_results)} context sources")
            return full_context
            
        except Exception as e:
            logger.error(f"ContextTool error: {e}")
            return f"Error retrieving context: {str(e)}"
    
    def _get_user_profile(self, query: str, user_id: str, org_id: str) -> str:
        """Get user profile information"""
        # This would integrate with your user management system
        # For now, return simulated data
        if user_id == "unknown":
            return "User profile: Anonymous user"
        
        return f"""
User ID: {user_id}
Organization: {org_id}
Subscription: Premium
Account Status: Active
Last Login: 2024-01-15
Previous Support Tickets: 0 open, 3 resolved
Preferred Language: English
Time Zone: UTC-8
        """.strip()
    
    def _get_system_status(self, query: str, user_id: str, org_id: str) -> str:
        """Get current system status"""
        # This would integrate with your monitoring system
        return """
API Gateway: Online (99.9% uptime)
Database: Online (response time: 45ms)
Authentication Service: Online
Search Service: Online
Rate Limiting: Active (1000 req/hour per user)
Maintenance Window: None scheduled
Recent Deployments: v2.1.0 deployed 2 days ago
        """.strip()
    
    def _get_organization_info(self, query: str, user_id: str, org_id: str) -> str:
        """Get organization-specific information"""
        # This would integrate with your organization management system
        if org_id == "unknown":
            return "Organization: Default/Public"
        
        return f"""
Organization: {org_id}
Plan: Enterprise
Features Enabled: SSO, Advanced Analytics, Priority Support
API Quota: 50,000 requests/month (used: 25,000)
Team Size: 25 users
Admin Contact: admin@{org_id}.com
Custom Integrations: Slack, GitHub, Jira
        """.strip()
    
    def _get_knowledge_base(self, query: str, user_id: str, org_id: str) -> str:
        """Get relevant information from knowledge base"""
        # This would integrate with your existing knowledge systems
        # For now, return query-relevant information
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["auth", "login", "token", "security"]):
            return """
Authentication Methods:
- JWT tokens with 24-hour expiry
- OAuth 2.0 for third-party integrations
- Multi-factor authentication available
- API keys for server-to-server communication

Common Issues:
- Token expiry: Refresh tokens automatically
- Invalid credentials: Check email/password format
- Rate limiting: Implement exponential backoff
            """.strip()
        
        elif any(term in query_lower for term in ["api", "rate", "limit", "quota"]):
            return """
API Rate Limiting:
- Standard: 1,000 requests/hour
- Premium: 5,000 requests/hour  
- Enterprise: 50,000 requests/hour
- Burst limit: 100 requests/minute

Best Practices:
- Implement exponential backoff
- Cache responses when possible
- Use webhooks instead of polling
- Monitor usage with headers
            """.strip()
        
        elif any(term in query_lower for term in ["price", "pricing", "cost", "plan"]):
            return """
Pricing Plans:
- Basic: $10/month (1,000 API calls, email support)
- Pro: $50/month (10,000 API calls, priority support)
- Enterprise: Custom pricing (unlimited calls, dedicated support)

Features Comparison:
- Basic: Core API access
- Pro: Advanced analytics, webhooks
- Enterprise: SSO, custom integrations, SLA
            """.strip()
        
        return "No specific knowledge base information found for this query."
    
    def _get_recent_activity(self, query: str, user_id: str, org_id: str) -> str:
        """Get recent user/organization activity"""
        # This would integrate with your activity tracking system
        return f"""
Recent Activity for {user_id}:
- Last API call: 2 hours ago (GET /api/users)
- Recent errors: None in last 24 hours
- Feature usage: Search API (high), Auth API (medium)
- Documentation views: Rate limiting guide, API reference

Organization Activity:
- Team API usage: 45% of monthly quota
- Most active users: {user_id}, user456, user789
- Recent support tickets: 1 resolved yesterday
        """.strip()


class AsyncContextTool:
    """Async version of ContextTool for integration with async systems"""
    
    def __init__(self, knowledge_sources: Optional[Dict[str, Any]] = None):
        self.sync_tool = ContextTool(knowledge_sources)
    
    async def get_context(self, query: str, source_types: List[str] = None, user_id: str = "unknown", org_id: str = "unknown") -> str:
        """Async wrapper for context retrieval"""
        # Run the sync method in a thread pool to avoid blocking
        return await asyncio.to_thread(
            self.sync_tool.get_context, 
            query, source_types, user_id, org_id
        )


# Integration helpers for existing systems
class IntegratedContextTool(ContextTool):
    """Context tool that integrates with your existing systems"""
    
    def __init__(self, learning_support=None, knowledge_query_func=None):
        """
        Initialize with existing system integrations
        
        Args:
            learning_support: Your LearningSupport instance
            knowledge_query_func: Your knowledge query function
        """
        super().__init__()
        self.learning_support = learning_support
        self.knowledge_query_func = knowledge_query_func
        
        # Override default sources with integrated ones
        self.knowledge_sources.update({
            "knowledge_base": self._get_integrated_knowledge,
            "learning_data": self._get_learning_data
        })
    
    def _get_integrated_knowledge(self, query: str, user_id: str, org_id: str) -> str:
        """Get knowledge from your existing knowledge systems"""
        if not self.knowledge_query_func:
            return "Knowledge system not available"
        
        try:
            # Check if we're in an async context and handle accordingly
            try:
                # Try to get the running loop
                loop = asyncio.get_running_loop()
                # If we're in an async context, run in thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.knowledge_query_func(query, bank_name="documents", top_k=3))
                    )
                    results = future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                results = asyncio.run(self.knowledge_query_func(query, bank_name="documents", top_k=3))
            
            if results:
                knowledge_text = "\n\n".join([result["raw"] for result in results])
                return f"Retrieved Knowledge:\n{knowledge_text}"
            return "No knowledge found in database"
        except Exception as e:
            return f"Error querying knowledge: {e}"
    
    def _get_learning_data(self, query: str, user_id: str, org_id: str) -> str:
        """Get data from your learning support system"""
        if not self.learning_support:
            return "Learning support system not available"
        
        try:
            # Check if we're in an async context and handle accordingly
            try:
                # Try to get the running loop
                loop = asyncio.get_running_loop()
                # If we're in an async context, run in thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.learning_support.search_knowledge(query, "", user_id, org_id=org_id))
                    )
                    result = future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                result = asyncio.run(self.learning_support.search_knowledge(
                    query, "", user_id, org_id=org_id
                ))
            
            knowledge_context = result.get("knowledge_context", "")
            if knowledge_context:
                return f"Learning Support Data:\n{knowledge_context}"
            return "No learning data found"
        except Exception as e:
            return f"Error querying learning support: {e}" 