# Context Tool Integration Guide

## 🎯 **Perfect Architecture - Tool-Based Context Retrieval**

Your suggestion was brilliant! Instead of injecting context as parameters, we now have a `get_context` **tool** that works exactly like the `search_tool`. This maintains a **simple, consistent architecture** where the LLM explicitly decides when to retrieve context.

## ✅ **What Was Implemented**

### **1. Context Tool (`context_tool.py`)**
- **ContextTool**: Retrieves context from multiple sources
- **IntegratedContextTool**: Integrates with your existing systems
- **AsyncContextTool**: Async wrapper for integration

### **2. Both LLM Tools Enhanced**
- **OpenAI Tool**: Now supports both `search_google` and `get_context` tools
- **Anthropic Tool**: Same dual-tool support with identical API
- **Executive Tool**: Automatically registers both tools

### **3. LLM Decides When to Get Context**
- LLM calls `get_context` when it needs additional information
- Explicit and controllable by the LLM
- No hidden parameter injection

## 🔧 **How It Works**

### **Available Tools for LLM**
```python
# Both tools are now available to the LLM
available_tools = [
    search_tool,    # search_google(query)
    context_tool    # get_context(query, source_types)
]
```

### **LLM Tool Definitions**
```python
# The LLM can now call:
{
    "name": "search_google",
    "description": "Search Google for information on any topic",
    "parameters": {
        "query": {"type": "string", "description": "The search query"}
    }
}

{
    "name": "get_context", 
    "description": "Retrieve relevant context including user profile, system status, organization info, and knowledge base information",
    "parameters": {
        "query": {"type": "string", "description": "The topic to get context for"},
        "source_types": {"type": "array", "items": {"type": "string"}, "description": "Specific sources: user_profile, system_status, organization_info, knowledge_base, recent_activity"}
    }
}
```

## 🚀 **Usage Examples**

### **1. Basic Usage (No Frontend Changes)**
```python
from exec_tool import execute_tool_stream

# Your existing API calls work exactly the same
async for chunk in execute_tool_stream(
    llm_provider="openai",
    user_query="What are your pricing options?",
    enable_tools=True  # Both search and context tools available
):
    print(chunk["content"], end="")

# LLM will automatically call get_context if it needs pricing information
```

### **2. Selective Tool Usage**
```python
# Only allow context tool (no search)
async for chunk in execute_tool_stream(
    llm_provider="anthropic",
    user_query="What's my current usage?",
    enable_tools=True,
    tools_whitelist=["context"]  # Only context tool available
):
    print(chunk["content"], end="")
```

### **3. Integration with Your Existing Systems**
```python
from context_tool import IntegratedContextTool
from learning_support import LearningSupport
from database import query_knowledge

# Create integrated context tool
learning_support = LearningSupport()
context_tool = IntegratedContextTool(
    learning_support=learning_support,
    knowledge_query_func=query_knowledge
)

# Use in your existing ExecutiveTool setup
executive_tool = ExecutiveTool()
executive_tool.available_tools["context"] = context_tool
```

### **4. LLM Conversation Flow**
```
User: "I'm having authentication issues"

LLM: I'll help you with authentication issues. Let me get some context about your account and our authentication system.

[LLM calls get_context("authentication issues", ["user_profile", "system_status", "knowledge_base"])]

Context Tool Returns:
=== USER PROFILE ===
User ID: user123
Organization: acme_corp
Subscription: Premium
Previous Support Tickets: 0 open, 2 resolved

=== SYSTEM STATUS ===
Authentication Service: Online
Recent Deployments: v2.1.0 deployed yesterday

=== KNOWLEDGE BASE ===
Authentication Methods:
- JWT tokens with 24-hour expiry
- OAuth 2.0 for third-party integrations
...

LLM: Based on your premium account status and our current system information, here are the most likely causes of your authentication issues...
```

## 🔗 **Integration with Your Systems**

### **Learning Support Integration**
```python
class CustomContextTool(ContextTool):
    def __init__(self, learning_support, user_id, org_id):
        super().__init__()
        self.learning_support = learning_support
        self.user_id = user_id
        self.org_id = org_id
    
    def _get_learning_data(self, query: str, user_id: str, org_id: str) -> str:
        # Use your existing learning support
        result = asyncio.run(self.learning_support.search_knowledge(
            query, "", user_id, org_id=org_id
        ))
        return result.get("knowledge_context", "No learning data found")
```

### **RAG System Integration**
```python
def _get_integrated_knowledge(self, query: str, user_id: str, org_id: str) -> str:
    # Use your existing RAG system
    documents = asyncio.run(query_knowledge(query, bank_name="documents", top_k=3))
    if documents:
        return "\n\n".join([doc["raw"] for doc in documents])
    return "No knowledge found"
```

## 📡 **Frontend Changes Required**

### **❌ ABSOLUTELY NONE**

```javascript
// Frontend code stays exactly the same
fetch('/tool/llm', {
  method: 'POST',
  body: JSON.stringify({
    llm_provider: 'openai',
    user_query: 'Help with API authentication',
    enable_tools: true
  })
})

// Backend now automatically provides both search and context tools
// LLM decides when to use each tool
```

## 🎯 **Architecture Benefits**

### **1. Explicit Control**
- LLM explicitly calls `get_context` when needed
- No hidden parameter injection
- Clear tool usage in logs

### **2. Consistent Pattern**
- Same pattern as `search_tool`
- Both tools work identically
- Simple architecture maintained

### **3. Flexible Sources**
- Multiple context sources available
- LLM can request specific sources
- Easy to add new context sources

### **4. No Breaking Changes**
- Existing APIs work unchanged
- Gradual enhancement possible
- Frontend remains untouched

## 🛠 **Configuration Options**

### **Enable Both Tools**
```python
# Default: Both search and context available
enable_tools=True
tools_whitelist=None  # All tools
```

### **Context Only** 
```python
# Only context tool (no web search)
enable_tools=True
tools_whitelist=["context"]
```

### **Search Only**
```python
# Only search tool (no context)
enable_tools=True  
tools_whitelist=["search"]
```

### **No Tools**
```python
# Pure LLM response (no tools)
enable_tools=False
```

## 📊 **Context Sources Available**

| Source | Description | Use Case |
|--------|-------------|-----------|
| `user_profile` | User account information | Personalized responses |
| `system_status` | Current system health | Troubleshooting |
| `organization_info` | Org settings & features | Feature availability |
| `knowledge_base` | Your knowledge systems | Domain expertise |
| `recent_activity` | User/org activity logs | Context-aware support |

## 🔧 **Quick Start**

### **1. Use Existing APIs**
Your current endpoints now support both tools automatically:

```python
# /tool/llm endpoint now includes context tool
POST /tool/llm
{
  "llm_provider": "openai",
  "user_query": "What are your pricing plans?",
  "enable_tools": true
}

# LLM will call get_context if it needs pricing information
```

### **2. Customize Context Sources**
```python
# In your API endpoint
from context_tool import IntegratedContextTool

# Create custom context tool
context_tool = IntegratedContextTool(
    learning_support=your_learning_support,
    knowledge_query_func=your_knowledge_query
)

# Use in executive tool
executive_tool = ExecutiveTool()
executive_tool.available_tools["context"] = context_tool
```

### **3. Monitor Tool Usage**
```python
# Tool calls are logged
"LLM called get_context with query: 'authentication issues'"
"Context sources: ['user_profile', 'system_status', 'knowledge_base']"
"Retrieved context for user: user123, org: acme_corp"
```

## 🎉 **Perfect Solution**

This tool-based approach is **exactly** what you wanted:

✅ **Explicit** - LLM explicitly calls `get_context`  
✅ **Simple** - Same pattern as `search_tool`  
✅ **Maintainable** - Clean, consistent architecture  
✅ **No Breaking Changes** - Frontend untouched  
✅ **Flexible** - LLM decides when to use context  
✅ **Integrated** - Works with your existing systems  

The LLM now has both `search_google` and `get_context` tools available and can use them intelligently based on the user's needs! 