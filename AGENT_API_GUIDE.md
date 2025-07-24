# Agent API Guide

## Overview

The `/api/tool/agent` endpoints provide specialized agent execution with deep reasoning capabilities. Unlike Ami (which builds and teaches agents), these endpoints execute tasks using trained agent capabilities.

## Key Features

- **🤖 Specialized Agents**: Different agent types (sales, support, analyst, etc.)
- **🧠 Deep Reasoning**: Enabled by default for comprehensive task analysis
- **🛠️ Tools Enabled**: Search, context, and brain vector tools available by default
- **🔍 Optional Search**: Search is available but not forced (agents decide when to use it)
- **⚡ Streaming Support**: Real-time response streaming for better UX

## Endpoints

### 1. Synchronous Agent Execution
```
POST /api/tool/agent
```

### 2. Streaming Agent Execution
```
POST /api/tool/agent/stream
```

## Request Format

```json
{
  "llm_provider": "openai",                    // Required: "openai" or "anthropic"
  "user_request": "Analyze Q4 sales data",    // Required: Task to execute
  "agent_id": "sales_analyst_001",             // Required: Unique agent instance ID
  "agent_type": "sales_agent",                 // Required: Agent type/specialization
  "system_prompt": "Custom prompt",            // Optional: Override system prompt
  "model": "gpt-4o",                          // Optional: Custom model
  "model_params": {                           // Optional: Model parameters
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "org_id": "your_org_id",                    // Optional: Organization ID
  "user_id": "your_user_id",                  // Optional: User ID
  
  // Agent-specific parameters (optimized defaults)
  "enable_tools": true,                       // Default: true (tools enabled)
  "enable_deep_reasoning": true,              // Default: true (deep reasoning)
  "reasoning_depth": "standard",              // "light", "standard", "deep"
  "task_focus": "execution",                  // "execution", "analysis", "communication"
  
  // Tool control
  "force_tools": false,                       // Default: false (agent decides)
  "tools_whitelist": null,                    // Optional: ["search", "context"]
  
  // Agent knowledge context
  "specialized_knowledge_domains": [          // Optional: Agent's expertise areas
    "sales", "market_analysis", "crm"
  ],
  "conversation_history": [],                 // Optional: Previous conversation
  "max_history_messages": 15,                 // Default: 15 (focused context)
  "max_history_tokens": 4000                  // Default: 4000 (efficient processing)
}
```

## Agent Types

### Sales Agent
```json
{
  "agent_type": "sales_agent",
  "specialized_knowledge_domains": ["sales", "market_analysis", "customer_relations", "product_knowledge"]
}
```

### Support Agent
```json
{
  "agent_type": "support_agent", 
  "specialized_knowledge_domains": ["customer_service", "technical_support", "problem_resolution", "product_troubleshooting"]
}
```

### Data Analyst Agent
```json
{
  "agent_type": "analyst_agent",
  "specialized_knowledge_domains": ["data_analysis", "business_intelligence", "reporting", "statistical_analysis"]
}
```

### Strategy Agent
```json
{
  "agent_type": "strategy_agent",
  "specialized_knowledge_domains": ["strategic_planning", "market_research", "competitive_analysis", "business_development"]
}
```

## Response Format

### Synchronous Response
```json
{
  "success": true,
  "result": "Detailed agent response...",
  "agent_id": "sales_analyst_001",
  "agent_type": "sales_agent", 
  "execution_time": 3.45,
  "tasks_completed": 1,
  "request_id": "abc12345",
  "total_elapsed_time": 3.67,
  "metadata": {
    "org_id": "your_org_id",
    "user_id": "your_user_id",
    "tools_used": ["search_factory", "brain_vector", "context"],
    "specialized_domains": ["sales", "market_analysis"]
  },
  "error": null
}
```

### Streaming Response (SSE)
```
data: {"type": "status", "content": "Agent sales_analyst_001 (sales_agent) starting task execution...", "agent_id": "sales_analyst_001"}

data: {"type": "thinking", "content": "🎯 Task Analysis: I need to analyze Q4 sales data comprehensively...", "thought_type": "task_understanding", "agent_id": "sales_analyst_001"}

data: {"type": "thinking", "content": "⚡ Execution Strategy: I'll use data analysis and market context... | Complexity: medium | Confidence: 85%", "thought_type": "execution_strategy", "agent_id": "sales_analyst_001"}

data: {"type": "thinking", "content": "🧠 Looking at the current sales trends and performance metrics...", "thought_type": "execution_analysis", "agent_id": "sales_analyst_001"}

data: {"type": "thinking", "content": "🛠️ Activating specialized tools: **search**, **brain_vector** to complete this task efficiently...", "thought_type": "tool_activation", "agent_id": "sales_analyst_001"}

data: {"type": "thinking", "content": "🚀 Beginning task execution with deep reasoning approach...", "thought_type": "execution_start", "agent_id": "sales_analyst_001"}

data: {"type": "response_chunk", "content": "Based on my analysis", "complete": false}

data: {"type": "response_chunk", "content": " of Q4 sales data...", "complete": false}

data: {"type": "response_complete", "content": "Based on my analysis of Q4 sales data, here are the key findings...", "complete": true}

data: {"type": "complete", "content": "Agent sales_analyst_001 task execution completed successfully", "execution_time": 4.23, "success": true, "agent_id": "sales_analyst_001"}
```

## Key Differences from Ami (`/tool/llm`)

| Feature | Ami (`/tool/llm`) | Agent (`/api/tool/agent`) |
|---------|-------------------|---------------------------|
| **Purpose** | Build & teach agents | Execute agent tasks |
| **Reasoning** | Analysis-focused | Task execution-focused |
| **Deep Reasoning** | Optional | **Default enabled** |
| **Tools** | Optional | **Default enabled** |
| **Search** | Optional/configurable | Available but not forced |
| **Context** | Teaching-oriented | Task completion-oriented |
| **History** | 25 messages (6000 tokens) | 15 messages (4000 tokens) |
| **Response Style** | Conversational/exploratory | Direct/action-oriented |

## Usage Examples

### Example 1: Sales Analysis Task
```bash
curl -X POST http://localhost:8000/api/tool/agent \
  -H "Content-Type: application/json" \
  -d '{
    "llm_provider": "openai",
    "user_request": "Analyze our Q4 sales performance and identify key growth opportunities",
    "agent_id": "sales_analyst_001",
    "agent_type": "sales_agent",
    "specialized_knowledge_domains": ["sales", "market_analysis", "performance_metrics"],
    "reasoning_depth": "deep",
    "org_id": "your_org_id",
    "user_id": "your_user_id"
  }'
```

### Example 2: Customer Support Task (Streaming)
```bash
curl -X POST http://localhost:8000/api/tool/agent/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "llm_provider": "anthropic",
    "user_request": "Help resolve a customer complaint about delayed shipping",
    "agent_id": "support_agent_002",
    "agent_type": "support_agent",
    "specialized_knowledge_domains": ["customer_service", "logistics", "problem_resolution"],
    "task_focus": "communication",
    "org_id": "your_org_id",
    "user_id": "your_user_id"
  }'
```

### Example 3: Data Analysis (Tools Disabled)
```bash
curl -X POST http://localhost:8000/api/tool/agent \
  -H "Content-Type: application/json" \
  -d '{
    "llm_provider": "openai",
    "user_request": "Create a KPI dashboard template for executive reporting",
    "agent_id": "analyst_agent_003",
    "agent_type": "analyst_agent",
    "specialized_knowledge_domains": ["business_intelligence", "reporting", "executive_dashboards"],
    "enable_tools": false,
    "reasoning_depth": "standard",
    "org_id": "your_org_id",
    "user_id": "your_user_id"
  }'
```

## Error Handling

### Common Error Response
```json
{
  "success": false,
  "result": "",
  "agent_id": "sales_analyst_001",
  "agent_type": "sales_agent",
  "execution_time": 0,
  "tasks_completed": 0,
  "request_id": "abc12345",
  "total_elapsed_time": 1.23,
  "error": "Agent execution failed: Invalid model specified",
  "metadata": null
}
```

### HTTP Status Codes
- `200`: Success
- `400`: Bad request (invalid parameters)
- `500`: Internal server error

## Performance Characteristics

### With Tools Enabled (Default)
- **Response Time**: 3-8 seconds
- **Deep Reasoning**: Comprehensive task analysis
- **Tool Usage**: Search, brain vectors, context as needed
- **Memory Usage**: Higher due to tool orchestration

### With Tools Disabled
- **Response Time**: 1-3 seconds
- **Deep Reasoning**: Still enabled, but faster
- **Tool Usage**: None (pure LLM reasoning)
- **Memory Usage**: Lower, more efficient

## Best Practices

### 1. Agent ID Naming
- Use descriptive, unique IDs: `sales_analyst_001`, `support_tier1_005`
- Include agent type and instance number
- Keep consistent across sessions

### 2. Specialized Knowledge Domains
- Be specific: `["crm_salesforce", "b2b_sales", "enterprise_accounts"]`
- Match domains to actual agent training/capabilities
- 3-5 domains optimal for focus

### 3. Task Requests
- Be clear and specific: "Analyze Q4 sales data and identify top 3 growth opportunities"
- Include context: "Based on our SaaS product portfolio..."
- Specify desired output format when needed

### 4. Reasoning Depth
- `"light"`: Quick responses, basic analysis
- `"standard"`: Balanced approach (recommended)
- `"deep"`: Comprehensive analysis, slower but thorough

### 5. Streaming vs Synchronous
- **Use Streaming**: For long-running tasks, better UX
- **Use Synchronous**: For simple tasks, API integrations

## Testing

Run the test script to verify endpoints:
```bash
python test_agent_endpoint.py
```

This will test:
- Basic agent execution
- Different agent types
- Tools enabled/disabled scenarios
- Streaming capabilities
- Error handling

## Integration Notes

### Frontend Integration
```javascript
// Synchronous call
const response = await fetch('/api/tool/agent', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    llm_provider: 'openai',
    user_request: 'Analyze customer feedback trends',
    agent_id: 'analyst_001',
    agent_type: 'analyst_agent',
    specialized_knowledge_domains: ['customer_analytics', 'sentiment_analysis']
  })
});

// Streaming call
const eventSource = new EventSource('/api/tool/agent/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(requestData)
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'thinking') {
    console.log('Agent thinking:', data.content);
  } else if (data.type === 'response_chunk') {
    console.log('Response:', data.content);
  }
};
```

### Backend Integration
```python
from agent import execute_agent_async, execute_agent_stream

# Direct agent execution
response = await execute_agent_async(
    llm_provider="openai",
    user_request="Analyze market trends",
    agent_id="market_analyst_001", 
    agent_type="analyst_agent",
    specialized_knowledge_domains=["market_research", "trend_analysis"]
)

# Streaming execution
async for chunk in execute_agent_stream(...):
    process_chunk(chunk)
```

This new agent API provides a powerful, specialized interface for task execution while maintaining the shared tool infrastructure with Ami. The default settings (tools enabled, deep reasoning) make it immediately productive while remaining highly configurable for specific use cases. 