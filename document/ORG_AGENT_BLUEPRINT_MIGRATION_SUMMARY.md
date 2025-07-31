# Org Agent Blueprint Migration Summary

## Overview

This migration transforms the org_agent system from a basic configuration-based approach to a comprehensive blueprint-based architecture that supports versioned, conversation-driven agent definitions.

## Key Changes

### 1. Database Architecture

**OLD SCHEMA:**
```sql
org_agents:
  - Basic agent info + embedded tools/knowledge configuration
  - Single table with JSONB fields for system_prompt, tools_list, knowledge_list
```

**NEW SCHEMA:**
```sql
org_agents (lightweight management):
  - id, agent_id, org_id, name, description, status
  - created_by, created_at, updated_at
  - current_blueprint_id → references active blueprint

agent_blueprints (versioned definitions):
  - id, agent_id, version
  - agent_blueprint (complete blueprint JSON)
  - created_by, created_at, conversation_id
  - Full versioning support
```

### 2. Blueprint Structure

The new `agent_blueprint` JSONB field can contain:

```json
{
  "identity": {
    "name": "Shop Buddy",
    "role": "Customer Service Assistant", 
    "personality": "Friendly and helpful",
    "primary_purpose": "Answer customer questions about clothing store"
  },
  "capabilities": [
    {
      "task": "Answer product questions",
      "description": "Tell customers about sizes, prices, stock",
      "examples": ["What sizes do you have?", "Is this in stock?"]
    }
  ],
  "tools": [
    {
      "name": "inventory_lookup",
      "purpose": "Check product availability",
      "triggers": ["stock", "availability", "sizes"]
    }
  ],
  "integrations": [
    {
      "app": "Slack",
      "actions": ["send_message"],
      "triggers": ["out_of_stock_inquiry"]
    }
  ],
  "knowledge_sources": [
    {
      "source": "Product Catalog Google Sheet",
      "type": "spreadsheet",
      "update_frequency": "daily"
    }
  ],
  "monitoring": {
    "reporting": "weekly_summary",
    "fallbacks": "escalate_to_human"
  },
  "test_scenarios": [
    {
      "question": "What's the price of blue jeans?",
      "expected": "The blue jeans are $39.99!"
    }
  ],
  "workflow": {
    "steps": ["receive_question", "check_knowledge", "respond_or_escalate"],
    "visual_flow": "[Customer] → [Question] → [Knowledge] → [Response]"
  }
}
```

### 3. Python Model Changes

**OLD Models:**
```python
class Agent:
    # Included system_prompt, tools_list, knowledge_list directly
```

**NEW Models:**
```python
class Agent:
    # Lightweight: basic info + current_blueprint_id reference
    
class AgentBlueprint:
    # Complete blueprint definition with versioning
```

### 4. API Endpoint Changes

**NEW ENDPOINTS:**
- `POST /org-agents/with-blueprint` - Create agent with initial blueprint
- `POST /org-agents/{id}/blueprints` - Create new blueprint version
- `GET /org-agents/{id}/blueprints` - Get all blueprint versions
- `GET /org-agents/{id}/blueprints/{blueprint_id}` - Get specific blueprint
- `POST /org-agents/{id}/blueprints/{blueprint_id}/activate` - Activate blueprint version

**UPDATED ENDPOINTS:**
- `GET /org-agents/{id}` - Now returns agent with current blueprint
- `POST /org-agents/` - Creates basic agent (no blueprint)
- `PUT /org-agents/{id}` - Updates basic info only (not blueprint)

## Migration Files

1. **migration_drop_old_org_agents.sql** - Drops existing tables
2. **migration_create_new_org_agents_schema.sql** - Creates new schema
3. **orgdb.py** - Updated with new models and CRUD functions
4. **org_agent_new.py** - New API endpoints for blueprint architecture
5. **replace_org_agent.py** - Script to backup and replace org_agent.py

## Migration Steps

### 1. Database Migration
```bash
# 1. Drop old tables
psql -f migration_drop_old_org_agents.sql

# 2. Create new schema
psql -f migration_create_new_org_agents_schema.sql
```

### 2. Code Migration
```bash
# Backup and replace org_agent.py
python replace_org_agent.py
```

### 3. Features Enabled

**Versioning:**
- Multiple blueprint versions per agent
- Conversation-linked blueprint creation
- Easy rollback to previous versions

**Blueprint-Driven Development:**
- Human-AMI conversations create blueprints
- Complete agent definition in single JSON
- Test scenarios and workflow documentation

**Better Organization:**
- Separation of agent management vs. blueprint content
- Clear versioning history
- Conversation traceability

## Blueprint Creation Flow

1. **Human-AMI Conversation** → Discusses agent requirements
2. **Agreement Reached** → AMI creates blueprint JSON
3. **Blueprint Stored** → New version created with conversation_id
4. **Agent Updated** → current_blueprint_id points to new version
5. **Agent Active** → Uses new blueprint for operations

## API Usage Examples

### Create Agent with Blueprint
```python
POST /org-agents/with-blueprint
{
  "name": "Shop Buddy",
  "description": "Customer service assistant",
  "agent_blueprint": {
    "identity": {...},
    "capabilities": [...],
    "tools": [...],
    // ... complete blueprint
  },
  "conversation_id": "conv_123"
}
```

### Update Agent Blueprint
```python
POST /org-agents/{agent_id}/blueprints
{
  "agent_blueprint": {
    // ... updated blueprint
  },
  "conversation_id": "conv_456"
}
```

### Activate Blueprint Version
```python
POST /org-agents/{agent_id}/blueprints/{blueprint_id}/activate
```

## Benefits

1. **Conversation-Driven:** Blueprints created through human-AMI discussions
2. **Versioned:** Full history of agent evolution
3. **Comprehensive:** Complete agent definition in structured format
4. **Traceable:** Links to conversations that created each version
5. **Flexible:** Easy to roll back, compare, or branch blueprints
6. **User-Friendly:** Blueprints match the AgentBlueprint.md format

## Next Steps

1. Test the new endpoints thoroughly
2. Update frontend to use blueprint structure
3. Implement conversation→blueprint creation flow
4. Add blueprint comparison and diff features
5. Create blueprint import/export functionality

---

This migration provides a solid foundation for conversation-driven, versioned agent management that aligns with the AgentBlueprint.md vision.