# Frontend Agent Name Management Guide

## 🎯 **Overview**

This guide explains how to handle agent name changes in the frontend. There are **two approaches** available, both fully supported and can be used simultaneously.

## 🏗️ **System Architecture**

### **Agent Name Storage**
Agent names are stored in **two places** that must stay synchronized:
- **Agent Table**: `org_agents.name` - The main agent record
- **Blueprint**: `agent_blueprint.identity.name` - The blueprint configuration

### **Automatic Synchronization**
As of the latest update, the system **automatically keeps both in sync** during refinement mode.

---

## 🔄 **Approach 1: Refinement Mode Name Changes**

### **How It Works**
Users can change agent names naturally through conversation during refinement mode. The system will automatically update both the blueprint and agent table.

### **Frontend Implementation**

#### **1. Detect When User Wants to Change Name**
```javascript
// User inputs during refinement mode:
const nameChangeRequests = [
  "Change the agent name to TechAlert",
  "Rename it to LogMonitor", 
  "Call it SlackBot instead",
  "I want to name it DataWatcher"
];

// Send to refinement endpoint with agent_id + blueprint_id
const refinementRequest = {
  user_input: "Change the agent name to TechAlert",
  agent_id: "8207c163-b19e-4e16-8c50-4fbff3e5e1fa",
  blueprint_id: "d5b6ccea-c66f-4cb2-a897-e9f08a50937e",
  conversation_history: [...], // Include recent messages
  org_id: "org-id",
  user_id: "user-id"
};
```

#### **2. Handle Name Change Response**
```javascript
const response = await fetch('/ami/collaborate', {
  method: 'POST',
  body: JSON.stringify(refinementRequest)
});

const result = await response.json();

if (result.success) {
  // Check if name was changed in the response
  const changes = result.data?.changes_made || [];
  const nameChanged = changes.some(change => 
    change.toLowerCase().includes('name') || 
    change.toLowerCase().includes('renamed')
  );
  
  if (nameChanged) {
    console.log('✅ Agent name updated automatically');
    // Refresh agent list or update UI
    await refreshAgentData(result.agent_id);
  }
}
```

#### **3. Backend Logs to Monitor**
```
🤝 [COLLAB] Agent name changed from 'LogWatch' to 'TechAlert' - syncing agent table
🤝 [COLLAB] Agent table updated successfully with new name: TechAlert
```

### **Advantages**
- ✅ **Natural UX**: Users just ask in conversation
- ✅ **Automatic Sync**: Backend handles both blueprint and agent table  
- ✅ **Context Aware**: LLM understands when name changes are requested
- ✅ **Conversation Flow**: Part of natural refinement process

### **Limitations**
- ⚠️ **LLM Dependent**: Relies on LLM understanding the request
- ⚠️ **Less Precise**: User intent interpretation may vary

---

## ⚙️ **Approach 2: Direct API Name Changes**

### **How It Works**
Direct API call to update agent name immediately with precise control.

### **Frontend Implementation**

#### **1. Direct Name Update**
```javascript
const updateAgentName = async (agentId, newName) => {
  try {
    const response = await fetch(`/org-agents/${agentId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify({
        name: newName  // Only update name, leave other fields unchanged
      })
    });
    
    if (response.ok) {
      const updatedAgent = await response.json();
      console.log('✅ Agent name updated:', updatedAgent.name);
      return updatedAgent;
    } else {
      throw new Error('Failed to update agent name');
    }
  } catch (error) {
    console.error('❌ Name update failed:', error);
    throw error;
  }
};
```

#### **2. Permission Handling**
```javascript
const updateAgentNameWithPermissionCheck = async (agentId, newName) => {
  try {
    const result = await updateAgentName(agentId, newName);
    return result;
  } catch (error) {
    if (error.status === 403) {
      showError('You need owner or admin permissions to rename agents');
    } else if (error.status === 404) {
      showError('Agent not found');
    } else {
      showError('Failed to update agent name');
    }
    throw error;
  }
};
```

#### **3. UI Integration Example**
```javascript
// In agent settings or profile component
const AgentNameEditor = ({ agent, onNameChange }) => {
  const [name, setName] = useState(agent.name);
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    if (name.trim() === agent.name) {
      setIsEditing(false);
      return;
    }

    setIsSaving(true);
    try {
      const updatedAgent = await updateAgentName(agent.id, name.trim());
      onNameChange(updatedAgent);
      setIsEditing(false);
      showSuccess(`Agent renamed to "${updatedAgent.name}"`);
    } catch (error) {
      setName(agent.name); // Reset on error
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="agent-name-editor">
      {isEditing ? (
        <div className="name-input-group">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSave()}
            disabled={isSaving}
            maxLength={255}
          />
          <button onClick={handleSave} disabled={isSaving}>
            {isSaving ? 'Saving...' : 'Save'}
          </button>
          <button onClick={() => {
            setName(agent.name);
            setIsEditing(false);
          }}>
            Cancel
          </button>
        </div>
      ) : (
        <div className="name-display" onClick={() => setIsEditing(true)}>
          <span>{agent.name}</span>
          <EditIcon />
        </div>
      )}
    </div>
  );
};
```

### **Advantages**
- ✅ **Immediate**: Instant name update
- ✅ **Precise Control**: Exact name specified
- ✅ **Error Handling**: Clear error responses
- ✅ **Permissions**: Built-in permission checking

### **Limitations**
- ⚠️ **Blueprint Sync**: Only updates agent table, not blueprint
- ⚠️ **Permission Required**: Only owners/admins can use this
- ⚠️ **Context Loss**: No conversation context

---

## 🔀 **Hybrid Approach (Recommended)**

### **Best Practice Implementation**
Use both approaches strategically:

```javascript
const AgentManager = {
  // For conversation-based refinement
  async refineAgent(agentId, blueprintId, userInput, conversationHistory) {
    return await fetch('/ami/collaborate', {
      method: 'POST',
      body: JSON.stringify({
        user_input: userInput,
        agent_id: agentId,
        blueprint_id: blueprintId,
        conversation_history: conversationHistory,
        // ... other fields
      })
    });
  },
  
  // For direct administrative changes
  async updateAgentName(agentId, newName) {
    return await fetch(`/org-agents/${agentId}`, {
      method: 'PUT',
      body: JSON.stringify({ name: newName })
    });
  },
  
  // Smart name change handler
  async handleNameChange(context) {
    const { agentId, blueprintId, newName, isInRefinementMode, userRole } = context;
    
    if (isInRefinementMode) {
      // Use conversational approach during active refinement
      return await this.refineAgent(
        agentId, 
        blueprintId, 
        `Change the agent name to ${newName}`,
        context.conversationHistory
      );
    } else if (userRole === 'owner' || userRole === 'admin') {
      // Use direct API for quick administrative changes
      return await this.updateAgentName(agentId, newName);
    } else {
      throw new Error('Permission denied: Only owners and admins can rename agents');
    }
  }
};
```

---

## 🔧 **Technical Details**

### **Database Schema**
```sql
-- Agent table (main record)
org_agents:
  id: uuid (primary key)
  name: varchar(255)  -- This gets updated by both approaches
  description: text
  status: varchar(50)
  
-- Blueprint table (configuration)
agent_blueprints:
  id: uuid (primary key)
  agent_id: uuid (foreign key)
  agent_blueprint: jsonb
    └── identity: 
        └── name: string  -- This gets updated during refinement
```

### **API Endpoints**

#### **Refinement Endpoint**
```
POST /ami/collaborate
- Updates: blueprint.identity.name AND org_agents.name (automatic sync)
- Permissions: Any user with agent access
- Context: Full conversation awareness
```

#### **Direct Update Endpoint**  
```
PUT /org-agents/{agent_id}
- Updates: org_agents.name only
- Permissions: Organization owners and admins only
- Context: Direct administrative action
```

### **Synchronization Logic**
```python
# In ami/collaborative_creator.py
if new_agent_name and new_agent_name != agent.name:
    collab_logger.info(f"Agent name changed from '{agent.name}' to '{new_agent_name}' - syncing agent table")
    updated_agent = update_agent(agent.id, name=new_agent_name)
    if updated_agent:
        collab_logger.info(f"Agent table updated successfully with new name: {new_agent_name}")
        agent.name = new_agent_name  # Update local object for response
```

---

## 🧪 **Testing Scenarios**

### **Test Case 1: Refinement Name Change**
```javascript
// 1. Create agent through conversation
// 2. Enter refinement mode with agent_id + blueprint_id  
// 3. Send: "Rename it to SuperBot"
// 4. Verify: Both agent.name and blueprint.identity.name updated
// 5. Check logs for sync confirmation
```

### **Test Case 2: Direct API Name Change**
```javascript
// 1. Call PUT /org-agents/{id} with new name
// 2. Verify: agent.name updated immediately
// 3. Note: blueprint.identity.name remains unchanged (expected)
```

### **Test Case 3: Permission Testing**
```javascript
// 1. Try direct API update as non-admin user
// 2. Expect: 403 Forbidden error
// 3. Try refinement approach as regular user  
// 4. Expect: Success (refinement allows all users)
```

---

## 📝 **Frontend Checklist**

### **Implementation Checklist**
- [ ] **Approach 1**: Handle refinement-based name changes
- [ ] **Approach 2**: Implement direct API name updates
- [ ] **Permissions**: Check user role before direct updates
- [ ] **Error Handling**: Handle 403, 404, and 500 errors
- [ ] **UI Updates**: Refresh agent data after name changes
- [ ] **Loading States**: Show saving/updating indicators
- [ ] **Validation**: Enforce name length limits (255 chars)
- [ ] **Confirmation**: Ask user confirmation for direct changes

### **Testing Checklist**
- [ ] **Happy Path**: Both approaches work correctly
- [ ] **Permissions**: Direct API respects role permissions
- [ ] **Edge Cases**: Empty names, very long names, special characters
- [ ] **Error Recovery**: UI handles API failures gracefully
- [ ] **Data Sync**: Verify both storage locations stay synchronized
- [ ] **Concurrent Changes**: Handle multiple users editing same agent

---

## 🚨 **Important Notes**

1. **Synchronization**: Refinement mode now automatically syncs both locations
2. **Permissions**: Direct API requires owner/admin, refinement mode allows all users
3. **Context**: Refinement preserves conversation context, direct API doesn't
4. **Error Handling**: Always handle permission errors gracefully in UI
5. **User Experience**: Use refinement for natural flow, direct API for admin tasks

## 📊 **Monitoring**

Watch for these log messages to monitor name changes:
```
🤝 [COLLAB] Agent name changed from 'OldName' to 'NewName' - syncing agent table
🤝 [COLLAB] Agent table updated successfully with new name: NewName
```

Any sync failures will show:
```
🤝 [COLLAB] Failed to update agent table with new name: NewName
```