# Frontend Guide: Collaborative Agent Creation

## 🎯 **IMPORTANT: New Simplified Workflow**

The `/ami/collaborate` endpoint now handles **both first calls and follow-up calls**. No need for separate draft creation!

---

## ✨ **First Call (No agent_id or blueprint_id needed)**

### Request Structure
```javascript
const startCollaboration = async (userIdea) => {
  const response = await fetch('/ami/collaborate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${userToken}`
    },
    body: JSON.stringify({
      user_input: userIdea,                    // ✅ Required
      conversation_id: generateConversationId(), // ✅ Optional - generate new
      org_id: currentOrgId,                    // ✅ Required
      user_id: currentUserId,                  // ✅ Required
      llm_provider: "anthropic",               // ✅ Optional - defaults to "anthropic"
      current_state: "initial_idea"            // ✅ Optional - defaults to "initial_idea"
      // ❌ NO agent_id or blueprint_id needed!
    })
  });
  
  return response.json();
};

// Example first call
const result = await startCollaboration(
  "Doc excel trong google drive"
);
```

### Response Structure
```json
{
  "success": true,
  "conversation_id": "f80319e3-2846-4ee6-9cc6-3e68db2592eb",
  "current_state": "skeleton_review",
  "ami_message": "I understand you want to work with Excel documents in Google Drive! Let me create an agent for that...",
  "agent_id": "bd88216b-c6e5-452a-b7f0-6ba30ecdfe94",      // ✨ NEW: Use for follow-ups
  "blueprint_id": "18c7fcff-6789-4e6b-9527-0344d1ec2d11",  // ✨ NEW: Use for follow-ups
  "data": {
    "context": {
      "agent_identity": {
        "name": "Doc Excel Agent",
        "purpose": "Handle Excel documents in Google Drive",
        // ... more context
      }
    }
  },
  "next_actions": [
    "Tell me more about what Excel operations you need",
    "Specify Google Drive folder structure",
    "Approve this agent design"
  ]
}
```

---

## 🔄 **Follow-up Calls (Include agent_id and blueprint_id)**

### Request Structure
```javascript
const continueCollaboration = async (userFeedback, conversationId, agentId, blueprintId) => {
  const response = await fetch('/ami/collaborate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${userToken}`
    },
    body: JSON.stringify({
      user_input: userFeedback,                // ✅ Required
      agent_id: agentId,                       // ✅ Required for follow-ups
      blueprint_id: blueprintId,               // ✅ Required for follow-ups
      conversation_id: conversationId,         // ✅ Continue same conversation
      org_id: currentOrgId,
      user_id: currentUserId,
      llm_provider: "anthropic",
      current_state: "skeleton_review"         // ✅ Update based on previous response
    })
  });
  
  return response.json();
};

// Example follow-up call
const result = await continueCollaboration(
  "Thêm khả năng tạo báo cáo từ Excel", 
  "f80319e3-2846-4ee6-9cc6-3e68db2592eb",
  "bd88216b-c6e5-452a-b7f0-6ba30ecdfe94",
  "18c7fcff-6789-4e6b-9527-0344d1ec2d11"
);
```

---

## 💡 **Complete Frontend Implementation**

### State Management
```javascript
const [collaborationSession, setCollaborationSession] = useState({
  conversationId: null,
  agentId: null,
  blueprintId: null,
  currentState: 'initial_idea',
  messages: []
});

const startNewAgent = async (userIdea) => {
  const result = await startCollaboration(userIdea);
  
  if (result.success) {
    setCollaborationSession({
      conversationId: result.conversation_id,
      agentId: result.agent_id,           // ✨ Save for follow-ups
      blueprintId: result.blueprint_id,   // ✨ Save for follow-ups
      currentState: result.current_state,
      messages: [
        { type: 'user', content: userIdea },
        { type: 'ami', content: result.ami_message }
      ]
    });
  }
};

const sendFollowUp = async (userMessage) => {
  const result = await continueCollaboration(
    userMessage,
    collaborationSession.conversationId,
    collaborationSession.agentId,      // ✨ Use saved IDs
    collaborationSession.blueprintId   // ✨ Use saved IDs
  );
  
  if (result.success) {
    setCollaborationSession(prev => ({
      ...prev,
      currentState: result.current_state,
      messages: [
        ...prev.messages,
        { type: 'user', content: userMessage },
        { type: 'ami', content: result.ami_message }
      ]
    }));
  }
};
```

### UI Component
```jsx
const CollaborativeAgentBuilder = () => {
  const [input, setInput] = useState('');
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    setLoading(true);
    try {
      if (!session) {
        // First call - no agent_id/blueprint_id needed
        await startNewAgent(input);
      } else {
        // Follow-up call - use existing IDs
        await sendFollowUp(input);
      }
      setInput('');
    } catch (error) {
      console.error('Collaboration failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="agent-builder">
      <div className="messages">
        {session?.messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.type}`}>
            {msg.content}
          </div>
        ))}
      </div>
      
      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            !session 
              ? "Describe your agent idea..." 
              : "Continue refining your agent..."
          }
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Send'}
        </button>
      </form>
    </div>
  );
};
```

---

## 🚨 **Key Changes for Frontend**

### ✅ **What Works Now**
- **First call**: Send user input only, get agent_id/blueprint_id back
- **Follow-up calls**: Include agent_id/blueprint_id from previous response
- **Automatic draft creation**: Backend handles everything
- **Seamless flow**: No need for separate `/create-draft` endpoint

### ❌ **What's No Longer Needed**
- Manual call to `/create-draft` endpoint
- Complex multi-step initialization
- Managing separate draft creation flow

### 🎯 **Required Response Handling**
Always save `agent_id` and `blueprint_id` from responses:

```javascript
// ✅ Correct handling
const result = await fetch('/ami/collaborate', { ... });
const data = await result.json();

if (data.success) {
  // Save these for follow-up calls
  const agentId = data.agent_id;
  const blueprintId = data.blueprint_id;
  
  // Use in next call
  const followUp = await fetch('/ami/collaborate', {
    body: JSON.stringify({
      user_input: "next message",
      agent_id: agentId,        // ✨ Include in follow-ups
      blueprint_id: blueprintId // ✨ Include in follow-ups
      // ...
    })
  });
}
```

---

## 🔥 **Example Complete Flow**

```javascript
// 1. First call - create agent from idea
const firstCall = await fetch('/ami/collaborate', {
  method: 'POST',
  body: JSON.stringify({
    user_input: "Doc excel trong google drive",
    org_id: "...",
    user_id: "...",
    llm_provider: "anthropic"
    // NO agent_id or blueprint_id
  })
});

const firstResponse = await firstCall.json();
// Gets: agent_id, blueprint_id, conversation_id

// 2. Follow-up call - refine the agent
const secondCall = await fetch('/ami/collaborate', {
  method: 'POST',
  body: JSON.stringify({
    user_input: "Thêm tính năng tạo báo cáo",
    agent_id: firstResponse.agent_id,           // ✨ Include from first response
    blueprint_id: firstResponse.blueprint_id,   // ✨ Include from first response
    conversation_id: firstResponse.conversation_id,
    org_id: "...",
    user_id: "...",
    llm_provider: "anthropic"
  })
});

// 3. Continue until approved...
```

**That's it!** The backend now handles the complexity, frontend just follows this simple pattern! 🎉