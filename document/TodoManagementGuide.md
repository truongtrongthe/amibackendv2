# Frontend Todo Management Guide
**Option A: Dedicated Todo APIs for Agent Creation**

## Overview

This guide explains how the frontend should handle todo management during the 8-step collaborative agent creation process. After blueprint approval (step 4), the system enters the **input collection phase** (steps 5-6) where users complete implementation todos before final compilation.

## 🎯 Complete Flow Architecture

### Steps 1-4: Collaborative Creation
```bash
POST /ami/collaborate  # Initial idea → skeleton → refinement → approval
```

### Steps 5-6: Todo Management (This Guide)
```bash
GET    /org-agents/{agent_id}/blueprints/{blueprint_id}/todos
POST   /org-agents/{agent_id}/blueprints/{blueprint_id}/todos/{todo_id}/collect-inputs
PUT    /org-agents/{agent_id}/blueprints/{blueprint_id}/todos/{todo_id}
```

### Steps 7-8: Final Compilation
```bash
POST /ami/collaborate  # Compile request → completed agent
```

---

## 📋 API Endpoints Reference

### 1. Get All Todos
```http
GET /org-agents/{agent_id}/blueprints/{blueprint_id}/todos
```

**Response:**
```json
{
  "success": true,
  "todos": [
    {
      "id": "todo_1",
      "title": "Configure Google Drive Integration",
      "description": "Set up OAuth credentials for Google Drive access",
      "status": "pending",
      "category": "integration",
      "priority": "high",
      "created_at": "2025-01-08T19:26:46.313346+00:00",
      "input_required": {
        "type": "oauth_credentials",
        "fields": [
          {
            "name": "client_id",
            "type": "text",
            "required": true,
            "description": "Google OAuth Client ID"
          },
          {
            "name": "client_secret",
            "type": "password",
            "required": true,
            "description": "Google OAuth Client Secret"
          },
          {
            "name": "redirect_uri",
            "type": "url",
            "required": true,
            "description": "OAuth redirect URI"
          }
        ]
      },
      "collected_inputs": {},
      "estimated_effort": "15-30 minutes"
    }
  ],
  "statistics": {
    "total": 3,
    "pending": 2,
    "in_progress": 1,
    "completed": 0,
    "cancelled": 0
  }
}
```

### 2. Update Todo Status
```http
PUT /org-agents/{agent_id}/blueprints/{blueprint_id}/todos/{todo_id}
```

**Request:**
```json
{
  "todo_id": "todo_1",
  "new_status": "in_progress"
}
```

**Valid Statuses:** `pending`, `in_progress`, `completed`, `cancelled`

### 3. Collect Todo Inputs (Primary Action)
```http
POST /org-agents/{agent_id}/blueprints/{blueprint_id}/todos/{todo_id}/collect-inputs
```

**Request:**
```json
{
  "collected_inputs": {
    "client_id": "123456789.apps.googleusercontent.com",
    "client_secret": "GOCSPX-abc123xyz789",
    "redirect_uri": "https://myapp.com/oauth/callback",
    "notes": "Configured for production environment"
  }
}
```

**Response:**
```json
{
  "message": "Inputs collected and todo completed successfully. All todos completed! Blueprint is now ready for compilation.",
  "todo_id": "todo_1",
  "new_status": "completed",
  "all_todos_completed": true
}
```

### 4. Validate Inputs (Optional)
```http
POST /org-agents/{agent_id}/blueprints/{blueprint_id}/todos/{todo_id}/validate-inputs
```

---

## 🚀 Frontend Implementation Guide

### Phase 1: Initial Todo Display

When `/ami/collaborate` returns `current_state: "building"`:

```javascript
// 1. Extract todo information from collaborate response
const { agent_id, blueprint_id, data } = collaborateResponse;
const { implementation_todos } = data;

// 2. Display todos in your UI
renderTodoList(implementation_todos);
```

### Phase 2: Todo Management UI

#### A. Todo List Component
```javascript
function TodoList({ agentId, blueprintId, todos }) {
  return (
    <div className="todo-container">
      <h3>Implementation Tasks ({todos.length})</h3>
      {todos.map(todo => (
        <TodoItem 
          key={todo.id}
          agentId={agentId}
          blueprintId={blueprintId}
          todo={todo}
          onComplete={handleTodoComplete}
        />
      ))}
    </div>
  );
}
```

#### B. Individual Todo Component
```javascript
function TodoItem({ agentId, blueprintId, todo, onComplete }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [inputs, setInputs] = useState({});
  const [status, setStatus] = useState(todo.status);

  const handleStatusChange = async (newStatus) => {
    try {
      const response = await fetch(
        `/org-agents/${agentId}/blueprints/${blueprintId}/todos/${todo.id}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            todo_id: todo.id,
            new_status: newStatus
          })
        }
      );
      
      if (response.ok) {
        setStatus(newStatus);
      }
    } catch (error) {
      console.error('Failed to update todo status:', error);
    }
  };

  const handleInputCollection = async () => {
    try {
      const response = await fetch(
        `/org-agents/${agentId}/blueprints/${blueprintId}/todos/${todo.id}/collect-inputs`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            collected_inputs: inputs
          })
        }
      );
      
      const result = await response.json();
      
      if (response.ok) {
        setStatus('completed');
        onComplete(todo.id, result.all_todos_completed);
        
        if (result.all_todos_completed) {
          // All todos done - show compilation option
          showCompilationPrompt();
        }
      } else {
        // Handle validation errors
        showValidationErrors(result.detail);
      }
    } catch (error) {
      console.error('Failed to collect inputs:', error);
    }
  };

  return (
    <div className={`todo-item priority-${todo.priority} status-${status}`}>
      <div className="todo-header" onClick={() => setIsExpanded(!isExpanded)}>
        <StatusBadge status={status} />
        <h4>{todo.title}</h4>
        <PriorityBadge priority={todo.priority} />
      </div>
      
      {isExpanded && (
        <div className="todo-details">
          <p>{todo.description}</p>
          <p><strong>Estimated effort:</strong> {todo.estimated_effort}</p>
          
          {/* Status Controls */}
          <div className="status-controls">
            <button 
              onClick={() => handleStatusChange('in_progress')}
              disabled={status === 'completed'}
            >
              Start Working
            </button>
          </div>
          
          {/* Input Form */}
          {todo.input_required && (
            <div className="input-form">
              <h5>Required Information:</h5>
              {todo.input_required.fields.map(field => (
                <FormField
                  key={field.name}
                  field={field}
                  value={inputs[field.name] || ''}
                  onChange={(value) => setInputs({...inputs, [field.name]: value})}
                />
              ))}
              
              <button 
                onClick={handleInputCollection}
                disabled={status === 'completed'}
                className="complete-todo-btn"
              >
                Complete Task
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

#### C. Form Field Component
```javascript
function FormField({ field, value, onChange }) {
  const inputProps = {
    id: field.name,
    value: value,
    onChange: (e) => onChange(e.target.value),
    required: field.required,
    placeholder: field.description
  };

  switch (field.type) {
    case 'password':
      return (
        <div className="form-field">
          <label htmlFor={field.name}>{field.name}</label>
          <input type="password" {...inputProps} />
        </div>
      );
    
    case 'url':
      return (
        <div className="form-field">
          <label htmlFor={field.name}>{field.name}</label>
          <input type="url" {...inputProps} />
        </div>
      );
    
    case 'textarea':
      return (
        <div className="form-field">
          <label htmlFor={field.name}>{field.name}</label>
          <textarea {...inputProps} rows="3" />
        </div>
      );
    
    default:
      return (
        <div className="form-field">
          <label htmlFor={field.name}>{field.name}</label>
          <input type="text" {...inputProps} />
        </div>
      );
  }
}
```

### Phase 3: Compilation Flow

#### A. Check Compilation Readiness
```javascript
function checkCompilationReadiness(agentId, blueprintId) {
  return fetch(`/org-agents/${agentId}/blueprints/${blueprintId}/todos`)
    .then(response => response.json())
    .then(data => {
      const { statistics } = data;
      return statistics.completed === statistics.total && statistics.total > 0;
    });
}
```

#### B. Trigger Compilation
```javascript
async function compileAgent(conversationId, agentId, blueprintId, conversationHistory) {
  // First check if ready
  const isReady = await checkCompilationReadiness(agentId, blueprintId);
  
  if (!isReady) {
    showError("Please complete all todos before compiling");
    return;
  }

  // Proceed with compilation
  const response = await fetch('/ami/collaborate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_input: "Compile my agent - all todos are completed",
      current_state: "building",
      conversation_id: conversationId,
      agent_id: agentId,
      blueprint_id: blueprintId,
      conversation_history: conversationHistory,
      llm_provider: "openai",
      model: "gpt-4o"
    })
  });

  const result = await response.json();
  
  if (result.current_state === "completed") {
    showSuccess("🎉 Agent compiled successfully!");
    navigateToAgentDashboard(agentId);
  } else if (result.current_state === "building") {
    // Still has pending todos
    showWarning(result.ami_message);
    refreshTodoList();
  }
}
```

---

## 🎨 UI/UX Best Practices

### 1. Visual Status Indicators
```css
.todo-item.status-pending { border-left: 4px solid #fbbf24; }
.todo-item.status-in_progress { border-left: 4px solid #3b82f6; }
.todo-item.status-completed { border-left: 4px solid #10b981; }
.todo-item.status-cancelled { border-left: 4px solid #ef4444; }

.priority-high { background: #fef3c7; }
.priority-critical { background: #fee2e2; }
```

### 2. Progress Tracking
```javascript
function TodoProgress({ statistics }) {
  const progress = (statistics.completed / statistics.total) * 100;
  
  return (
    <div className="todo-progress">
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${progress}%` }}
        />
      </div>
      <span>{statistics.completed}/{statistics.total} completed</span>
    </div>
  );
}
```

### 3. Input Validation Feedback
```javascript
function showValidationErrors(errors) {
  const errorList = Array.isArray(errors) ? errors : [errors];
  
  errorList.forEach(error => {
    toast.error(error, {
      duration: 5000,
      position: 'top-right'
    });
  });
}
```

---

## 🔄 State Management

### Redux/Context State Structure
```javascript
const todoState = {
  todos: [],
  statistics: {
    total: 0,
    pending: 0,
    in_progress: 0,
    completed: 0,
    cancelled: 0
  },
  loading: false,
  error: null,
  compilationReady: false
};

// Actions
const todoActions = {
  FETCH_TODOS_START: 'FETCH_TODOS_START',
  FETCH_TODOS_SUCCESS: 'FETCH_TODOS_SUCCESS',
  FETCH_TODOS_ERROR: 'FETCH_TODOS_ERROR',
  UPDATE_TODO_STATUS: 'UPDATE_TODO_STATUS',
  COMPLETE_TODO: 'COMPLETE_TODO',
  SET_COMPILATION_READY: 'SET_COMPILATION_READY'
};
```

---

## 🚨 Error Handling

### Common Error Scenarios
1. **Invalid inputs** - Show field-specific validation errors
2. **Network errors** - Show retry option
3. **Permission errors** - Redirect to login or show access denied
4. **Todo not found** - Refresh todo list
5. **Compilation attempted too early** - Show pending todos

### Error Response Handling
```javascript
async function handleApiResponse(response) {
  if (!response.ok) {
    const error = await response.json();
    
    switch (response.status) {
      case 400:
        throw new Error(`Validation Error: ${error.detail}`);
      case 403:
        throw new Error('You don\'t have permission to perform this action');
      case 404:
        throw new Error('Todo not found. Please refresh the page.');
      case 500:
        throw new Error('Server error. Please try again later.');
      default:
        throw new Error(`Unexpected error: ${error.detail || 'Unknown error'}`);
    }
  }
  
  return response.json();
}
```

---

## 🎯 Complete Integration Example

```javascript
// Main Todo Management Component
function AgentTodoManager({ conversationId, agentId, blueprintId, conversationHistory }) {
  const [todos, setTodos] = useState([]);
  const [statistics, setStatistics] = useState({});
  const [loading, setLoading] = useState(true);

  // Fetch todos on mount
  useEffect(() => {
    fetchTodos();
  }, [agentId, blueprintId]);

  const fetchTodos = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/org-agents/${agentId}/blueprints/${blueprintId}/todos`);
      const data = await handleApiResponse(response);
      
      setTodos(data.todos);
      setStatistics(data.statistics);
    } catch (error) {
      console.error('Failed to fetch todos:', error);
      showError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTodoComplete = async (todoId, allCompleted) => {
    // Refresh todos to get updated statistics
    await fetchTodos();
    
    if (allCompleted) {
      showSuccess("🎉 All todos completed! Ready to compile your agent.");
      setCompilationReady(true);
    }
  };

  const handleCompile = () => {
    compileAgent(conversationId, agentId, blueprintId, conversationHistory);
  };

  if (loading) return <LoadingSpinner />;

  return (
    <div className="agent-todo-manager">
      <div className="header">
        <h2>Implementation Tasks</h2>
        <TodoProgress statistics={statistics} />
      </div>
      
      <TodoList 
        agentId={agentId}
        blueprintId={blueprintId}
        todos={todos}
        onComplete={handleTodoComplete}
      />
      
      {statistics.completed === statistics.total && statistics.total > 0 && (
        <div className="compilation-section">
          <button 
            onClick={handleCompile}
            className="compile-agent-btn"
          >
            🚀 Compile Agent
          </button>
        </div>
      )}
    </div>
  );
}
```

---

## 📝 Summary Checklist

✅ **Display todos** from collaborate response  
✅ **Implement status updates** (pending → in_progress → completed)  
✅ **Create input forms** based on `input_required` fields  
✅ **Handle input collection** via dedicated API  
✅ **Show progress tracking** with statistics  
✅ **Validate inputs** before submission  
✅ **Handle errors gracefully** with user feedback  
✅ **Check compilation readiness** before allowing compile  
✅ **Trigger final compilation** via collaborate endpoint  
✅ **Navigate to success** when agent is completed  

This approach provides a **robust, user-friendly todo management system** that integrates seamlessly with the collaborative agent creation flow! 🚀
