# Deep Reasoning Frontend Integration Guide

## Overview
The Deep Reasoning System provides **Cursor-style multi-step reasoning** with **brain vector reading** capabilities. This guide shows how to integrate the enhanced reasoning features into your frontend application.

## 🧠 What is Deep Reasoning?

Instead of simple tool execution, the AI now:
1. **Reads the user's agent brain vectors** (like Cursor reading source code)
2. **Plans contextual investigation** based on request type
3. **Executes multi-step reasoning** with rich thought streaming
4. **Synthesizes personalized strategies** based on discovered context

## 📡 API Integration

### Enhanced Request Parameters

Add these new parameters to your existing `/tool/llm` requests:

```typescript
interface LLMToolExecuteRequest {
  // ... existing parameters ...
  
  // NEW: Deep reasoning controls
  enable_deep_reasoning?: boolean;     // Enable multi-step reasoning (default: false)
  reasoning_depth?: "light" | "standard" | "deep";  // Reasoning complexity (default: "standard")
  brain_reading_enabled?: boolean;     // Read user's brain vectors (default: true)
  max_investigation_steps?: number;    // Limit reasoning steps (default: 5)
}
```

### Example Request

```javascript
const deepReasoningRequest = {
  llm_provider: "anthropic",
  user_query: "How should I test my agent?",
  cursor_mode: true,                    // Required for deep reasoning
  enable_deep_reasoning: true,          // Enable the new system
  reasoning_depth: "standard",          // Balanced reasoning
  brain_reading_enabled: true,          // Read agent's brain vectors
  max_investigation_steps: 5,           // Up to 5 reasoning steps
  user_id: "user123",                   // Required for brain vector access
  org_id: "org456"                      // Required for brain vector access
};

// Send request
const response = await fetch('/tool/llm', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(deepReasoningRequest)
});
```

## 🎭 Enhanced Thought Streaming

### New Thought Types

The system now streams **rich reasoning thoughts** with these new types:

```typescript
interface ReasoningThought {
  type: "thinking";
  content: string;
  thought_type: 
    | "brain_reading_start"          // 🧠 Starting to read brain vectors
    | "brain_vectors_loaded"         // 📚 Found X knowledge vectors
    | "brain_analysis_complete"      // 💡 Analysis of agent capabilities
    | "investigation_planning"       // 🔍 Planning investigation steps
    | "investigation_execution"      // 🧠 Executing investigation step
    | "discovery_sharing"           // 💡 Sharing what was discovered
    | "strategy_formation"          // 🎯 Synthesizing final strategy
    | "understanding"               // 💭 Initial understanding
    | "intent_analysis"             // 🔍 Intent classification
    | "detailed_analysis";          // 🧠 Detailed reasoning steps
  
  reasoning_step: number | string;    // Step number in reasoning chain
  timestamp: string;                  // ISO timestamp
  metadata?: {                        // Additional context
    vector_count?: number;
    search_context?: string;
    brain_access?: boolean;
  };
}
```

### Frontend Handling

```javascript
const eventSource = new EventSource('/tool/llm');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === "thinking") {
    handleReasoningThought(data);
  }
};

function handleReasoningThought(thought) {
  const { thought_type, content, reasoning_step } = thought;
  
  switch (thought_type) {
    case "brain_reading_start":
      showBrainReadingIndicator();
      appendThought(`🧠 ${content}`, "brain-reading");
      break;
      
    case "brain_vectors_loaded":
      const vectorCount = thought.metadata?.vector_count || 0;
      appendThought(`📚 ${content}`, "brain-analysis");
      updateProgressIndicator(`Analyzing ${vectorCount} knowledge pieces...`);
      break;
      
    case "investigation_planning":
      appendThought(`🔍 ${content}`, "investigation");
      break;
      
    case "strategy_formation":
      appendThought(`🎯 ${content}`, "strategy");
      break;
      
    default:
      appendThought(content, "general");
  }
}
```

## 🎨 UI/UX Recommendations

### 1. Brain Reading Visualization

```css
.brain-reading-indicator {
  display: flex;
  align-items: center;
  padding: 12px;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border-radius: 8px;
  color: white;
  margin-bottom: 16px;
}

.brain-reading-indicator::before {
  content: "🧠";
  margin-right: 8px;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
```

### 2. Reasoning Step Progress

```javascript
function createReasoningProgressBar(totalSteps) {
  return `
    <div class="reasoning-progress">
      <div class="progress-header">
        <span>🔍 Deep Analysis in Progress</span>
        <span class="step-counter">Step 1 of ${totalSteps}</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: 0%"></div>
      </div>
    </div>
  `;
}

function updateReasoningProgress(currentStep, totalSteps) {
  const percentage = (currentStep / totalSteps) * 100;
  document.querySelector('.progress-fill').style.width = `${percentage}%`;
  document.querySelector('.step-counter').textContent = `Step ${currentStep} of ${totalSteps}`;
}
```

### 3. Thought Categorization

```javascript
const thoughtStyles = {
  "brain_reading_start": { icon: "🧠", color: "#667eea", label: "Reading Brain" },
  "brain_vectors_loaded": { icon: "📚", color: "#764ba2", label: "Analyzing Knowledge" },
  "investigation_planning": { icon: "🔍", color: "#f093fb", label: "Planning Investigation" },
  "investigation_execution": { icon: "🧠", color: "#f5576c", label: "Investigating" },
  "discovery_sharing": { icon: "💡", color: "#4facfe", label: "Discovery" },
  "strategy_formation": { icon: "🎯", color: "#43e97b", label: "Strategy" }
};

function renderThought(thought) {
  const style = thoughtStyles[thought.thought_type] || thoughtStyles.default;
  
  return `
    <div class="reasoning-thought" data-type="${thought.thought_type}">
      <div class="thought-header">
        <span class="thought-icon">${style.icon}</span>
        <span class="thought-label" style="color: ${style.color}">${style.label}</span>
        <span class="thought-step">Step ${thought.reasoning_step}</span>
      </div>
      <div class="thought-content">${thought.content}</div>
    </div>
  `;
}
```

## 🔧 Configuration Options

### Reasoning Depth Settings

```javascript
const reasoningConfigs = {
  light: {
    enable_deep_reasoning: true,
    reasoning_depth: "light",
    max_investigation_steps: 2,
    description: "Quick analysis with basic brain reading"
  },
  
  standard: {
    enable_deep_reasoning: true,
    reasoning_depth: "standard", 
    max_investigation_steps: 5,
    description: "Balanced reasoning with comprehensive analysis"
  },
  
  deep: {
    enable_deep_reasoning: true,
    reasoning_depth: "deep",
    max_investigation_steps: 8,
    description: "Thorough investigation with detailed brain analysis"
  }
};

// Let user choose reasoning level
function selectReasoningLevel(level) {
  return {
    ...baseRequest,
    ...reasoningConfigs[level]
  };
}
```

### Conditional Enabling

```javascript
function shouldEnableDeepReasoning(userQuery) {
  const deepReasoningTriggers = [
    /how to test/i,
    /evaluate/i,
    /analyze/i,
    /strategy/i,
    /recommend/i,
    /best approach/i
  ];
  
  return deepReasoningTriggers.some(pattern => pattern.test(userQuery));
}

// Auto-enable for complex queries
const request = {
  ...baseRequest,
  enable_deep_reasoning: shouldEnableDeepReasoning(userQuery),
  cursor_mode: true // Always required for deep reasoning
};
```

## 📊 Analytics & Monitoring

### Track Reasoning Performance

```javascript
function trackReasoningMetrics(thought) {
  if (thought.thought_type === "brain_vectors_loaded") {
    analytics.track("brain_vectors_accessed", {
      vector_count: thought.metadata?.vector_count,
      user_id: currentUser.id
    });
  }
  
  if (thought.thought_type === "strategy_formation") {
    analytics.track("reasoning_completed", {
      total_steps: thought.reasoning_step,
      reasoning_depth: currentRequest.reasoning_depth
    });
  }
}
```

### User Feedback Collection

```javascript
function collectReasoningFeedback() {
  return `
    <div class="reasoning-feedback">
      <h4>How was the deep analysis?</h4>
      <div class="feedback-options">
        <button onclick="submitFeedback('helpful')">🎯 Very helpful</button>
        <button onclick="submitFeedback('good')">👍 Good</button>
        <button onclick="submitFeedback('too_slow')">⏰ Too slow</button>
        <button onclick="submitFeedback('not_relevant')">❌ Not relevant</button>
      </div>
    </div>
  `;
}
```

## 🚀 Example Implementation

### Complete React Component

```jsx
import React, { useState, useEffect } from 'react';

const DeepReasoningChat = () => {
  const [thoughts, setThoughts] = useState([]);
  const [isReasoning, setIsReasoning] = useState(false);
  const [reasoningProgress, setReasoningProgress] = useState({ current: 0, total: 0 });

  const sendDeepReasoningQuery = async (query) => {
    setIsReasoning(true);
    setThoughts([]);

    const request = {
      llm_provider: "anthropic",
      user_query: query,
      cursor_mode: true,
      enable_deep_reasoning: true,
      reasoning_depth: "standard",
      brain_reading_enabled: true,
      max_investigation_steps: 5,
      user_id: "user123",
      org_id: "org456"
    };

    const response = await fetch('/tool/llm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          
          if (data.type === "thinking") {
            setThoughts(prev => [...prev, data]);
            
            if (data.reasoning_step) {
              setReasoningProgress(prev => ({
                ...prev,
                current: Math.max(prev.current, data.reasoning_step)
              }));
            }
          }
        }
      }
    }

    setIsReasoning(false);
  };

  return (
    <div className="deep-reasoning-chat">
      {isReasoning && (
        <div className="reasoning-indicator">
          🧠 Deep analysis in progress... 
          Step {reasoningProgress.current} of {reasoningProgress.total}
        </div>
      )}
      
      <div className="thoughts-container">
        {thoughts.map((thought, index) => (
          <div key={index} className={`thought thought-${thought.thought_type}`}>
            <span className="thought-content">{thought.content}</span>
            <span className="thought-timestamp">{thought.timestamp}</span>
          </div>
        ))}
      </div>
      
      <input 
        type="text" 
        placeholder="Ask about your agent..."
        onKeyPress={(e) => {
          if (e.key === 'Enter') {
            sendDeepReasoningQuery(e.target.value);
          }
        }}
      />
    </div>
  );
};

export default DeepReasoningChat;
```

## 🎯 Best Practices

### 1. **Always Enable cursor_mode**
Deep reasoning requires `cursor_mode: true` to function properly.

### 2. **Provide User Context**
Always include `user_id` and `org_id` for brain vector access.

### 3. **Handle Reasoning Gracefully**
Show progress indicators and allow users to understand what's happening.

### 4. **Collect Feedback**
Monitor which reasoning approaches work best for your users.

### 5. **Performance Considerations**
Deep reasoning takes longer - set appropriate user expectations.

## 🔄 Migration from Simple Mode

### Before (Simple Mode)
```javascript
const request = {
  llm_provider: "anthropic",
  user_query: "How to test my agent?",
  cursor_mode: false
};
```

### After (Deep Reasoning Mode)
```javascript
const request = {
  llm_provider: "anthropic", 
  user_query: "How to test my agent?",
  cursor_mode: true,                    // Required
  enable_deep_reasoning: true,          // Enable new system
  reasoning_depth: "standard",          // Choose depth
  brain_reading_enabled: true,          // Read brain vectors
  user_id: "user123",                   // Required for brain access
  org_id: "org456"                      // Required for brain access
};
```

## 📈 Expected Benefits

1. **More Relevant Responses** - AI reads actual agent knowledge
2. **Transparent Reasoning** - Users see the thinking process
3. **Personalized Guidance** - Recommendations based on specific setup
4. **Better User Trust** - Clear explanation of how conclusions are reached
5. **Improved Outcomes** - Context-aware strategies vs generic advice

---

**Ready to implement?** Start with the basic integration and gradually add the enhanced UI components for the best user experience! 🚀 