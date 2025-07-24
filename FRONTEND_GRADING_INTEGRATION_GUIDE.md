# 🎯 Frontend Grading Integration Guide

## Overview

This document provides complete technical and UX guidelines for integrating the **Agent Grading & Capability Showcase** feature into the frontend application.

## 🔌 **Technical Integration**

### **API Endpoint**
The grading feature uses the **existing** `/tool/llm` endpoint - no new endpoints needed!

```javascript
POST /tool/llm
```

### **Triggering Grading Flow**

**Method**: Natural language triggers in user messages

```javascript
// Example API calls that trigger grading
const gradingRequests = [
  "I want to try out my agent's capabilities",
  "Can you test my agent and show its best performance?", 
  "Help me evaluate what my agent can do",
  "Let's grade my agent",
  "Show me what my agent is capable of",
  "Demonstrate my agent's abilities",
  "I want to benchmark my agent"
];

// Standard API call - same as regular chat
const response = await fetch('/tool/llm', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    llm_provider: "openai", // or "anthropic"
    user_query: "I want to try out my agent's capabilities", // Natural trigger
    enable_tools: true,
    cursor_mode: true,
    org_id: currentUser.orgId,
    user_id: currentUser.userId
  })
});
```

### **🔑 Approval Mechanism (CRITICAL)**

**After receiving a grading scenario proposal, the frontend MUST send an approval message to trigger the demonstration:**

```javascript
// STEP 1: User approves scenario via UI button click
const handleScenarioApproval = async (scenarioData) => {
  // STEP 2: Send approval message to same /tool/llm endpoint
  const approvalResponse = await fetch('/tool/llm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      llm_provider: "openai",
      user_query: "Yes, proceed with the grading scenario demonstration",
      // STEP 3: Include scenario data in grading_context
      grading_context: {
        approved_scenario: scenarioData, // The full scenario data from proposal
        approval_action: "execute_demonstration"
      },
      enable_tools: true,
      org_id: currentUser.orgId,
      user_id: currentUser.userId
    })
  });
  
  // STEP 4: Stream the demonstration results
  streamResponse(approvalResponse, handleDemonstrationChunk);
};
```

**⚠️ Important**: Without sending the approval message, the demonstration will **NOT** execute!

### **Response Flow & Types**

The grading flow produces **4 main response types** in sequence:

#### **1. Thinking Steps** (Standard)
```javascript
{
  "type": "thinking",
  "content": "🧠 Analyzing your agent's brain vectors comprehensively...",
  "thought_type": "grading_analysis",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### **2. Grading Scenario Proposal** (New!)
```javascript
{
  "type": "grading_scenario_proposal",
  "content": "🎯 **Optimal Grading Scenario Generated**\n\n**Scenario:** Financial Due Diligence Analysis...",
  "scenario_data": {
    "scenario_name": "Financial Due Diligence Analysis",
    "description": "Analyze company financial statements and provide investment recommendations",
    "agent_role_play": "As FinanceGuru Agent, I specialize in financial analysis...",
    "test_inputs": [
      {"type": "excel_file", "description": "Company financial statements"},
      {"type": "requirements", "description": "Analysis focus areas"}
    ],
    "expected_outputs": [
      {"type": "financial_summary", "description": "Key financial metrics"},
      {"type": "risk_assessment", "description": "Investment risks"},
      {"type": "recommendation", "description": "Investment recommendation"}
    ],
    "showcased_capabilities": ["financial_analysis", "excel_automation", "risk_assessment"],
    "difficulty_level": "intermediate",
    "estimated_time": "15-20 minutes",
    "success_criteria": ["Shows domain expertise", "Processes inputs correctly", "Provides accurate outputs"],
    // NEW: Diagram support (Cursor-style visual diagrams)
    "scenario_diagram": "flowchart TD\n    A[\"Financial Analysis Scenario\"] --> B[\"Agent Analysis\"]\n    B --> C[\"Input Processing\"]...",
    "capability_map": "mindmap\n  root)🤖 Agent Capabilities(\n    Financial Analysis\n      🟢 Excel Automation\n      🟢 Risk Assessment...",
    "process_diagrams": [
      {
        "title": "Input Processing Flow",
        "description": "How the agent processes test inputs using its knowledge",
        "diagram": "sequenceDiagram\n    participant U as User\n    participant A as Agent..."
      }
    ]
  },
  "requires_approval": true,
  "timestamp": "2024-01-15T10:30:15Z"
}
```

#### **3. Agent Demonstration** (After Approval)
```javascript
{
  "type": "agent_demonstration", 
  "content": "🤖 **Agent Introduction:**\n\nAs FinanceGuru Agent, I specialize in financial analysis...",
  "demo_step": "introduction", // "introduction", "capability_visualization", "input_processing_1", "process_diagram_1"
  "demo_data": {
    "step": "input_processing_1",
    "input": {"type": "excel_file", "description": "Financial data"},
    "output": {
      "analysis": "Processing financial data using my knowledge of ratio analysis...",
      "steps_taken": ["Analyzed requirements", "Applied knowledge", "Generated response"],  
      "confidence": 0.85
    }
  },
  // NEW: Diagram data (when demo_step includes diagrams)
  "diagram_data": {
    "type": "capability_map", // "capability_map", "process_flow", "assessment_results"
    "diagram": "mindmap\n  root)🤖 Agent Capabilities(\n    Financial Analysis\n      🟢 Excel Automation...",
    "title": "Agent Capabilities Overview",
    "description": "Visual overview of agent's capabilities"
  },
  "timestamp": "2024-01-15T10:35:00Z"
}
```

#### **4. Grading Assessment** (Final)
```javascript
{
  "type": "grading_assessment",
  "content": "🎯 **Grading Assessment Complete**\n\n**Overall Score:** 82%...",
  "assessment_data": {
    "overall_score": 0.82,
    "criteria_met": 4,
    "total_criteria": 5,
    "strengths": ["Domain expertise", "Structured responses", "Knowledge application"],
    "areas_for_improvement": ["More specific examples", "Faster processing"],
    "recommendation": "Strong performance - excellent capabilities demonstrated"
  },
  // NEW: Assessment diagram data (when included)
  "diagram_data": {
    "type": "assessment_results",
    "diagram": "flowchart TD\n    A[\"Assessment\"] --> B[\"82% Score\"]\n    B --> C{\"Performance Level\"}...",
    "title": "Assessment Results Visualization"
  },
  "timestamp": "2024-01-15T10:40:00Z"
}
```

## 📊 **Diagram Integration (Cursor-Style Visual Enhancement)**

### **Diagram Types & Rendering**

The grading system generates **4 types of Mermaid diagrams** similar to Cursor's diagram capabilities:

#### **1. Scenario Workflow Diagram**
```javascript
// Shows the complete grading scenario flow
const scenarioData = response.scenario_data;
if (scenarioData.scenario_diagram) {
  renderDiagram(scenarioData.scenario_diagram, 'scenario-workflow');
}
```

#### **2. Capability Map Diagram**  
```javascript
// Mindmap showing agent's capabilities
if (scenarioData.capability_map) {
  renderDiagram(scenarioData.capability_map, 'capability-map');
}
```

#### **3. Process Flow Diagrams**
```javascript
// Step-by-step process visualizations
scenarioData.process_diagrams?.forEach((diagram, index) => {
  renderDiagram(diagram.diagram, `process-flow-${index}`, {
    title: diagram.title,
    description: diagram.description
  });
});
```

#### **4. Assessment Results Diagram**
```javascript
// Final assessment visualization
if (chunk.diagram_data?.type === 'assessment_results') {
  renderDiagram(chunk.diagram_data.diagram, 'assessment-results');
}
```

### **Mermaid Integration**

```javascript
// Install Mermaid if not already available
import mermaid from 'mermaid';

// Initialize Mermaid with theme
mermaid.initialize({
  theme: 'default',
  themeVariables: {
    primaryColor: '#ff6b6b',
    primaryTextColor: '#fff',
    primaryBorderColor: '#ff6b6b',
    lineColor: '#333'
  }
});

// Diagram rendering function
const renderDiagram = async (diagramCode, containerId, options = {}) => {
  try {
    const element = document.getElementById(containerId);
    if (!element) return;
    
    // Generate unique ID for diagram
    const diagramId = `diagram-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Render the diagram
    const { svg } = await mermaid.render(diagramId, diagramCode);
    element.innerHTML = `
      <div class="diagram-container">
        ${options.title ? `<h4 class="diagram-title">${options.title}</h4>` : ''}
        ${options.description ? `<p class="diagram-description">${options.description}</p>` : ''}
        <div class="diagram-content">${svg}</div>
      </div>
    `;
    
    // Add interactivity
    addDiagramInteractivity(element);
    
  } catch (error) {
    console.error('Diagram rendering failed:', error);
    // Fallback to text representation
    element.innerHTML = `
      <div class="diagram-fallback">
        <p>📊 Diagram: ${options.title || 'Visual representation'}</p>
        <small>Visual diagram could not be rendered</small>
      </div>
    `;
  }
};

// Add diagram interactivity (zoom, pan, etc.)
const addDiagramInteractivity = (container) => {
  const svg = container.querySelector('svg');
  if (!svg) return;
  
  // Add click-to-zoom functionality
  svg.addEventListener('click', () => {
    svg.classList.toggle('diagram-zoomed');
  });
  
  // Add responsive behavior
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', 'auto');
};
```

### **Diagram Component Examples**

#### **Scenario Proposal with Diagrams**
```javascript
const ScenarioProposalWithDiagrams = ({ scenarioData, onApprove, onReject }) => {
  const [activeTab, setActiveTab] = useState('overview');
  
  return (
    <div className="scenario-proposal-card enhanced">
      {/* Standard proposal content */}
      <div className="scenario-overview">
        <h4>{scenarioData.scenario_name}</h4>
        <p>{scenarioData.description}</p>
      </div>

      {/* NEW: Diagram tabs */}
      <div className="diagram-tabs">
        <button 
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          📋 Overview
        </button>
        <button 
          className={`tab ${activeTab === 'workflow' ? 'active' : ''}`}
          onClick={() => setActiveTab('workflow')}
        >
          🔄 Workflow
        </button>
        <button 
          className={`tab ${activeTab === 'capabilities' ? 'active' : ''}`}
          onClick={() => setActiveTab('capabilities')}
        >
          🧠 Capabilities
        </button>
      </div>

      <div className="diagram-content">
        {activeTab === 'workflow' && (
          <div id="scenario-workflow" className="diagram-panel">
            {/* Mermaid diagram will be rendered here */}
          </div>
        )}
        
        {activeTab === 'capabilities' && (
          <div id="capability-map" className="diagram-panel">
            {/* Capability map will be rendered here */}
          </div>
        )}
      </div>

      {/* Approval actions */}
      <div className="approval-actions">
        <button className="btn-primary" onClick={onApprove}>
          ✅ Start Visual Demo
        </button>
        <button className="btn-secondary" onClick={onReject}>
          🔄 Generate Different Scenario
        </button>
      </div>
    </div>
  );
};

// Effect to render diagrams when component mounts
useEffect(() => {
  if (scenarioData.scenario_diagram && activeTab === 'workflow') {
    renderDiagram(scenarioData.scenario_diagram, 'scenario-workflow', {
      title: 'Scenario Workflow'
    });
  }
  
  if (scenarioData.capability_map && activeTab === 'capabilities') {
    renderDiagram(scenarioData.capability_map, 'capability-map', {
      title: 'Agent Capabilities'
    });
  }
}, [activeTab, scenarioData]);
```

#### **Enhanced Agent Demonstration**
```javascript
const AgentDemonstrationWithDiagrams = ({ demoSteps }) => {
  return (
    <div className="agent-demonstration enhanced">
      {demoSteps.map((step, index) => (
        <div key={index} className={`demo-step ${step.demo_step}`}>
          <div className="step-content">
            <ReactMarkdown>{step.content}</ReactMarkdown>
          </div>
          
          {/* NEW: Diagram rendering for visual steps */}
          {step.diagram_data && (
            <div className="step-diagram">
              <div 
                id={`demo-diagram-${index}`}
                className={`diagram-container ${step.diagram_data.type}`}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

// Effect to render step diagrams
useEffect(() => {
  demoSteps.forEach((step, index) => {
    if (step.diagram_data) {
      renderDiagram(step.diagram_data.diagram, `demo-diagram-${index}`, {
        title: step.diagram_data.title,
        description: step.diagram_data.description
      });
    }
  });
}, [demoSteps]);
```

## 🎨 **UX Guidelines**

### **Phase 1: Grading Detection & Analysis**

#### **Visual Design**
```javascript
// Show thinking steps with grading-specific icons
const GradingThoughts = ({ thought }) => (
  <div className="thinking-step grading-analysis">
    <div className="icon">🧠</div>
    <div className="content">
      <span className="thought-type">Capability Analysis</span>
      <p>{thought.content}</p>
    </div>
    <div className="progress-indicator">
      <div className="spinner" />
    </div>
  </div>
);
```

#### **Progress Indicators**
```javascript
const gradingSteps = [
  { step: 1, label: "Analyzing Capabilities", icon: "🧠", status: "active" },
  { step: 2, label: "Designing Scenario", icon: "🎯", status: "pending" },
  { step: 3, label: "Agent Demonstration", icon: "🤖", status: "pending" },
  { step: 4, label: "Performance Assessment", icon: "📊", status: "pending" }
];

<ProgressTracker steps={gradingSteps} currentStep={1} />
```

### **Phase 2: Scenario Proposal & Approval**

#### **Scenario Proposal Card**
```javascript
const ScenarioProposal = ({ scenarioData, onApprove, onReject }) => (
  <div className="scenario-proposal-card">
    <div className="header">
      <h3>🎯 Optimal Grading Scenario</h3>
      <span className="difficulty-badge">{scenarioData.difficulty_level}</span>
    </div>
    
    <div className="scenario-overview">
      <h4>{scenarioData.scenario_name}</h4>
      <p className="description">{scenarioData.description}</p>
      
      <div className="meta-info">
        <span className="time">⏱️ {scenarioData.estimated_time}</span>
        <span className="capabilities">
          🎖️ {scenarioData.showcased_capabilities.length} capabilities
        </span>
      </div>
    </div>

    <div className="agent-introduction">
      <h5>🤖 Your Agent Will Say:</h5>
      <blockquote>{scenarioData.agent_role_play}</blockquote>
    </div>

    <div className="test-details">
      <div className="inputs">
        <h5>📋 Test Components:</h5>
        <ul>
          {scenarioData.test_inputs.map((input, i) => (
            <li key={i}>{input.description}</li>
          ))}
        </ul>
      </div>
      
      <div className="outputs">
        <h5>✅ Expected Demonstrations:</h5>
        <ul>
          {scenarioData.expected_outputs.map((output, i) => (
            <li key={i}>{output.description}</li>
          ))}
        </ul>
      </div>
    </div>

    <div className="capabilities-showcase">
      <h5>🎖️ Capabilities Showcased:</h5>
      <div className="capability-tags">
        {scenarioData.showcased_capabilities.map(cap => (
          <span key={cap} className="capability-tag">{cap}</span>
        ))}
      </div>
    </div>

    <div className="approval-actions">
      <button 
        className="btn-primary approve-btn"
        onClick={onApprove}
      >
        ✅ Start Grading Demo
      </button>
      <button 
        className="btn-secondary reject-btn" 
        onClick={onReject}
      >
        🔄 Generate Different Scenario
      </button>
    </div>
  </div>
);
```

#### **Approval Flow State Management**
```javascript
const [gradingState, setGradingState] = useState({
  phase: 'analyzing', // 'analyzing' | 'proposal' | 'demonstration' | 'assessment'
  scenarioData: null,
  approvalRequired: false,
  demonstrationSteps: [],
  finalAssessment: null
});

const handleApproval = async (scenarioData) => {
  // Update UI state immediately
  setGradingState(prev => ({
    ...prev,
    phase: 'demonstration',
    approvalRequired: false
  }));
  
  // NEW: Send approval message to backend with scenario context
  const approvalResponse = await fetch('/tool/llm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      llm_provider: "openai", // or user's preferred provider
      user_query: `Execute the approved grading scenario: ${scenarioData.scenario_name}`,
      grading_context: {
        approved_scenario: scenarioData,
        approval_action: "execute_demonstration",
        test_inputs: {} // Optional: custom test inputs
      },
      enable_tools: true,
      cursor_mode: true,
      org_id: currentUser.orgId,
      user_id: currentUser.userId
    })
  });
  
  // Stream the demonstration response
  const reader = approvalResponse.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          handleStreamChunk(data);
        } catch (e) {
          console.error('Failed to parse SSE data:', e);
        }
      }
    }
  }
};
```

### **Phase 3: Agent Demonstration**

#### **Demonstration Display**
```javascript
const AgentDemonstration = ({ demoSteps }) => (
  <div className="agent-demonstration">
    <div className="demo-header">
      <h3>🤖 Agent Demonstration in Progress</h3>
      <div className="agent-avatar">
        <img src="/icons/agent-avatar.png" alt="Agent" />
        <span className="status-indicator active" />
      </div>
    </div>

    <div className="demo-timeline">
      {demoSteps.map((step, index) => (
        <div key={index} className={`demo-step ${step.demo_step}`}>
          <div className="step-header">
            <span className="step-number">{index + 1}</span>
            <h4>{step.demo_step.replace('_', ' ').title()}</h4>
          </div>
          
          <div className="step-content">
            <ReactMarkdown>{step.content}</ReactMarkdown>
          </div>
          
          {step.demo_data && (
            <div className="step-details">
              <div className="confidence-meter">
                <span>Confidence: {(step.demo_data.output?.confidence * 100 || 0).toFixed(0)}%</span>
                <div className="meter">
                  <div 
                    className="fill" 
                    style={{ width: `${step.demo_data.output?.confidence * 100 || 0}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  </div>
);
```

#### **Real-time Demo Updates**
```javascript
// Handle streaming demo steps
useEffect(() => {
  if (chunk.type === 'agent_demonstration') {
    setGradingState(prev => ({
      ...prev,
      demonstrationSteps: [...prev.demonstrationSteps, chunk]
    }));
    
    // Auto-scroll to latest step
    scrollToBottom();
    
    // Show step completion animation
    animateStepCompletion(chunk.demo_step);
  }
}, [chunk]);
```

### **Phase 4: Assessment Results**

#### **Assessment Dashboard**
```javascript
const AssessmentResults = ({ assessment }) => (
  <div className="assessment-results">
    <div className="score-section">
      <div className="overall-score">
        <div className="score-circle">
          <CircularProgress 
            value={assessment.overall_score * 100}
            color={getScoreColor(assessment.overall_score)}
          />
          <span className="score-text">
            {(assessment.overall_score * 100).toFixed(0)}%
          </span>
        </div>
        <h3>Overall Performance</h3>
      </div>
      
      <div className="criteria-breakdown">
        <h4>Criteria Met: {assessment.criteria_met}/{assessment.total_criteria}</h4>
        <div className="criteria-bar">
          <div 
            className="fill"
            style={{ width: `${(assessment.criteria_met / assessment.total_criteria) * 100}%` }}
          />
        </div>
      </div>
    </div>

    <div className="strengths-section">
      <h4>✅ Strengths</h4>
      <ul className="strengths-list">
        {assessment.strengths.map((strength, i) => (
          <li key={i} className="strength-item">
            <span className="icon">🎯</span>
            {strength}
          </li>
        ))}
      </ul>
    </div>

    <div className="improvements-section">
      <h4>🔄 Areas for Improvement</h4>
      <ul className="improvements-list">
        {assessment.areas_for_improvement.map((area, i) => (
          <li key={i} className="improvement-item">
            <span className="icon">💡</span>
            {area}
          </li>
        ))}
      </ul>
    </div>

    <div className="recommendation">
      <h4>📋 Recommendation</h4>
      <p className="recommendation-text">{assessment.recommendation}</p>
    </div>

    <div className="actions">
      <button className="btn-primary" onClick={shareResults}>
        📤 Share Results
      </button>
      <button className="btn-secondary" onClick={runAnotherTest}>
        🔄 Run Another Test
      </button>
    </div>
  </div>
);
```

## 🎯 **Complete Integration Example**

```javascript
import React, { useState, useEffect } from 'react';

const GradingInterface = () => {
  const [gradingState, setGradingState] = useState({
    phase: 'idle',
    scenarioData: null,
    demonstrationSteps: [],
    assessment: null,
    isLoading: false
  });

  const handleGradingRequest = async (message) => {
    setGradingState(prev => ({ ...prev, phase: 'analyzing', isLoading: true }));
    
    const response = await fetch('/tool/llm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        llm_provider: "openai",
        user_query: message, // e.g., "try out my agent"
        enable_tools: true,
        cursor_mode: true,
        org_id: user.orgId,
        user_id: user.userId
      })
    });

    const stream = response.body.getReader();
    
    while (true) {
      const { done, value } = await stream.read();
      if (done) break;
      
      const chunk = JSON.parse(new TextDecoder().decode(value));
      
      switch (chunk.type) {
        case 'thinking':
          // Show thinking steps with special grading styling
          break;
          
        case 'grading_scenario_proposal':
          setGradingState(prev => ({
            ...prev,
            phase: 'proposal',
            scenarioData: chunk.scenario_data,
            isLoading: false
          }));
          break;
          
        case 'agent_demonstration':
          setGradingState(prev => ({
            ...prev,
            phase: 'demonstration',
            demonstrationSteps: [...prev.demonstrationSteps, chunk]
          }));
          break;
          
        case 'grading_assessment':
          setGradingState(prev => ({
            ...prev,
            phase: 'assessment',
            assessment: chunk.assessment_data,
            isLoading: false
          }));
          break;
      }
    }
  };

  const handleScenarioApproval = async (scenarioData) => {
  // Update UI state immediately
  setGradingState(prev => ({ ...prev, phase: 'demonstration' }));
  
  // CRITICAL: Send approval to backend to trigger demonstration
  const approvalResponse = await fetch('/tool/llm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      llm_provider: currentUser.preferredProvider || "openai",
      user_query: `Yes, proceed with the grading scenario demonstration`,
      grading_context: {
        approved_scenario: scenarioData,
        approval_action: "execute_demonstration"
      },
      enable_tools: true,
      org_id: currentUser.orgId,
      user_id: currentUser.userId
    })
  });
  
  // Stream demonstration results
  streamResponse(approvalResponse, handleStreamChunk);
};

  return (
    <div className="grading-interface">
      {gradingState.phase === 'analyzing' && (
        <GradingAnalysis isLoading={gradingState.isLoading} />
      )}
      
      {gradingState.phase === 'proposal' && (
        <ScenarioProposal 
          scenarioData={gradingState.scenarioData}
          onApprove={handleScenarioApproval}
          onReject={() => {/* Generate new scenario */}}
        />
      )}
      
      {gradingState.phase === 'demonstration' && (
        <AgentDemonstration 
          demoSteps={gradingState.demonstrationSteps}
        />
      )}
      
      {gradingState.phase === 'assessment' && (
        <AssessmentResults 
          assessment={gradingState.assessment}
        />
      )}
    </div>
  );
};
```

## 🚨 **Error Handling**

```javascript
// Handle grading-specific errors
const handleGradingError = (error) => {
  const errorMessages = {
    'capability_analysis_failed': 'Could not analyze your agent\'s capabilities. Please try again.',
    'scenario_generation_failed': 'Failed to generate grading scenario. Please try again.',
    'demonstration_failed': 'Agent demonstration failed. Please check your agent\'s configuration.',
    'no_capabilities_found': 'No capabilities found for your agent. Please add more knowledge first.'
  };
  
  showToast({
    type: 'error',
    title: 'Grading Failed',
    message: errorMessages[error.code] || 'An unexpected error occurred.',
    action: { text: 'Try Again', onClick: retryGrading }
  });
};
```

## 📱 **Mobile Responsiveness**

```css
/* Mobile-first grading interface */
.scenario-proposal-card {
  @media (max-width: 768px) {
    padding: 16px;
    margin: 8px;
    
    .approval-actions {
      flex-direction: column;
      gap: 12px;
      
      button {
        width: 100%;
        padding: 14px;
      }
    }
  }
}

.assessment-results {
  @media (max-width: 768px) {
    .score-section {
      flex-direction: column;
      text-align: center;
    }
    
    .overall-score .score-circle {
      width: 120px;
      height: 120px;
    }
  }
}
```

## 🎯 **Key UX Principles**

1. **Progressive Disclosure**: Show information in logical phases
2. **Clear Progress**: Always show where user is in the grading flow  
3. **Engaging Animations**: Use subtle animations for step transitions
4. **Human Approval**: Always require explicit approval before demonstrations
5. **Results Sharing**: Make it easy to share impressive results
6. **Error Recovery**: Provide clear paths to retry on failures
7. **Mobile Optimization**: Ensure great experience on all devices

## 🎨 **Diagram Styling**

```css
/* Diagram container styling */
.diagram-container {
  margin: 20px 0;
  padding: 16px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background: #fafafa;
}

.diagram-title {
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.diagram-description {
  margin: 0 0 16px 0;
  font-size: 14px;
  color: #666;
}

.diagram-content svg {
  width: 100%;
  height: auto;
  cursor: pointer;
  transition: transform 0.3s ease;
}

.diagram-content svg.diagram-zoomed {
  transform: scale(1.2);
}

/* Diagram tabs styling */
.diagram-tabs {
  display: flex;
  gap: 8px;
  margin: 16px 0;
  border-bottom: 1px solid #e0e0e0;
}

.diagram-tabs .tab {
  padding: 8px 16px;
  border: none;
  background: none;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}

.diagram-tabs .tab.active {
  border-bottom-color: #007bff;
  color: #007bff;
  font-weight: 600;
}

.diagram-tabs .tab:hover {
  background: #f5f5f5;
}

/* Diagram panels */
.diagram-panel {
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Diagram type specific styling */
.diagram-container.capability_map {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.diagram-container.process_flow {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
}

.diagram-container.assessment_results {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  color: white;
}

/* Fallback styling */
.diagram-fallback {
  text-align: center;
  padding: 40px;
  color: #666;
  background: #f9f9f9;
  border-radius: 8px;
}

/* Mobile responsive */
@media (max-width: 768px) {
  .diagram-container {
    margin: 16px 0;
    padding: 12px;
  }
  
  .diagram-tabs {
    flex-wrap: wrap;
    gap: 4px;
  }
  
  .diagram-tabs .tab {
    padding: 6px 12px;
    font-size: 14px;
  }
  
  .diagram-content svg.diagram-zoomed {
    transform: scale(1.1);
  }
}
```

## 🔧 **CSS Classes Reference**

```css
/* Core grading interface classes */
.grading-interface { /* Main container */ }
.scenario-proposal-card { /* Scenario proposal styling */ }
.scenario-proposal-card.enhanced { /* Enhanced with diagrams */ }
.agent-demonstration { /* Demo display container */ }
.agent-demonstration.enhanced { /* Enhanced with diagrams */ }
.assessment-results { /* Final results dashboard */ }
.capability-tag { /* Individual capability badges */ }
.demo-step { /* Individual demonstration steps */ }
.step-diagram { /* Diagram within demo steps */ }
.score-circle { /* Circular progress for scores */ }
.approval-actions { /* Approval button container */ }

/* NEW: Diagram-specific classes */
.diagram-container { /* Diagram wrapper */ }
.diagram-title { /* Diagram title */ }
.diagram-description { /* Diagram description */ }
.diagram-content { /* Diagram SVG container */ }
.diagram-tabs { /* Tab navigation for diagrams */ }
.diagram-panel { /* Individual diagram panels */ }
.diagram-fallback { /* Fallback when diagram fails */ }
```

This integration provides a seamless, engaging experience for users to test and showcase their agent's capabilities! 🚀 