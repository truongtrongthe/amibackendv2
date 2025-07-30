# Frontend Event Types - Modular Agent Architecture

## 🎯 **New Event Types from Modular Architecture**

The modular agent architecture introduces new SSE event types that provide better visibility into multi-step execution. Your frontend currently shows "Unknown event type" for these - here's how to handle them.

## 📡 **Event Types Reference**

### **1. `plan` Event**
**Purpose**: Notifies when execution plan is generated

**Structure:**
```typescript
{
  type: 'plan',
  content: '✅ Execution plan generated: 5 steps',
  execution_plan: {
    plan_id: string,
    total_steps: number,
    complexity_score: number,
    complexity_level: string,
    estimated_time: string,
    execution_steps: Array<{
      step_number: number,
      name: string,
      description: string,
      tools_needed: string[],
      estimated_time: string
    }>
  }
}
```

**Frontend Handling:**
```typescript
case 'plan':
  // Show execution plan UI
  setExecutionPlan(chunk.execution_plan);
  addMessage({
    type: 'system',
    content: chunk.content,
    metadata: { 
      type: 'plan',
      steps: chunk.execution_plan.total_steps,
      complexity: chunk.execution_plan.complexity_level
    }
  });
  break;
```

### **2. `step_start` Event**
**Purpose**: Indicates a step is beginning execution

**Structure:**
```typescript
{
  type: 'step_start',
  content: '🔄 Step 1/5: Context Analysis & Knowledge Activation',
  provider: 'openai',
  agent_id: string,
  step_info: {
    step_number: number,
    name: string,
    description: string,
    estimated_time: string,
    tools_needed: string[],
    dependencies: number[]
  }
}
```

**Frontend Handling:**
```typescript
case 'step_start':
  // Update step progress UI
  setCurrentStep(chunk.step_info.step_number);
  setStepStatus(chunk.step_info.step_number, 'in_progress');
  
  addMessage({
    type: 'system',
    content: chunk.content,
    metadata: {
      type: 'step_start',
      step: chunk.step_info
    }
  });
  break;
```

### **3. `step_error` Event**
**Purpose**: Reports step execution failure

**Structure:**
```typescript
{
  type: 'step_error',
  content: '❌ Step 1 failed: Context Analysis & Knowledge Activation - Error details',
  provider: 'openai',
  agent_id: string,
  step_result: {
    step_number: number,
    name: string,
    status: 'failed',
    error: string,
    execution_time: number
  }
}
```

**Frontend Handling:**
```typescript
case 'step_error':
  // Mark step as failed
  setStepStatus(chunk.step_result.step_number, 'failed');
  
  addMessage({
    type: 'error',
    content: chunk.content,
    metadata: {
      type: 'step_error',
      step_number: chunk.step_result.step_number,
      error: chunk.step_result.error
    }
  });
  break;
```

### **4. `step_complete` Event**
**Purpose**: Confirms successful step completion

**Structure:**
```typescript
{
  type: 'step_complete',
  content: '✅ Step 1 completed: Context Analysis (2.3s)',
  provider: 'openai',
  agent_id: string,
  step_result: {
    step_number: number,
    name: string,
    status: 'completed',
    execution_time: number,
    tools_used: string[],
    success_criteria_met: boolean
  }
}
```

### **5. `checkpoint` Event**
**Purpose**: Quality checkpoint validation

**Structure:**
```typescript
{
  type: 'checkpoint',
  content: '🔍 Quality Checkpoint: Knowledge-Based Quality Check',
  provider: 'openai',
  agent_id: string,
  checkpoint: {
    checkpoint_name: string,
    trigger_after_step: number,
    validation_criteria: string,
    success_threshold: string
  }
}
```

### **6. `execution_summary` Event**
**Purpose**: Final multi-step execution summary

**Structure:**
```typescript
{
  type: 'execution_summary',
  content: '📋 Multi-step execution summary: 4/5 steps completed, 1 failed, 27.3s total',
  provider: 'openai',
  agent_id: string,
  summary: {
    total_steps: number,
    completed_steps: number,
    failed_steps: number,
    total_execution_time: number,
    success_rate: number
  }
}
```

## 🎨 **Frontend Implementation Guide**

### **Enhanced Event Handler**
```typescript
// Update your useAgentChat.ts handleSSEEvent function
const handleSSEEvent = (chunk: any) => {
  switch (chunk.type) {
    case 'plan':
      handlePlanEvent(chunk);
      break;
      
    case 'step_start':
      handleStepStartEvent(chunk);
      break;
      
    case 'step_error':
      handleStepErrorEvent(chunk);
      break;
      
    case 'step_complete':
      handleStepCompleteEvent(chunk);
      break;
      
    case 'checkpoint':
      handleCheckpointEvent(chunk);
      break;
      
    case 'execution_summary':
      handleExecutionSummaryEvent(chunk);
      break;
      
    // Existing event types
    case 'response_chunk':
    case 'tool_execution':
    case 'error':
    case 'complete':
      // Your existing handlers
      break;
      
    default:
      console.warn('[useAgentChat] 🔍 Unknown event type:', chunk.type, chunk);
  }
};
```

### **State Management for Multi-Step Execution**
```typescript
interface MultiStepState {
  executionPlan?: ExecutionPlan;
  currentStep?: number;
  stepStatuses: Record<number, 'pending' | 'in_progress' | 'completed' | 'failed'>;
  executionSummary?: ExecutionSummary;
}

const [multiStepState, setMultiStepState] = useState<MultiStepState>({
  stepStatuses: {}
});
```

### **UI Components for Multi-Step Visualization**

#### **Execution Plan Display**
```tsx
const ExecutionPlanDisplay = ({ plan }: { plan: ExecutionPlan }) => (
  <div className="execution-plan">
    <div className="plan-header">
      📋 Execution Plan: {plan.total_steps} steps 
      ({plan.complexity_level} complexity)
    </div>
    <div className="steps-list">
      {plan.execution_steps.map(step => (
        <div key={step.step_number} className="step-item">
          <span className="step-number">{step.step_number}</span>
          <span className="step-name">{step.name}</span>
          <span className="step-time">{step.estimated_time}</span>
        </div>
      ))}
    </div>
  </div>
);
```

#### **Step Progress Indicator**
```tsx
const StepProgressIndicator = ({ 
  currentStep, 
  totalSteps, 
  stepStatuses 
}: {
  currentStep: number;
  totalSteps: number;
  stepStatuses: Record<number, string>;
}) => (
  <div className="step-progress">
    {Array.from({length: totalSteps}, (_, i) => i + 1).map(stepNum => (
      <div 
        key={stepNum}
        className={`step-indicator ${stepStatuses[stepNum] || 'pending'}`}
      >
        {stepNum}
        {stepStatuses[stepNum] === 'completed' && '✅'}
        {stepStatuses[stepNum] === 'failed' && '❌'}
        {stepStatuses[stepNum] === 'in_progress' && '🔄'}
      </div>
    ))}
  </div>
);
```

## 🎯 **Priority Implementation**

### **Immediate (Critical)**
1. **Add event handlers** for `plan`, `step_start`, `step_error`
2. **Remove "Unknown event type" warnings**
3. **Basic progress indication** for multi-step execution

### **Short Term (Enhanced UX)**
1. **Execution plan visualization**
2. **Step progress indicators**
3. **Error handling with retry options**

### **Long Term (Advanced Features)**
1. **Interactive step debugging**
2. **Execution timeline visualization**
3. **Step dependency graphs**

## 🚀 **Benefits After Implementation**

- **Better User Experience**: Clear visibility into multi-step processes
- **Improved Debugging**: Step-by-step error tracking
- **Enhanced Trust**: Users see what the agent is doing
- **Professional UI**: No more "Unknown event type" messages

## 📝 **Testing Events**

You can test these events by triggering complex agent requests that require multi-step execution (complexity score ≥ 5).

Example requests that trigger multi-step:
- "Analyze the automotive industry and create a comprehensive market report"
- "Research competitor pricing and develop a strategic recommendation"
- "Process this document and create executive summary with action items"

The modular architecture will automatically break these into steps and emit the new event types.