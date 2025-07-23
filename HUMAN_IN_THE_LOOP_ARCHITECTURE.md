# Human-in-the-Loop Architecture Guide

## 🔄 **New Architecture Flow**

### **Before (Direct Response):**
```
User Input → Reasoning → LLM Response → Done
```

### **After (Human-in-the-Loop):**
```
User Input → Reasoning → Knowledge Extraction → Human Approval → Copilot Summary
```

## 🎯 **Core Changes**

### **1. Enhanced Reasoning Steps (Steps 2-5)**

**❌ Before (Generic Logging):**
```
🔍 I need to break down this task and check what resources are available...
```

**✅ Now (Specific Analysis):**
```
🔍 Let me analyze the appointment scheduling process you shared:
   - Identity verification step
   - Calendar integration requirements  
   - Exception handling needs
   - I'll check my knowledge about clinic automation systems...
```

### **2. Human-in-the-Loop Knowledge Approval**

**New Event Types:**
- `knowledge_approval_request` - Main approval request
- `knowledge_piece` - Individual knowledge pieces for approval
- `awaiting_approval` - System waits for human input
- `summary_response` - Copilot-style summary after approval

### **3. Copilot-Style Response**

**❌ Before (Chatbot Style):**
```
"Cảm ơn bạn đã chia sẻ quy trình đặt lịch hẹn của phòng khám. 
Với thông tin này, chúng ta có thể tiến hành tạo Agent..."
```

**✅ Now (Copilot Style):**
```
"Perfect! I've successfully enhanced your agent's knowledge base with 9 new processes.

**Your Agent Now Knows:**
- Patient identity verification workflow
- Calendar integration requirements
- Exception handling procedures

**Next Steps:**
1. Set up API integration with your clinic system
2. Configure automated reminder system

Which step would you like to tackle first?"
```

## 🔧 **Frontend Integration Changes**

### **1. New Event Handlers**

```javascript
// Handle knowledge approval request
eventSource.addEventListener('knowledge_approval_request', (event) => {
  const data = JSON.parse(event.data);
  showKnowledgeApprovalUI(data.knowledge_pieces);
});

// Handle individual knowledge pieces
eventSource.addEventListener('knowledge_piece', (event) => {
  const data = JSON.parse(event.data);
  addKnowledgePieceToUI(data);
});

// Handle awaiting approval state
eventSource.addEventListener('awaiting_approval', (event) => {
  showWaitingForApprovalState();
});

// Handle final summary
eventSource.addEventListener('summary_response', (event) => {
  const data = JSON.parse(event.data);
  showCopilotSummary(data.content);
});
```

### **2. Knowledge Approval UI Component**

```jsx
const KnowledgeApprovalUI = ({ knowledgePieces, onApproval }) => {
  const [selectedPieces, setSelectedPieces] = useState([]);

  return (
    <div className="knowledge-approval-container">
      <h3>📚 Review Knowledge Pieces</h3>
      <p>I've extracted {knowledgePieces.length} knowledge pieces. Please approve which ones should be learned:</p>
      
      {knowledgePieces.map((piece, index) => (
        <div key={piece.piece_id} className="knowledge-piece">
          <div className="piece-header">
            <input 
              type="checkbox"
              checked={selectedPieces.includes(piece.piece_id)}
              onChange={(e) => handlePieceSelection(piece.piece_id, e.target.checked)}
            />
            <strong>{piece.title || `Knowledge ${index + 1}`}</strong>
            <span className="quality-badge">{piece.quality_score}% quality</span>
            <span className="category-badge">{piece.category}</span>
          </div>
          <div className="piece-content">
            {piece.content}
          </div>
        </div>
      ))}
      
      <div className="approval-actions">
        <button onClick={() => onApproval(selectedPieces)}>
          Approve Selected ({selectedPieces.length})
        </button>
        <button onClick={() => onApproval([])}>
          Skip All
        </button>
        <button onClick={() => onApproval(knowledgePieces.map(p => p.piece_id))}>
          Approve All
        </button>
      </div>
    </div>
  );
};
```

### **3. New API Endpoint**

```javascript
// Send approval to backend
const sendKnowledgeApproval = async (approvedIds, allKnowledgePieces, originalRequest) => {
  const response = await fetch('/tool/knowledge-approval', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      approved_knowledge_ids: approvedIds,
      all_knowledge_pieces: allKnowledgePieces,
      original_request: originalRequest
    })
  });
  
  // This will return a streaming response with the copilot summary
  return response;
};
```

### **4. Complete Flow Implementation**

```javascript
class ConversationManager {
  constructor() {
    this.currentKnowledgePieces = [];
    this.originalRequest = null;
    this.isAwaitingApproval = false;
  }

  handleLLMStream(eventSource) {
    // Standard reasoning events
    eventSource.addEventListener('thinking', (event) => {
      const data = JSON.parse(event.data);
      this.displayThought(data);
    });

    // Knowledge approval request
    eventSource.addEventListener('knowledge_approval_request', (event) => {
      const data = JSON.parse(event.data);
      this.currentKnowledgePieces = data.knowledge_pieces;
      this.showKnowledgeApprovalUI(data.knowledge_pieces);
    });

    // Awaiting approval state
    eventSource.addEventListener('awaiting_approval', (event) => {
      this.isAwaitingApproval = true;
      this.showWaitingState();
    });

    // Final summary after approval
    eventSource.addEventListener('summary_response', (event) => {
      const data = JSON.parse(event.data);
      this.showCopilotSummary(data.content);
      this.isAwaitingApproval = false;
    });
  }

  async handleKnowledgeApproval(approvedIds) {
    if (!this.isAwaitingApproval) return;

    // Send approval to backend
    const summaryStream = await sendKnowledgeApproval(
      approvedIds, 
      this.currentKnowledgePieces, 
      this.originalRequest
    );

    // Handle the summary response stream
    this.handleSummaryStream(summaryStream);
  }

  showKnowledgeApprovalUI(knowledgePieces) {
    // Hide the regular response area
    this.hideResponseArea();
    
    // Show knowledge approval UI
    const approvalUI = new KnowledgeApprovalUI({
      knowledgePieces,
      onApproval: (approvedIds) => this.handleKnowledgeApproval(approvedIds)
    });
    
    this.displayApprovalUI(approvalUI);
  }

  showCopilotSummary(content) {
    // Hide approval UI
    this.hideApprovalUI();
    
    // Show copilot-style summary
    this.displayCopilotResponse(content);
  }
}
```

## 🚀 **Backend API Changes**

### **New Endpoint: `/tool/knowledge-approval`**

```python
@app.post("/tool/knowledge-approval")
async def handle_knowledge_approval(
    approved_knowledge_ids: List[str],
    all_knowledge_pieces: List[Dict[str, Any]],
    original_request: Dict[str, Any]
):
    """Handle human approval of knowledge pieces and return copilot summary"""
    
    executive_tool = ExecutiveTool()
    request = ToolExecutionRequest(**original_request)
    
    return StreamingResponse(
        executive_tool.handle_knowledge_approval(
            approved_knowledge_ids, 
            all_knowledge_pieces, 
            request
        ),
        media_type="text/plain"
    )
```

## 🎯 **User Experience Flow**

### **1. User Sends Message**
```
User: "Here's our appointment scheduling process..."
```

### **2. Enhanced Reasoning (More Detailed)**
```
💭 Understanding: The user is sharing their clinic's appointment scheduling workflow...
🔍 Investigation: Let me analyze the appointment scheduling process you shared:
   - Identity verification requirements
   - Calendar system integration needs  
   - Exception handling procedures
   I'll check my knowledge about clinic automation systems...
📚 Knowledge Gathering: Checking what I know about healthcare automation...
💡 Analysis: Analyzing the information to understand your specific workflow...
🎯 Strategy: Based on my analysis, I'll extract key processes and wait for your approval...
```

### **3. Knowledge Extraction & Approval**
```
🔍 Extracting actionable knowledge pieces from my response...
🔍 Extracted 9 knowledge pieces from response

📚 I've extracted 9 knowledge pieces from your input. Please review and approve:

☐ Patient Identity Verification (process - 90% quality)
☐ Find Available Time Slots (process - 90% quality)  
☐ Check Additional Service Requests (process - 80% quality)
...

[Approve Selected] [Skip All] [Approve All]
```

### **4. Human Approves**
```
User selects 7 out of 9 pieces and clicks "Approve Selected"
```

### **5. Copilot Summary**
```
✅ Processing 7 approved knowledge pieces...

Perfect! I've successfully enhanced your agent's knowledge base with 7 new processes.

**Your Agent Now Knows:**
- Patient identity verification workflow
- Calendar integration requirements
- Exception handling procedures
- Automated reminder system setup

**Next Steps:**
1. Set up API integration with your clinic system
2. Configure automated reminder notifications
3. Test the appointment booking flow

Which step would you like to tackle first?
```

## 📋 **Implementation Checklist**

### **Backend (Completed ✅)**
- ✅ Enhanced reasoning with specific analysis
- ✅ Human-in-the-loop knowledge approval flow
- ✅ Copilot-style summary generation
- ✅ New event types for approval process
- ✅ Knowledge approval handler method

### **Frontend (Required)**
- ❓ Knowledge approval UI component
- ❓ New event handlers for approval flow
- ❓ API integration for approval endpoint
- ❓ State management for approval process
- ❓ Copilot-style response display

### **API (Required)**
- ❓ New `/tool/knowledge-approval` endpoint
- ❓ Request/response models for approval
- ❓ Integration with existing streaming system

## 🎉 **Benefits of New Architecture**

1. **Human Control**: Users approve what the agent learns
2. **Transparency**: Clear visibility into extracted knowledge
3. **Quality**: Only approved, high-quality knowledge is learned
4. **Copilot Experience**: Cursor-style summaries and next steps
5. **Iterative Learning**: One conversation round = one learning cycle

This architecture transforms the AI from a chatbot into a true **copilot** that collaborates with humans on building intelligent agents! 🚀 