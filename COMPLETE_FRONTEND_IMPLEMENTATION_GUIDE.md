# Complete Frontend Implementation Guide: Human-in-the-Loop

## 🎯 **Core Questions Answered**

1. **Does frontend send something to save knowledge?** → **YES** - New API call to `/tool/knowledge-approval`
2. **Should frontend show loading while waiting?** → **YES** - Multiple loading states needed
3. **How does the complete flow work?** → **Detailed implementation below**

## 🔄 **Complete Flow Overview**

```
User Message → Reasoning Stream → Knowledge Extraction → Human Approval → Knowledge Saving → Copilot Summary
```

## 📋 **Step-by-Step Implementation**

### **Step 1: Enhanced Event Handling**

```javascript
class HumanInTheLoopManager {
  constructor() {
    this.currentKnowledgePieces = [];
    this.originalRequest = null;
    this.isAwaitingApproval = false;
    this.isProcessingApproval = false;
    this.eventSource = null;
  }

  initializeStream(requestData) {
    this.originalRequest = requestData;
    this.eventSource = new EventSource('/tool/llm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });

    this.setupEventHandlers();
  }

  setupEventHandlers() {
    // Standard reasoning events
    this.eventSource.addEventListener('thinking', (event) => {
      const data = JSON.parse(event.data);
      this.displayReasoningThought(data);
    });

    // Knowledge extraction starts
    this.eventSource.addEventListener('knowledge_extraction', (event) => {
      const data = JSON.parse(event.data);
      this.showKnowledgeExtractionProgress(data);
    });

    // Knowledge pieces ready for approval
    this.eventSource.addEventListener('knowledge_approval_request', (event) => {
      const data = JSON.parse(event.data);
      this.currentKnowledgePieces = data.knowledge_pieces;
      this.showKnowledgeApprovalUI(data);
    });

    // System waiting for human approval
    this.eventSource.addEventListener('awaiting_approval', (event) => {
      this.isAwaitingApproval = true;
      this.showAwaitingApprovalState();
    });

    // Handle any errors
    this.eventSource.addEventListener('error', (event) => {
      this.handleStreamError(event);
    });
  }
}
```

### **Step 2: Knowledge Approval UI Component**

```jsx
import React, { useState, useEffect } from 'react';

const KnowledgeApprovalInterface = ({ 
  knowledgePieces, 
  onApproval, 
  isProcessing = false 
}) => {
  const [selectedPieces, setSelectedPieces] = useState([]);
  const [selectAll, setSelectAll] = useState(false);

  useEffect(() => {
    // Pre-select high-quality pieces (80%+)
    const highQualityPieces = knowledgePieces
      .filter(piece => piece.quality_score >= 80)
      .map(piece => piece.piece_id);
    setSelectedPieces(highQualityPieces);
  }, [knowledgePieces]);

  const handlePieceToggle = (pieceId) => {
    setSelectedPieces(prev => 
      prev.includes(pieceId)
        ? prev.filter(id => id !== pieceId)
        : [...prev, pieceId]
    );
  };

  const handleSelectAll = () => {
    if (selectAll) {
      setSelectedPieces([]);
    } else {
      setSelectedPieces(knowledgePieces.map(p => p.piece_id));
    }
    setSelectAll(!selectAll);
  };

  const getQualityColor = (score) => {
    if (score >= 90) return '#10b981'; // green
    if (score >= 70) return '#f59e0b'; // yellow
    return '#ef4444'; // red
  };

  const getCategoryIcon = (category) => {
    const icons = {
      'process': '⚙️',
      'requirement': '📋',
      'configuration': '🔧',
      'general': '📄'
    };
    return icons[category] || '📄';
  };

  return (
    <div className="knowledge-approval-container">
      <div className="approval-header">
        <h3>📚 Review Knowledge Pieces</h3>
        <p>I've extracted <strong>{knowledgePieces.length}</strong> knowledge pieces from your input. Please review and approve which ones should be learned by your agent:</p>
      </div>

      <div className="approval-controls">
        <label className="select-all-control">
          <input
            type="checkbox"
            checked={selectAll}
            onChange={handleSelectAll}
            disabled={isProcessing}
          />
          Select All ({knowledgePieces.length})
        </label>
        <div className="selection-summary">
          {selectedPieces.length} of {knowledgePieces.length} selected
        </div>
      </div>

      <div className="knowledge-pieces-list">
        {knowledgePieces.map((piece, index) => (
          <div 
            key={piece.piece_id} 
            className={`knowledge-piece ${selectedPieces.includes(piece.piece_id) ? 'selected' : ''}`}
          >
            <div className="piece-header">
              <label className="piece-selector">
                <input
                  type="checkbox"
                  checked={selectedPieces.includes(piece.piece_id)}
                  onChange={() => handlePieceToggle(piece.piece_id)}
                  disabled={isProcessing}
                />
                <div className="piece-info">
                  <span className="piece-title">
                    {getCategoryIcon(piece.category)} {piece.title || `Knowledge ${index + 1}`}
                  </span>
                  <div className="piece-metadata">
                    <span 
                      className="quality-badge"
                      style={{ backgroundColor: getQualityColor(piece.quality_score) }}
                    >
                      {piece.quality_score}% quality
                    </span>
                    <span className="category-badge">
                      {piece.category}
                    </span>
                  </div>
                </div>
              </label>
            </div>
            <div className="piece-content">
              {piece.content}
            </div>
          </div>
        ))}
      </div>

      <div className="approval-actions">
        <button 
          className="btn-primary"
          onClick={() => onApproval(selectedPieces)}
          disabled={isProcessing || selectedPieces.length === 0}
        >
          {isProcessing ? (
            <>
              <span className="spinner"></span>
              Processing {selectedPieces.length} pieces...
            </>
          ) : (
            `Approve Selected (${selectedPieces.length})`
          )}
        </button>
        
        <button 
          className="btn-secondary"
          onClick={() => onApproval(knowledgePieces.map(p => p.piece_id))}
          disabled={isProcessing}
        >
          Approve All
        </button>
        
        <button 
          className="btn-outline"
          onClick={() => onApproval([])}
          disabled={isProcessing}
        >
          Skip All
        </button>
      </div>
    </div>
  );
};
```

### **Step 3: Loading States Management**

```javascript
class LoadingStateManager {
  constructor() {
    this.states = {
      REASONING: 'reasoning',
      EXTRACTING_KNOWLEDGE: 'extracting_knowledge', 
      AWAITING_APPROVAL: 'awaiting_approval',
      PROCESSING_APPROVAL: 'processing_approval',
      GENERATING_SUMMARY: 'generating_summary',
      COMPLETE: 'complete'
    };
    
    this.currentState = null;
    this.stateCallbacks = {};
  }

  setState(newState, data = {}) {
    this.currentState = newState;
    this.updateUI(newState, data);
    
    // Trigger callbacks
    if (this.stateCallbacks[newState]) {
      this.stateCallbacks[newState].forEach(callback => callback(data));
    }
  }

  updateUI(state, data) {
    const loadingElement = document.getElementById('loading-indicator');
    const contentElement = document.getElementById('main-content');

    switch (state) {
      case this.states.REASONING:
        this.showLoadingSpinner('🧠 Analyzing your input...', false);
        break;

      case this.states.EXTRACTING_KNOWLEDGE:
        this.showLoadingSpinner('🔍 Extracting knowledge pieces...', false);
        break;

      case this.states.AWAITING_APPROVAL:
        this.hideLoadingSpinner();
        this.showAwaitingApprovalState();
        break;

      case this.states.PROCESSING_APPROVAL:
        this.showLoadingSpinner(
          `✅ Processing ${data.approvedCount || 0} approved knowledge pieces...`, 
          true
        );
        break;

      case this.states.GENERATING_SUMMARY:
        this.showLoadingSpinner('✨ Generating copilot summary...', true);
        break;

      case this.states.COMPLETE:
        this.hideLoadingSpinner();
        break;
    }
  }

  showLoadingSpinner(message, showProgress = false) {
    const loadingHTML = `
      <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-message">${message}</div>
        ${showProgress ? '<div class="loading-progress-bar"><div class="progress-fill"></div></div>' : ''}
      </div>
    `;
    
    document.getElementById('loading-indicator').innerHTML = loadingHTML;
    document.getElementById('loading-indicator').style.display = 'block';
  }

  showAwaitingApprovalState() {
    const awaitingHTML = `
      <div class="awaiting-approval-state">
        <div class="awaiting-icon">⏳</div>
        <div class="awaiting-message">
          <h4>Waiting for your approval...</h4>
          <p>Please review the knowledge pieces above and choose which ones to approve.</p>
        </div>
      </div>
    `;
    
    document.getElementById('loading-indicator').innerHTML = awaitingHTML;
    document.getElementById('loading-indicator').style.display = 'block';
  }

  hideLoadingSpinner() {
    document.getElementById('loading-indicator').style.display = 'none';
  }
}
```

### **Step 4: Knowledge Approval API Handler**

```javascript
class KnowledgeApprovalAPI {
  constructor() {
    this.baseUrl = '/tool';
  }

  async sendApproval(approvedIds, allKnowledgePieces, originalRequest) {
    try {
      const response = await fetch(`${this.baseUrl}/knowledge-approval`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({
          approved_knowledge_ids: approvedIds,
          all_knowledge_pieces: allKnowledgePieces,
          original_request: originalRequest
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response;
    } catch (error) {
      console.error('Knowledge approval API error:', error);
      throw error;
    }
  }

  handleApprovalStream(response, callbacks = {}) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    const processStream = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                // Handle different event types
                switch (data.type) {
                  case 'thinking':
                    callbacks.onThinking?.(data);
                    break;
                  case 'processing_approval':
                    callbacks.onProcessing?.(data);
                    break;
                  case 'summary_response':
                    callbacks.onSummary?.(data);
                    break;
                  default:
                    callbacks.onOther?.(data);
                }
              } catch (parseError) {
                console.error('Error parsing stream data:', parseError);
              }
            }
          }
        }
      } catch (streamError) {
        console.error('Stream processing error:', streamError);
        callbacks.onError?.(streamError);
      }
    };

    processStream();
  }
}
```

### **Step 5: Complete Integration**

```javascript
class CompleteHumanInTheLoopSystem {
  constructor() {
    this.knowledgeManager = new HumanInTheLoopManager();
    this.loadingManager = new LoadingStateManager();
    this.approvalAPI = new KnowledgeApprovalAPI();
    this.isActive = false;
  }

  async startConversation(userMessage, requestConfig = {}) {
    this.isActive = true;
    
    const requestData = {
      llm_provider: 'openai',
      user_query: userMessage,
      cursor_mode: true,
      enable_intent_classification: true,
      enable_request_analysis: true,
      enable_deep_reasoning: false, // Can be enabled for deeper analysis
      ...requestConfig
    };

    // Start the initial stream
    this.knowledgeManager.initializeStream(requestData);
    this.setupStreamHandlers();
    
    // Set initial loading state
    this.loadingManager.setState(this.loadingManager.states.REASONING);
  }

  setupStreamHandlers() {
    const eventSource = this.knowledgeManager.eventSource;

    // Reasoning phase
    eventSource.addEventListener('thinking', (event) => {
      const data = JSON.parse(event.data);
      this.displayThought(data);
    });

    // Knowledge extraction phase
    eventSource.addEventListener('knowledge_extraction', (event) => {
      this.loadingManager.setState(this.loadingManager.states.EXTRACTING_KNOWLEDGE);
    });

    // Knowledge approval phase
    eventSource.addEventListener('knowledge_approval_request', (event) => {
      const data = JSON.parse(event.data);
      this.loadingManager.setState(this.loadingManager.states.AWAITING_APPROVAL);
      this.showKnowledgeApprovalInterface(data.knowledge_pieces);
    });

    // Stream complete (waiting for human)
    eventSource.addEventListener('awaiting_approval', (event) => {
      // UI already updated in knowledge_approval_request handler
      console.log('Stream paused - awaiting human approval');
    });
  }

  showKnowledgeApprovalInterface(knowledgePieces) {
    const approvalInterface = new KnowledgeApprovalInterface({
      knowledgePieces,
      onApproval: (approvedIds) => this.handleKnowledgeApproval(approvedIds),
      isProcessing: false
    });

    // Mount the React component or update DOM
    ReactDOM.render(approvalInterface, document.getElementById('approval-container'));
  }

  async handleKnowledgeApproval(approvedIds) {
    try {
      // Update UI to show processing state
      this.loadingManager.setState(
        this.loadingManager.states.PROCESSING_APPROVAL, 
        { approvedCount: approvedIds.length }
      );

      // Update approval interface to show loading
      this.updateApprovalInterface({ isProcessing: true });

      // Send approval to backend
      const response = await this.approvalAPI.sendApproval(
        approvedIds,
        this.knowledgeManager.currentKnowledgePieces,
        this.knowledgeManager.originalRequest
      );

      // Handle the summary stream
      this.approvalAPI.handleApprovalStream(response, {
        onThinking: (data) => {
          this.displayThought(data);
        },
        onProcessing: (data) => {
          // Already in processing state
        },
        onSummary: (data) => {
          this.loadingManager.setState(this.loadingManager.states.COMPLETE);
          this.showCopilotSummary(data);
          this.cleanup();
        },
        onError: (error) => {
          this.handleApprovalError(error);
        }
      });

    } catch (error) {
      this.handleApprovalError(error);
    }
  }

  updateApprovalInterface(props) {
    const approvalInterface = new KnowledgeApprovalInterface({
      knowledgePieces: this.knowledgeManager.currentKnowledgePieces,
      onApproval: (approvedIds) => this.handleKnowledgeApproval(approvedIds),
      ...props
    });

    ReactDOM.render(approvalInterface, document.getElementById('approval-container'));
  }

  showCopilotSummary(summaryData) {
    const summaryHTML = `
      <div class="copilot-summary">
        <div class="summary-header">
          <h3>✅ Knowledge Integration Complete</h3>
        </div>
        <div class="summary-content">
          ${summaryData.content}
        </div>
        <div class="summary-metadata">
          <span class="knowledge-count">
            ${summaryData.approved_knowledge_count} knowledge pieces integrated
          </span>
        </div>
      </div>
    `;

    document.getElementById('summary-container').innerHTML = summaryHTML;
    document.getElementById('approval-container').style.display = 'none';
  }

  handleApprovalError(error) {
    console.error('Approval process error:', error);
    
    this.loadingManager.setState(this.loadingManager.states.COMPLETE);
    
    const errorHTML = `
      <div class="error-message">
        <h4>❌ Approval Process Error</h4>
        <p>There was an error processing your approval. Please try again.</p>
        <button onclick="location.reload()">Refresh Page</button>
      </div>
    `;
    
    document.getElementById('error-container').innerHTML = errorHTML;
  }

  displayThought(thoughtData) {
    const thoughtElement = document.createElement('div');
    thoughtElement.className = `thought thought-${thoughtData.thought_type}`;
    thoughtElement.innerHTML = `
      <div class="thought-content">${thoughtData.content}</div>
      <div class="thought-meta">Step ${thoughtData.reasoning_step}</div>
    `;
    
    document.getElementById('thoughts-container').appendChild(thoughtElement);
  }

  cleanup() {
    if (this.knowledgeManager.eventSource) {
      this.knowledgeManager.eventSource.close();
    }
    this.isActive = false;
  }
}
```

### **Step 6: CSS Styles**

```css
/* Knowledge Approval Interface */
.knowledge-approval-container {
  max-width: 800px;
  margin: 20px auto;
  padding: 24px;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #ffffff;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.approval-header h3 {
  color: #1f2937;
  margin-bottom: 8px;
}

.approval-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 20px 0;
  padding: 12px;
  background: #f9fafb;
  border-radius: 8px;
}

.knowledge-pieces-list {
  max-height: 400px;
  overflow-y: auto;
  margin: 20px 0;
}

.knowledge-piece {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  margin-bottom: 12px;
  transition: all 0.2s ease;
}

.knowledge-piece.selected {
  border-color: #3b82f6;
  background: #eff6ff;
}

.piece-header {
  padding: 16px;
  border-bottom: 1px solid #f3f4f6;
}

.piece-selector {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  cursor: pointer;
}

.piece-info {
  flex: 1;
}

.piece-title {
  font-weight: 600;
  color: #1f2937;
  display: block;
  margin-bottom: 8px;
}

.piece-metadata {
  display: flex;
  gap: 8px;
}

.quality-badge {
  padding: 2px 8px;
  border-radius: 12px;
  color: white;
  font-size: 12px;
  font-weight: 500;
}

.category-badge {
  padding: 2px 8px;
  border-radius: 12px;
  background: #f3f4f6;
  color: #6b7280;
  font-size: 12px;
}

.piece-content {
  padding: 16px;
  color: #4b5563;
  line-height: 1.5;
}

.approval-actions {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  margin-top: 24px;
}

.btn-primary {
  background: #3b82f6;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-primary:disabled {
  background: #9ca3af;
  cursor: not-allowed;
}

.btn-secondary {
  background: #10b981;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
}

.btn-outline {
  background: transparent;
  color: #6b7280;
  border: 1px solid #d1d5db;
  padding: 12px 24px;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
}

/* Loading States */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px;
  text-align: center;
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #f3f4f6;
  border-top: 3px solid #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-message {
  color: #6b7280;
  font-weight: 500;
}

.awaiting-approval-state {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  background: #fef3c7;
  border: 1px solid #f59e0b;
  border-radius: 8px;
  margin: 20px 0;
}

.awaiting-icon {
  font-size: 24px;
}

.copilot-summary {
  max-width: 800px;
  margin: 20px auto;
  padding: 24px;
  border: 1px solid #10b981;
  border-radius: 12px;
  background: #f0fdf4;
}

.summary-header h3 {
  color: #065f46;
  margin-bottom: 16px;
}

.summary-content {
  color: #1f2937;
  line-height: 1.6;
  margin-bottom: 16px;
}

.summary-metadata {
  padding-top: 16px;
  border-top: 1px solid #bbf7d0;
  color: #059669;
  font-size: 14px;
}
```

### **Step 7: Usage Example**

```javascript
// Initialize the system
const humanInTheLoop = new CompleteHumanInTheLoopSystem();

// Start a conversation
document.getElementById('send-button').addEventListener('click', () => {
  const userMessage = document.getElementById('user-input').value;
  
  if (userMessage.trim()) {
    humanInTheLoop.startConversation(userMessage, {
      llm_provider: 'openai',
      cursor_mode: true,
      enable_deep_reasoning: true // Enable for complex queries
    });
    
    // Clear input
    document.getElementById('user-input').value = '';
  }
});
```

## 🎯 **Key Points Summary**

### **YES - Frontend Sends Approval Data:**
- New API call to `/tool/knowledge-approval`
- Includes approved IDs, all pieces, and original request
- Returns streaming copilot summary

### **YES - Multiple Loading States:**
1. **Reasoning** - "🧠 Analyzing your input..."
2. **Extracting** - "🔍 Extracting knowledge pieces..."
3. **Awaiting** - "⏳ Waiting for your approval..."
4. **Processing** - "✅ Processing X approved pieces..."
5. **Summary** - "✨ Generating copilot summary..."

### **Complete User Experience:**
- Real-time reasoning thoughts
- Immediate knowledge piece display
- Interactive approval interface
- Loading states throughout
- Copilot-style summary
- Error handling

**This creates a seamless, professional human-in-the-loop experience!** 🎉 