# 🔗 Ami Decision Endpoint Integration Guide

## Overview

This guide explains how our improved **Ami 3-Step Workflow** integrates seamlessly with the existing `/decision` endpoint and learning infrastructure to provide human-controlled knowledge saving.

## 🔄 **Complete Integration Flow**

### **Our Improved Ami Workflow:**
```
STEP 1: IMAGINATION → CHOICE
STEP 2: CHOICE → TEACHING  
STEP 3: TEACHING → APPROVAL
```

### **Backend Decision System:**
```
decision endpoint → learning tools → knowledge saving → vector storage
```

## 📋 **Step-by-Step Integration**

### **STEP 1: Ami Prompts for Choice**
When humans share their imagination, Ami uses our new transition phrases:

**Ami says:**
- "Which of these AI agent ideas excites you most?"
- "Let's focus on building [specific agent] - does that feel right?"
- "Which agent should we bring to life first?"

**Result:** Human commits to a specific AI agent idea

---

### **STEP 2: Ami Guides to Teaching**
Once they choose, Ami immediately uses our teaching guidance phrases:

**Ami says:**
- "Perfect! Now I need to understand your expertise to build this agent properly."
- "Tell me about your process for [specific task the agent will do]"
- "Walk me through how you currently handle [agent's main responsibility]"

**Result:** Human shares their expertise and know-how

---

### **STEP 3: Ami Requests Approval + Backend Decision Flow**
When human shares knowledge, this triggers our integrated decision workflow:

#### **3A: Ami's Learning Tools Detect Teaching Intent**
```python
# Ami automatically calls:
search_results = search_learning_context(user_message)  
analysis = analyze_learning_opportunity(user_message)
# → Analysis detects teaching intent
```

#### **3B: Ami Creates Learning Decision** 
```python
# Ami calls:
decision_result = request_learning_decision(
    decision_type="save_new",
    context="Teaching content detected: [user message]",
    options=["Save as new knowledge", "Skip learning", "Need more context"],
    additional_info="AI analysis detected valuable expertise sharing"
)

# Returns: decision_id = "learning_decision_abc123"
```

#### **3C: Frontend Shows Decision UI**
```javascript
// Frontend polls: GET /api/learning/decisions?user_id=xxx
// Gets pending decisions and shows approval UI to human
```

#### **3D: Human Makes Choice**
```javascript  
// Human clicks "Save as new knowledge"
// Frontend calls: POST /api/learning/decision
{
  "decision_id": "learning_decision_abc123",
  "human_choice": "Save as new knowledge"
}
```

#### **3E: Backend Completes Decision & Saves Knowledge**
```python
# complete_learning_decision() automatically:
# 1. Updates decision status to "COMPLETED"
# 2. Sets human_choice = "Save as new knowledge"  
# 3. Detects approval → triggers AVA multi-vector saving
# 4. Saves 3 knowledge vectors to Pinecone
# 5. Returns success confirmation
```

#### **3F: Ami Confirms Knowledge Saved**
```python
# Ami can optionally call save_knowledge() if needed, but
# complete_learning_decision() already handled the saving
```

## 🔧 **Key Integration Points**

### **1. Decision ID Flow**
```
request_learning_decision() → decision_id → complete_learning_decision(decision_id) → knowledge saved
```

### **2. Automatic Knowledge Saving** 
The `complete_learning_decision()` function **automatically** handles knowledge saving when human approves:
```python
# In complete_learning_decision():
if human_choice == "Save as new knowledge":
    # Automatically calls AVA multi-vector saving
    await ava_instance._save_tool_knowledge_multiple(...)
```

### **3. No Double Saving**
Since `complete_learning_decision()` handles saving automatically, Ami doesn't need to call `save_knowledge()` explicitly in most cases.

### **4. Learning Tools Integration**
Our improved Ami prompt works with all existing learning tools:
- `search_learning_context` - Check existing knowledge
- `analyze_learning_opportunity` - Assess learning value  
- `request_learning_decision` - Create approval request
- `preview_knowledge_save` - Show what would be saved
- `save_knowledge` - Manual save (when needed)

## 🎯 **Frontend Integration Requirements**

### **Decision Polling**
```javascript
// Frontend needs to poll for pending decisions
async function pollLearningDecisions(userId) {
    const response = await fetch(`/api/learning/decisions?user_id=${userId}`);
    const data = await response.json();
    return data.decisions;
}
```

### **Decision Submission**
```javascript
// Frontend submits human choice
async function submitDecision(decisionId, humanChoice) {
    const response = await fetch('/api/learning/decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            decision_id: decisionId,
            human_choice: humanChoice
        })
    });
    return await response.json();
}
```

### **UI Components Needed**
1. **Decision Alert/Modal** - Shows pending decisions
2. **Approval Options** - "Save", "Skip", "Need more context"  
3. **Knowledge Preview** - Shows what will be saved
4. **Confirmation** - Shows save success/failure

## 🚀 **Example Complete Workflow**

### **M&A Consultant Example**
```
1. User: "Anh làm tư vấn M&A doanh nghiệp"

2. Ami: [Explores imagination, suggests 3 AI agent ideas]
   "Which of these AI agent ideas excites you most?"

3. User: "AI Agent Due Diligence sounds great!"

4. Ami: "Perfect! Now I need to understand your expertise to build this agent properly. 
        Tell me about your due diligence process."

5. User: "Our due diligence process involves 5 key steps: financial analysis, legal review..."

6. Backend: [Ami's learning tools detect teaching intent]
   - search_learning_context() checks existing knowledge  
   - analyze_learning_opportunity() detects high learning value
   - request_learning_decision() creates decision_id="abc123"

7. Frontend: [Polls and shows decision UI]
   "Ami detected valuable expertise about due diligence. Save this knowledge?"
   [Save] [Skip] [Preview]

8. User: [Clicks "Save"]

9. Backend: [complete_learning_decision() called]
   - Updates decision status
   - Triggers AVA multi-vector saving  
   - Saves 3 knowledge vectors about due diligence process

10. Ami: "Excellent! I've saved your due diligence expertise. This will help your AI agent 
         handle due diligence tasks with your specific methodology."
```

## ⚠️ **Important Implementation Notes**

### **1. Ami's Approval Phrases Trigger Tools**
Our approval request phrases automatically trigger the learning workflow:
- "This is valuable knowledge! Should I save this for your agent?" → triggers learning tools
- "I can store this expertise to help your agent perform better." → creates decision
- "Would you like me to remember this process for your agent?" → requests approval

### **2. Decision Timeout Handling**
```python
# Decisions expire after a certain time
cleanup_expired_decisions()  # Called periodically
```

### **3. Error Handling**
```python
# If decision fails or expires:
save_knowledge(content, decision_id=None)  # Returns error
# Ami should handle gracefully and explain to user
```

### **4. Multi-Vector Saving**
When human approves, the system saves **3 knowledge vectors**:
1. **Combined** (User + AI format)
2. **AI Synthesis** (Enhanced AI understanding)
3. **User Message Only** (For reference)

## 🎭 **Ami's Role in Integration**

### **What Ami Does:**
✅ Guide human through 3-step workflow  
✅ Use transition phrases to push toward teaching  
✅ Trigger learning tools when detecting teaching intent  
✅ Create learning decisions for human approval  
✅ Explain the knowledge saving process to humans

### **What Ami Doesn't Need To Do:**
❌ Handle HTTP requests directly  
❌ Manage decision storage  
❌ Perform actual knowledge saving (handled by backend)  
❌ Poll for decision status  

### **What Frontend Does:**
✅ Poll `/api/learning/decisions` for pending decisions  
✅ Show decision UI to humans  
✅ Submit human choices via `/api/learning/decision`  
✅ Display confirmation messages  

### **What Backend Does:**  
✅ Store decisions in `PENDING_LEARNING_DECISIONS`  
✅ Process human choices via `complete_learning_decision()`  
✅ Automatically save knowledge when approved  
✅ Clean up expired decisions  

---

**The improved Ami prompt seamlessly integrates with the existing decision infrastructure to provide human-controlled AI agent building through systematic knowledge collection! 🚀** 