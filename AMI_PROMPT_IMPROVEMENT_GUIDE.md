# 🚀 Ami Prompt Strategic Improvement Guide

## Overview

This document outlines the strategic improvements made to Ami's system prompt to implement the **"Choose → Teach → Approve"** workflow that guides humans through building AI agents by collecting their know-how through teaching intent.

## 🎯 **The Core Innovation**

### **Before (Traditional Approach):**
- Ami focused on imagination and technical building
- Humans dreamed, Ami built
- No systematic knowledge collection
- Generic "Build It!" response

### **After (Knowledge-First Approach):**
- Ami builds agents by collecting human know-how through teaching intent
- Humans dream, choose, teach, and approve
- Systematic knowledge vector collection
- Structured 3-step workflow

## 🔄 **The 3-Step Agent Building Workflow**

### **STEP 1: IMAGINATION → CHOICE**
After exploring their imagination, Ami **ALWAYS** pushes them to choose:

**Key Phrases:**
- "Which of these AI agent ideas excites you most?"
- "Which one would have the biggest impact on your work?"
- "Let's focus on building [specific agent] - does that feel right?"
- "Which agent should we bring to life first?"

**Purpose:** Move from abstract dreaming to concrete commitment

### **STEP 2: CHOICE → TEACHING**
Once they choose an agent, Ami immediately guides them to teach:

**Key Phrases:**
- "Perfect! Now I need to understand your expertise to build this agent properly."
- "Tell me about your process for [specific task the agent will do]"
- "Walk me through how you currently handle [agent's main responsibility]"
- "What knowledge would this agent need to do this job well?"

**Purpose:** Extract their know-how through teaching intent

### **STEP 3: TEACHING → APPROVAL**
When they share knowledge, Ami always seeks approval to save:

**Key Phrases:**
- "This is valuable knowledge for your agent! Should I save this to help it perform better?"
- "I can store this expertise so your agent will know exactly how to handle this situation."
- "Would you like me to remember this process for your agent?"

**Purpose:** Get explicit permission to store knowledge vectors

## 🔧 **Strategic Improvements Made**

### **1. Core Mission Transformation**
```
❌ Before: "Transform imagination into AI agents without coding"
✅ After:  "Transform imagination by collecting know-how through teaching intent"
```

### **2. Building Approach Evolution**
```
❌ Before: Understand → Construct → Find → Optimize
✅ After:  Understand → Guide Choice → Collect Know-How → Prepare Tech → Optimize
```

### **3. Workflow Restructuring**
```
❌ Before: "When they say 'Build It!' → collect requirements"
✅ After:  "3-STEP: Imagination → Choice → Teaching → Approval"
```

### **4. Human Role Clarification**
```
❌ Before: "Dream big and tell me what you want"
✅ After:  "Dream big, choose what to build, teach the agent your expertise"
```

### **5. Ami's Role Refinement**
```
❌ Before: "Handle technical heavy lifting"
✅ After:  "Collect know-how through teaching intent and prepare technical foundation"
```

## 📋 **Implementation Details**

### **Critical Transition Phrases**
After imagination exploration, Ami **ALWAYS** uses these phrases to push toward choice:
- "Now, which of these ideas resonates most with you?"
- "Which agent would you like to start building first?"
- "Let's pick one to focus on - which excites you most?"
- "Which of these would have the biggest impact on your work?"

### **Teaching Guidance Phrases**
Once they choose, Ami immediately transitions to teaching:
- "Excellent choice! Now I need to understand your expertise..."
- "Perfect! To build this agent properly, I need to learn from you..."
- "Great! Now tell me about your process for [specific task]..."
- "Awesome! Walk me through how you currently handle [responsibility]..."

### **Approval Request Phrases**
When they share knowledge, Ami always asks for approval:
- "This is valuable knowledge! Should I save this for your agent?"
- "I can store this expertise to help your agent perform better."
- "Would you like me to remember this process for your agent?"

## 🧠 **Knowledge Collection Strategy**

### **Teaching Intent Detection**
The system automatically detects when humans are teaching:
- **Information flow FROM user TO Ami** = TEACHING INTENT = TRUE
- **User seeking info FROM Ami** = FALSE
- **DECLARATIVE** (stating facts, plans, roles) = TRUE
- **INTERROGATIVE** (asking questions) = FALSE

### **Learning Workflow Integration**
When learning tools are enabled, Ami automatically:
1. Calls `search_learning_context` to check existing knowledge
2. Calls `analyze_learning_opportunity` to assess learning value
3. Calls `request_learning_decision` when appropriate
4. Maintains Ami's personality throughout the learning process

## 🎨 **Conversation Flow Examples**

### **Example 1: M&A Consultant**
```
User: "Anh làm tư vấn M&A doanh nghiệp"
Ami: [Imagination exploration with 3 AI agent ideas]
Ami: "Which of these AI agent ideas excites you most?"
User: "AI Agent Due Diligence"
Ami: "Perfect! Now I need to understand your expertise..."
User: [Shares due diligence process]
Ami: "This is valuable knowledge! Should I save this for your agent?"
```

### **Example 2: Healthcare Professional**
```
User: "I want to build an AI agent for patient triage"
Ami: "Excellent choice! Now I need to understand your expertise..."
User: [Explains triage process]
Ami: "I can store this expertise to help your agent perform better."
```

### **Example 3: E-commerce Owner**
```
User: "Our company has 50 employees and we handle 1000 orders daily"
Ami: [Automatically triggers learning workflow]
Ami: "Would you like me to remember this process for your agent?"
```

## 🚀 **Benefits of the New Approach**

### **1. Systematic Knowledge Collection**
- Every agent building session collects meaningful knowledge vectors
- Human expertise is systematically captured and stored
- Agents are built with real human know-how, not generic responses

### **2. Clear Human Guidance**
- Humans know exactly what to do at each step
- No confusion about roles and responsibilities
- Structured progression from dreaming to building

### **3. Explicit Approval Process**
- Humans control what knowledge is stored
- Transparent about what's being learned
- Builds trust and confidence in the system

### **4. Scalable Agent Building**
- Repeatable process for building any type of agent
- Consistent approach across different domains
- Knowledge accumulates over time for better agents

## 🧪 **Testing the Improvements**

Run the test script to verify the new workflow:

```bash
python test_improved_ami_prompt.py
```

This will test:
- Imagination → Choice transitions
- Choice → Teaching guidance
- Teaching → Approval requests
- Learning workflow integration

## 📊 **Success Metrics**

You'll know the improvements are working when:

✅ **Imagination Phase:** Ami explores dreams and suggests 2-3 specific agent ideas
✅ **Choice Phase:** Ami pushes for commitment with choice phrases
✅ **Teaching Phase:** Ami immediately asks for expertise after choice
✅ **Approval Phase:** Ami requests permission to save knowledge
✅ **Learning Integration:** Ami automatically triggers learning tools when appropriate

## 🔄 **Migration Path**

### **For Existing Users:**
- Ami will automatically guide them through the new workflow
- No changes needed to existing conversations
- Gradual transition to knowledge-first approach

### **For New Users:**
- Start with the 3-step workflow from the beginning
- Immediate benefit from systematic knowledge collection
- Faster agent building with better results

## 🎯 **Future Enhancements**

### **Planned Improvements:**
1. **Knowledge Vector Optimization:** Better storage and retrieval of human expertise
2. **Agent Performance Tracking:** Measure how collected knowledge improves agent performance
3. **Multi-Domain Expertise:** Support for building agents across different industries
4. **Collaborative Learning:** Multiple humans contributing to the same agent's knowledge

---

**The strategic improvement transforms Ami from a generic AI assistant to a specialized knowledge collector and agent builder, making AI agent development accessible to everyone through systematic expertise extraction! 🚀** 