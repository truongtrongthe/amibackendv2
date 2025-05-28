# Complete Flow Diagram: ami.py → ava.py (with UPDATE Implementation)

## 🚀 Entry Point: `ami.py`

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 ami.py                                          │
│                          convo_stream_learning()                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. Initialize State & Conversation History                                      │
│    • Load checkpoint from thread_id                                             │
│    • Convert messages to conversation_history format                            │
│    • Create state with user_id, graph_version_id, thread_id                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. Create AVA Instance                                                          │
│    • ava = AVA()                                                                │
│    • await ava.initialize()                                                     │
│    • Initialize LearningSupport & KnowledgeExplorer                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. Stream Processing                                                            │
│    • ava.read_human_input(user_input, context, user_id, thread_id)              │
│    • Stream chunks to frontend as they arrive                                   │
│    • Store final_response for post-processing                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼

## 🧠 AVA Processing: `ava.py`

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ava.read_human_input()                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. Knowledge Exploration                                                        │
│    • knowledge_explorer.explore(message, context, user_id, thread_id)           │
│    • Search for relevant knowledge using iterative exploration                  │
│    • Calculate similarity scores with existing knowledge                        │
│    • Return: analysis_knowledge with similarity, query_results, etc.            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. Streaming LLM Response                                                       │
│    • _active_learning_streaming()                                               │
│    • Stream chunks to frontend in real-time                                     │
│    • Extract teaching intent, priority topics, tool calls                       │
│    • Return final_response with metadata                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. Similarity-Based Decision Gate                                               │
│    • should_save_knowledge_with_similarity_gate()                               │
│    • Evaluate: teaching_intent + similarity_score                               │
│    • Return save_decision with action_type                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ Similarity Gate │
                              │   Decision      │
                              └─────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
        ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
        │ High Similarity │  │Medium Similarity│  │ Low Similarity  │
        │    (≥65%)       │  │   (35%-65%)     │  │    (<35%)       │
        │ + Teaching      │  │ + Teaching      │  │ + Teaching      │
        └─────────────────┘  └─────────────────┘  └─────────────────┘
                │                       │                       │
                ▼                       ▼                       ▼
        ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
        │   AUTO SAVE     │  │ UPDATE vs CREATE│  │   AUTO SAVE     │
        │  (3-4 vectors)  │  │   DECISION      │  │  (3-4 vectors)  │
        └─────────────────┘  └─────────────────┘  └─────────────────┘
                │                       │                       │
                ▼                       ▼                       ▼
        ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
        │handle_teaching_ │  │handle_teaching_ │  │handle_teaching_ │
        │intent()         │  │intent_with_     │  │intent()         │
        │                 │  │update_flow()    │  │                 │
        └─────────────────┘  └─────────────────┘  └─────────────────┘

## 🔄 UPDATE vs CREATE Flow (Medium Similarity)

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    handle_teaching_intent_with_update_flow()                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. Identify Update Candidates                                                   │
│    • identify_update_candidates(message, similarity, query_results)             │
│    • Filter results in medium similarity range (35%-65%)                        │
│    • Sort by similarity, take top 3 candidates                                  │
│    • Return: List[{vector_id, content, similarity, metadata}]                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. Present Options to Human                                                     │
│    • present_update_options(message, new_content, candidates)                   │
│    • Create decision_request with request_id                                    │
│    • Store in _pending_decisions cache                                          │
│    • Add decision prompt to response message                                    │
│    • Return: decision_request with UPDATE/CREATE options                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. Response to Frontend                                                         │
│    • Include update_decision_request in response metadata                       │
│    • Frontend displays UPDATE vs CREATE options to user                         │
│    • User makes decision and calls handle_update_decision tool                  │
└─────────────────────────────────────────────────────────────────────────────────┘

## 🔧 Tool Execution: Back to `ami.py`

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Tool Execution Loop                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Check for tool_calls in response.metadata                                       │
│ • save_knowledge → execute_save_knowledge_tool()                                │
│ • save_teaching_synthesis → execute_save_teaching_synthesis_tool()              │
│ • request_save_approval → handle_save_approval_request()                        │
│ • handle_update_decision → handle_update_decision_tool() ⭐ NEW                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        handle_update_decision_tool()                            │
│ • Extract request_id, action, target_id from parameters                         │
│ • Create AVA instance and call ava.handle_update_decision()                     │
│ • Return result with success, action, vector_ids                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼

## 🔄 UPDATE Decision Processing: Back to `ava.py`

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ava.handle_update_decision()                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ Human Decision  │
                              └─────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                                       ▼
        ┌─────────────────┐                     ┌─────────────────┐
        │  CREATE_NEW     │                     │ UPDATE_EXISTING │
        │                 │                     │                 │
        └─────────────────┘                     └─────────────────┘
                │                                       │
                ▼                                       ▼
        ┌─────────────────┐                     ┌─────────────────┐
        │ Use existing    │                     │ 1. Find target  │
        │ save_knowledge  │                     │    candidate    │
        │ logic           │                     │ 2. merge_       │
        │                 │                     │    knowledge()  │
        │ Result:         │                     │ 3. update_      │
        │ • 1 new vector  │                     │    existing_    │
        │ • vector_id     │                     │    knowledge()  │
        └─────────────────┘                     └─────────────────┘
                │                                       │
                ▼                                       ▼
        ┌─────────────────┐                     ┌─────────────────┐
        │ Return:         │                     │ Return:         │
        │ {               │                     │ {               │
        │   success: true │                     │   success: true │
        │   action: "CREATE_NEW"                │   action: "UPDATE_EXISTING"
        │   vector_id: "new_id"                 │   original_vector_id: "old_id"
        │   message: "New knowledge created"    │   new_vector_id: "merged_id"
        │ }               │                     │   merged_content: "..."
        └─────────────────┘                     │   message: "Knowledge updated"
                                                │ }
                                                └─────────────────┘

## 🔀 Knowledge Merging Process

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              merge_knowledge()                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. LLM-Based Intelligent Merge                                                  │
│    • Send existing_content + new_content to LLM                                 │
│    • Instructions: preserve valuable info, integrate new, note contradictions   │
│    • Format: "User: [combined] \n\n AI: [merged content]"                       │
│    • Fallback: structured concatenation if LLM fails                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         update_existing_knowledge()                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. Create New Vector with Merged Content                                        │
│    • save_knowledge(merged_content, user_id, "conversation", ...)               │
│    • Categories: ["updated_knowledge", "teaching_intent"]                       │
│    • TTL: 365 days                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. Delete Old Vector (TODO: Not Yet Implemented)                                │
│    • delete_vector(vector_id, bank_name) - FUTURE IMPLEMENTATION                │
│    • For now: old vector remains in database                                    │
│    • Result: 2 vectors temporarily (old + new)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

## 📊 Vector Count Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              VECTOR COUNTS                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ HIGH SIMILARITY (≥65%) + Teaching Intent:                                       │
│ ├── Combined Knowledge Vector                                                   │
│ ├── AI Synthesis Vector                                                         │
│ ├── Standalone Summary Vector (if available)                                    │
│ └── Teaching Synthesis Vector (alpha.py)                                        │
│ TOTAL: 3-4 vectors                                                              │
│                                                                                 │
│ MEDIUM SIMILARITY (35%-65%) + Teaching Intent:                                  │
│ ├── UPDATE Decision → 1-2 vectors (merged + old until deletion)                 │
│ └── CREATE Decision → 3-4 vectors (same as high similarity)                     │
│                                                                                 │
│ LOW SIMILARITY (<35%) + Teaching Intent:                                        │
│ ├── Combined Knowledge Vector                                                   │
│ ├── AI Synthesis Vector                                                         │
│ ├── Standalone Summary Vector (if available)                                    │
│ └── Teaching Synthesis Vector (alpha.py)                                        │
│ TOTAL: 3-4 vectors                                                              │
│                                                                                 │
│ HIGH SIMILARITY (≥70%) + No Teaching Intent:                                    │
│ └── High-Quality Conversation Vector                                            │
│ TOTAL: 1 vector                                                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Complete Flow Summary

```
Frontend Request
       │
       ▼
┌─────────────────┐
│    ami.py       │
│ convo_stream_   │
│ learning()      │
└─────────────────┘
       │
       ▼ Create AVA instance
┌─────────────────┐
│    ava.py       │
│ read_human_     │
│ input()         │
└─────────────────┘
       │
       ▼ Knowledge exploration
┌─────────────────┐
│ Knowledge       │
│ Explorer        │
│ (similarity)    │
└─────────────────┘
       │
       ▼ LLM streaming
┌─────────────────┐
│ _active_        │
│ learning_       │
│ streaming()     │
└─────────────────┘
       │
       ▼ Similarity gate
┌─────────────────┐
│ should_save_    │
│ knowledge_with_ │
│ similarity_gate │
└─────────────────┘
       │
       ▼ Decision routing
┌─────────────────┐
│ Medium          │
│ Similarity?     │
│ UPDATE vs CREATE│
└─────────────────┘
       │
       ▼ Present options
┌─────────────────┐
│ Frontend shows  │
│ UPDATE/CREATE   │
│ options to user │
└─────────────────┘
       │
       ▼ Human decision
┌─────────────────┐
│    ami.py       │
│ handle_update_  │
│ decision_tool() │
└─────────────────┘
       │
       ▼ Process decision
┌─────────────────┐
│    ava.py       │
│ handle_update_  │
│ decision()      │
└─────────────────┘
       │
       ▼ Execute action
┌─────────────────┐
│ CREATE_NEW or   │
│ UPDATE_EXISTING │
│ (merge + save)  │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Vector Database │
│ Updated         │
└─────────────────┘
```

## 🎯 Key Integration Points

1. **Entry Point**: `ami.py:convo_stream_learning()` - Main API endpoint
2. **Core Processing**: `ava.py:read_human_input()` - Streaming LLM processing
3. **Decision Gate**: `ava.py:should_save_knowledge_with_similarity_gate()` - Smart routing
4. **Human-in-Loop**: `ami.py:handle_update_decision_tool()` - Tool execution framework
5. **Knowledge Merging**: `ava.py:merge_knowledge()` - LLM-based intelligent merging
6. **Vector Operations**: `ava.py:update_existing_knowledge()` - Database updates

## 🚀 Benefits of UPDATE Implementation

- **Reduces Vector Pollution**: Merges similar knowledge instead of creating duplicates
- **Improves Knowledge Quality**: LLM intelligently combines information
- **Human Control**: User decides when to merge vs create new
- **Preserves Context**: Maintains relationship between related knowledge
- **Efficient Storage**: Fewer vectors for similar topics 