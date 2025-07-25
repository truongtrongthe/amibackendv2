# Agent Execution Logging Guide

## Overview
The Agent system now has comprehensive logging to trace exactly what happens during execution. This guide shows what logs you'll see and how to interpret them.

## Log Categories

### 🤖 [AGENT] - Core Agent Operations
- **Purpose**: Basic agent lifecycle and tool initialization
- **Format**: `🤖 [AGENT] HH:MM:SS - LEVEL - message`

### 🔍 [EXEC] - Detailed Execution Tracing  
- **Purpose**: Step-by-step execution flow with timing and metrics
- **Format**: `🔍 [EXEC] HH:MM:SS - message`

### 📁 [FILE] - File Access Operations
- **Purpose**: Google Drive and local file operations
- **Format**: `📁 [FILE] HH:MM:SS - message`

### 📊 [BUSINESS] - Business Logic Operations
- **Purpose**: Sales analysis and business intelligence operations
- **Format**: `📊 [BUSINESS] HH:MM:SS - message`

## Example: Google Drive Document Analysis

When a user submits: *"Analyse for me this: https://docs.google.com/document/d/1vtazm6L4vVkwBym9smqIAFaI3RmigLo0CyvpSJePEtc/edit?usp=sharing"*

### Expected Log Flow:

```bash
# === EXECUTION START ===
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] === AGENT EXECUTION START ===
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] Agent: business_analyzer (business_analyst_agent)
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] Provider: anthropic
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] Model: default
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] User Request: 'Analyse for me this: https://docs.google.com/document/d/1vtazm6L4vVkwBym9smqIAFaI3R...'
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] Specialized Domains: ['business_analysis', 'strategic_planning']
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] Tools Enabled: True
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] Org ID: org_123, User ID: user_456

# === VALIDATION ===
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] VALIDATION PASSED - Provider anthropic supported
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] STREAMING STATUS - Initial processing message

# === TOOL SETUP ===
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] TOOL CONVERSION - Converting agent request to tool request
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] TOOL CONVERSION COMPLETE - Available tools: ['search_factory', 'context_tool', 'brain_vector_tool', 'learning_tools_factory', 'file_access', 'business_logic']
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] SYSTEM PROMPT: You are business_analyzer, a specialized business_analyst_agent. Your specialization areas: business_analysis, strategic_planning You have access to tools when needed...

# === LLM EXECUTION ===
🔍 [EXEC] 14:30:15 - [business_analyzer-143015] LLM EXECUTION START - Provider: anthropic

# === TOOL EXECUTION: File Access ===
🔍 [EXEC] 14:30:16 - [business_analyzer-143015] TOOL CALL #1 - read_gdrive_link_docx: I need to read and analyze this Google Drive document.

📁 [FILE] 14:30:16 - READ_GDRIVE_LINK_DOCX - Starting with link: https://docs.google.com/document/d/1vtazm6L4vVkwBym9smqIAFaI3RmigLo0CyvpSJePEtc/edit?usp=sharing
📁 [FILE] 14:30:16 - READ_GDRIVE_LINK_DOCX - Extracting file ID from link
📁 [FILE] 14:30:16 - EXTRACT_FILE_ID - Processing link: https://docs.google.com/document/d/1vtazm6L4vVkwBym9smqIAFaI3RmigLo0CyvpSJePEtc/edit?usp=sharing
📁 [FILE] 14:30:16 - EXTRACT_FILE_ID - Matched Pattern 2 (docs.google.com/document): 1vtazm6L4vVkwBym9smqIAFaI3RmigLo0CyvpSJePEtc
📁 [FILE] 14:30:16 - READ_GDRIVE_LINK_DOCX - Extracted file ID: 1vtazm6L4vVkwBym9smqIAFaI3RmigLo0CyvpSJePEtc
📁 [FILE] 14:30:16 - READ_GDRIVE_LINK_DOCX - Reading file with ID: 1vtazm6L4vVkwBym9smqIAFaI3RmigLo0CyvpSJePEtc
📁 [FILE] 14:30:17 - READ_GDRIVE_LINK_DOCX - Successfully read 2847 characters from file

# === TOOL EXECUTION: Business Analysis ===
🔍 [EXEC] 14:30:18 - [business_analyzer-143015] TOOL CALL #2 - sale_summarize: Now I'll analyze this business plan document.

📊 [BUSINESS] 14:30:18 - SALE_SUMMARIZE - Starting analysis
📊 [BUSINESS] 14:30:18 - SALE_SUMMARIZE - Data length: 2847 chars
📊 [BUSINESS] 14:30:18 - SALE_SUMMARIZE - Instructions: Provide comprehensive business plan analysis focusing on market opportunity, competitive...
📊 [BUSINESS] 14:30:18 - SALE_SUMMARIZE - Parsed instructions, analyzing requirements
📊 [BUSINESS] 14:30:18 - SALE_SUMMARIZE - Generated summary report (1543 chars)
📊 [BUSINESS] 14:30:18 - SALE_SUMMARIZE - Analysis completed successfully

# === LLM RESPONSE ===
🔍 [EXEC] 14:30:19 - [business_analyzer-143015] RESPONSE CHUNK #1: 'I've successfully analyzed the TONMAT CC Business Plan...'
🔍 [EXEC] 14:30:19 - [business_analyzer-143015] RESPONSE CHUNK #2: 'document from your Google Drive link. Here's my comp...'
🔍 [EXEC] 14:30:19 - [business_analyzer-143015] RESPONSE CHUNK #3: 'rehensive analysis: ## DOCUMENT OVERVIEW **Title:**...'
🔍 [EXEC] 14:30:19 - [business_analyzer-143015] RESPONSE CHUNK #4: '(1st Draft) TONMAT CC- Business Plan **Type:** Busi...'
🔍 [EXEC] 14:30:19 - [business_analyzer-143015] RESPONSE CHUNK #5: 'ness planning document **Status:** Draft version ##...'

# === EXECUTION COMPLETE ===
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] LLM EXECUTION COMPLETE - Processed 47 chunks
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] EXECUTION SUMMARY:
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] - Total chunks: 47
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] - Tool calls: 2
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] - Response chunks: 23
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] - Execution time: 4.73s
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] === AGENT EXECUTION SUCCESS ===
```

## Key Information You Can Track

### 1. **Execution Flow**
- ✅ Request validation and setup
- ✅ Tool initialization and availability  
- ✅ System prompt construction
- ✅ LLM provider selection and execution
- ✅ Tool call sequence and results
- ✅ Response generation and streaming
- ✅ Completion status and timing

### 2. **File Operations**
- ✅ Google Drive link processing
- ✅ File ID extraction (with pattern matching)
- ✅ Google Drive API calls
- ✅ File content reading (with size metrics)
- ✅ Error handling and validation

### 3. **Business Logic**
- ✅ Analysis instruction parsing
- ✅ Data processing metrics
- ✅ Summary generation
- ✅ Output formatting

### 4. **Performance Metrics**
- ✅ Total execution time
- ✅ Chunk processing counts
- ✅ Tool execution counts  
- ✅ Response generation timing
- ✅ Data transfer sizes

## Error Scenarios

### Google Drive Access Error:
```bash
📁 [FILE] 14:30:16 - READ_GDRIVE_LINK_DOCX - Google Drive service not initialized
🔍 [EXEC] 14:30:16 - [business_analyzer-143015] TOOL CALL #1 - read_gdrive_link_docx: Error: Google Drive service not initialized. Check credentials.
```

### Invalid Google Drive Link:
```bash
📁 [FILE] 14:30:16 - EXTRACT_FILE_ID - No pattern matched for link: https://invalid-link.com/document
📁 [FILE] 14:30:16 - READ_GDRIVE_LINK_DOCX - Could not extract file ID from: https://invalid-link.com/document
```

### Agent Execution Failure:
```bash
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] === AGENT EXECUTION FAILED ===
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] Error: Agent execution failed: Connection timeout
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] Execution time: 12.34s
🔍 [EXEC] 14:30:20 - [business_analyzer-143015] Exception details: Connection timeout after 10 seconds
```

## Using the Logs

### **For Debugging:**
1. Search for execution ID (e.g., `business_analyzer-143015`) to trace a specific request
2. Look for `ERROR` level messages for failures
3. Check tool call sequences to understand agent decisions
4. Review timing information for performance issues

### **For Monitoring:**
1. Track execution times across different requests
2. Monitor tool usage patterns
3. Watch for recurring errors or failures
4. Analyze response generation performance

### **For Development:**
1. Understand agent decision-making process
2. Debug tool integration issues
3. Optimize system prompt effectiveness
4. Validate file access operations

## Configuration

The logging is configured in each respective module:
- **Agent logging**: `agent.py` (lines ~15-30)
- **File access logging**: `file_access_tool.py` (lines ~10-17)
- **Business logic logging**: `business_logic_tool.py` (lines ~10-17)

All logs use INFO level by default and stream to console with timestamp formatting. 