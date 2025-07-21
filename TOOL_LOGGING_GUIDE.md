# 🔧 Tool Call Logging System

## Overview

The comprehensive tool call logging system provides **real-time visibility** into all tool executions with detailed console logging and streaming events for both server-side debugging and frontend user experience.

## 🎯 Key Features

### ✅ **Dual Logging Approach**
- **Console Logging**: Detailed server-side logs with timestamps
- **Streaming Events**: Real-time frontend updates with tool status

### ✅ **Comprehensive Coverage**  
- **Before Execution**: Tool name, parameters, start time
- **During Execution**: Progress updates and status
- **After Execution**: Results, execution time, success/error status
- **Summary**: Total tools, success rate, total execution time

### ✅ **Multi-Provider Support**
- **Anthropic Tools**: Full logging in `anthropic_tool.py`
- **OpenAI Tools**: Full logging in `openai_tool.py`
- **Executive Tool**: Orchestration-level logging in `exec_tool.py`

## 📋 Console Logging Format

### **Log Structure**
```
🔧 [PROVIDER_TOOL] HH:MM:SS - MESSAGE
```

### **Example Console Output**
```
🔧 [ANTHROPIC_TOOL] 14:23:15 - 🚀 Starting tool execution for query: 'What is AI in 2024?'
🔧 [ANTHROPIC_TOOL] 14:23:15 - 📋 Available tools: 3 tools
🔧 [ANTHROPIC_TOOL] 14:23:15 - 🔍 EXECUTING: search_google
🔧 [ANTHROPIC_TOOL] 14:23:15 -    Parameters: {'query': 'What is AI in 2024?'}
🔧 [ANTHROPIC_TOOL] 14:23:17 - ✅ SUCCESS: search_google completed in 1.85s
🔧 [ANTHROPIC_TOOL] 14:23:17 -    Result preview: AI in 2024 has seen remarkable advances...
🔧 [ANTHROPIC_TOOL] 14:23:18 - 📚 EXECUTING: search_learning_context  
🔧 [ANTHROPIC_TOOL] 14:23:18 -    Parameters: {'query': 'What is AI in 2024?'}
🔧 [ANTHROPIC_TOOL] 14:23:19 - ✅ SUCCESS: search_learning_context completed in 0.95s
🔧 [ANTHROPIC_TOOL] 14:23:19 - 📊 TOOL EXECUTION SUMMARY:
🔧 [ANTHROPIC_TOOL] 14:23:19 -    Total tools: 2
🔧 [ANTHROPIC_TOOL] 14:23:19 -    Successful: 2
🔧 [ANTHROPIC_TOOL] 14:23:19 -    Failed: 0
🔧 [ANTHROPIC_TOOL] 14:23:19 -    Total time: 2.80s
```

## 📡 Streaming Events

### **Event Types**

#### **1. Tool Execution Event**
```json
{
  "type": "tool_execution",
  "content": "🔍 Search completed (1.9s) - Found 2847 chars of results",
  "tool_name": "search_google", 
  "status": "completed",
  "execution_time": 1.85
}
```

#### **2. Tools Summary Event**
```json
{
  "type": "tools_summary",
  "content": "🏁 Tools completed: 2/2 successful (2.8s total)",
  "tools_executed": [
    {
      "name": "search_google",
      "status": "success",
      "execution_time": 1.85,
      "result_length": 2847
    },
    {
      "name": "search_learning_context", 
      "status": "success",
      "execution_time": 0.95,
      "result_length": 1243
    }
  ],
  "total_execution_time": 2.80
}
```

#### **3. Error Event**
```json
{
  "type": "tool_execution",
  "content": "❌ Search failed: API key invalid",
  "tool_name": "search_google",
  "status": "error", 
  "execution_time": 0.12
}
```

## 🛠️ Implementation Details

### **Files Modified**

#### **1. `exec_tool.py`**
- Added `tool_logger` configuration
- Enhanced console logging format
- Added execution time tracking

#### **2. `anthropic_tool.py`**
- Comprehensive logging in `_stream_tool_execution`
- Before/during/after execution logging
- Error handling with detailed context
- Summary statistics generation

#### **3. `openai_tool.py`**
- Tool call logging in `_handle_tool_calls` 
- Streaming tool execution logging
- Parameter and result tracking
- Error categorization

### **Logger Configuration**
```python
# Configure detailed logging for tool calls
tool_logger = logging.getLogger("tool_calls")
tool_logger.setLevel(logging.INFO)
if not tool_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('🔧 [PROVIDER_TOOL] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    tool_logger.addHandler(handler)
```

## 📊 Logged Information

### **For Each Tool Execution**

#### **Before Execution**
- ✅ Tool name and provider
- ✅ Input parameters (with truncation for readability)
- ✅ Start timestamp
- ✅ Available tools count

#### **During Execution**
- ✅ Real-time status updates
- ✅ Progress indicators for long-running tools
- ✅ Parameter validation results

#### **After Execution**
- ✅ Success/error status
- ✅ Execution time (precise to milliseconds)
- ✅ Result preview (first 200 chars)
- ✅ Result length statistics
- ✅ Error messages with context

#### **Summary Statistics**
- ✅ Total tools executed
- ✅ Success/failure counts
- ✅ Total execution time
- ✅ Performance metrics

## 🧪 Testing

### **Run Comprehensive Tests**
```bash
python test_tool_logging.py
```

### **Test Scenarios Covered**
1. **Learning Requests**: Tools like `search_learning_context`, `analyze_learning_opportunity`
2. **Search Requests**: Web search with `search_google`
3. **Context Requests**: Context retrieval with `get_context` 
4. **Error Scenarios**: API failures, invalid parameters, tool not found
5. **Multi-tool Scenarios**: Multiple tools in sequence

### **Expected Output**
The test script demonstrates:
- Console logging for all tool executions
- Streaming events with detailed status
- Error handling and recovery
- Performance metrics and summaries

## 🎛️ Configuration Options

### **Log Levels**
- `INFO`: Standard tool execution logging (default)
- `DEBUG`: Verbose parameter and result logging  
- `WARNING`: Only errors and warnings
- `ERROR`: Only critical failures

### **Streaming Control**
- Enable/disable tool execution events
- Configure result preview length
- Control summary frequency

### **Console Output**
- Customize log format and timestamps
- Filter by tool type or provider
- Adjust verbosity levels

## 📈 Benefits for Debugging

### **Server-Side Debugging**
- ✅ **Trace Tool Execution**: See exactly which tools are called and when
- ✅ **Performance Analysis**: Identify slow-performing tools
- ✅ **Error Diagnosis**: Get detailed context for failures
- ✅ **Parameter Validation**: Verify tool inputs are correct

### **Frontend Development**
- ✅ **Real-time Updates**: Show users what's happening
- ✅ **Progress Indicators**: Display tool execution progress  
- ✅ **Error Handling**: Provide meaningful error messages
- ✅ **Performance Feedback**: Show execution times to users

### **System Monitoring** 
- ✅ **Tool Usage Statistics**: Track which tools are used most
- ✅ **Performance Metrics**: Monitor average execution times
- ✅ **Error Rates**: Track tool failure rates
- ✅ **Resource Usage**: Understand tool resource consumption

## 🔍 Example: Learning Tool Execution

### **User Request**
```
"Công ty của tôi có 100 nhân viên và chúng tôi đang phát triển sản phẩm AI"
(My company has 100 employees and we're developing AI products)
```

### **Console Log Output**
```
🔧 [ANTHROPIC_TOOL] 14:30:15 - 🚀 Starting tool execution for query: 'Công ty của tôi có 100 nhân viên...'
🔧 [ANTHROPIC_TOOL] 14:30:15 - 📋 Available tools: 5 tools
🔧 [ANTHROPIC_TOOL] 14:30:15 - 📚 EXECUTING: search_learning_context
🔧 [ANTHROPIC_TOOL] 14:30:15 -    Parameters: {'query': 'Công ty của tôi có 100 nhân viên...'}
🔧 [ANTHROPIC_TOOL] 14:30:16 - ✅ SUCCESS: search_learning_context completed in 0.83s
🔧 [ANTHROPIC_TOOL] 14:30:16 -    Result preview: No similar knowledge found in database...
🔧 [ANTHROPIC_TOOL] 14:30:16 - 🧠 EXECUTING: analyze_learning_opportunity
🔧 [ANTHROPIC_TOOL] 14:30:16 -    Parameters: {'user_message': 'Công ty của tôi có 100 nhân viên...'}
🔧 [ANTHROPIC_TOOL] 14:30:18 - ✅ SUCCESS: analyze_learning_opportunity completed in 1.45s
🔧 [ANTHROPIC_TOOL] 14:30:18 -    Result preview: SHOULD_LEARN: High-value company information detected...
🔧 [ANTHROPIC_TOOL] 14:30:18 - 🤝 EXECUTING: request_learning_decision  
🔧 [ANTHROPIC_TOOL] 14:30:18 -    Reason: Learning analysis suggested saving content
🔧 [ANTHROPIC_TOOL] 14:30:19 - ✅ SUCCESS: request_learning_decision completed in 0.67s
🔧 [ANTHROPIC_TOOL] 14:30:19 -    Decision created for human approval
🔧 [ANTHROPIC_TOOL] 14:30:19 - 📊 TOOL EXECUTION SUMMARY:
🔧 [ANTHROPIC_TOOL] 14:30:19 -    Total tools: 3
🔧 [ANTHROPIC_TOOL] 14:30:19 -    Successful: 3
🔧 [ANTHROPIC_TOOL] 14:30:19 -    Failed: 0
🔧 [ANTHROPIC_TOOL] 14:30:19 -    Total time: 2.95s
```

### **Frontend Streaming Events**
```json
{"type": "tool_execution", "content": "📚 Learning context search completed (0.8s)", "tool_name": "search_learning_context", "status": "completed"}
{"type": "tool_execution", "content": "🧠 Learning analysis completed (1.5s)", "tool_name": "analyze_learning_opportunity", "status": "completed"}  
{"type": "tool_execution", "content": "🤝 Learning decision created (0.7s) - Awaiting human approval", "tool_name": "request_learning_decision", "status": "completed"}
{"type": "tools_summary", "content": "🏁 Tools completed: 3/3 successful (3.0s total)", "total_execution_time": 2.95}
```

## 🚀 Usage in Production

### **Enable Tool Logging**
Tool logging is **enabled by default** in the system. No additional configuration needed.

### **View Logs in Real-Time** 
```bash
# Watch server logs
tail -f app.log | grep "TOOL"

# Filter by tool type
tail -f app.log | grep "search_google"

# Monitor errors only  
tail -f app.log | grep "❌"
```

### **Frontend Integration**
Listen for tool execution events in your frontend:
```javascript
eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'tool_execution') {
    console.log(`Tool ${data.tool_name}: ${data.status}`);
    updateProgressIndicator(data);
  }
  
  if (data.type === 'tools_summary') {
    console.log(`Tools completed: ${data.content}`);
    hideProgressIndicators();
  }
});
```

## 🎯 Key Benefits Summary

### **🔍 Complete Visibility**
- See exactly when each tool is called
- Track all parameters and results
- Monitor execution times and performance
- Get detailed error context

### **🐛 Enhanced Debugging** 
- Trace tool execution flows
- Identify bottlenecks and failures
- Validate tool inputs and outputs
- Monitor system health

### **👥 Better User Experience**
- Real-time progress updates
- Transparent tool execution
- Meaningful error messages
- Performance feedback

### **📊 Data-Driven Insights**
- Tool usage statistics
- Performance metrics
- Error analysis
- Resource optimization

The tool call logging system transforms the `/tool/llm` endpoint from a "black box" into a **fully transparent, debuggable, and monitorable** system that provides clear visibility into every aspect of tool execution. 