# Google Drive Folder Reading Guide

## Overview

Your AI agent can now read entire Google Drive folders, not just individual files! This powerful capability allows you to analyze multiple documents at once for comprehensive insights.

## How the Agent Knows About Folder Reading

The agent is automatically aware of folder reading capabilities through:

1. **System Prompt Instructions**: The agent's system prompt includes specific instructions for folder reading
2. **Tool Descriptions**: The `read_gdrive_folder` tool is included in the available tools
3. **Request Detection**: The agent detects folder-related keywords and automatically enables file access tools

## Usage Examples

### Basic Folder Reading

**User Request:**
```
"Read all documents in the 'Reports' folder"
```

**Agent Action:**
- Automatically calls `read_gdrive_folder(folder_name="Reports")`
- Reads all DOCX and PDF files in the folder
- Returns combined content for analysis

### Specific File Types

**User Request:**
```
"Read all PDF files in the 'Sales' folder and provide a summary"
```

**Agent Action:**
- Calls `read_gdrive_folder(folder_name="Sales", file_types=["pdf"])`
- Reads only PDF files from the folder
- Provides analysis of PDF content only

### Folder by ID

**User Request:**
```
"Read all documents from Google Drive folder ID 1ABC123DEF456"
```

**Agent Action:**
- Calls `read_gdrive_folder(folder_id="1ABC123DEF456")`
- Reads all supported files from the specified folder

### Mixed Requests

**User Request:**
```
"First read the 'Business Plans' folder, then analyze this specific document: https://docs.google.com/document/d/ABC123/edit"
```

**Agent Action:**
1. Calls `read_gdrive_folder(folder_name="Business Plans")`
2. Calls `read_gdrive_link_docx(drive_link="https://docs.google.com/document/d/ABC123/edit")`
3. Provides comprehensive analysis of both folder content and specific document

## Supported File Types

The folder reading tool supports:
- **DOCX files** (Microsoft Word documents)
- **PDF files** (Portable Document Format)
- **Google Docs** (automatically exported as DOCX)

## Agent Detection Keywords

The agent automatically detects folder reading requests when you use these keywords:
- "folder" or "folders"
- "directory"
- "documents in"
- "files in"
- "google drive"
- "gdrive"

## Output Format

When reading a folder, the agent returns:
```
=== FOLDER CONTENTS ===
Reading X supported files:

--- FILE 1: document1.docx ---
[Content of first file]

--- FILE 2: report2.pdf ---
[Content of second file]

[Additional files...]
```

## Benefits of Folder Reading

1. **Comprehensive Analysis**: Analyze multiple related documents at once
2. **Pattern Recognition**: Identify trends across multiple documents
3. **Efficient Processing**: No need to read files one by one
4. **Context Building**: Understand relationships between documents

## Error Handling

The agent handles various error scenarios:
- **Folder not found**: Clear error message with suggestions
- **No supported files**: Lists available file types
- **Permission issues**: Guidance on sharing settings
- **Individual file errors**: Continues processing other files

## Best Practices

1. **Be Specific**: Mention the folder name clearly
2. **Specify File Types**: If you only want certain file types, mention them
3. **Provide Context**: Tell the agent what kind of analysis you need
4. **Use Folder Names**: Use descriptive folder names for better results

## Example Use Cases

### Business Analysis
```
"Read all documents in the 'Q4 Reports' folder and provide a quarterly business summary"
```

### Research Projects
```
"Read all PDF files in the 'Research Papers' folder and identify key findings"
```

### Project Management
```
"Read all documents in the 'Project Documentation' folder and create a project overview"
```

### Compliance Review
```
"Read all files in the 'Compliance Documents' folder and check for any missing requirements"
```

## Technical Details

### Tool Parameters
- `folder_name`: Name of the Google Drive folder
- `folder_id`: Google Drive folder ID (alternative to folder_name)
- `file_types`: List of file types to read (default: ['docx', 'pdf'])

### Security
- Only reads files that the service account has access to
- Respects Google Drive sharing permissions
- No write access - read-only operations only

### Performance
- Processes files sequentially for reliability
- Handles large folders with multiple files
- Provides progress feedback for long operations

## Troubleshooting

### Common Issues

1. **"Folder not found"**
   - Check folder name spelling
   - Ensure folder is shared with service account
   - Verify folder exists in Google Drive

2. **"No supported files found"**
   - Check if folder contains DOCX or PDF files
   - Verify file permissions
   - Try listing files first with `list_gdrive_files`

3. **"Service not initialized"**
   - Check Google Drive credentials setup
   - Verify service account configuration
   - Ensure required dependencies are installed

### Getting Help

If you encounter issues:
1. Check the Google Drive setup guide
2. Verify folder sharing permissions
3. Test with a simple file reading request first
4. Check the logs for detailed error messages

## Testing

You can test folder reading capabilities using the provided test script:

```bash
python test_folder_reading.py
```

This will run various test scenarios to verify the functionality works correctly. 