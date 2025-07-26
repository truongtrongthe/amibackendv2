# Google Drive Permission Fix Guide

## 🐛 **Error Summary**

**Error Type**: Google Drive API 404 - File Not Found
**Error Message**: 
```
HttpError 404 when requesting https://www.googleapis.com/drive/v3/files/1Dq78EV3y0EK98cJ-7oAg5GK8FN-bbEVbh0HRXV0oz5A?fields=id%2Cname%2CmimeType&alt=json returned "File not found: 1Dq78EV3y0EK98cJ-7oAg5GK8FN-bbEVbh0HRXV0oz5A."
```

## 🔍 **Root Cause Analysis**

### **What Happened:**
1. ✅ **Agent Detection**: Successfully detected Google Drive request
2. ✅ **File ID Extraction**: Correctly extracted file ID `1Dq78EV3y0EK98cJ-7oAg5GK8FN-bbEVbh0HRXV0oz5A`
3. ❌ **Permission Denied**: Service account cannot access the file

### **The Problem:**
- **Service Account**: The Google Drive service account lacks permission to access the file
- **File Owner**: The file is owned by a different Google account
- **Sharing**: The file is not shared with the service account

## 🛠️ **Solutions to Fix This**

### **Solution 1: Share File with Service Account**

#### **Step 1: Find Service Account Email**
```bash
# Check your credentials file
cat amiai-462810-e87677bfd1b0.json | grep client_email
```

**Expected Output:**
```json
"client_email": "file-access-agent@your-project.iam.gserviceaccount.com"
```

#### **Step 2: Share the File**
1. **Open the Google Drive file**:
   ```
   https://docs.google.com/document/d/1Dq78EV3y0EK98cJ-7oAg5GK8FN-bbEVbh0HRXV0oz5A/edit
   ```

2. **Click "Share" button** (top right)

3. **Add Service Account**:
   - Enter the service account email from Step 1
   - Set permission to "Viewer"
   - Uncheck "Notify people" (optional)
   - Click "Share"

4. **Verify Sharing**:
   - The file should now be accessible to the service account

### **Solution 2: Use the New Check File Access Tool**

The agent now has a `check_file_access` tool to diagnose permission issues:

**Usage Examples:**
```
User: "Check if I can access this file: https://docs.google.com/document/d/1Dq78EV3y0EK98cJ-7oAg5GK8FN-bbEVbh0HRXV0oz5A/edit"

Agent will call: check_file_access(drive_link="https://docs.google.com/document/d/1Dq78EV3y0EK98cJ-7oAg5GK8FN-bbEVbh0HRXV0oz5A/edit")
```

**Expected Output:**
```
❌ File Access: DENIED

Error: File not found or access denied.
File ID: 1Dq78EV3y0EK98cJ-7oAg5GK8FN-bbEVbh0HRXV0oz5A

To fix this:
1. Ensure the file exists
2. Share the file with the service account
3. Check file permissions
```

### **Solution 3: Batch Share Multiple Files**

If you have many files to share:

#### **Using Google Drive Web Interface:**
1. Select multiple files/folders
2. Right-click → "Share"
3. Add service account email
4. Set permissions to "Viewer"
5. Click "Share"

#### **Using Google Drive API (Advanced):**
```python
# Example script to share files with service account
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

def share_files_with_service_account(file_ids, service_account_email):
    credentials = Credentials.from_service_account_file(
        'amiai-462810-e87677bfd1b0.json',
        scopes=['https://www.googleapis.com/auth/drive']
    )
    
    service = build('drive', 'v3', credentials=credentials)
    
    for file_id in file_ids:
        permission = {
            'type': 'user',
            'role': 'reader',
            'emailAddress': service_account_email
        }
        
        service.permissions().create(
            fileId=file_id,
            body=permission,
            fields='id'
        ).execute()
        
        print(f"Shared file {file_id} with {service_account_email}")
```

## 🔧 **Enhanced Error Handling**

### **New Error Messages:**
The system now provides specific error messages for different issues:

1. **404 - File Not Found**:
   ```
   Error: File not found or access denied. Please ensure the file is shared with the service account.
   ```

2. **403 - Forbidden**:
   ```
   Error: Access denied. Please check file sharing permissions.
   ```

3. **401 - Unauthorized**:
   ```
   Error: Authentication failed. Please check Google Drive credentials.
   ```

### **Agent Behavior:**
- **Automatic Detection**: Agent detects permission errors
- **Diagnostic Tool**: Uses `check_file_access` to diagnose issues
- **Clear Guidance**: Provides specific steps to fix the problem

## 📋 **Troubleshooting Checklist**

### **For Individual Files:**
- [ ] File exists and is not deleted
- [ ] File is shared with service account email
- [ ] Service account has "Viewer" or higher permissions
- [ ] File is not in a restricted shared drive

### **For Folders:**
- [ ] Folder is shared with service account email
- [ ] All files in folder are accessible
- [ ] Folder permissions are inherited correctly

### **For Service Account:**
- [ ] Service account credentials are valid
- [ ] Google Drive API is enabled
- [ ] Service account has proper scopes

## 🚀 **Quick Fix Commands**

### **1. Check File Access:**
```bash
# Test if a specific file is accessible
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "https://www.googleapis.com/drive/v3/files/FILE_ID?fields=id,name"
```

### **2. List Service Account Permissions:**
```bash
# Check what files the service account can access
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "https://www.googleapis.com/drive/v3/files?fields=files(id,name)"
```

### **3. Share File via API:**
```bash
# Share a file with the service account
curl -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"role":"reader","type":"user","emailAddress":"SERVICE_ACCOUNT_EMAIL"}' \
  "https://www.googleapis.com/drive/v3/files/FILE_ID/permissions"
```

## 📊 **Expected Results After Fix**

### **Before Fix:**
```
❌ File Access: DENIED
Error: File not found or access denied.
```

### **After Fix:**
```
✅ File Access: SUCCESS
📄 File Information:
   Name: Document Name
   ID: 1Dq78EV3y0EK98cJ-7oAg5GK8FN-bbEVbh0HRXV0oz5A
   Type: application/vnd.google-apps.document
   Size: 45.2 KB
   Owner: User Name
   Status: File is accessible and ready to read
```

## 🎯 **Best Practices**

1. **Pre-share Files**: Share files with service account before requesting analysis
2. **Use Check Tool**: Use `check_file_access` to verify permissions first
3. **Batch Operations**: Share entire folders instead of individual files
4. **Monitor Access**: Regularly check service account permissions
5. **Error Handling**: Always provide clear error messages with actionable steps

The enhanced error handling and new diagnostic tools will help you quickly identify and resolve permission issues! 🚀 