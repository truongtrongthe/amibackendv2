# Google Drive Setup Guide

This guide shows how to set up Google Drive integration for the File Access Tool.

## 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing project
3. Enable the Google Drive API:
   - Go to "APIs & Services" → "Library"
   - Search for "Google Drive API"
   - Click "Enable"

## 2. Create Service Account

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "Service Account"
3. Fill in service account details:
   - Name: `file-access-agent`
   - Description: `Service account for agent file access`
4. Click "Create and Continue"
5. Skip role assignment (optional)
6. Click "Done"

## 3. Generate Service Account Key

1. Click on the created service account
2. Go to "Keys" tab
3. Click "Add Key" → "Create new key"
4. Select "JSON" format
5. Download the JSON file
6. Rename it to `google_drive_credentials.json`
7. Place it in your project root directory

## 4. Share Google Drive Folders

For the agent to access files, you need to share folders with the service account:

1. Open Google Drive
2. Right-click on the folder you want to share
3. Click "Share"
4. Add the service account email (found in the JSON credentials file)
5. Give "Viewer" permissions
6. Click "Send"

## 5. Environment Setup

Set the credentials path (optional):
```bash
export GOOGLE_DRIVE_CREDENTIALS_PATH="/path/to/google_drive_credentials.json"
```

If not set, it defaults to `google_drive_credentials.json` in the project root.

## 6. Install Dependencies

```bash
pip install -r file_access_requirements.txt
```

## 7. Test the Setup

You can test if everything works:

```python
from file_access_tool import FileAccessTool

tool = FileAccessTool()
print(tool.list_gdrive_files(folder_name="Reports"))
```

## Usage Examples

### List files in Google Drive folder:
```python
# By folder name
result = tool.list_gdrive_files(folder_name="Sales Reports")

# By folder ID
result = tool.list_gdrive_files(folder_id="1ABCdefGHI...")
```

### Read files from Google Drive:
```python
# Read DOCX by name
content = tool.read_gdrive_docx(file_name="sales-info.docx", folder_name="Reports")

# Read PDF by file ID
content = tool.read_gdrive_pdf(file_id="1XYZabc...")
```

### Read files using Google Drive links (Easiest method):
```python
# Read DOCX from sharing link
content = tool.read_gdrive_link_docx("https://drive.google.com/file/d/1ABC123xyz/view?usp=sharing")

# Read PDF from sharing link  
content = tool.read_gdrive_link_pdf("https://drive.google.com/file/d/1XYZ789abc/view?usp=sharing")

# Get file information from link
info = tool.get_file_info_from_link("https://drive.google.com/file/d/1ABC123xyz/view")
```

**Supported Link Formats:**
- Standard sharing links: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
- Google Docs links: `https://docs.google.com/document/d/FILE_ID/edit?usp=sharing`
- Open links: `https://drive.google.com/open?id=FILE_ID`
- Direct file IDs: `1ABC123xyz789` (just the ID itself)

### Find files:
```python
# Find file info
info = tool.find_file_by_name("quarterly-report.docx", folder_name="Sales")
```

## Security Notes

- Keep your `google_drive_credentials.json` file secure
- Add it to `.gitignore` to avoid committing to version control
- The service account only has read access to shared folders
- Files outside shared folders cannot be accessed

## Troubleshooting

### "Credentials not found" error:
- Check if `google_drive_credentials.json` exists in project root
- Verify the `GOOGLE_DRIVE_CREDENTIALS_PATH` environment variable

### "File not found" error:
- Ensure the folder/file is shared with the service account email
- Check the folder name or ID is correct
- Verify the file exists in Google Drive

### "Permission denied" error:
- Make sure the service account has at least "Viewer" permission
- Check if the Google Drive API is enabled in your project 