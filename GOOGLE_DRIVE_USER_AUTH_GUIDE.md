# Google Drive User Authentication Guide

## Overview

This guide explains the new user-based Google Drive integration approach that uses the user's Google OAuth credentials to directly access their Google Drive instead of requiring database integrations.

## Architecture Changes

### Previous Approach (Database Integrations)
- Required organization-level Google Drive integrations stored in database
- Users had to create and manage integrations through admin interface
- API endpoints required `integration_id` parameters
- Complex setup with multiple database tables

### New Approach (User Authentication)
- Uses user's existing Google OAuth credentials
- No database integrations required
- Direct access to user's Google Drive with their permission
- Simplified API using access tokens

## API Endpoints

### 1. Get Auth Configuration
```
GET /google-drive/auth-config
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "client_id": "your-google-client-id",
  "scopes": [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile"
  ],
  "redirect_uri": "http://localhost:8000/auth/google/callback",
  "response_type": "code",
  "access_type": "offline",
  "prompt": "consent"
}
```

### 2. Test Connection
```
POST /google-drive/test-connection
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "access_token": "ya29.a0AfH6SMC...",
  "refresh_token": "1//04-rK5..."
}
```

**Response:**
```json
{
  "connected": true,
  "user_email": "user@example.com",
  "user_name": "John Doe",
  "storage_quota": {
    "limit": "17179869184",
    "usage": "1234567890"
  },
  "message": "Google Drive connection successful"
}
```

### 3. List Folder Contents
```
POST /google-drive/list-folder
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "access_token": "ya29.a0AfH6SMC...",
  "refresh_token": "1//04-rK5...",
  "folder_id": "root"
}
```

**Response:**
```json
{
  "folder": {
    "id": "root",
    "name": "Root",
    "webViewLink": null,
    "files": [
      {
        "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        "name": "Example Document",
        "mimeType": "application/vnd.google-apps.document",
        "size": null,
        "modifiedTime": "2024-01-15T10:30:00.000Z",
        "webViewLink": "https://docs.google.com/document/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
        "parents": ["root"],
        "isFolder": false,
        "thumbnailLink": null
      }
    ],
    "subfolders": [
      {
        "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        "name": "My Folder",
        "webViewLink": "https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        "files": [],
        "subfolders": []
      }
    ]
  }
}
```

### 4. Search Folders
```
POST /google-drive/search-folders
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "access_token": "ya29.a0AfH6SMC...",
  "refresh_token": "1//04-rK5...",
  "query": "Documents"
}
```

### 5. Ingest Folder (Background Processing)
```
POST /google-drive/ingest-folder
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "access_token": "ya29.a0AfH6SMC...",
  "refresh_token": "1//04-rK5...",
  "brainGraphId": "brain-graph-uuid",
  "folderId": "folder-id-to-ingest",
  "folderName": "My Documents",
  "fileTypes": [
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/pdf"
  ],
  "recursive": true
}
```

## Frontend Implementation

### 1. Google OAuth Setup

```javascript
// Get auth configuration from backend
const getAuthConfig = async () => {
  const response = await fetch('/google-drive/auth-config', {
    headers: {
      'Authorization': `Bearer ${userToken}`
    }
  });
  return response.json();
};

// Initiate Google OAuth flow
const initiateGoogleAuth = async () => {
  const config = await getAuthConfig();
  
  const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
    `client_id=${config.client_id}&` +
    `redirect_uri=${encodeURIComponent(config.redirect_uri)}&` +
    `response_type=${config.response_type}&` +
    `scope=${encodeURIComponent(config.scopes.join(' '))}&` +
    `access_type=${config.access_type}&` +
    `prompt=${config.prompt}`;
  
  window.location.href = authUrl;
};
```

### 2. Handle OAuth Callback

```javascript
// Handle the OAuth callback to exchange code for tokens
const handleGoogleCallback = async (code) => {
  const tokenResponse = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      client_id: config.client_id,
      client_secret: config.client_secret, // Should be handled by backend
      code: code,
      grant_type: 'authorization_code',
      redirect_uri: config.redirect_uri,
    }),
  });
  
  const tokens = await tokenResponse.json();
  
  // Store tokens securely (consider using secure storage)
  localStorage.setItem('google_access_token', tokens.access_token);
  localStorage.setItem('google_refresh_token', tokens.refresh_token);
  
  return tokens;
};
```

### 3. Use Google Drive API

```javascript
// Test connection
const testConnection = async () => {
  const response = await fetch('/google-drive/test-connection', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${userToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      access_token: localStorage.getItem('google_access_token'),
      refresh_token: localStorage.getItem('google_refresh_token')
    })
  });
  
  return response.json();
};

// List folder contents
const listFolder = async (folderId = 'root') => {
  const response = await fetch('/google-drive/list-folder', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${userToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      access_token: localStorage.getItem('google_access_token'),
      refresh_token: localStorage.getItem('google_refresh_token'),
      folder_id: folderId
    })
  });
  
  return response.json();
};
```

## Environment Variables Required

```env
# Google OAuth Configuration
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
BASE_URL=http://localhost:8000

# JWT Configuration (existing)
JWT_SECRET=your-jwt-secret
```

## Security Considerations

1. **Token Storage**: Store Google OAuth tokens securely on the frontend (consider using secure httpOnly cookies)
2. **Token Refresh**: Implement automatic token refresh when access tokens expire
3. **Scope Minimization**: Only request necessary scopes (drive.readonly for read-only access)
4. **HTTPS**: Always use HTTPS in production for OAuth flows
5. **CORS**: Ensure proper CORS configuration for your frontend domain

## Benefits of New Approach

1. **Simplified Setup**: No database integrations required
2. **User-Centric**: Each user manages their own Google Drive access
3. **Real-time Access**: Direct access to user's current Google Drive state
4. **Reduced Complexity**: Fewer database tables and management overhead
5. **Better UX**: Users can immediately access their own files without admin setup

## Migration Guide

If you're migrating from the old integration-based approach:

1. Update frontend to use new OAuth flow
2. Remove integration management UI
3. Update API calls to use access tokens instead of integration IDs
4. Clean up old integration database records (optional)

## Troubleshooting

### Common Issues

1. **"Invalid or expired credentials"**: Refresh tokens or re-authenticate
2. **"Google OAuth client credentials not configured"**: Check environment variables
3. **"Token has expired"**: Implement token refresh logic
4. **CORS errors**: Ensure proper CORS configuration

### Debug Steps

1. Check browser console for OAuth errors
2. Verify environment variables are set
3. Test token validity with test-connection endpoint
4. Check backend logs for detailed error messages 