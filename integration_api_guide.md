# Organization Integrations API Guide

## Overview

This document provides implementation details for integrating with the Organization Integrations API. These endpoints allow for managing third-party service integrations (like Facebook, Hubspot, Salesforce) for your organization.

## Base URL

All endpoints are relative to the base URL: `http://your-api-domain/`

## Authentication

Authentication details should be included with each request (to be implemented).

## API Endpoints

### 1. List Organization Integrations

Retrieves all integrations for an organization.

**Endpoint:** `GET /organization-integrations`

**Query Parameters:**
- `org_id` (required): UUID of the organization
- `active_only` (optional): Boolean to filter only active integrations

**Example Request:**
```javascript
const getIntegrations = async (orgId, activeOnly = false) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/organization-integrations?org_id=${orgId}&active_only=${activeOnly}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          // Add authorization headers
        },
      }
    );
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    const data = await response.json();
    return data.integrations;
  } catch (error) {
    console.error('Failed to fetch integrations:', error);
    throw error;
  }
};
```

**Response:** 
```json
{
  "integrations": [
    {
      "id": "55fcad75-d3d2-4cf2-9313-8974f57dbc93",
      "org_id": "3f7cda9c-a6f2-4475-8b1e-77a51f4e0a6b",
      "integration_type": "facebook",
      "name": "Facebook Messenger Integration",
      "is_active": true,
      "api_base_url": "https://graph.facebook.com/v17.0",
      "webhook_url": "https://example.com/webhook",
      "created_at": "2023-06-14T12:30:45.123Z",
      "updated_at": "2023-06-14T12:30:45.123Z"
    }
  ]
}
```

### 2. Get Integration Details

Retrieves details for a specific integration.

**Endpoint:** `GET /organization-integration/{integration_id}`

**Path Parameters:**
- `integration_id` (required): UUID of the integration

**Example Request:**
```javascript
const getIntegrationDetails = async (integrationId) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/organization-integration/${integrationId}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          // Add authorization headers
        },
      }
    );
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    const data = await response.json();
    return data.integration;
  } catch (error) {
    console.error('Failed to fetch integration details:', error);
    throw error;
  }
};
```

**Response:**
```json
{
  "integration": {
    "id": "55fcad75-d3d2-4cf2-9313-8974f57dbc93",
    "org_id": "3f7cda9c-a6f2-4475-8b1e-77a51f4e0a6b",
    "integration_type": "facebook",
    "name": "Facebook Messenger Integration",
    "is_active": true,
    "api_base_url": "https://graph.facebook.com/v17.0",
    "webhook_url": "https://example.com/webhook",
    "api_key": "abc123def456",
    "api_secret": "••••••",
    "access_token": "••••••",
    "refresh_token": "••••••",
    "token_expires_at": "2024-06-14T12:30:45.123Z",
    "config": {
      "page_id": "123456789012345",
      "verify_token": "your_verification_token"
    },
    "created_at": "2023-06-14T12:30:45.123Z",
    "updated_at": "2023-06-14T12:30:45.123Z"
  }
}
```

### 3. Create Integration

Creates a new integration for an organization.

**Endpoint:** `POST /create-organization-integration`

**Request Body:**
```json
{
  "org_id": "3f7cda9c-a6f2-4475-8b1e-77a51f4e0a6b",
  "integration_type": "facebook", 
  "name": "Facebook Messenger Integration",
  "api_base_url": "https://graph.facebook.com/v17.0",
  "webhook_url": "https://example.com/webhook",
  "webhook_verify_token": "your_custom_verify_token",
  "api_key": "your_api_key",
  "api_secret": "your_api_secret",
  "access_token": "your_access_token",
  "refresh_token": "your_refresh_token",
  "config": {
    "page_id": "123456789012345",
    "verify_token": "your_verification_token"
  },
  "is_active": false
}
```

**Example Request:**
```javascript
const createIntegration = async (integrationData) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/create-organization-integration`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add authorization headers
        },
        body: JSON.stringify(integrationData),
      }
    );
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    const data = await response.json();
    return data.integration;
  } catch (error) {
    console.error('Failed to create integration:', error);
    throw error;
  }
};
```

**Response:**
```json
{
  "message": "Integration created successfully",
  "integration": {
    "id": "55fcad75-d3d2-4cf2-9313-8974f57dbc93",
    "org_id": "3f7cda9c-a6f2-4475-8b1e-77a51f4e0a6b",
    "integration_type": "facebook",
    "name": "Facebook Messenger Integration",
    "is_active": false,
    "api_base_url": "https://graph.facebook.com/v17.0",
    "webhook_url": "https://example.com/webhook",
    "api_key": "your_api_key",
    "api_secret": "••••••",
    "access_token": "••••••",
    "refresh_token": "••••••",
    "config": {
      "page_id": "123456789012345",
      "verify_token": "your_verification_token"
    },
    "created_at": "2023-06-14T12:30:45.123Z",
    "updated_at": "2023-06-14T12:30:45.123Z"
  }
}
```

### 4. Update Integration

Updates an existing integration.

**Endpoint:** `POST /update-organization-integration`

**Request Body:**
```json
{
  "id": "55fcad75-d3d2-4cf2-9313-8974f57dbc93",
  "name": "Updated Facebook Integration",
  "webhook_url": "https://example.com/updated-webhook",
  "config": {
    "page_id": "987654321098765",
    "verify_token": "updated_verify_token"
  }
}
```

**Example Request:**
```javascript
const updateIntegration = async (updateData) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/update-organization-integration`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add authorization headers
        },
        body: JSON.stringify(updateData),
      }
    );
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    const data = await response.json();
    return data.integration;
  } catch (error) {
    console.error('Failed to update integration:', error);
    throw error;
  }
};
```

**Response:**
```json
{
  "message": "Integration updated successfully",
  "integration": {
    "id": "55fcad75-d3d2-4cf2-9313-8974f57dbc93",
    "org_id": "3f7cda9c-a6f2-4475-8b1e-77a51f4e0a6b",
    "integration_type": "facebook",
    "name": "Updated Facebook Integration",
    "is_active": false,
    "webhook_url": "https://example.com/updated-webhook",
    "config": {
      "page_id": "987654321098765",
      "verify_token": "updated_verify_token"
    },
    "updated_at": "2023-06-14T13:45:12.789Z"
  }
}
```

### 5. Toggle Integration

Activates or deactivates an integration.

**Endpoint:** `POST /toggle-organization-integration`

**Request Body:**
```json
{
  "id": "55fcad75-d3d2-4cf2-9313-8974f57dbc93",
  "active": true
}
```

**Example Request:**
```javascript
const toggleIntegration = async (integrationId, active) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/toggle-organization-integration`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add authorization headers
        },
        body: JSON.stringify({
          id: integrationId,
          active: active
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    const data = await response.json();
    return data.integration;
  } catch (error) {
    console.error('Failed to toggle integration:', error);
    throw error;
  }
};
```

**Response:**
```json
{
  "message": "Integration activated successfully",
  "integration": {
    "id": "55fcad75-d3d2-4cf2-9313-8974f57dbc93",
    "org_id": "3f7cda9c-a6f2-4475-8b1e-77a51f4e0a6b",
    "integration_type": "facebook",
    "name": "Updated Facebook Integration",
    "is_active": true,
    "updated_at": "2023-06-14T14:15:32.456Z"
  }
}
```

### 6. Delete Integration

Deletes an integration.

**Endpoint:** `POST /delete-organization-integration`

**Request Body:**
```json
{
  "id": "55fcad75-d3d2-4cf2-9313-8974f57dbc93"
}
```

**Example Request:**
```javascript
const deleteIntegration = async (integrationId) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/delete-organization-integration`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add authorization headers
        },
        body: JSON.stringify({
          id: integrationId
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    const data = await response.json();
    return data.message;
  } catch (error) {
    console.error('Failed to delete integration:', error);
    throw error;
  }
};
```

**Response:**
```json
{
  "message": "Integration deleted successfully"
}
```

## Integration Types

The API currently supports the following integration types:
- `odoo_crm`
- `hubspot`
- `salesforce`
- `facebook`
- `other`

## Facebook Messenger Integration

### Overview

The Facebook Messenger integration allows your organization to send and receive messages through Facebook Messenger. Each organization can connect their own Facebook page and receive messages from users who interact with that page.

### Setting Up Facebook Integration (Simplified Workflow)

1. **Create a Facebook App**:
   - Go to [Facebook Developers](https://developers.facebook.com/) and create a new app
   - Choose "Business" as the app type
   - Set up Messenger product in your app
   - Configure necessary permissions

2. **Get Page Access Token and Page ID**:
   - Generate a Page Access Token from the Facebook app
   - Note your Facebook Page ID (found in your Facebook Page settings or URL) - this is optional but recommended

3. **Create Integration in Your System**:
   - Use the `/create-organization-integration` endpoint to create a Facebook integration
   - **Only these fields are required**:
     - `org_id`: Your organization UUID
     - `integration_type`: Set to "facebook"
     - `name`: A display name for the integration
     - `access_token`: The Page Access Token from Facebook
   - **Optional fields**:
     - `config`: A JSON object that can contain `{"page_id": "123456789012345"}` (recommended but optional)
     - `is_active`: Boolean indicating if the integration should be active immediately
   - Everything else is auto-generated

4. **Configure Webhook in Facebook**:
   - After creating the integration, you'll receive the auto-generated:
     - `webhook_url`: URL where Facebook will send webhook events
     - `webhook_verify_token`: Security token for webhook verification
   - Go back to your Facebook App settings
   - Set up a webhook subscription for your page:
     - Paste the `webhook_url` as your Callback URL
     - Paste the `webhook_verify_token` as your Verify Token
     - Select these webhook fields: `messages`, `messaging_postbacks`, `message_deliveries`

### Example Integration Creation Request (Minimal)

```json
{
  "org_id": "3f7cda9c-a6f2-4475-8b1e-77a51f4e0a6b",
  "integration_type": "facebook", 
  "name": "Company Facebook Page",
  "access_token": "EAABbmNuhsFYBACl1zUIfRQP7rqZASPqBbAMy...",
  "is_active": true
}
```

### Example Integration Creation Response

```json
{
  "message": "Integration created successfully",
  "integration": {
    "id": "55fcad75-d3d2-4cf2-9313-8974f57dbc93",
    "org_id": "3f7cda9c-a6f2-4475-8b1e-77a51f4e0a6b",
    "integration_type": "facebook",
    "name": "Company Facebook Page",
    "is_active": true,
    "webhook_url": "https://api.yourdomain.com/webhook?org_id=3f7cda9c-a6f2-4475-8b1e-77a51f4e0a6b",
    "webhook_verify_token": "BW6GjtM8pL3xK7Qr94zFhN5sT2dYcX_VJk1aZoD",
    "access_token": "••••••",
    "config": {},
    "created_at": "2023-06-14T12:30:45.123Z",
    "updated_at": "2023-06-14T12:30:45.123Z"
  }
}
```

### Understanding API_BASE_URL

There are two different URL concepts in this integration:

1. **Your API Base URL**: This is your server's URL where:
   - Your API endpoints are hosted (e.g., `/create-organization-integration`)
   - Facebook will send webhook events to your `/webhook` endpoint
   - This URL is used to generate the webhook_url value automatically
   - It's configured in your server's environment variables

2. **Frontend API Base URL**: In frontend code, when making requests to your API:
   ```javascript
   const API_BASE_URL = 'https://api.yourdomain.com';
   // Used to call your endpoints
   fetch(`${API_BASE_URL}/create-organization-integration`, {...});
   ```

Never expose or request the user to provide their own URL - your system handles all the URL generation automatically.

## Error Handling

All API requests should include proper error handling. The API returns appropriate HTTP status codes for different error conditions:

- `400 Bad Request`: Invalid input (missing required fields, invalid format)
- `404 Not Found`: Resource (integration) not found
- `500 Internal Server Error`: Server-side errors

Example error response:
```json
{
  "error": "org_id, integration_type, and name are required"
}
```

## React Implementation Example

Here's an example of a React component that manages integrations:

```jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://your-api-domain';

const IntegrationsManager = ({ orgId }) => {
  const [integrations, setIntegrations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch integrations on component mount
  useEffect(() => {
    const fetchIntegrations = async () => {
      try {
        setLoading(true);
        const response = await axios.get(
          `${API_BASE_URL}/organization-integrations?org_id=${orgId}`
        );
        setIntegrations(response.data.integrations);
        setError(null);
      } catch (err) {
        setError('Failed to fetch integrations');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchIntegrations();
  }, [orgId]);

  // Toggle integration active status
  const handleToggleIntegration = async (integrationId, currentStatus) => {
    try {
      const response = await axios.post(
        `${API_BASE_URL}/toggle-organization-integration`,
        {
          id: integrationId,
          active: !currentStatus
        }
      );
      
      // Update the local state with the updated integration
      setIntegrations(integrations.map(integration => 
        integration.id === integrationId 
          ? { ...integration, is_active: !currentStatus } 
          : integration
      ));
      
    } catch (err) {
      setError('Failed to toggle integration');
      console.error(err);
    }
  };

  // Delete integration
  const handleDeleteIntegration = async (integrationId) => {
    if (window.confirm('Are you sure you want to delete this integration?')) {
      try {
        await axios.post(
          `${API_BASE_URL}/delete-organization-integration`,
          {
            id: integrationId
          }
        );
        
        // Remove the deleted integration from the state
        setIntegrations(integrations.filter(
          integration => integration.id !== integrationId
        ));
        
      } catch (err) {
        setError('Failed to delete integration');
        console.error(err);
      }
    }
  };

  if (loading) return <div>Loading integrations...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="integrations-container">
      <h2>Organization Integrations</h2>
      
      {integrations.length === 0 ? (
        <p>No integrations found. Create one to get started.</p>
      ) : (
        <ul className="integrations-list">
          {integrations.map(integration => (
            <li key={integration.id} className="integration-item">
              <div className="integration-header">
                <h3>{integration.name}</h3>
                <span className={`status ${integration.is_active ? 'active' : 'inactive'}`}>
                  {integration.is_active ? 'Active' : 'Inactive'}
                </span>
              </div>
              
              <div className="integration-details">
                <p>Type: {integration.integration_type}</p>
                {integration.webhook_url && (
                  <p>Webhook URL: {integration.webhook_url}</p>
                )}
              </div>
              
              <div className="integration-actions">
                <button
                  onClick={() => handleToggleIntegration(integration.id, integration.is_active)}
                  className="toggle-btn"
                >
                  {integration.is_active ? 'Deactivate' : 'Activate'}
                </button>
                <button
                  onClick={() => handleDeleteIntegration(integration.id)}
                  className="delete-btn"
                >
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
      
      <button className="add-integration-btn">Add New Integration</button>
    </div>
  );
};

export default IntegrationsManager;
```

## Forms for Creating/Editing Integrations

For creating and editing integrations, you'll need forms that collect the appropriate information based on the integration type. Each integration type might require different fields.

### Example Simplified Facebook Integration Form

```jsx
// Form fields for Facebook integration - minimal required fields
const facebookFormFields = [
  { name: 'name', label: 'Integration Name', type: 'text', required: true },
  { name: 'access_token', label: 'Page Access Token', type: 'text', required: true },
  { name: 'page_id', label: 'Facebook Page ID', type: 'text', required: false, configField: true },
  { name: 'is_active', label: 'Activate Now', type: 'checkbox', defaultValue: true }
];

// Example form component
const FacebookIntegrationForm = ({ orgId, onSubmit }) => {
  const [formData, setFormData] = useState({
    name: '',
    access_token: '',
    page_id: '',
    is_active: true
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Prepare the request data
    const integrationData = {
      org_id: orgId,
      integration_type: 'facebook',
      name: formData.name,
      access_token: formData.access_token,
      is_active: formData.is_active
    };
    
    // Only add page_id to config if it's provided
    if (formData.page_id) {
      integrationData.config = {
        page_id: formData.page_id
      };
    }
    
    onSubmit(integrationData);
  };

  // Render form fields...
};
```
The frontend implementation should provide a user-friendly interface for managing these integrations while handling all API interactions and error cases appropriately. 
The frontend implementation should provide a user-friendly interface for managing these integrations while handling all API interactions and error cases appropriately. 