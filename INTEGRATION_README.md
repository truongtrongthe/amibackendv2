# Organization Integrations

This module provides functionality for managing integrations between your organizations and third-party services like Odoo CRM, Hubspot, Salesforce, and Facebook.

## Features

- Create, read, update, and delete integration configurations
- Support for multiple integration types with a flexible configuration structure
- Toggle integration active status
- Securely store authentication credentials
- API endpoints for managing integrations

## Setup

### 1. Create the Database Table

First, you need to create the necessary table in your Supabase database. You can do this using the provided script:

```bash
# Make sure SUPABASE_URL and SUPABASE_KEY environment variables are set
export SUPABASE_URL="your-supabase-url"
export SUPABASE_KEY="your-supabase-key"

# Run the script to create the table
python create_integration_table.py
```

### 2. API Endpoints

The following API endpoints are available for managing integrations:

- `GET /organization-integrations?org_id=<uuid>` - List all integrations for an organization
- `GET /organization-integration/<uuid>` - Get details of a specific integration
- `POST /create-organization-integration` - Create a new integration
- `POST /update-organization-integration` - Update an existing integration
- `POST /delete-organization-integration` - Delete an integration
- `POST /toggle-organization-integration` - Toggle an integration's active status

## Testing

You can test the integration API endpoints using the provided test script:

```bash
# Run the test script with an organization ID
python test_org_integrations.py <organization_uuid>

# Optionally specify a different API base URL
python test_org_integrations.py <organization_uuid> --url http://your-api-url:port
```

The test script will walk you through:
1. Viewing existing integrations for the organization
2. Creating a new integration
3. Retrieving integration details
4. Updating integration settings
5. Toggling the active status
6. Optionally deleting the integration

## Example Usage

### Creating a new integration

```json
// POST /create-organization-integration
{
  "org_id": "your-org-uuid",
  "integration_type": "facebook",
  "name": "Facebook Messenger Integration",
  "webhook_url": "https://your-domain.com/webhook",
  "api_key": "your-facebook-app-id",
  "api_secret": "your-facebook-app-secret",
  "config": {
    "page_id": "your-facebook-page-id",
    "verify_token": "your-custom-verify-token"
  }
}
```

### Toggling an integration

```json
// POST /toggle-organization-integration
{
  "id": "integration-uuid",
  "active": true
}
```

## Integration Types

The system currently supports these integration types:

1. **odoo_crm** - Odoo CRM integration
   - Typically requires: `api_base_url`, `api_key`, `api_secret`
   
2. **hubspot** - Hubspot integration
   - Typically requires: `api_key`, `access_token`, `refresh_token`
   
3. **salesforce** - Salesforce integration
   - Typically requires: `api_base_url`, `access_token`, `refresh_token`
   
4. **facebook** - Facebook Messenger integration
   - Typically requires: `webhook_url`, `api_key`, `api_secret`
   - Config fields: `page_id`, `verify_token`
   
5. **other** - For custom integrations not covered by the above types

You can extend the system by adding new integration types to the check constraint in the SQL file and updating the validation in the Python code.

## Security Notes

- API keys, secrets and tokens are stored in the database
- When returning integration details, sensitive fields are masked in the API responses
- Consider implementing additional encryption for sensitive fields in production 