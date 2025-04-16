# Chatwoot Webhook Integration with Organization Support

## Overview

The Chatwoot webhook integration has been enhanced to support proper organization separation based on inbox ID. This allows multiple Chatwoot inboxes (potentially from different Facebook pages) to be configured with different organizations in the same API instance.

## Configuration

### 1. The Mapping File

A configuration file `inbox_mapping.json` defines the relationship between Chatwoot inboxes and organizations:

```json
{
  "inboxes": [
    {
      "inbox_id": "123",
      "facebook_page_id": "1234567890",
      "organization_id": "fa2ff944-36cc-4ef8-99bc-5c6dd08ded36",
      "page_name": "FB Page 1"
    },
    {
      "inbox_id": "456",
      "facebook_page_id": "0987654321",
      "organization_id": "60a2445b-95f2-4c2f-a3c5-973c7bce6c6e",
      "page_name": "FB Page 2"
    }
  ]
}
```

This mapping ensures that:
- Messages from Chatwoot inbox 123 are associated with organization fa2ff944-36cc-4ef8-99bc-5c6dd08ded36
- Messages from Chatwoot inbox 456 are associated with organization 60a2445b-95f2-4c2f-a3c5-973c7bce6c6e

### 2. Configuring the Webhook in Chatwoot

There are two ways to configure the Chatwoot webhook:

#### Option 1: Through Query Parameters (Recommended)

Configure your webhook URL in Chatwoot as:
```
https://your-api.example.com/webhook/chatwoot?organization_id=YOUR_ORG_ID
```

#### Option 2: Using Headers

In Chatwoot's webhook configuration, add a custom header:
- Header Name: `X-Organization-Id`
- Header Value: `YOUR_ORG_ID`

### 3. Automatic Organization ID Detection

If you've set up the `inbox_mapping.json` file, the system can automatically determine the organization based on the inbox_id in the webhook payload, even if no organization_id is provided in the query parameters or headers.

## Security Features

1. **Organization Validation**: If organization_id is provided both in the webhook and in the mapping file, they must match.

2. **Inbox Validation**: The system validates that the incoming webhook is from a recognized inbox.

3. **Facebook Page Validation**: When Facebook page IDs are available, they're validated against the expected values.

4. **Duplicate Prevention**: The system prevents duplicate webhook processing.

## Error Handling

The webhook endpoint responds with appropriate HTTP status codes:

- **200 OK**: Successfully processed the webhook
- **400 Bad Request**: Invalid or missing required data
- **500 Internal Server Error**: Server-side error

## Implementation Details

1. When a webhook is received, the system:
   - Extracts the inbox_id from the payload
   - Looks up the expected organization_id from the mapping
   - Validates the provided organization_id (if any) against the expected value
   - Associates contacts and conversations with the correct organization

2. Contacts created from Chatwoot webhooks will be associated with the appropriate organization, ensuring data separation between organizations.

## Logging

Detailed logging is implemented for debugging:
- Webhook receipt and processing
- Organization and inbox ID information
- Validation results
- Error conditions

## Example Response

A successful webhook processing returns:
```json
{
  "status": "success",
  "message": "Processed message_created event",
  "organization_id": "fa2ff944-36cc-4ef8-99bc-5c6dd08ded36",
  "inbox_id": "123"
}
``` 