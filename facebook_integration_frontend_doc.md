# Facebook Messenger Integration - Frontend Guide

## Overview
We've significantly enhanced our Facebook Messenger integration to automatically collect user profile data and create rich contact profiles. This document outlines the changes and how the frontend can leverage this new functionality.

## 1. Enhanced Contact Data from Facebook

### What's New
When a user messages us through Facebook Messenger, our system now:
- Retrieves their Facebook profile picture, ensuring it's always available
- Collects additional profile information when available (subject to Facebook privacy settings)
- Creates a comprehensive contact profile automatically
- Associates all conversations with their contact record

### Available Contact Fields
The `/contact-details` endpoint now includes:

```json
{
  "contact": {
    "id": 123,
    "facebook_id": "12345678901234567",
    "profile_picture_url": "https://graph.facebook.com/12345678901234567/picture?type=large&access_token=...",
    "first_name": "John",
    "last_name": "Doe",
    "email": "john@example.com",  // If shared through Facebook
    "created_at": "2023-06-01T10:15:23Z"
  }
}
```

### Implementation Notes
1. **Profile Pictures**: 
   - Always use `profile_picture_url` to display the user's profile photo
   - The URL points directly to Facebook's CDN and will always show their current profile picture
   - No need to download and re-host the images

2. **Contact Identification**:
   - `facebook_id` uniquely identifies the Facebook user
   - Use this for any Facebook-specific integrations

## 2. Rich Profile Information

### What's New
The system now automatically creates profile records for Facebook users containing:
- Profile summary constructed from their public information
- Structured general info including location, gender, timezone (when available)
- Default personality traits and goals
- Social media links, including their Facebook profile

### Profile Data Structure
The `/profile-details` endpoint includes:

```json
{
  "profile": {
    "contact_id": 123,
    "profile_summary": "Facebook user John Doe from San Francisco.",
    "general_info": {
      "gender": "male",
      "locale": "en_US",
      "timezone": -7,
      "location": "San Francisco, California",
      "source": "Facebook Messenger"
    },
    "personality": "John is a Facebook Messenger user who initiated contact with our platform.",
    "social_media_urls": [
      {
        "platform": "facebook",
        "url": "https://facebook.com/12345678901234567"
      },
      {
        "platform": "website",
        "url": "https://johndoe.com"
      }
    ],
    "best_goals": [
      {
        "goal": "Get information or assistance",
        "deadline": "Ongoing",
        "importance": "High"
      }
    ]
  }
}
```

### Implementation Notes
1. **Profile Summary**:
   - Display in the contact header section
   - Provides concise information about the user

2. **Social Media Links**:
   - Can be displayed as clickable icons
   - The "facebook" platform always has a valid URL

3. **General Info**:
   - Can populate user details view
   - The "source" field indicates the user came from Facebook 

## 3. Facebook Conversation History

### What's New
All Facebook messages are now:
- Stored in conversations with full context
- Associated with the correct contact
- Standardized with proper timestamps and message types
- Accessible via the same API endpoints as other conversations

### Message Data Structure
Facebook messages appear in the conversation data like this:

```json
{
  "messages": [
    {
      "id": "fb_1686581447_user123",
      "platform_msg_id": "m_AbCdEfGhIjKlMnOp",
      "sender_id": "12345678901234567",
      "sender_type": "contact", 
      "content": "Hello there!",
      "content_type": "text",
      "timestamp": "2023-06-12T15:30:47Z",
      "platform": "facebook",
      "status": "received",
      "attachments": [],
      "metadata": {
        "facebook": { /*original Facebook message structure*/ }
      }
    }
  ]
}
```

### Implementation Notes
1. **Message Display**:
   - Use `sender_type` to determine message alignment (left for "contact", right for "system")
   - Use `content_type` to render appropriate message templates (text, image, etc.)
   - Use `timestamp` for displaying time information

2. **Rich Media**:
   - For image messages, `content` includes a placeholder with the image URL
   - `attachments` array contains detailed information about media
   - Attachments can be displayed inline or as downloadable items

3. **Message Status**:
   - Facebook replies have "received" status
   - Outgoing messages have "sent" status

## 4. Frontend Implementation Guidelines

### Contact List View
1. Each contact can now have:
   - Profile picture from Facebook
   - Full name from their Facebook profile
   - Conversational history from Facebook Messenger

2. Contact card should display:
   - Profile picture using `profile_picture_url`
   - Name using `first_name` and `last_name`
   - Indicator if the contact came from Facebook

### Contact Detail View
1. Contact information section:
   - Display Facebook profile picture prominently
   - Show profile summary from their Facebook data
   - Include a "View on Facebook" link using their `facebook_id`

2. Conversation section:
   - Display conversation thread as normal
   - Support Facebook-specific message types (images, attachments)
   - Show timestamps in user's local timezone

### Design Considerations
1. **Profile Pictures**:
   - Always available for Facebook users
   - Use fallback avatars only for non-Facebook contacts

2. **Attribution**:
   - Consider indicating which contacts came from Facebook
   - Helpful for support staff to know the communication channel

3. **Rich Messages**:
   - Support displaying images, videos, and other attachments from Facebook
   - Present them in a mobile-friendly format

## 5. Example Implementation

### Contact Card with Facebook Data
```jsx
function ContactCard({ contact }) {
  return (
    <div className="contact-card">
      <img 
        src={contact.profile_picture_url || '/default-avatar.png'} 
        alt={`${contact.first_name} ${contact.last_name}`} 
        className="avatar"
      />
      <div className="contact-info">
        <h3>{contact.first_name} {contact.last_name}</h3>
        {contact.facebook_id && (
          <span className="source-badge">Facebook</span>
        )}
      </div>
    </div>
  );
}
```

### Facebook Message Rendering
```jsx
function MessageBubble({ message }) {
  const isContact = message.sender_type === 'contact';
  
  // Determine if it's an image message
  const isImage = message.content_type === 'image' || 
                 (message.content_type === 'text' && message.content.startsWith('[Image]'));
  
  return (
    <div className={`message ${isContact ? 'contact' : 'system'}`}>
      {isImage ? (
        <div className="image-message">
          {message.attachments && message.attachments[0] && (
            <img src={message.attachments[0].url} alt="Shared image" />
          )}
        </div>
      ) : (
        <div className="text-message">{message.content}</div>
      )}
      <div className="timestamp">
        {new Date(message.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
}
```

## 6. API Endpoints to Use

| Endpoint | Purpose | Key Fields for Facebook |
|----------|---------|-------------------------|
| `GET /contact-details?contact_id={id}` | Get contact details | `facebook_id`, `profile_picture_url` |
| `GET /profile-details?contact_id={id}` | Get profile info | `general_info`, `social_media_urls` |
| `GET /contact-conversations?contact_id={id}` | List conversations | Filter for `platform: "facebook"` |
| `GET /conversation?conversation_id={id}` | Get conversation | Check `conversation_data.messages` | 