-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
  id BIGSERIAL PRIMARY KEY,
  uuid UUID DEFAULT gen_random_uuid() NOT NULL UNIQUE,
  contact_id BIGINT NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
  title VARCHAR(255),
  status VARCHAR(50) NOT NULL DEFAULT 'active',
  platform VARCHAR(50) DEFAULT 'web', -- 'facebook', 'web', 'whatsapp', etc.
  platform_conversation_id VARCHAR(255), -- External ID from the platform (e.g., Facebook thread ID)
  last_message_text TEXT, -- Quick access to last message content
  last_message_timestamp TIMESTAMPTZ, -- Quick access to last message time
  unread_count INTEGER DEFAULT 0, -- Track unread messages
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  conversation_data JSONB DEFAULT '{}'::jsonb -- Stores all messages and metadata
);

-- Create index on contact_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_conversations_contact_id ON conversations(contact_id);

-- Create index on platform_conversation_id for lookups from external platforms
CREATE INDEX IF NOT EXISTS idx_conversations_platform_id ON conversations(platform, platform_conversation_id);

-- Create index on last_message_timestamp for sorting conversations by recency
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(last_message_timestamp DESC);

-- Add comment to table
COMMENT ON TABLE conversations IS 'Stores conversation history with contacts including messaging platform data';

-- Suggested message structure (stored in conversation_data.messages JSON array):
/*
{
  "messages": [
    {
      "id": "msg123456",            -- Unique message ID
      "platform_msg_id": "fb12345", -- Facebook's message ID
      "sender_id": "user123",       -- ID of the sender
      "sender_type": "user",        -- "user", "contact", "system", "bot", etc.
      "content": "Hello there!",    -- Message content
      "content_type": "text",       -- "text", "image", "audio", "video", "file", etc.
      "attachments": [              -- Optional array of attachments
        {
          "type": "image",
          "url": "https://example.com/image.jpg",
          "name": "image.jpg"
        }
      ],
      "timestamp": "2023-06-20T15:30:45Z", -- When message was sent
      "status": "delivered",         -- "sent", "delivered", "read", "failed"
      "metadata": {                  -- Platform-specific metadata
        "facebook": {
          "is_sponsored": false,
          "reaction": null
        }
      }
    }
  ]
}
*/
