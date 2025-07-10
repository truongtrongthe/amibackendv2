-- Chat System Tables for AMI Backend
-- This creates the database schema for the chat system similar to Cursor's chat interface

-- Chats table - represents individual chat sessions
CREATE TABLE chats (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id VARCHAR(255) NOT NULL,
  org_id UUID,
  title VARCHAR(500),
  status VARCHAR(50) DEFAULT 'active',
  chat_type VARCHAR(50) DEFAULT 'conversation',
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Messages table - individual messages within chats
CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chat_id UUID NOT NULL,
  user_id VARCHAR(255) NOT NULL,
  role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
  content TEXT NOT NULL,
  message_type VARCHAR(50) DEFAULT 'text',
  metadata JSONB DEFAULT '{}',
  thread_id VARCHAR(255),
  parent_message_id UUID,
  token_count INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (parent_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

-- Function to automatically update updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to call the function before any update
CREATE TRIGGER update_chats_updated_at 
    BEFORE UPDATE ON chats 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_messages_updated_at 
    BEFORE UPDATE ON messages 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create indexes for better performance
CREATE INDEX idx_chats_user_id ON chats(user_id);
CREATE INDEX idx_chats_org_id ON chats(org_id);
CREATE INDEX idx_chats_created_at ON chats(created_at);
CREATE INDEX idx_chats_status ON chats(status);

CREATE INDEX idx_messages_chat_id ON messages(chat_id);
CREATE INDEX idx_messages_user_id ON messages(user_id);
CREATE INDEX idx_messages_role ON messages(role);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_messages_thread_id ON messages(thread_id);
CREATE INDEX idx_messages_parent_message_id ON messages(parent_message_id);

-- Create composite indexes for common queries
CREATE INDEX idx_chats_user_status ON chats(user_id, status);
CREATE INDEX idx_messages_chat_role ON messages(chat_id, role);
CREATE INDEX idx_messages_chat_created ON messages(chat_id, created_at);

-- Optional: Add RLS (Row Level Security) if needed
-- ALTER TABLE chats ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Create policy examples (uncomment if RLS is needed)
-- CREATE POLICY "Users can view their own chats" ON chats
--   FOR SELECT USING (auth.uid() = user_id);

-- CREATE POLICY "Users can insert their own chats" ON chats
--   FOR INSERT WITH CHECK (auth.uid() = user_id);

-- CREATE POLICY "Users can update their own chats" ON chats
--   FOR UPDATE USING (auth.uid() = user_id);

-- CREATE POLICY "Users can delete their own chats" ON chats
--   FOR DELETE USING (auth.uid() = user_id);

-- Create views for common queries
CREATE VIEW chat_with_latest_message AS
SELECT 
    c.*,
    m.content as latest_message_content,
    m.role as latest_message_role,
    m.created_at as latest_message_time,
    (SELECT COUNT(*) FROM messages WHERE chat_id = c.id) as message_count
FROM chats c
LEFT JOIN messages m ON c.id = m.chat_id 
WHERE m.created_at = (
    SELECT MAX(created_at) 
    FROM messages 
    WHERE chat_id = c.id
)
OR m.id IS NULL;

CREATE VIEW message_thread_view AS
SELECT 
    m.*,
    c.title as chat_title,
    c.status as chat_status,
    u.name as user_name,
    u.email as user_email
FROM messages m
JOIN chats c ON m.chat_id = c.id
JOIN users u ON m.user_id = u.id
ORDER BY m.created_at; 