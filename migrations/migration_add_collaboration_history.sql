-- Migration: Add Agent Collaboration History Table
-- This creates a table to store conversation history for agent collaboration sessions

-- Create agent_collaboration_history table
CREATE TABLE agent_collaboration_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL,
    blueprint_id UUID NOT NULL,
    message_type VARCHAR(20) NOT NULL CHECK (message_type IN ('user', 'ami')),
    message_content TEXT NOT NULL,
    context_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key constraints
    FOREIGN KEY (agent_id) REFERENCES org_agents(id) ON DELETE CASCADE,
    FOREIGN KEY (blueprint_id) REFERENCES agent_blueprints(id) ON DELETE CASCADE
);

-- Create indexes for faster queries
CREATE INDEX idx_collaboration_history_agent_blueprint ON agent_collaboration_history(agent_id, blueprint_id);
CREATE INDEX idx_collaboration_history_created_at ON agent_collaboration_history(created_at);
CREATE INDEX idx_collaboration_history_message_type ON agent_collaboration_history(message_type);

-- Add comments for documentation
COMMENT ON TABLE agent_collaboration_history IS 'Stores conversation history for agent collaboration sessions between users and AMI';
COMMENT ON COLUMN agent_collaboration_history.agent_id IS 'Reference to the agent being collaborated on';
COMMENT ON COLUMN agent_collaboration_history.blueprint_id IS 'Reference to the specific blueprint being refined';
COMMENT ON COLUMN agent_collaboration_history.message_type IS 'Type of message: user or ami';
COMMENT ON COLUMN agent_collaboration_history.message_content IS 'The actual message content';
COMMENT ON COLUMN agent_collaboration_history.context_data IS 'Additional context data in JSON format';