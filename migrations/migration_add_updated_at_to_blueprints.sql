-- Migration: Add updated_at column to agent_blueprints table
-- This adds the missing updated_at column that the code expects

-- Add updated_at column to agent_blueprints table
ALTER TABLE agent_blueprints 
ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Update existing records to have updated_at = created_at
UPDATE agent_blueprints 
SET updated_at = created_at 
WHERE updated_at IS NULL;

-- Create trigger to automatically update updated_at on blueprint changes
CREATE TRIGGER update_agent_blueprints_updated_at 
    BEFORE UPDATE ON agent_blueprints 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create index for updated_at queries
CREATE INDEX idx_agent_blueprints_updated_at ON agent_blueprints(updated_at);

-- Add comment for documentation
COMMENT ON COLUMN agent_blueprints.updated_at IS 'Timestamp when the blueprint was last modified';