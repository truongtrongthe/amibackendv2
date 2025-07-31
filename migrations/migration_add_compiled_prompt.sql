-- Migration Script: Add Compiled System Prompt Support
-- This script adds compiled system prompt fields to the agent_blueprints table
-- Run this AFTER the main schema migration

-- Add compiled system prompt fields to agent_blueprints table
ALTER TABLE agent_blueprints ADD COLUMN compiled_system_prompt TEXT;
ALTER TABLE agent_blueprints ADD COLUMN compiled_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE agent_blueprints ADD COLUMN compiled_by VARCHAR(255);
ALTER TABLE agent_blueprints ADD COLUMN compilation_status VARCHAR(50) DEFAULT 'draft' CHECK (compilation_status IN ('draft', 'compiled', 'failed'));

-- Add foreign key for compiled_by (references users table)
ALTER TABLE agent_blueprints 
ADD CONSTRAINT fk_compiled_by 
FOREIGN KEY (compiled_by) REFERENCES users(id) ON DELETE SET NULL;

-- Create index for compilation queries
CREATE INDEX idx_agent_blueprints_compilation_status ON agent_blueprints(compilation_status);
CREATE INDEX idx_agent_blueprints_compiled_at ON agent_blueprints(compiled_at);
CREATE INDEX idx_agent_blueprints_compiled_by ON agent_blueprints(compiled_by);

-- Create composite index for agent + compilation status
CREATE INDEX idx_agent_blueprints_agent_compilation ON agent_blueprints(agent_id, compilation_status);

-- Create view for compiled blueprints only
CREATE VIEW compiled_agent_blueprints AS
SELECT 
    id,
    agent_id,
    version,
    agent_blueprint,
    compiled_system_prompt,
    compiled_at,
    compiled_by,
    created_at,
    created_by,
    conversation_id
FROM agent_blueprints 
WHERE compilation_status = 'compiled';

-- Create view for agents with their current compiled blueprints
CREATE VIEW agents_with_compiled_blueprints AS
SELECT 
    a.id,
    a.agent_id,
    a.org_id,
    a.name,
    a.description,
    a.status,
    a.created_by,
    a.created_at,
    a.updated_at,
    b.id as blueprint_id,
    b.version as blueprint_version,
    b.agent_blueprint,
    b.compiled_system_prompt,
    b.compiled_at,
    b.compiled_by,
    b.conversation_id
FROM org_agents a
LEFT JOIN agent_blueprints b ON a.current_blueprint_id = b.id
WHERE a.status != 'delete' AND (b.compilation_status = 'compiled' OR b.compilation_status IS NULL);

-- Function to update compilation status
CREATE OR REPLACE FUNCTION update_blueprint_compilation(
    blueprint_uuid UUID,
    compiled_prompt TEXT,
    compiler_id VARCHAR(255)
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE agent_blueprints 
    SET 
        compiled_system_prompt = compiled_prompt,
        compiled_at = NOW(),
        compiled_by = compiler_id,
        compilation_status = 'compiled'
    WHERE id = blueprint_uuid;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Function to mark compilation as failed
CREATE OR REPLACE FUNCTION mark_compilation_failed(
    blueprint_uuid UUID,
    compiler_id VARCHAR(255)
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE agent_blueprints 
    SET 
        compiled_at = NOW(),
        compiled_by = compiler_id,
        compilation_status = 'failed'
    WHERE id = blueprint_uuid;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Function to get compilation statistics for an organization
CREATE OR REPLACE FUNCTION get_org_compilation_stats(org_uuid UUID)
RETURNS TABLE(
    total_blueprints BIGINT,
    compiled_blueprints BIGINT,
    draft_blueprints BIGINT,
    failed_blueprints BIGINT,
    compilation_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_blueprints,
        COUNT(*) FILTER (WHERE b.compilation_status = 'compiled') as compiled_blueprints,
        COUNT(*) FILTER (WHERE b.compilation_status = 'draft') as draft_blueprints,
        COUNT(*) FILTER (WHERE b.compilation_status = 'failed') as failed_blueprints,
        ROUND(
            (COUNT(*) FILTER (WHERE b.compilation_status = 'compiled')::NUMERIC / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as compilation_rate
    FROM agent_blueprints b
    JOIN org_agents a ON b.agent_id = a.id
    WHERE a.org_id = org_uuid AND a.status != 'delete';
END;
$$ LANGUAGE plpgsql;

-- Confirmation message
SELECT 'Compiled system prompt support added successfully' as migration_status;