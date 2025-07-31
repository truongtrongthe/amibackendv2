-- Migration Script: Drop Old Org Agent Tables
-- This script removes the current org_agent related tables to prepare for new architecture
-- Run this BEFORE creating the new schema

-- Drop existing triggers first
DROP TRIGGER IF EXISTS update_org_agents_updated_at ON org_agents;

-- Drop existing views
DROP VIEW IF EXISTS active_org_agents;

-- Drop existing indexes
DROP INDEX IF EXISTS idx_org_agents_org_created_by;
DROP INDEX IF EXISTS idx_org_agents_org_status;
DROP INDEX IF EXISTS idx_org_agents_name;
DROP INDEX IF EXISTS idx_org_agents_created_at;
DROP INDEX IF EXISTS idx_org_agents_created_by;
DROP INDEX IF EXISTS idx_org_agents_status;
DROP INDEX IF EXISTS idx_org_agents_org_id;

-- Drop the main table
DROP TABLE IF EXISTS org_agents;

-- Drop the update function if it exists and is only used by org_agents
-- (Be careful - check if other tables use this function)
-- DROP FUNCTION IF EXISTS update_updated_at_column();

-- Confirmation message
SELECT 'Old org_agents related tables and objects have been dropped successfully' as migration_status;