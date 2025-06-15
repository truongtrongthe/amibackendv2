-- Migration Script: Fix Organization Management System
-- This script creates the proper organization relationships and handles existing data

-- 1. Create user_organizations table for proper user-organization relationships
CREATE TABLE IF NOT EXISTS user_organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    org_id UUID NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member')),
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique user-org relationships
    UNIQUE(user_id, org_id)
);

-- 2. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_organizations_user_id ON user_organizations(user_id);
CREATE INDEX IF NOT EXISTS idx_user_organizations_org_id ON user_organizations(org_id);
CREATE INDEX IF NOT EXISTS idx_user_organizations_role ON user_organizations(role);

-- 3. Add triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_organizations_updated_at 
    BEFORE UPDATE ON user_organizations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 4. Migration of existing data
-- Note: This handles users who have broken org_id values from the old system
-- We'll need to handle this carefully since the old org_id values are random strings

-- First, let's identify users with organization names but broken org_ids
-- These users should either create new organizations or join existing ones
-- For now, we'll clear their org_id and let them rejoin/create organizations

-- Clear broken org_id values (they start with "org_" and are random strings)
UPDATE users 
SET org_id = NULL 
WHERE org_id IS NOT NULL 
AND org_id LIKE 'org_%' 
AND org_id NOT IN (SELECT id::text FROM organization);

-- 5. Optional: Create a function to find users by organization name for migration
CREATE OR REPLACE FUNCTION migrate_user_organizations()
RETURNS void AS $$
DECLARE
    user_record RECORD;
    org_record RECORD;
BEGIN
    -- For users who have organization names but no org_id
    FOR user_record IN 
        SELECT id, organization, name, email 
        FROM users 
        WHERE organization IS NOT NULL 
        AND organization != ''
        AND org_id IS NULL
    LOOP
        -- Try to find existing organization by name
        SELECT * INTO org_record 
        FROM organization 
        WHERE LOWER(name) = LOWER(user_record.organization)
        LIMIT 1;
        
        IF org_record IS NOT NULL THEN
            -- Organization exists, add user as member
            INSERT INTO user_organizations (user_id, org_id, role)
            VALUES (user_record.id, org_record.id, 'member')
            ON CONFLICT (user_id, org_id) DO NOTHING;
            
            -- Update user's org_id
            UPDATE users 
            SET org_id = org_record.id 
            WHERE id = user_record.id;
            
            RAISE NOTICE 'Added user % to existing organization %', user_record.email, org_record.name;
        ELSE
            -- Organization doesn't exist, create it with user as owner
            INSERT INTO organization (name, created_date)
            VALUES (user_record.organization, NOW())
            RETURNING * INTO org_record;
            
            -- Add user as owner
            INSERT INTO user_organizations (user_id, org_id, role)
            VALUES (user_record.id, org_record.id, 'owner')
            ON CONFLICT (user_id, org_id) DO NOTHING;
            
            -- Update user's org_id
            UPDATE users 
            SET org_id = org_record.id 
            WHERE id = user_record.id;
            
            RAISE NOTICE 'Created new organization % with user % as owner', org_record.name, user_record.email;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- 6. Run the migration (uncomment when ready)
-- SELECT migrate_user_organizations();

-- 7. Add helpful views for organization management
CREATE OR REPLACE VIEW user_organization_details AS
SELECT 
    u.id as user_id,
    u.email,
    u.name as user_name,
    o.id as org_id,
    o.name as org_name,
    o.description as org_description,
    uo.role,
    uo.joined_at
FROM users u
LEFT JOIN user_organizations uo ON u.id = uo.user_id
LEFT JOIN organization o ON uo.org_id = o.id;

-- 8. Helpful queries for debugging
-- Find users with broken org_ids:
-- SELECT id, email, name, organization, org_id FROM users WHERE org_id IS NOT NULL AND org_id LIKE 'org_%';

-- Find users with organization names but no org_id:
-- SELECT id, email, name, organization FROM users WHERE organization IS NOT NULL AND org_id IS NULL;

-- View all user-organization relationships:
-- SELECT * FROM user_organization_details ORDER BY org_name, user_name;

COMMENT ON TABLE user_organizations IS 'Junction table for user-organization relationships with roles';
COMMENT ON COLUMN user_organizations.role IS 'User role in organization: owner, admin, or member';
COMMENT ON FUNCTION migrate_user_organizations() IS 'Migrates existing users with organization names to proper organization relationships'; 