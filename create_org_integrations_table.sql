-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Organization Integration Settings Table
CREATE TABLE organization_integrations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
    integration_type VARCHAR(50) NOT NULL CHECK (integration_type IN ('odoo_crm', 'hubspot', 'salesforce', 'facebook', 'other')),
    name VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    api_base_url VARCHAR(255),
    webhook_url VARCHAR(255),
    api_key TEXT,
    api_secret TEXT,
    access_token TEXT,
    refresh_token TEXT,
    token_expires_at TIMESTAMP WITH TIME ZONE,
    config JSONB DEFAULT '{}'::jsonb,  -- Flexible JSON field for additional configuration
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Each organization can have only one integration of each type
    UNIQUE (org_id, integration_type)
);

-- Create index for faster queries
CREATE INDEX idx_org_integrations_org_id ON organization_integrations(org_id);
CREATE INDEX idx_org_integrations_type ON organization_integrations(integration_type);

-- Automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_organization_integrations_timestamp
BEFORE UPDATE ON organization_integrations
FOR EACH ROW
EXECUTE PROCEDURE update_modified_column();

-- Add comments for documentation
COMMENT ON TABLE organization_integrations IS 'Stores configuration for third-party service integrations by organization';
COMMENT ON COLUMN organization_integrations.integration_type IS 'Type of integration (odoo_crm, hubspot, salesforce, facebook, other)';
COMMENT ON COLUMN organization_integrations.api_base_url IS 'Base URL for API calls to the integrated service';
COMMENT ON COLUMN organization_integrations.webhook_url IS 'URL for receiving webhooks/callbacks from the integrated service';
COMMENT ON COLUMN organization_integrations.api_key IS 'API key for authentication with the integrated service';
COMMENT ON COLUMN organization_integrations.api_secret IS 'API secret for authentication with the integrated service';
COMMENT ON COLUMN organization_integrations.access_token IS 'OAuth access token for the integrated service';
COMMENT ON COLUMN organization_integrations.refresh_token IS 'OAuth refresh token for renewing access tokens';
COMMENT ON COLUMN organization_integrations.token_expires_at IS 'Expiration timestamp for the access token';
COMMENT ON COLUMN organization_integrations.config IS 'Additional JSON configuration specific to each integration type'; 