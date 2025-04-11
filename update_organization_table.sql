-- Update Organization Table: Add contact information fields
-- This migration adds email, phone, and address fields to the organization table

-- Add email column
ALTER TABLE organization
ADD COLUMN IF NOT EXISTS email VARCHAR(255);

-- Add phone column
ALTER TABLE organization
ADD COLUMN IF NOT EXISTS phone VARCHAR(50);

-- Add address column
ALTER TABLE organization
ADD COLUMN IF NOT EXISTS address TEXT;

-- Comment on columns to document their purpose
COMMENT ON COLUMN organization.email IS 'Contact email for the organization';
COMMENT ON COLUMN organization.phone IS 'Contact phone number for the organization';
COMMENT ON COLUMN organization.address IS 'Physical address for the organization';
