-- Add facebook_id to contacts table
ALTER TABLE contacts 
ADD COLUMN IF NOT EXISTS facebook_id VARCHAR(255);

-- Add profile_picture_url to contacts table
ALTER TABLE contacts 
ADD COLUMN IF NOT EXISTS profile_picture_url VARCHAR(512);

-- Add unique constraint on facebook_id
ALTER TABLE contacts 
ADD CONSTRAINT unique_facebook_id UNIQUE (facebook_id);

-- Add index for efficient facebook_id lookups
CREATE INDEX IF NOT EXISTS idx_contacts_facebook_id 
ON contacts(facebook_id);

-- Add comment on column
COMMENT ON COLUMN contacts.facebook_id IS 'Facebook user ID for associating contacts with Facebook Messenger';
COMMENT ON COLUMN contacts.profile_picture_url IS 'URL to profile picture (from Facebook or other sources)'; 