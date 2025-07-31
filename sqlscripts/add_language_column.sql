-- Add language column to users table
-- Migration script for adding user language preferences

-- Add language column with default value 'en' (English)
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS language VARCHAR(5) DEFAULT 'en';

-- Add comment to the column
COMMENT ON COLUMN users.language IS 'User preferred language code (ISO 639-1)';

-- Create index on language column for better performance
CREATE INDEX IF NOT EXISTS idx_users_language ON users(language);

-- Update existing users to have default language 'en' if null
UPDATE users 
SET language = 'en' 
WHERE language IS NULL;

-- Add constraint to ensure only supported language codes are allowed
ALTER TABLE users 
ADD CONSTRAINT chk_users_language 
CHECK (language IN ('en', 'vi', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'th', 'id', 'ms', 'tl'));

-- Optionally: Set Vietnamese language for existing users who have Vietnamese content in their names
-- This is a smart heuristic but should be reviewed before executing
-- UPDATE users 
-- SET language = 'vi' 
-- WHERE (name ~ '[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]'
--        OR email ~ '[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]')
--        AND language = 'en';

-- Display summary of language distribution after migration
SELECT 
    language,
    COUNT(*) as user_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM users 
GROUP BY language
ORDER BY user_count DESC; 