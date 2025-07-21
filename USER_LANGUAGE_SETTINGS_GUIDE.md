# User Language Settings Guide

## 🌍 **Overview**

This guide explains the new user language preference system that allows users to set their preferred language for AI responses and application interface.

## ✨ **Features**

### **🔧 Language Preference Storage**
- Users can set language preference during signup
- Language preference stored in user profile
- Persistent across sessions and devices
- Integrates with existing language detection system

### **🎯 Supported Languages**
```python
"en": English          "vi": Vietnamese       "es": Spanish
"fr": French           "de": German           "zh": Chinese
"ja": Japanese         "ko": Korean           "th": Thai
"id": Indonesian       "ms": Malay            "tl": Tagalog
```

### **🔄 Smart Language Handling**
- **Priority 1**: User's saved language preference
- **Priority 2**: Dynamic language detection from message
- **Priority 3**: Conversation context language
- **Fallback**: English (default)

## 🛠️ **API Endpoints**

### **1. Signup with Language** - `POST /auth/signup`
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "password123",
  "confirmPassword": "password123",
  "language": "vi"  // Optional: defaults to "en"
}
```

**Response:**
```json
{
  "message": "Registration successful! Please check your email to verify your account.",
  "email": "john@example.com",
  "emailSent": true
}
```

### **2. Update Language Only** - `POST /auth/update-language`
```json
{
  "language": "vi"
}
```

**Response:**
```json
{
  "message": "Language updated successfully",
  "user": {
    "id": "user_xxx",
    "email": "john@example.com",
    "name": "John Doe",
    "language": "vi",
    // ... other user fields
  }
}
```

### **3. Update Full Profile** - `POST /auth/update-profile`
```json
{
  "name": "John Doe Updated",
  "phone": "+84123456789",
  "language": "vi"
}
```

**Response:**
```json
{
  "message": "Profile updated successfully",
  "user": {
    // Updated user object
  }
}
```

### **4. Get Current User** - `GET /auth/me`
```json
{
  "id": "user_xxx",
  "email": "john@example.com",
  "name": "John Doe",
  "language": "vi",  // User's preferred language
  "emailVerified": true,
  // ... other fields
}
```

### **5. Get Supported Languages** - `GET /auth/supported-languages`
```json
{
  "languages": {
    "en": {"name": "English", "native": "English"},
    "vi": {"name": "Vietnamese", "native": "Tiếng Việt"},
    "es": {"name": "Spanish", "native": "Español"}
    // ... more languages
  },
  "default": "en",
  "message": "Supported languages for user preferences"
}
```

## 🔧 **Backend Integration**

### **Database Schema Update**
```sql
-- Add language column to users table
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS language VARCHAR(5) DEFAULT 'en';

-- Add constraint for supported languages
ALTER TABLE users 
ADD CONSTRAINT chk_users_language 
CHECK (language IN ('en', 'vi', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'th', 'id', 'ms', 'tl'));
```

### **Using Language Preference in Your Code**
```python
from login import get_user_language_preference, get_language_display_name

# Get user's preferred language
user_lang = get_user_language_preference(user_id)  # Returns "en", "vi", etc.

# Get display names
lang_info = get_language_display_name(user_lang)
# Returns: {"name": "Vietnamese", "native": "Tiếng Việt"}

# Integration with existing language detection
# In your AI response functions:
def generate_response(message, user_id):
    # Get user preference first
    preferred_lang = get_user_language_preference(user_id)
    
    # Use existing language detection as backup
    detected_lang = detect_language_with_llm(message)
    
    # Smart priority handling
    response_language = preferred_lang if preferred_lang != "en" else detected_lang
    
    # Generate response in appropriate language
    return create_response(message, response_language)
```

## 🎨 **Frontend Integration Examples**

### **React Language Settings Component**
```jsx
import { useState, useEffect } from 'react';

const LanguageSettings = ({ user, onUpdate }) => {
  const [languages, setLanguages] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch supported languages
    fetch('/auth/supported-languages')
      .then(res => res.json())
      .then(data => setLanguages(data.languages));
  }, []);

  const updateLanguage = async (languageCode) => {
    setLoading(true);
    try {
      const response = await fetch('/auth/update-language', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({ language: languageCode })
      });

      const data = await response.json();
      if (response.ok) {
        onUpdate(data.user);
        alert('Language updated successfully!');
      }
    } catch (error) {
      alert('Failed to update language');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="language-settings">
      <h3>Preferred Language</h3>
      <select 
        value={user.language}
        onChange={(e) => updateLanguage(e.target.value)}
        disabled={loading}
      >
        {Object.entries(languages).map(([code, info]) => (
          <option key={code} value={code}>
            {info.native} ({info.name})
          </option>
        ))}
      </select>
    </div>
  );
};
```

### **Signup Form with Language Selection**
```jsx
const SignupForm = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    language: 'en'
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await fetch('/auth/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });

    const data = await response.json();
    if (response.ok) {
      alert('Registration successful! Please check your email.');
    } else {
      alert(data.detail || 'Registration failed');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Name"
        value={formData.name}
        onChange={(e) => setFormData({...formData, name: e.target.value})}
        required
      />
      <input
        type="email"
        placeholder="Email"
        value={formData.email}
        onChange={(e) => setFormData({...formData, email: e.target.value})}
        required
      />
      <input
        type="password"
        placeholder="Password"
        value={formData.password}
        onChange={(e) => setFormData({...formData, password: e.target.value})}
        required
      />
      <input
        type="password"
        placeholder="Confirm Password"
        value={formData.confirmPassword}
        onChange={(e) => setFormData({...formData, confirmPassword: e.target.value})}
        required
      />
      
      <label>Preferred Language:</label>
      <select
        value={formData.language}
        onChange={(e) => setFormData({...formData, language: e.target.value})}
      >
        <option value="en">English</option>
        <option value="vi">Tiếng Việt</option>
        <option value="es">Español</option>
        <option value="fr">Français</option>
        {/* Add more languages */}
      </select>

      <button type="submit">Sign Up</button>
    </form>
  );
};
```

## 🔗 **Integration with Existing Systems**

### **AI Response Generation**
```python
# In your AI response modules (mc.py, learning.py, etc.)
from login import get_user_language_preference

async def generate_ai_response(message, user_id, context):
    # Get user's preferred language
    user_preferred_lang = get_user_language_preference(user_id)
    
    # Use existing language detection
    detected_lang = await detect_language_with_llm(message)
    
    # Smart language selection
    if user_preferred_lang != "en":
        # User has set a specific preference
        response_language = user_preferred_lang
        confidence = 0.9
    else:
        # Use dynamic detection
        response_language = detected_lang["language"]
        confidence = detected_lang["confidence"]
    
    # Create language-aware prompt
    system_prompt = f"""
    LANGUAGE REQUIREMENT:
    - User preferred language: {user_preferred_lang}
    - Detected language: {detected_lang["language"]}
    - Respond in: {response_language}
    - Language confidence: {confidence}
    
    Respond naturally in {response_language} while maintaining cultural context.
    """
    
    return await generate_response(message, system_prompt)
```

### **Email Notifications**
```python
def send_verification_email(email, name, token, user_language="en"):
    # Get language-specific email templates
    templates = {
        "en": {
            "subject": "Verify your email address",
            "greeting": f"Hi {name}",
            "message": "Thank you for signing up! Please verify your email address."
        },
        "vi": {
            "subject": "Xác thực địa chỉ email của bạn",
            "greeting": f"Xin chào {name}",
            "message": "Cảm ơn bạn đã đăng ký! Vui lòng xác thực địa chỉ email của bạn."
        }
        # Add more language templates
    }
    
    template = templates.get(user_language, templates["en"])
    
    # Use template to send localized email
    send_email(email, template["subject"], template["message"])
```

## 🚀 **Migration Guide**

### **For Existing Users**
1. Run the SQL migration script: `add_language_column.sql`
2. All existing users will default to "en" (English)
3. Users can update their preference via `/auth/update-language`

### **For New Features**
1. Always check user's language preference first
2. Fall back to detection if preference is "en" (default)
3. Maintain language consistency across the session
4. Store language-specific content appropriately

## 🔍 **Testing**

### **Test Language Preference Storage**
```bash
# Test signup with language
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test User",
    "email": "test@example.com",
    "password": "password123",
    "confirmPassword": "password123",
    "language": "vi"
  }'

# Test language update
curl -X POST "http://localhost:8000/auth/update-language" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"language": "es"}'
```

### **Test Language Integration**
```python
# Test helper functions
user_lang = get_user_language_preference("user_123")
assert user_lang in ["en", "vi", "es", "fr", "de", "zh", "ja", "ko", "th", "id", "ms", "tl"]

lang_info = get_language_display_name("vi")
assert lang_info["name"] == "Vietnamese"
assert lang_info["native"] == "Tiếng Việt"
```

## 📊 **Analytics & Monitoring**

```sql
-- Monitor language distribution
SELECT 
    language,
    COUNT(*) as user_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM users 
GROUP BY language
ORDER BY user_count DESC;

-- Track language changes
SELECT 
    u.language,
    COUNT(*) as recent_updates
FROM users u
WHERE u.updated_at > NOW() - INTERVAL '7 days'
  AND u.language != 'en'
GROUP BY u.language;
```

## 🎯 **Best Practices**

### **1. Language Priority**
- User preference > Message detection > Context > Default (en)

### **2. Consistency**
- Maintain language throughout the conversation
- Don't switch languages mid-response unless user switches

### **3. Cultural Context**
- Use appropriate pronouns for Vietnamese (em/anh)
- Respect cultural communication patterns
- Use native expressions when possible

### **4. Fallbacks**
- Always have English as fallback
- Handle unsupported language codes gracefully
- Log language detection issues for improvement

This language preference system enhances user experience by providing personalized, culturally appropriate responses while maintaining the sophisticated language detection capabilities of your existing system. 