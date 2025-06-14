# SendGrid Setup Guide

## 🚀 **Quick Setup (5 minutes)**

### **Step 1: Create SendGrid Account**
1. Go to https://sendgrid.com/
2. Sign up for free account (100 emails/day free tier)
3. Verify your email address
4. Complete account setup

### **Step 2: Create API Key**
1. Login to SendGrid dashboard
2. Go to **Settings** → **API Keys**
3. Click **"Create API Key"**
4. Choose **"Restricted Access"**
5. Give permissions:
   - ✅ **Mail Send** → Full Access
   - ✅ **Template Engine** → Read Access (optional)
6. Click **"Create & View"**
7. **COPY THE API KEY** (starts with `SG.`) - you won't see it again!

### **Step 3: Set Up Sender Authentication**
1. Go to **Settings** → **Sender Authentication**
2. Choose **Single Sender Verification** (easiest)
3. Fill in:
   - **From Name**: Your App Name
   - **From Email**: your-email@yourdomain.com
   - **Reply To**: your-email@yourdomain.com
   - **Company**: Your Company
   - **Address**: Any valid address
4. Click **"Create"**
5. **Check your email** and click verification link

## 🔧 **Environment Variables**

Update your `.env` file with these settings:

### **Local Development (.env)**
```env
# SendGrid Configuration
SENDGRID_API_KEY=SG.your-api-key-here
EMAIL_FROM=your-verified-email@domain.com
EMAIL_FROM_NAME=Your App Name
BASE_URL=http://localhost:3000
```

### **Production (.env)**
```env
# SendGrid Configuration
SENDGRID_API_KEY=SG.your-api-key-here
EMAIL_FROM=your-verified-email@domain.com
EMAIL_FROM_NAME=Your App Name
BASE_URL=https://yourdomain.com
```

## 🧪 **Test Your Setup**

### **Method 1: Quick Test Script**
```python
# test_sendgrid.py
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def test_sendgrid():
    api_key = os.getenv('SENDGRID_API_KEY')
    from_email = os.getenv('EMAIL_FROM')
    
    if not api_key or not from_email:
        print("❌ Missing environment variables")
        return
    
    message = Mail(
        from_email=from_email,
        to_emails='test@example.com',  # Change to your email
        subject='SendGrid Test',
        html_content='<strong>Test email from SendGrid!</strong>'
    )
    
    try:
        sg = SendGridAPIClient(api_key=api_key)
        response = sg.send(message)
        print(f"✅ Email sent! Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_sendgrid()
```

Run: `python test_sendgrid.py`

### **Method 2: Test via API**
```bash
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test User",
    "email": "your-email@domain.com",
    "password": "TestPass123",
    "confirmPassword": "TestPass123"
  }'
```

## 🔍 **Troubleshooting**

### **Issue 1: "Forbidden" Error**
```
Error: HTTP Error 403: Forbidden
```
**Solution**: Check API key permissions
- Go to SendGrid → Settings → API Keys
- Edit your key and ensure "Mail Send" has "Full Access"

### **Issue 2: "From Email Not Verified"**
```
Error: The from address does not match a verified Sender Identity
```
**Solution**: 
- Go to Settings → Sender Authentication
- Verify your from email address
- Use exact same email in `EMAIL_FROM`

### **Issue 3: "API Key Not Found"**
```
Error: SendGrid API key not configured
```
**Solution**: 
- Check `.env` file has `SENDGRID_API_KEY=SG.your-key`
- Restart your application after updating `.env`

## 📊 **SendGrid Dashboard Features**

### **Monitor Email Delivery**
1. Go to **Activity** → **Email Activity**
2. See delivery status, opens, clicks
3. Debug bounces and blocks

### **Email Templates (Optional)**
1. Go to **Email API** → **Dynamic Templates**
2. Create reusable email templates
3. Use in code with template IDs

### **Domain Authentication (Production)**
For better deliverability:
1. Go to **Settings** → **Sender Authentication**
2. Choose **Domain Authentication**
3. Add DNS records to your domain
4. Improves email reputation

## 🎯 **Benefits Over SMTP**

| Feature | SMTP | SendGrid API |
|---------|------|--------------|
| **Cloud Compatibility** | ❌ Often blocked | ✅ Always works |
| **Deliverability** | ⚠️ Variable | ✅ Optimized |
| **Analytics** | ❌ None | ✅ Detailed tracking |
| **Rate Limits** | ⚠️ Provider dependent | ✅ Predictable |
| **Error Handling** | ⚠️ Complex | ✅ Clear responses |
| **Scalability** | ⚠️ Limited | ✅ High volume |

## 🚀 **Production Deployment**

1. **Update production `.env`** with SendGrid credentials
2. **Deploy updated code** with SendGrid integration
3. **Test email flow** end-to-end
4. **Monitor delivery** via SendGrid dashboard

No more SMTP port issues! 🎉 