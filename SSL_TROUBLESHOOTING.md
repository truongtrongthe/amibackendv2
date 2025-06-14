# SSL Certificate Troubleshooting Guide

## 🚨 **Error: SSL Certificate Verification Failed**

If you see this error:
```
ERROR: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain>
```

This means your production server has SSL certificate validation issues.

## 🔧 **Quick Fixes (In Order of Preference)**

### **Fix 1: Update SSL Certificates (Recommended)**

```bash
# SSH to your production server

# Ubuntu/Debian
sudo apt update
sudo apt install ca-certificates
sudo update-ca-certificates

# CentOS/RHEL
sudo yum update ca-certificates
# OR
sudo dnf update ca-certificates

# Restart your application after update
```

### **Fix 2: Set Environment Variable (Temporary)**

If Fix 1 doesn't work, temporarily disable SSL verification:

```bash
# Add to your production .env file
DISABLE_SSL_VERIFICATION=true
```

⚠️ **Warning**: This reduces security. Only use if other fixes fail.

### **Fix 3: Install Python SSL Packages**

```bash
# On your production server
pip install --upgrade certifi requests urllib3

# Restart your application
```

### **Fix 4: System-Level SSL Update**

```bash
# Check current SSL/TLS setup
python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"
python3 -c "import certifi; print(certifi.where())"

# Update Python SSL
pip install --upgrade pip setuptools

# Force reinstall SSL packages
pip install --force-reinstall certifi requests
```

## 🏥 **Cloud Provider Specific Fixes**

### **AWS EC2**
```bash
# Update Amazon Linux
sudo yum update -y
sudo yum install ca-certificates -y

# Update Ubuntu on AWS
sudo apt update && sudo apt upgrade -y
sudo apt install ca-certificates -y
```

### **Google Cloud Platform**
```bash
# Update GCP instance
sudo apt update && sudo apt upgrade -y
sudo apt install ca-certificates -y

# Restart services
sudo systemctl restart your-app-service
```

### **DigitalOcean**
```bash
# Update droplet
sudo apt update && sudo apt upgrade -y
sudo apt install ca-certificates -y
```

### **Azure**
```bash
# Update Azure VM
sudo apt update && sudo apt upgrade -y
sudo apt install ca-certificates -y
```

## 🧪 **Test SSL Configuration**

Create this test script on your production server:

```python
# test_ssl.py
import ssl
import certifi
import urllib.request

def test_ssl_connection():
    print("Testing SSL configuration...")
    
    # Test 1: Basic SSL info
    print(f"OpenSSL Version: {ssl.OPENSSL_VERSION}")
    print(f"Certifi path: {certifi.where()}")
    
    # Test 2: Try connecting to SendGrid
    try:
        url = "https://api.sendgrid.com"
        response = urllib.request.urlopen(url)
        print(f"✅ SSL connection to SendGrid: Success ({response.status})")
    except Exception as e:
        print(f"❌ SSL connection failed: {e}")
    
    # Test 3: Test with custom SSL context
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        request = urllib.request.Request("https://api.sendgrid.com")
        response = urllib.request.urlopen(request, context=ssl_context)
        print(f"✅ Custom SSL context: Success ({response.status})")
    except Exception as e:
        print(f"❌ Custom SSL context failed: {e}")

if __name__ == "__main__":
    test_ssl_connection()
```

Run: `python3 test_ssl.py`

## 🔍 **Environment Variables for SSL Issues**

Add these to your production `.env`:

```env
# SSL Configuration
DISABLE_SSL_VERIFICATION=false  # Set to 'true' only if other fixes fail

# SendGrid with SSL bypass (if needed)
SENDGRID_API_KEY=SG.your-key-here
EMAIL_FROM=your-email@domain.com
EMAIL_FROM_NAME=Your App

# Alternative: Use HTTP proxy if available
HTTP_PROXY=http://your-proxy:port
HTTPS_PROXY=http://your-proxy:port
```

## 🎯 **Root Cause Analysis**

### **Common Causes:**
1. **Outdated CA certificates** - Most common in older servers
2. **Corporate firewall** - Self-signed certificates in proxy
3. **Python environment issues** - Missing/corrupted SSL packages
4. **System clock drift** - SSL certificates appear invalid

### **Check System Clock:**
```bash
# Ensure system time is correct
date
sudo ntpdate -s time.nist.gov  # Sync time
```

### **Check Corporate Firewall:**
```bash
# Test if you're behind a corporate proxy
curl -I https://api.sendgrid.com
curl -I https://google.com

# If these fail, you may need proxy configuration
```

## 📊 **Verification Steps**

After applying fixes:

1. **Restart your application**
2. **Test email sending**: Try signup flow
3. **Check logs**: Should see "SendGrid response: Status 202"
4. **Verify email delivery**: Check recipient inbox

## 🚀 **Production Deployment Checklist**

- [ ] Update CA certificates on server
- [ ] Install certifi package: `pip install certifi`
- [ ] Test SSL connection: `python3 test_ssl.py`
- [ ] Deploy updated code with SSL handling
- [ ] Test email verification flow
- [ ] Monitor logs for SSL errors
- [ ] Only use `DISABLE_SSL_VERIFICATION=true` as last resort

## 💡 **Prevention for Future Deployments**

Include in your deployment scripts:
```bash
# Always update certificates
sudo apt update && sudo apt install ca-certificates -y
pip install --upgrade certifi requests

# Test SSL before deploying app
python3 -c "import ssl; import certifi; print('SSL OK')"
```

This ensures SSL issues are caught early in deployment process. 