# Production Email Service Alternatives

## 🚀 **Recommended Services for Production**

### **1. SendGrid (Easiest)**
```env
EMAIL_SMTP_SERVER=smtp.sendgrid.net
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=apikey
EMAIL_PASSWORD=SG.your-sendgrid-api-key
EMAIL_USE_TLS=true
EMAIL_USE_SSL=false
```

**Setup:**
1. Sign up at sendgrid.com
2. Create API key
3. Update environment variables
4. No firewall issues!

### **2. AWS SES (If using AWS)**
```env
EMAIL_SMTP_SERVER=email-smtp.us-east-1.amazonaws.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your-smtp-username
EMAIL_PASSWORD=your-smtp-password
EMAIL_USE_TLS=true
```

**Setup:**
1. Enable SES in AWS Console
2. Verify your domain
3. Create SMTP credentials
4. Request production access

### **3. Mailgun**
```env
EMAIL_SMTP_SERVER=smtp.mailgun.org
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=postmaster@your-domain.mailgun.org
EMAIL_PASSWORD=your-mailgun-password
EMAIL_USE_TLS=true
```

### **4. Gmail (Production Workaround)**
If you must use Gmail in production:

```env
# Try these combinations:
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=465
EMAIL_USE_SSL=true
EMAIL_USE_TLS=false

# Or alternative Gmail server:
EMAIL_SMTP_SERVER=aspmx.l.google.com
EMAIL_SMTP_PORT=25
EMAIL_USE_TLS=false
EMAIL_USE_SSL=false
```

## 🔧 **Cloud Provider Specific Solutions**

### **AWS EC2:**
- Request limit increase: https://aws.amazon.com/ses/sending-limits/
- Use AWS SES instead of Gmail
- Configure security groups to allow outbound 587/465

### **Google Cloud:**
- Use SendGrid add-on from GCP Marketplace
- Enable "Allow SMTP" in VPC firewall rules
- Use Google Cloud SMTP relay

### **DigitalOcean:**
- Usually works with Gmail
- Check droplet firewall settings
- Use DO's managed email services

### **Azure:**
- Use Azure Communication Services
- Allow SMTP in Network Security Groups
- Consider SendGrid (owned by Microsoft)

## 🧪 **Testing Commands**

```bash
# Test SMTP connectivity
telnet smtp.gmail.com 587
telnet smtp.gmail.com 465
telnet smtp.sendgrid.net 587

# Test DNS resolution
nslookup smtp.gmail.com
curl -I smtp.gmail.com:587

# Test from Python
python3 -c "import smtplib; smtplib.SMTP('smtp.gmail.com', 587).quit()"
```

## 🚨 **Troubleshooting Steps**

1. **Check Network Connectivity**
2. **Verify DNS Resolution**
3. **Test Different Ports**
4. **Check Firewall Rules**
5. **Try Alternative SMTP Providers**
6. **Review Cloud Provider SMTP Policies** 