#!/usr/bin/env python3
"""
SendGrid Test Script
Quick way to test your SendGrid configuration
"""

import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def test_sendgrid():
    print("🧪 Testing SendGrid Configuration")
    print("=" * 40)
    
    # Check environment variables
    api_key = os.getenv('SENDGRID_API_KEY')
    from_email = os.getenv('EMAIL_FROM')
    from_name = os.getenv('EMAIL_FROM_NAME', 'Test App')
    
    if not api_key:
        print("❌ SENDGRID_API_KEY not found in environment")
        print("   Set: export SENDGRID_API_KEY=SG.your-key-here")
        return
    
    if not from_email:
        print("❌ EMAIL_FROM not found in environment")
        print("   Set: export EMAIL_FROM=your-verified-email@domain.com")
        return
    
    print(f"✅ API Key: {api_key[:20]}...")
    print(f"✅ From Email: {from_email}")
    print(f"✅ From Name: {from_name}")
    
    # Get test email from user
    test_email = input("\n📧 Enter your email to send test: ")
    if not test_email:
        test_email = from_email
    
    print(f"\n🚀 Sending test email to {test_email}...")
    
    # Create test email
    message = Mail(
        from_email=(from_email, from_name),
        to_emails=test_email,
        subject='SendGrid Test - Email Verification System',
        html_content=f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>🎉 SendGrid Test Successful!</h2>
            <p>Your email verification system is working correctly.</p>
            <ul>
                <li>✅ SendGrid API connection: Working</li>
                <li>✅ From email: {from_email}</li>
                <li>✅ Email delivery: Success</li>
            </ul>
            <p>You can now use this for email verification in production!</p>
        </div>
        """,
        plain_text_content=f"""
        SendGrid Test Successful!
        
        Your email verification system is working correctly.
        - SendGrid API connection: Working
        - From email: {from_email}
        - Email delivery: Success
        
        You can now use this for email verification in production!
        """
    )
    
    try:
        # Send email
        sg = SendGridAPIClient(api_key=api_key)
        response = sg.send(message)
        
        print(f"✅ Email sent successfully!")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Headers: {dict(response.headers)}")
        
        if response.status_code in [200, 201, 202]:
            print("\n🎉 SendGrid is working perfectly!")
            print("   Check your email inbox for the test message.")
            print("   Your email verification system is ready for production!")
        else:
            print(f"\n⚠️  Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")
        
        # Common error solutions
        if "403" in str(e):
            print("\n💡 Troubleshooting:")
            print("   - Check API key permissions (Mail Send → Full Access)")
            print("   - Verify the API key is correct")
        elif "400" in str(e):
            print("\n💡 Troubleshooting:")
            print("   - Verify sender email in SendGrid dashboard")
            print("   - Check email format is valid")
        else:
            print("\n💡 Check the SendGrid documentation for more details")

def check_environment():
    """Check if all required environment variables are set"""
    print("🔧 Environment Check")
    print("-" * 20)
    
    required_vars = {
        'SENDGRID_API_KEY': 'Your SendGrid API key (starts with SG.)',
        'EMAIL_FROM': 'Verified sender email address',
        'EMAIL_FROM_NAME': 'Sender name (optional)',
        'BASE_URL': 'Your application URL'
    }
    
    all_good = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            if var == 'SENDGRID_API_KEY':
                print(f"✅ {var}: {value[:20]}...")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set ({description})")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("SendGrid Email Verification Test")
    print("=" * 50)
    
    if check_environment():
        print("\n" + "=" * 50)
        test_sendgrid()
    else:
        print("\n❌ Please set all required environment variables first.")
        print("See SENDGRID_SETUP_GUIDE.md for instructions.") 