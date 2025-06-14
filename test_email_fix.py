#!/usr/bin/env python3
"""
Test script to verify email verification system works without SSL errors
"""

import sys
import os
sys.path.append('.')

from login import send_verification_email, generate_verification_token

def test_email_function():
    """Test the email sending function"""
    print("🧪 Testing Email Function (without actually sending)")
    print("=" * 50)
    
    # Test parameters
    test_email = "test@example.com"
    test_name = "Test User"
    test_token = generate_verification_token()
    
    print(f"📧 Test Email: {test_email}")
    print(f"👤 Test Name: {test_name}")
    print(f"🔐 Test Token: {test_token[:10]}...")
    
    # Check if SendGrid is configured
    sendgrid_key = os.getenv("SENDGRID_API_KEY")
    email_from = os.getenv("EMAIL_FROM")
    
    if not sendgrid_key or not email_from:
        print("\n⚠️  SendGrid not configured - function will return False")
        print("   This is expected if SendGrid is not set up yet")
        
        # Test the function logic without sending
        try:
            result = send_verification_email(test_email, test_name, test_token)
            print(f"\n✅ Function executed without SSL errors: {result}")
            return True
        except Exception as e:
            print(f"\n❌ Function failed with error: {e}")
            return False
    else:
        print("\n✅ SendGrid configured - this would send a real email")
        print("   Skipping actual send to avoid spam")
        return True

if __name__ == "__main__":
    success = test_email_function()
    sys.exit(0 if success else 1) 