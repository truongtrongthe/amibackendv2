# Quick script to check user verification status
# Run this to see if admin@thefusionlab.ai is actually verified

import os
from supabase import create_client

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")
supabase = create_client(spb_url, spb_key)

def check_user_status(email):
    try:
        response = supabase.table("users").select("*").eq("email", email).execute()
        
        if response.data:
            user = response.data[0]
            print(f"✅ User found: {email}")
            print(f"📧 Email verified: {user.get('email_verified')}")
            print(f"🔄 Provider: {user.get('provider')}")
            print(f"👤 User ID: {user.get('id')}")
            print(f"📅 Created: {user.get('created_at')}")
            print(f"🔄 Updated: {user.get('updated_at')}")
            
            if user.get('email_verified'):
                print("🎉 User is VERIFIED - they can login!")
            else:
                print("❌ User is NOT verified - needs new verification token")
                
        else:
            print(f"❌ User not found: {email}")
            
    except Exception as e:
        print(f"Error checking user: {e}")

if __name__ == "__main__":
    check_user_status("admin@thefusionlab.ai") 