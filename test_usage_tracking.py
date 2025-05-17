#!/usr/bin/env python3
"""
Test script for organization usage tracking functionality.
This script simulates usage tracking and displays the results.
"""

import os
import sys
import uuid
from datetime import datetime, timedelta
from usage import OrganizationUsage

def test_usage_tracking():
    """Test the usage tracking functionality"""
    # Create a test organization ID (or use a real one if provided)
    test_org_id = os.environ.get("TEST_ORG_ID") or str(uuid.uuid4())
    print(f"🧪 Testing usage tracking with organization ID: {test_org_id}")
    
    # Initialize the usage tracker
    org_usage = OrganizationUsage(test_org_id)
    
    # Add some test usage data
    print("📝 Adding test usage data...")
    
    # Simulate adding message usage
    message_count = 5
    org_usage.add_message(message_count)
    print(f"✓ Added {message_count} messages")
    
    # Simulate adding reasoning usage
    reasoning_count = 2
    org_usage.add_reasoning(reasoning_count)
    print(f"✓ Added {reasoning_count} reasoning units")
    
    # Get and display usage data
    print("\n📊 Usage Summary:")
    
    # Get daily usage
    daily_messages = org_usage.get_message_count('day')
    daily_reasoning = org_usage.get_reasoning_count('day')
    print(f"• Today: {daily_messages} messages, {daily_reasoning} reasoning units")
    
    # Get weekly usage
    weekly_messages = org_usage.get_message_count('week')
    weekly_reasoning = org_usage.get_reasoning_count('week')
    print(f"• This week: {weekly_messages} messages, {weekly_reasoning} reasoning units")
    
    # Get monthly usage
    monthly_messages = org_usage.get_message_count('month')
    monthly_reasoning = org_usage.get_reasoning_count('month')
    print(f"• This month: {monthly_messages} messages, {monthly_reasoning} reasoning units")
    
    # Get summary for the day
    daily_summary = org_usage.get_usage_summary('day')
    print(f"• Daily summary: {daily_summary}")
    
    # Get detailed usage records
    print("\n📝 Recent Usage Details:")
    details = org_usage.get_usage_details(limit=10)
    
    if details:
        for i, detail in enumerate(details):
            print(f"{i+1}. Type: {detail.get('type')}, Count: {detail.get('count')}, Time: {detail.get('timestamp')}")
    else:
        print("No usage details found")
    
    print("\n✅ Usage tracking test completed")
    
    return {
        "organization_id": test_org_id,
        "daily_messages": daily_messages,
        "daily_reasoning": daily_reasoning,
        "weekly_messages": weekly_messages,
        "weekly_reasoning": weekly_reasoning,
        "monthly_messages": monthly_messages,
        "monthly_reasoning": monthly_reasoning,
        "daily_summary": daily_summary,
        "details_count": len(details)
    }

def simulate_chatwoot_response(org_id, message_chunks=3, reasoning_count=1):
    """Simulate the usage tracking as it would happen in chatwoot.py"""
    print(f"\n🤖 Simulating AI response with {message_chunks} chunks for organization {org_id}")
    
    try:
        # Simulate the usage tracking code from generate_ai_response
        org_usage = OrganizationUsage(org_id)
        
        # Track message usage - one count per message chunk
        org_usage.add_message(message_chunks)
        
        # Track reasoning usage
        org_usage.add_reasoning(reasoning_count)
        
        print(f"✓ Tracked simulated usage: {message_chunks} messages, {reasoning_count} reasoning")
        return True
    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if test organization ID is provided as command line argument
    test_org_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    if test_org_id:
        os.environ["TEST_ORG_ID"] = test_org_id
        print(f"Using provided organization ID: {test_org_id}")
    
    results = test_usage_tracking()
    
    # If an org ID is detected with existing data, simulate a chatwoot response
    if results["daily_messages"] > 0 or results["daily_reasoning"] > 0:
        simulate_chatwoot_response(results["organization_id"]) 