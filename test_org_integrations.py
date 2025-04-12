#!/usr/bin/env python3
"""
Test script for Organization Integrations API

This script tests the API endpoints for the organization integrations.
Run with: python test_org_integrations.py <org_id>
"""

import argparse
import requests
import json
import time
import sys
from uuid import UUID
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:5001"  # Change this to match your API host

# Color codes for console output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def log_success(message):
    print(f"{GREEN}✓ {message}{RESET}")

def log_warning(message):
    print(f"{YELLOW}⚠ {message}{RESET}")

def log_error(message):
    print(f"{RED}✗ {message}{RESET}")

def is_valid_uuid(uuid_str):
    try:
        UUID(uuid_str)
        return True
    except ValueError:
        return False

def test_create_integration(org_id):
    """Test creating a new integration"""
    print("\n🔍 Testing create integration...")
    
    # Test data
    test_data = {
        "org_id": org_id,
        "integration_type": "facebook",
        "name": f"Test Facebook Integration {int(time.time())}",
        "webhook_url": "https://example.com/webhook",
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "config": {
            "page_id": "123456789012345",
            "verify_token": "test_verify_token"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/create-organization-integration",
            json=test_data
        )
        
        if response.status_code == 201:
            data = response.json()
            integration_id = data["integration"].get("id")
            
            if integration_id and is_valid_uuid(integration_id):
                log_success(f"Created integration with ID: {integration_id}")
                return integration_id
            else:
                log_error("Integration created but returned invalid ID")
                return None
        else:
            log_error(f"Failed to create integration: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        log_error(f"Exception during create test: {str(e)}")
        return None

def test_get_integrations(org_id):
    """Test listing all integrations for an organization"""
    print("\n🔍 Testing get all integrations...")
    
    try:
        response = requests.get(
            f"{BASE_URL}/organization-integrations",
            params={"org_id": org_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            integrations = data.get("integrations", [])
            log_success(f"Found {len(integrations)} integrations")
            
            # Print a summary of each integration
            for i, integration in enumerate(integrations):
                print(f"  {i+1}. {integration['name']} ({integration['integration_type']}) - Active: {integration['is_active']}")
            
            return integrations
        else:
            log_error(f"Failed to get integrations: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        log_error(f"Exception during get list test: {str(e)}")
        return []

def test_get_integration_details(integration_id):
    """Test getting details of a specific integration"""
    print(f"\n🔍 Testing get integration details for ID: {integration_id}")
    
    try:
        response = requests.get(
            f"{BASE_URL}/organization-integration/{integration_id}"
        )
        
        if response.status_code == 200:
            data = response.json()
            integration = data.get("integration", {})
            log_success(f"Retrieved integration: {integration['name']}")
            
            # Print all details
            print(json.dumps(integration, indent=2))
            
            return integration
        else:
            log_error(f"Failed to get integration details: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        log_error(f"Exception during get details test: {str(e)}")
        return None

def test_update_integration(integration_id):
    """Test updating an integration"""
    print(f"\n🔍 Testing update integration for ID: {integration_id}")
    
    # Test data for update
    update_data = {
        "id": integration_id,
        "name": f"Updated Integration {int(time.time())}",
        "webhook_url": "https://example.com/updated-webhook",
        "config": {
            "page_id": "987654321098765",
            "verify_token": "updated_verify_token",
            "additional_setting": "new_value"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/update-organization-integration",
            json=update_data
        )
        
        if response.status_code == 200:
            data = response.json()
            integration = data.get("integration", {})
            log_success(f"Updated integration: {integration['name']}")
            
            # Verify update was successful
            if integration['webhook_url'] == update_data['webhook_url']:
                log_success("Webhook URL updated successfully")
            else:
                log_warning("Webhook URL not updated correctly")
                
            return integration
        else:
            log_error(f"Failed to update integration: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        log_error(f"Exception during update test: {str(e)}")
        return None

def test_toggle_integration(integration_id, active_status):
    """Test toggling the active status of an integration"""
    print(f"\n🔍 Testing toggle integration to {active_status} for ID: {integration_id}")
    
    toggle_data = {
        "id": integration_id,
        "active": active_status
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/toggle-organization-integration",
            json=toggle_data
        )
        
        if response.status_code == 200:
            data = response.json()
            integration = data.get("integration", {})
            status_text = "activated" if active_status else "deactivated"
            log_success(f"{integration['name']} {status_text} successfully")
            
            # Verify toggle was successful
            if integration['is_active'] == active_status:
                log_success(f"Status change confirmed: is_active = {active_status}")
            else:
                log_error(f"Status not updated correctly. Expected: {active_status}, Got: {integration['is_active']}")
                
            return integration
        else:
            log_error(f"Failed to toggle integration: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        log_error(f"Exception during toggle test: {str(e)}")
        return None

def test_delete_integration(integration_id):
    """Test deleting an integration"""
    print(f"\n🔍 Testing delete integration for ID: {integration_id}")
    
    delete_data = {
        "id": integration_id
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/delete-organization-integration",
            json=delete_data
        )
        
        if response.status_code == 200:
            data = response.json()
            log_success(f"Integration deleted: {data.get('message')}")
            
            # Verify deletion by trying to get the integration
            verify_response = requests.get(
                f"{BASE_URL}/organization-integration/{integration_id}"
            )
            if verify_response.status_code == 404:
                log_success("Verified integration no longer exists")
            else:
                log_warning("Integration may still exist after deletion")
                
            return True
        else:
            log_error(f"Failed to delete integration: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        log_error(f"Exception during delete test: {str(e)}")
        return False

def run_tests(org_id):
    """Run all tests in sequence"""
    print(f"\n==== STARTING INTEGRATION API TESTS FOR ORG: {org_id} ====\n")
    
    # Test getting all integrations first to see what's there
    current_integrations = test_get_integrations(org_id)
    
    # Prompt user to select an existing integration or create a new one
    if current_integrations:
        print("\nDo you want to test with an existing integration or create a new one?")
        print("1. Use existing integration")
        print("2. Create new integration")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\nSelect an integration to use:")
            for i, integration in enumerate(current_integrations):
                print(f"{i+1}. {integration['name']} ({integration['integration_type']})")
            
            selection = int(input("Enter number: ").strip()) - 1
            if 0 <= selection < len(current_integrations):
                integration_id = current_integrations[selection]["id"]
                log_success(f"Selected integration ID: {integration_id}")
            else:
                log_error("Invalid selection. Creating a new integration instead.")
                integration_id = test_create_integration(org_id)
        else:
            integration_id = test_create_integration(org_id)
    else:
        log_warning("No existing integrations found. Creating a new one.")
        integration_id = test_create_integration(org_id)
    
    if not integration_id:
        log_error("Cannot proceed without a valid integration ID")
        return
    
    # Run the remaining tests
    test_get_integration_details(integration_id)
    test_update_integration(integration_id)
    
    # Toggle twice to test both activating and deactivating
    current_status = False  # Assume default is inactive
    test_toggle_integration(integration_id, not current_status)
    test_toggle_integration(integration_id, current_status)
    
    # Ask if user wants to delete the integration
    delete_choice = input("\nDo you want to delete this integration? (y/n): ").strip().lower()
    if delete_choice == 'y':
        test_delete_integration(integration_id)
    else:
        log_warning("Skipping delete test")
    
    print("\n==== ALL TESTS COMPLETED ====\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test organization integration APIs')
    parser.add_argument('org_id', help='Organization UUID to use for testing')
    parser.add_argument('--url', help='Base URL for the API (default: http://localhost:5001)', default='http://localhost:5001')
    
    args = parser.parse_args()
    
    if not is_valid_uuid(args.org_id):
        log_error("Invalid organization ID format. Must be a valid UUID.")
        sys.exit(1)
    
    BASE_URL = args.url
    run_tests(args.org_id) 