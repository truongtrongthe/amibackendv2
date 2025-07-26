#!/usr/bin/env python3
"""
Test script to demonstrate Google Drive folder reading capabilities
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import execute_agent_async

async def test_folder_reading():
    """Test the agent's ability to read Google Drive folders"""
    
    print("🧪 Testing Google Drive Folder Reading Capabilities")
    print("=" * 60)
    
    # Test cases for folder reading
    test_cases = [
        {
            "name": "Read entire folder by name",
            "request": "Read and analyze all documents in the 'Reports' folder",
            "expected_tool": "read_gdrive_folder",
            "expected_params": {"folder_name": "Reports"}
        },
        {
            "name": "Read specific file types in folder",
            "request": "Read all PDF files in the 'Sales' folder and provide a summary",
            "expected_tool": "read_gdrive_folder", 
            "expected_params": {"folder_name": "Sales", "file_types": ["pdf"]}
        },
        {
            "name": "Read folder by ID",
            "request": "Read all documents from Google Drive folder ID 1ABC123DEF456",
            "expected_tool": "read_gdrive_folder",
            "expected_params": {"folder_id": "1ABC123DEF456"}
        },
        {
            "name": "Mixed folder and file request",
            "request": "First read the 'Business Plans' folder, then analyze this specific document: https://docs.google.com/document/d/ABC123/edit",
            "expected_tools": ["read_gdrive_folder", "read_gdrive_link_docx"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"Request: {test_case['request']}")
        print("-" * 40)
        
        try:
            # Execute the agent
            response = await execute_agent_async(
                llm_provider="openai",
                user_request=test_case['request'],
                agent_id="test_folder_agent",
                agent_type="document_analyzer",
                org_id="test_org",
                user_id="test_user"
            )
            
            if response.success:
                print("✅ Agent executed successfully")
                print(f"Result preview: {response.result[:200]}...")
            else:
                print(f"❌ Agent execution failed: {response.error}")
                
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
        
        print()

async def test_single_folder_request():
    """Test a single folder reading request"""
    
    print("🎯 Testing Single Folder Reading Request")
    print("=" * 50)
    
    request = "Read all documents in the 'Project Documents' folder and provide a comprehensive analysis"
    
    print(f"Request: {request}")
    print("-" * 40)
    
    try:
        response = await execute_agent_async(
            llm_provider="openai",
            user_request=request,
            agent_id="folder_analyzer",
            agent_type="document_analyzer",
            org_id="test_org",
            user_id="test_user"
        )
        
        if response.success:
            print("✅ Folder reading request executed successfully")
            print(f"Execution time: {response.execution_time:.2f}s")
            print(f"Tasks completed: {response.tasks_completed}")
            print(f"\nResult:\n{response.result}")
        else:
            print(f"❌ Folder reading failed: {response.error}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Google Drive Folder Reading Tests")
    print("Make sure you have Google Drive credentials set up!")
    print()
    
    # Run the tests
    asyncio.run(test_folder_reading())
    print("\n" + "="*60 + "\n")
    asyncio.run(test_single_folder_request()) 