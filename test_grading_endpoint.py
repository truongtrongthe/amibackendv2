#!/usr/bin/env python3
"""
Test script for the /conversation/grading endpoint.
This demonstrates how to call the API to get the three comprehensive knowledge bases.
"""

import asyncio
import aiohttp
import json
from datetime import datetime

async def test_grading_endpoint():
    """Test the /conversation/grading endpoint."""
    
    # Endpoint configuration
    base_url = "http://localhost:5001"  # Adjust if your server runs on a different port
    endpoint = "/conversation/grading"
    url = f"{base_url}{endpoint}"
    
    # Request payload
    payload = {
        "graph_version_id": ""  # Use empty string for default, or specify a specific version
    }
    
    print("🧪 Testing /conversation/grading endpoint")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 60)
    
    try:
        async with aiohttp.ClientSession() as session:
            start_time = datetime.now()
            
            async with session.post(url, json=payload) as response:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                
                print(f"⏱️ Response time: {elapsed_time:.2f}s")
                print(f"📊 Status code: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    
                    print("✅ Success! Response structure:")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Request ID: {data.get('request_id')}")
                    print(f"   Server elapsed time: {data.get('elapsed_time', 'N/A'):.2f}s")
                    print(f"   Graph Version ID: {data.get('graph_version_id', 'N/A')}")
                    
                    # Validation info
                    validation = data.get('validation', {})
                    print(f"\n📋 Validation Results:")
                    print(f"   Overall Status: {validation.get('overall_status')}")
                    print(f"   Basic Knowledge Valid: {validation.get('basic_knowledge_valid')}/3")
                    print(f"   Comprehensive Knowledge Valid: {validation.get('comprehensive_knowledge_valid')}/3")
                    
                    # Knowledge bases
                    knowledge_bases = data.get('knowledge_bases', {})
                    print(f"\n🧠 Knowledge Bases:")
                    
                    for name, details in knowledge_bases.items():
                        print(f"\n   📚 {name.replace('_', ' ').title()}:")
                        print(f"      Content Length: {details.get('content_length', 0)} characters")
                        print(f"      Is Valid: {details.get('is_valid', False)}")
                        print(f"      Source: {details.get('metadata', {}).get('source', 'Unknown')}")
                        print(f"      Compiled At: {details.get('metadata', {}).get('timestamp', 'Unknown')}")
                        
                        # Show preview of content
                        content = details.get('knowledge_context', '')
                        if content:
                            preview = content[:200].replace('\n', ' ')
                            print(f"      Preview: {preview}...")
                        else:
                            print(f"      Preview: No content available")
                    
                    # Show the three main knowledge bases specifically
                    print(f"\n🎯 The Three Main Knowledge Components:")
                    
                    profiling = knowledge_bases.get('profiling_instinct', {})
                    communication = knowledge_bases.get('communication_instinct', {})
                    business = knowledge_bases.get('business_objectives_instinct', {})
                    
                    print(f"   1. Profiling Knowledge as Instinct: {profiling.get('content_length', 0)} chars")
                    print(f"   2. Communication Knowledge as Instinct: {communication.get('content_length', 0)} chars")
                    print(f"   3. Business Objectives Knowledge as Instinct: {business.get('content_length', 0)} chars")
                    
                    print(f"\n📄 Full Response JSON:")
                    print(json.dumps(data, indent=2, ensure_ascii=False)[:500] + "..." if len(json.dumps(data)) > 500 else json.dumps(data, indent=2, ensure_ascii=False))
                    
                else:
                    error_text = await response.text()
                    print(f"❌ Error response:")
                    print(f"   Status: {response.status}")
                    print(f"   Response: {error_text}")
                    
    except aiohttp.ClientError as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def test_with_specific_graph_version():
    """Test with a specific graph version ID."""
    
    # You can specify a specific graph version ID here
    specific_graph_version = "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"  # Example from the code
    
    print(f"\n🔍 Testing with specific graph version: {specific_graph_version}")
    
    base_url = "http://localhost:5001"
    endpoint = "/conversation/grading"
    url = f"{base_url}{endpoint}"
    
    payload = {
        "graph_version_id": specific_graph_version
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Success with specific graph version!")
                    print(f"   Used Graph Version: {data.get('graph_version_id')}")
                    print(f"   Validation Status: {data.get('validation', {}).get('overall_status')}")
                else:
                    error_text = await response.text()
                    print(f"❌ Error with specific graph version: {error_text}")
                    
    except Exception as e:
        print(f"❌ Error testing specific graph version: {e}")

def show_usage_examples():
    """Show examples of how to use the endpoint in different scenarios."""
    
    print("\n" + "=" * 80)
    print("💡 Usage Examples")
    print("=" * 80)
    
    print("\n1. Basic curl command:")
    print("curl -X POST http://localhost:5001/conversation/grading \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"graph_version_id\": \"\"}'")
    
    print("\n2. With specific graph version:")
    print("curl -X POST http://localhost:5001/conversation/grading \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"graph_version_id\": \"bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc\"}'")
    
    print("\n3. Python requests example:")
    print("""
import requests

response = requests.post(
    'http://localhost:5001/conversation/grading',
    json={'graph_version_id': ''}
)

if response.status_code == 200:
    data = response.json()
    
    # Access the three knowledge bases
    profiling = data['knowledge_bases']['profiling_instinct']['knowledge_context']
    communication = data['knowledge_bases']['communication_instinct']['knowledge_context']
    business = data['knowledge_bases']['business_objectives_instinct']['knowledge_context']
    
    print("Profiling Knowledge:", len(profiling), "characters")
    print("Communication Knowledge:", len(communication), "characters")
    print("Business Knowledge:", len(business), "characters")
""")
    
    print("\n4. JavaScript/fetch example:")
    print("""
fetch('http://localhost:5001/conversation/grading', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        graph_version_id: ''
    })
})
.then(response => response.json())
.then(data => {
    if (data.status === 'success') {
        console.log('Profiling Instinct:', data.knowledge_bases.profiling_instinct.content_length, 'chars');
        console.log('Communication Instinct:', data.knowledge_bases.communication_instinct.content_length, 'chars');
        console.log('Business Objectives Instinct:', data.knowledge_bases.business_objectives_instinct.content_length, 'chars');
    }
});
""")

if __name__ == "__main__":
    print("🚀 Starting /conversation/grading endpoint tests...")
    
    # Show usage examples first
    show_usage_examples()
    
    # Run the tests
    print("\n" + "=" * 80)
    print("🧪 Running Tests")
    print("=" * 80)
    
    asyncio.run(test_grading_endpoint())
    asyncio.run(test_with_specific_graph_version()) 