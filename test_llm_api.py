#!/usr/bin/env python3
"""
Test script for the new LLM Tool Execution API endpoint
Demonstrates how to use the /api/llm/execute endpoint with different providers and configurations
"""

import requests
import json
from typing import Dict, Any, Optional


def test_llm_api(
    base_url: str = "http://localhost:8000",
    llm_provider: str = "openai",
    user_query: str = "What is the latest news about artificial intelligence?",
    system_prompt: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "test_org",
    user_id: str = "test_user"
):
    """
    Test the LLM tool execution API endpoint
    
    Args:
        base_url: The base URL of the API server
        llm_provider: Either 'anthropic' or 'openai'
        user_query: The user's query
        system_prompt: Optional custom system prompt
        model_params: Optional model parameters (temperature, max_tokens, etc.)
        org_id: Organization ID
        user_id: User ID
    """
    
    endpoint = f"{base_url}/api/llm/execute"
    
    # Prepare the request payload
    payload = {
        "llm_provider": llm_provider,
        "user_query": user_query,
        "org_id": org_id,
        "user_id": user_id
    }
    
    if system_prompt:
        payload["system_prompt"] = system_prompt
    
    if model_params:
        payload["model_params"] = model_params
    
    print(f"\n{'='*60}")
    print(f"Testing LLM API with {llm_provider.upper()}")
    print(f"{'='*60}")
    print(f"Endpoint: {endpoint}")
    print(f"Provider: {llm_provider}")
    print(f"Query: {user_query}")
    print(f"System Prompt: {system_prompt if system_prompt else 'Default'}")
    print(f"Model Params: {model_params if model_params else 'Default'}")
    print(f"{'='*60}")
    
    try:
        # Make the API request
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"Provider: {result.get('provider')}")
            print(f"Model Used: {result.get('model_used')}")
            print(f"Execution Time: {result.get('execution_time'):.2f}s")
            print(f"Total Time: {result.get('total_elapsed_time'):.2f}s")
            print(f"\n📋 Result:")
            print("-" * 40)
            print(result.get('result', ''))
            print("-" * 40)
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error Message: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    """Run various test scenarios"""
    
    print("🚀 Testing LLM Tool Execution API")
    print("Make sure your server is running on http://localhost:8000")
    
    # Test 1: Basic OpenAI request
    test_llm_api(
        llm_provider="openai",
        user_query="What is machine learning?",
        system_prompt="You are a helpful AI tutor. Explain concepts clearly and concisely."
    )
    
    # Test 2: OpenAI with custom model parameters
    test_llm_api(
        llm_provider="openai",
        user_query="Tell me a creative story about a robot",
        system_prompt="You are a creative storyteller. Write engaging and imaginative stories.",
        model_params={
            "temperature": 0.8,
            "max_tokens": 300
        }
    )
    
    # Test 3: Anthropic Claude request
    test_llm_api(
        llm_provider="anthropic",
        user_query="Explain quantum computing in simple terms",
        system_prompt="You are a science educator. Make complex topics accessible to everyone."
    )
    
    # Test 4: Search query that should trigger tool usage
    test_llm_api(
        llm_provider="openai",
        user_query="What are the latest developments in renewable energy?",
        system_prompt="You are a research assistant. Provide current and accurate information."
    )
    
    print(f"\n{'='*60}")
    print("🎯 All tests completed!")
    print("The API supports dynamic system prompts and model parameters")
    print("Both Anthropic Claude and OpenAI GPT-4 are available")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 