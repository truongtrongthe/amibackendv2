#!/usr/bin/env python3
"""
Test script to demonstrate complete Ami + Decision Endpoint Integration
Shows the full workflow: Imagination → Choice → Teaching → Approval → Knowledge Saving
"""

import asyncio
import json
import aiohttp
from datetime import datetime
from typing import Dict, Any
from exec_tool import ExecutiveTool, ToolExecutionRequest

async def test_complete_integration():
    """Test the complete integration between Ami and decision endpoint"""
    
    print("🔗 COMPLETE AMI + DECISION ENDPOINT INTEGRATION TEST")
    print("=" * 80)
    print("Testing the full workflow: IMAGINATION → CHOICE → TEACHING → APPROVAL → SAVING")
    print()
    
    # Initialize the executive tool
    executive_tool = ExecutiveTool()
    
    # Test scenarios that demonstrate the complete integration
    integration_scenarios = [
        {
            "name": "M&A Consultant - Complete Workflow",
            "steps": [
                {
                    "phase": "IMAGINATION",
                    "query": "I work in M&A consulting for mid-size companies",
                    "expected": ["imagination exploration", "AI agent suggestions", "choice push"]
                },
                {
                    "phase": "CHOICE",
                    "query": "I love the AI Agent Due Diligence idea! Let's build that one.",
                    "expected": ["teaching guidance", "expertise request", "process questions"]
                },
                {
                    "phase": "TEACHING",
                    "query": "Our due diligence process has 5 key phases: 1) Financial analysis - we examine 3 years of statements, 2) Legal review - check contracts and IP, 3) Market analysis - competitive positioning, 4) Management evaluation - team capabilities, 5) Risk assessment - identify red flags.",
                    "expected": ["learning tools triggered", "decision creation", "approval request"]
                }
            ]
        },
        {
            "name": "E-commerce Owner - Knowledge Collection",
            "steps": [
                {
                    "phase": "IMAGINATION",
                    "query": "I run an online electronics store with 15 employees",
                    "expected": ["agent suggestions", "choice options", "business focus"]
                },
                {
                    "phase": "CHOICE", 
                    "query": "The Customer Service Agent sounds perfect for my business",
                    "expected": ["expertise questions", "process inquiry", "domain knowledge request"]
                },
                {
                    "phase": "TEACHING",
                    "query": "Our customer service process: Level 1 handles basic questions via chatbot, Level 2 deals with returns and exchanges, Level 3 handles technical support and warranty issues. We use a priority system: VIP customers get 1-hour response, regular customers within 4 hours.",
                    "expected": ["decision workflow", "approval UI trigger", "knowledge preview"]
                }
            ]
        }
    ]
    
    for scenario in integration_scenarios:
        await test_integration_scenario(executive_tool, scenario)
    
    # Test decision endpoint integration
    await test_decision_endpoint_integration()

async def test_integration_scenario(executive_tool: ExecutiveTool, scenario: Dict[str, Any]):
    """Test a complete integration scenario"""
    
    print(f"🧪 TESTING SCENARIO: {scenario['name']}")
    print("=" * 60)
    
    for step in scenario['steps']:
        print(f"\n📍 PHASE: {step['phase']}")
        print(f"🔵 Query: '{step['query']}'")
        print(f"🎯 Expected: {step['expected']}")
        print("-" * 40)
        
        # Create request with learning tools enabled
        request = ToolExecutionRequest(
            llm_provider="anthropic",
            user_query=step['query'],
            enable_tools=True,
            cursor_mode=True,
            enable_intent_classification=True,
            org_id="test_integration",
            user_id="test_user_integration"
        )
        
        # Capture all response chunks
        response_data = {
            "thinking_steps": [],
            "tool_calls": [],
            "response_chunks": [],
            "decisions_created": [],
            "learning_triggered": False
        }
        
        # Execute and analyze response
        async for chunk in executive_tool.execute_tool_stream(request):
            await analyze_response_chunk(chunk, response_data)
        
        # Report results
        print_phase_results(step['phase'], response_data)
        print()
    
    print("✅ Scenario completed")
    print("=" * 60)
    print()

async def analyze_response_chunk(chunk: Dict[str, Any], response_data: Dict[str, Any]):
    """Analyze each response chunk for integration testing"""
    
    chunk_type = chunk.get("type", "")
    content = chunk.get("content", "")
    
    if chunk_type == "thinking":
        response_data["thinking_steps"].append(content)
        # Show real-time thinking
        print(f"💭 {content}")
    
    elif chunk_type == "response_chunk":
        response_data["response_chunks"].append(content)
        # Show real-time response
        print(f"💬 Ami: {content}", end="")
    
    elif chunk_type == "tool_call":
        tool_name = chunk.get("tool_name", "")
        status = chunk.get("status", "")
        response_data["tool_calls"].append({"tool": tool_name, "status": status})
        
        # Detect learning-related tool calls
        if tool_name in ["search_learning_context", "analyze_learning_opportunity", "request_learning_decision"]:
            response_data["learning_triggered"] = True
        
        if tool_name == "request_learning_decision" and status == "completed":
            # Extract decision ID if available
            result = chunk.get("result", "")
            if "Decision ID:" in result:
                decision_id = extract_decision_id(result)
                response_data["decisions_created"].append(decision_id)
        
        print(f"🔧 Tool: {tool_name} - {status}")

def extract_decision_id(result: str) -> str:
    """Extract decision ID from tool result"""
    lines = result.split("\n")
    for line in lines:
        if "Decision ID:" in line:
            return line.split("Decision ID:")[1].strip()
    return "unknown"

def print_phase_results(phase: str, response_data: Dict[str, Any]):
    """Print the results of a test phase"""
    
    print("\n📊 PHASE RESULTS:")
    print(f"   Thinking steps: {len(response_data['thinking_steps'])}")
    print(f"   Tool calls: {len(response_data['tool_calls'])}")
    print(f"   Learning triggered: {'✅' if response_data['learning_triggered'] else '❌'}")
    print(f"   Decisions created: {len(response_data['decisions_created'])}")
    
    if response_data['decisions_created']:
        print(f"   Decision IDs: {response_data['decisions_created']}")
    
    # Check for key phrases based on phase
    response_text = " ".join(response_data['response_chunks'])
    
    if phase == "IMAGINATION":
        choice_phrases = ["Which", "choose", "excites you most", "biggest impact"]
        found_choice = any(phrase.lower() in response_text.lower() for phrase in choice_phrases)
        print(f"   Choice push detected: {'✅' if found_choice else '❌'}")
    
    elif phase == "CHOICE":
        teaching_phrases = ["expertise", "process", "walk me through", "tell me about"]
        found_teaching = any(phrase.lower() in response_text.lower() for phrase in teaching_phrases)
        print(f"   Teaching guidance detected: {'✅' if found_teaching else '❌'}")
    
    elif phase == "TEACHING":
        approval_phrases = ["save", "remember", "store this", "valuable knowledge"]
        found_approval = any(phrase.lower() in response_text.lower() for phrase in approval_phrases)
        print(f"   Approval request detected: {'✅' if found_approval else '❌'}")

async def test_decision_endpoint_integration():
    """Test the decision endpoint integration"""
    
    print("🌐 TESTING DECISION ENDPOINT INTEGRATION")
    print("=" * 60)
    
    base_url = "http://localhost:5000"  # Adjust as needed
    
    # Test scenarios for decision endpoints
    endpoint_tests = [
        {
            "name": "Get Pending Decisions",
            "method": "GET", 
            "url": f"{base_url}/api/learning/decisions?user_id=test_user_integration",
            "expected": "List of pending decisions"
        },
        {
            "name": "Submit Learning Decision",
            "method": "POST",
            "url": f"{base_url}/api/learning/decision",
            "data": {
                "decision_id": "learning_decision_test123",
                "human_choice": "Save as new knowledge"
            },
            "expected": "Decision submission confirmation"
        },
        {
            "name": "Cleanup Expired Decisions",
            "method": "POST",
            "url": f"{base_url}/api/learning/decisions/cleanup",
            "expected": "Cleanup status"
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for test in endpoint_tests:
            await test_endpoint(session, test)

async def test_endpoint(session: aiohttp.ClientSession, test: Dict[str, Any]):
    """Test a specific endpoint"""
    
    print(f"\n🔗 Testing: {test['name']}")
    print(f"   Method: {test['method']}")
    print(f"   URL: {test['url']}")
    
    try:
        if test['method'] == "GET":
            async with session.get(test['url']) as response:
                result = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Result: {json.dumps(result, indent=2)[:200]}...")
        
        elif test['method'] == "POST":
            data = test.get('data')
            async with session.post(test['url'], json=data) as response:
                result = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Result: {json.dumps(result, indent=2)[:200]}...")
        
        print(f"   Expected: {test['expected']}")
        print("   ✅ Endpoint accessible")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        print("   ⚠️  Note: Make sure the backend server is running")

def demonstrate_complete_workflow():
    """Demonstrate the complete workflow integration"""
    
    print("📋 COMPLETE WORKFLOW DEMONSTRATION")
    print("=" * 50)
    
    workflow_steps = [
        {
            "step": "1. IMAGINATION → CHOICE",
            "ami_action": "Ami suggests 3 AI agent ideas and pushes for choice",
            "human_action": "Human selects preferred AI agent",
            "backend_action": "Intent analysis detects choice commitment",
            "frontend_action": "Display conversation normally"
        },
        {
            "step": "2. CHOICE → TEACHING",
            "ami_action": "Ami immediately asks for expertise and processes",
            "human_action": "Human shares their know-how and processes",
            "backend_action": "Learning tools detect teaching intent",
            "frontend_action": "Show conversation flow"
        },
        {
            "step": "3. TEACHING → APPROVAL",
            "ami_action": "Ami triggers request_learning_decision()",
            "human_action": "Human sees approval UI and makes choice",
            "backend_action": "complete_learning_decision() saves knowledge",
            "frontend_action": "Poll decisions, show UI, submit choice"
        },
        {
            "step": "4. APPROVAL → CONFIRMATION",
            "ami_action": "Ami confirms knowledge saved",
            "human_action": "Human continues building their AI agent",
            "backend_action": "3 knowledge vectors saved to Pinecone",
            "frontend_action": "Show success confirmation"
        }
    ]
    
    for workflow in workflow_steps:
        print(f"\n🔄 {workflow['step']}")
        print(f"   🤖 Ami: {workflow['ami_action']}")
        print(f"   👤 Human: {workflow['human_action']}")
        print(f"   🔧 Backend: {workflow['backend_action']}")
        print(f"   💻 Frontend: {workflow['frontend_action']}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    async def main():
        demonstrate_complete_workflow()
        print()
        await test_complete_integration()
    
    asyncio.run(main()) 