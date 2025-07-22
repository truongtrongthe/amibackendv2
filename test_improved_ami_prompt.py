#!/usr/bin/env python3
"""
Test script to demonstrate the improved Ami prompt with the 3-step workflow:
IMAGINATION → CHOICE → TEACHING → APPROVAL
"""

import asyncio
import json
from datetime import datetime
from exec_tool import ExecutiveTool, ToolExecutionRequest

async def test_improved_ami_workflow():
    """Test the improved Ami prompt with the strategic 3-step workflow"""
    
    print("🚀 IMPROVED AMI PROMPT TEST - 3-STEP WORKFLOW")
    print("=" * 70)
    print("Testing the strategic improvement: IMAGINATION → CHOICE → TEACHING → APPROVAL")
    print()
    
    # Initialize the executive tool
    executive_tool = ExecutiveTool()
    
    # Test scenarios that should trigger the 3-step workflow
    test_scenarios = [
        {
            "name": "M&A Consultant - Imagination Phase",
            "query": "Anh làm tư vấn M&A doanh nghiệp",
            "expected_flow": [
                "imagination exploration",
                "choice push",
                "teaching guidance",
                "approval request"
            ]
        },
        {
            "name": "Healthcare Professional - Choice Phase", 
            "query": "I want to build an AI agent for patient triage",
            "expected_flow": [
                "immediate teaching guidance",
                "process questions",
                "approval request"
            ]
        },
        {
            "name": "E-commerce Owner - Teaching Phase",
            "query": "Our company has 50 employees and we handle 1000 orders daily",
            "expected_flow": [
                "learning workflow",
                "approval request",
                "knowledge storage"
            ]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"🧪 TESTING: {scenario['name']}")
        print(f"📝 Query: '{scenario['query']}'")
        print(f"🎯 Expected Flow: {' → '.join(scenario['expected_flow'])}")
        print("-" * 50)
        
        # Create request
        request = ToolExecutionRequest(
            llm_provider="anthropic",
            user_query=scenario['query'],
            enable_tools=True,
            cursor_mode=True,
            enable_intent_classification=True,
            org_id="test_org",
            user_id="test_user"
        )
        
        # Execute and collect response
        response_chunks = []
        async for chunk in executive_tool.execute_tool_stream(request):
            response_chunks.append(chunk)
            
            # Show real-time progress
            if chunk.get("type") == "response_chunk":
                print(f"💬 Ami: {chunk.get('content', '')}")
            elif chunk.get("type") == "thinking":
                print(f"💭 {chunk.get('content', '')}")
            elif chunk.get("type") == "tool_call":
                print(f"🔧 Tool: {chunk.get('tool_name', '')} - {chunk.get('status', '')}")
        
        print()
        print("✅ Test completed")
        print("=" * 70)
        print()

def demonstrate_workflow_phases():
    """Demonstrate the key phases of the improved workflow"""
    
    print("📋 IMPROVED WORKFLOW PHASES")
    print("=" * 50)
    
    phases = [
        {
            "phase": "STEP 1: IMAGINATION → CHOICE",
            "description": "After exploring imagination, ALWAYS push them to choose",
            "key_phrases": [
                "Which of these AI agent ideas excites you most?",
                "Which one would have the biggest impact on your work?",
                "Let's focus on building [specific agent] - does that feel right?",
                "Which agent should we bring to life first?"
            ]
        },
        {
            "phase": "STEP 2: CHOICE → TEACHING", 
            "description": "Once they choose, immediately guide them to teach",
            "key_phrases": [
                "Perfect! Now I need to understand your expertise...",
                "Tell me about your process for [specific task]",
                "Walk me through how you currently handle [responsibility]",
                "What knowledge would this agent need to do this job well?"
            ]
        },
        {
            "phase": "STEP 3: TEACHING → APPROVAL",
            "description": "When they share knowledge, always seek approval to save",
            "key_phrases": [
                "This is valuable knowledge! Should I save this for your agent?",
                "I can store this expertise to help your agent perform better.",
                "Would you like me to remember this process for your agent?"
            ]
        }
    ]
    
    for phase in phases:
        print(f"\n🎯 {phase['phase']}")
        print(f"📝 {phase['description']}")
        print("💬 Key Phrases:")
        for phrase in phase['key_phrases']:
            print(f"   • {phrase}")
    
    print("\n" + "=" * 50)

def show_strategic_improvements():
    """Show the strategic improvements made to the prompt"""
    
    print("🔧 STRATEGIC IMPROVEMENTS")
    print("=" * 50)
    
    improvements = [
        {
            "aspect": "Core Mission",
            "before": "Transform imagination into AI agents without coding",
            "after": "Transform imagination by collecting know-how through teaching intent"
        },
        {
            "aspect": "Building Approach", 
            "before": "Understand → Construct → Find → Optimize",
            "after": "Understand → Guide Choice → Collect Know-How → Prepare Tech → Optimize"
        },
        {
            "aspect": "Workflow",
            "before": "When they say 'Build It!' → collect requirements",
            "after": "3-STEP: Imagination → Choice → Teaching → Approval"
        },
        {
            "aspect": "Human Role",
            "before": "Dream big and tell me what you want",
            "after": "Dream big, choose what to build, teach the agent your expertise"
        },
        {
            "aspect": "Ami's Role",
            "before": "Handle technical heavy lifting",
            "after": "Collect know-how through teaching intent and prepare technical foundation"
        }
    ]
    
    for improvement in improvements:
        print(f"\n📊 {improvement['aspect']}")
        print(f"❌ Before: {improvement['before']}")
        print(f"✅ After:  {improvement['after']}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    async def main():
        show_strategic_improvements()
        demonstrate_workflow_phases()
        await test_improved_ami_workflow()
    
    asyncio.run(main()) 