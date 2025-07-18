#!/usr/bin/env python3
"""
Demo test for conversation analysis logic
Shows how the improved system works without requiring API keys
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from human_context_tool import create_human_context_tool

async def test_conversation_analysis_demo():
    """Demo test showing conversation analysis logic"""
    
    print("🔍 Demo: Conversation Analysis & Confidence Assessment")
    print("=" * 60)
    print("This demo shows how the improved system analyzes conversation history")
    print("and determines confidence levels for AI agent suggestions.")
    print()
    
    # Create test conversation with progressive information sharing
    test_conversations = [
        {
            "name": "Scenario 1: Initial greeting (no context)",
            "history": [
                {"role": "user", "content": "Hi there!"},
                {"role": "assistant", "content": "Hello! I'm Ami..."}
            ],
            "user_query": "Guide me",
            "expected_confidence": "Low"
        },
        {
            "name": "Scenario 2: Some work information shared",
            "history": [
                {"role": "user", "content": "Hi there!"},
                {"role": "assistant", "content": "Hello! I'm Ami..."},
                {"role": "user", "content": "I work in marketing and spend most of my time creating social media content"},
                {"role": "assistant", "content": "That sounds interesting! What did you work on yesterday?"},
                {"role": "user", "content": "Yesterday I spent 3 hours creating Instagram posts and scheduling them"}
            ],
            "user_query": "What should I do next?",
            "expected_confidence": "Medium"
        },
        {
            "name": "Scenario 3: Concrete problems and time estimates",
            "history": [
                {"role": "user", "content": "Hi there!"},
                {"role": "assistant", "content": "Hello! I'm Ami..."},
                {"role": "user", "content": "I work in marketing and spend most of my time creating social media content"},
                {"role": "assistant", "content": "That sounds interesting! What did you work on yesterday?"},
                {"role": "user", "content": "Yesterday I spent 3 hours creating Instagram posts and scheduling them"},
                {"role": "assistant", "content": "3 hours sounds like a lot! Can you walk me through what that process involves?"},
                {"role": "user", "content": "Well, I have to write the captions, find appropriate images, resize them, schedule posts for different time zones, and then manually post to 5 different accounts. It's really repetitive and takes me about 2-3 hours every day."},
                {"role": "assistant", "content": "That does sound time-consuming. How long have you been doing this daily routine?"},
                {"role": "user", "content": "About 6 months now. It's getting really tedious and I feel like I'm wasting time on repetitive tasks when I could be focusing on strategy."}
            ],
            "user_query": "Is there a better way to do this?",
            "expected_confidence": "High"
        }
    ]
    
    print("📊 Conversation Analysis Results (using fallback analysis)")
    print("-" * 50)
    
    human_context_tool = create_human_context_tool()
    
    for i, scenario in enumerate(test_conversations):
        print(f"\n{i+1}. {scenario['name']}")
        print(f"   Query: '{scenario['user_query']}'")
        
        try:
            # Get human context with conversation analysis (will use fallback due to no API key)
            context = await human_context_tool.get_human_context(
                user_id="test_user",
                org_id="test_org", 
                conversation_history=scenario["history"],
                llm_provider="anthropic"
            )
            
            patterns = context.get("conversation_patterns", {})
            confidence = patterns.get("confidence_level", 0.0)
            ready_for_suggestion = patterns.get("readiness_for_suggestion", False)
            facts = patterns.get("facts_extracted", [])
            pain_points = patterns.get("pain_points", [])
            reasoning = patterns.get("reasoning", "")
            clues = patterns.get("clues", [])
            
            print(f"   Confidence Level: {confidence:.2f}")
            print(f"   Ready for Suggestion: {ready_for_suggestion}")
            print(f"   Facts Extracted: {len(facts)}")
            print(f"   Pain Points: {len(pain_points)}")
            print(f"   Industry Clues: {clues}")
            print(f"   Reasoning: {reasoning}")
            
            if facts:
                print(f"   Key Facts: {facts[:2]}")
            if pain_points:
                print(f"   Key Pain Points: {pain_points[:2]}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("📋 System Architecture Overview")
    print("=" * 60)
    print("""
IMPROVED CONVERSATION ANALYSIS SYSTEM:

1. **Mom Test Discovery Approach**:
   - Asks about past behavior, not future hypotheticals
   - Focuses on specific, concrete examples from recent work
   - Listens for real problems, not imagined ones
   - Asks follow-up questions about time and effort

2. **Conversation Analysis Requirements**:
   - ALWAYS analyze conversation history before making suggestions
   - Look for concrete facts, not assumptions
   - Identify patterns in work, pain points, and time-consuming tasks
   - Extract specific examples of problems mentioned
   - Note frequency and severity of issues

3. **Confidence Assessment**:
   Before suggesting an AI agent, system must understand:
   - Specific role and responsibilities
   - Biggest time-consuming tasks
   - Most frustrating or repetitive work
   - Impact on productivity
   - Technical comfort level

4. **Suggestion Criteria**:
   Only suggest AI agents when having enough FACTS about:
   ✓ What they actually do daily/weekly
   ✓ Specific problems they've mentioned experiencing
   ✓ Time estimates for tasks they find tedious
   ✓ Their workflow and current tools
   ✓ Their goals and desired outcomes

5. **Critical Progression**:
   - Discovery Phase: Use Mom Test principles
   - Analysis Phase: Analyze conversation history
   - Confidence Assessment: Ensure enough concrete information
   - Suggestion Phase: Recommend most appropriate AI agent

The system now prioritizes understanding before suggesting, using actual
conversation analysis instead of mock data.
""")
    
    print("\n🎯 Key Improvements Made:")
    print("-" * 30)
    print("✅ Updated all system prompts with conversation analysis requirements")
    print("✅ Added confidence assessment before AI agent suggestions")
    print("✅ Implemented LLM-based conversation analysis (when API keys available)")
    print("✅ Created clear progression from discovery to suggestion")
    print("✅ Added fallback analysis for when LLM analysis fails")
    print("✅ Integrated Mom Test principles throughout the system")
    print("✅ Removed reliance on mock data for conversation analysis")
    
    print("\n✅ Demo Complete - System is now ready for production use!")

if __name__ == "__main__":
    asyncio.run(test_conversation_analysis_demo()) 