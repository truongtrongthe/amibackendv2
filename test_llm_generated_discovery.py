#!/usr/bin/env python3
"""
Test script to verify LLM-generated Mom Test discovery system
"""

import asyncio
import json
from human_context_tool import HumanContextTool


async def test_llm_generated_discovery():
    """Test LLM-generated Mom Test discovery questions and approaches"""
    
    print("🧪 Testing LLM-Generated Mom Test Discovery")
    print("=" * 60)
    
    # Initialize the tool
    tool = HumanContextTool()
    
    # Test scenarios with different contexts
    test_contexts = [
        {
            "name": "Finance M&A Consultant",
            "user_id": "finance_user",
            "org_id": "test_org",  # This maps to TechCorp finance
            "expected_elements": ["deal", "analysis", "client", "time", "last"]
        },
        {
            "name": "Healthcare Worker",
            "user_id": "healthcare_user",
            "org_id": "healthcare_org",
            "mock_override": {
                "org_profile": {
                    "name": "MedCenter",
                    "industry": "healthcare",
                    "size": "200+ employees",
                    "challenges": ["patient documentation", "scheduling", "compliance"],
                    "focus_areas": ["patient care", "efficiency", "quality"],
                    "tech_maturity": "basic"
                },
                "user_profile": {
                    "name": "Dr. Smith",
                    "role": "physician",
                    "department": "emergency",
                    "interests": ["patient care", "efficiency"],
                    "skills": ["clinical", "EMR systems"],
                    "previous_projects": ["workflow optimization"]
                }
            },
            "expected_elements": ["patient", "shift", "documentation", "time", "last"]
        },
        {
            "name": "Retail Manager",
            "user_id": "retail_user",
            "org_id": "retail_org",
            "mock_override": {
                "org_profile": {
                    "name": "RetailCorp",
                    "industry": "retail",
                    "size": "100-500 employees",
                    "challenges": ["inventory management", "customer service", "sales tracking"],
                    "focus_areas": ["customer experience", "operations", "growth"],
                    "tech_maturity": "intermediate"
                },
                "user_profile": {
                    "name": "Manager",
                    "role": "store manager",
                    "department": "operations",
                    "interests": ["customer service", "efficiency"],
                    "skills": ["management", "POS systems"],
                    "previous_projects": ["process improvement"]
                }
            },
            "expected_elements": ["customer", "inventory", "sales", "time", "yesterday"]
        }
    ]
    
    for i, context in enumerate(test_contexts, 1):
        print(f"\n📋 Test {i}: {context['name']}")
        print("-" * 40)
        
        try:
            # Get human context
            if "mock_override" in context:
                # Use mock override for testing
                human_context = context["mock_override"]
                human_context["conversation_patterns"] = {"patterns": [], "topics": [], "clues": []}
            else:
                # Use the tool's context retrieval
                human_context = await tool.get_human_context(
                    context["user_id"], 
                    context["org_id"]
                )
            
            print(f"Context: {human_context['org_profile']['name']} ({human_context['org_profile']['industry']})")
            print(f"Role: {human_context['user_profile']['role']}")
            
            # Generate discovery strategy using LLM
            print("\n🤖 Generating LLM-based discovery strategy...")
            strategy = await tool.generate_discovery_strategy(human_context, "openai")
            
            print(f"\n📝 Generated Opener:")
            print(f"   {strategy['opener']}")
            
            print(f"\n❓ Generated Questions:")
            for j, question in enumerate(strategy['discovery_questions'], 1):
                print(f"   {j}. {question}")
            
            print(f"\n💬 Conversational Approach:")
            print(f"   {strategy['conversational_approach']}")
            
            # Analyze quality
            print(f"\n📊 Quality Analysis:")
            opener_quality = analyze_opener_quality(strategy['opener'])
            questions_quality = analyze_questions_quality(strategy['discovery_questions'], context['expected_elements'])
            approach_quality = analyze_approach_quality(strategy['conversational_approach'])
            
            print(f"   Opener Quality: {opener_quality['score']:.1f}% - {opener_quality['feedback']}")
            print(f"   Questions Quality: {questions_quality['score']:.1f}% - {questions_quality['feedback']}")
            print(f"   Approach Quality: {approach_quality['score']:.1f}% - {approach_quality['feedback']}")
            
            overall_score = (opener_quality['score'] + questions_quality['score'] + approach_quality['score']) / 3
            print(f"   Overall Score: {overall_score:.1f}%")
            
            if overall_score >= 70:
                print("   ✅ Good LLM-generated discovery strategy!")
            else:
                print("   ❌ Strategy needs improvement")
                
        except Exception as e:
            print(f"❌ Error in test {i}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 LLM-Generated Discovery Testing Complete!")


def analyze_opener_quality(opener: str) -> dict:
    """Analyze the quality of the generated opener"""
    opener_lower = opener.lower()
    
    checks = {
        "mentions_ami": "ami" in opener_lower,
        "mentions_co_builder": any(word in opener_lower for word in ["co-builder", "co builder", "cobuilder"]),
        "warm_tone": any(word in opener_lower for word in ["hi", "hello", "nice", "great", "excited"]),
        "concise": len(opener.split()) <= 30,
        "no_ai_pitching": not any(word in opener_lower for word in ["ai agent", "automate", "build", "solution"])
    }
    
    score = sum(checks.values()) / len(checks) * 100
    
    if score >= 80:
        feedback = "Excellent opener"
    elif score >= 60:
        feedback = "Good opener"
    else:
        feedback = "Needs improvement"
    
    return {"score": score, "feedback": feedback, "details": checks}


def analyze_questions_quality(questions: list, expected_elements: list) -> dict:
    """Analyze the quality of generated questions"""
    if not questions:
        return {"score": 0, "feedback": "No questions generated", "details": {}}
    
    all_questions = " ".join(questions).lower()
    
    checks = {
        "past_behavior": any(word in all_questions for word in ["what did", "yesterday", "last", "spent", "took"]),
        "specific_examples": any(word in all_questions for word in ["walk me through", "tell me about", "how", "what happened"]),
        "time_focus": any(word in all_questions for word in ["how long", "time", "took", "spent"]),
        "context_relevant": any(element.lower() in all_questions for element in expected_elements),
        "no_ai_mention": not any(word in all_questions for word in ["ai", "automate", "agent", "build"])
    }
    
    score = sum(checks.values()) / len(checks) * 100
    
    if score >= 80:
        feedback = "Excellent Mom Test questions"
    elif score >= 60:
        feedback = "Good Mom Test questions"
    else:
        feedback = "Questions need improvement"
    
    return {"score": score, "feedback": feedback, "details": checks}


def analyze_approach_quality(approach: str) -> dict:
    """Analyze the quality of the conversational approach"""
    approach_lower = approach.lower()
    
    checks = {
        "mentions_curiosity": any(word in approach_lower for word in ["curious", "listen", "ask", "learn"]),
        "conversational": any(word in approach_lower for word in ["conversational", "natural", "genuine"]),
        "no_pitching": not any(word in approach_lower for word in ["sell", "pitch", "solution", "ai agent"]),
        "follow_up_focus": any(word in approach_lower for word in ["follow up", "dig deeper", "ask more"]),
        "concise": len(approach.split()) <= 100
    }
    
    score = sum(checks.values()) / len(checks) * 100
    
    if score >= 80:
        feedback = "Excellent conversational approach"
    elif score >= 60:
        feedback = "Good conversational approach"
    else:
        feedback = "Approach needs improvement"
    
    return {"score": score, "feedback": feedback, "details": checks}


if __name__ == "__main__":
    print("🚀 Starting LLM-Generated Mom Test Discovery Tests...")
    
    # Run the test
    asyncio.run(test_llm_generated_discovery())
    
    print("\n🎉 LLM-generated discovery tests completed!")
    print("\n💡 Benefits of LLM-Generated Approach:")
    print("   1. Dynamic and contextual questions")
    print("   2. Natural, conversational tone")
    print("   3. Adapts to any industry/role combination")
    print("   4. Maintains Mom Test principles")
    print("   5. No rigid templates - flexible and creative") 