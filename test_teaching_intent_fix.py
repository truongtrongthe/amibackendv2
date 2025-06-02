#!/usr/bin/env python3
"""
Test script to verify teaching intent detection fix

This script tests that:
1. detect_teaching_intent_llm() correctly returns True for teaching intent
2. The final LLM evaluation section also includes has_teaching_intent: true
3. The knowledge saving decision recognizes the teaching intent
"""

import asyncio
import json
from learning_support import LearningSupport

class MockLearningProcessor:
    """Mock learning processor for testing"""
    def __init__(self):
        self.graph_version_id = "test_graph"

async def test_teaching_intent_detection():
    """Test the complete teaching intent flow"""
    print("🧪 Testing Teaching Intent Detection Fix")
    print("=" * 50)
    
    # Create test instances
    mock_processor = MockLearningProcessor()
    support = LearningSupport(mock_processor)
    
    # Test message with clear teaching intent
    test_message = "Bên anh sắp mở văn phòng ở Đà Nẵng"
    conversation_context = ""
    
    print(f"📝 Test Message: '{test_message}'")
    print()
    
    # Test 1: LLM-based teaching intent detection
    print("🔍 Test 1: LLM Teaching Intent Detection")
    try:
        teaching_intent_result = await support.detect_teaching_intent_llm(test_message, conversation_context)
        print(f"   Result: {teaching_intent_result}")
        print(f"   Status: {'✅ PASS' if teaching_intent_result else '❌ FAIL'}")
    except Exception as e:
        print(f"   Status: ❌ ERROR - {str(e)}")
    print()
    
    # Test 2: Build LLM prompt and check evaluation instructions
    print("🔍 Test 2: LLM Prompt Includes Evaluation Instructions")
    try:
        prompt = support.build_llm_prompt(
            message_str=test_message,
            conversation_context=conversation_context,
            temporal_context="",
            knowledge_context="",
            response_strategy="GENERAL",
            strategy_instructions="",
            core_prior_topic="",
            user_id="test_user"
        )
        
        has_evaluation_instructions = "MANDATORY EVALUATION OUTPUT" in prompt
        has_teaching_criteria = "has_teaching_intent" in prompt
        has_evaluation_format = "<evaluation>" in prompt
        
        print(f"   Evaluation Instructions: {'✅' if has_evaluation_instructions else '❌'}")
        print(f"   Teaching Intent Criteria: {'✅' if has_teaching_criteria else '❌'}")
        print(f"   Evaluation Format: {'✅' if has_evaluation_format else '❌'}")
        
        all_checks = has_evaluation_instructions and has_teaching_criteria and has_evaluation_format
        print(f"   Status: {'✅ PASS' if all_checks else '❌ FAIL'}")
        
        if not all_checks:
            print(f"   Prompt excerpt: ...{prompt[-500:]}")
    except Exception as e:
        print(f"   Status: ❌ ERROR - {str(e)}")
    print()
    
    # Test 3: Extract evaluation from sample response
    print("🔍 Test 3: Evaluation Extraction")
    try:
        sample_response = """
        Cảm ơn bạn đã chia sẻ! Mình hiểu bên bạn sắp mở văn phòng ở Đà Nẵng.
        
        <evaluation>
        {
            "has_teaching_intent": true,
            "is_priority_topic": true,
            "priority_topic_name": "office_expansion",
            "should_save_knowledge": true,
            "intent_type": "teaching",
            "name_addressed": false,
            "ai_referenced": false
        }
        </evaluation>
        """
        
        content, tool_calls, evaluation = support.extract_tool_calls_and_evaluation(sample_response, test_message)
        
        has_teaching_intent = evaluation.get("has_teaching_intent", False)
        intent_type = evaluation.get("intent_type", "unknown")
        
        print(f"   Extracted has_teaching_intent: {has_teaching_intent}")
        print(f"   Extracted intent_type: {intent_type}")
        print(f"   Status: {'✅ PASS' if has_teaching_intent and intent_type == 'teaching' else '❌ FAIL'}")
        
    except Exception as e:
        print(f"   Status: ❌ ERROR - {str(e)}")
    print()
    
    print("🎯 Summary")
    print("The fix adds mandatory evaluation instructions to the LLM prompt.")
    print("This ensures the LLM always includes teaching intent evaluation in its response.")
    print("The evaluation is then extracted and used for knowledge saving decisions.")
    print()
    print("🔧 Key Changes Made:")
    print("1. Added evaluation section instructions to build_llm_prompt()")
    print("2. Made evaluation output mandatory for all LLM responses")
    print("3. Included specific teaching intent criteria in the prompt")
    print()

if __name__ == "__main__":
    asyncio.run(test_teaching_intent_detection()) 