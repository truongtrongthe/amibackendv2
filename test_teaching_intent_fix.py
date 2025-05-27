#!/usr/bin/env python3
"""
Test script to verify the teaching intent classification fix.
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tool_learning import LearningProcessor

async def test_teaching_intent_classification():
    """Test that teaching intent is correctly classified and not overridden by similarity gating."""
    print("🧪 Testing Teaching Intent Classification Fix")
    print("=" * 60)
    
    # Initialize the learning processor
    processor = LearningProcessor()
    await processor.initialize()
    
    # Test cases: requests vs teaching
    test_cases = [
        {
            "message": "Anh khó nghĩ quá làm thế nào để lấy được số của khách hàng đây",
            "expected_intent": False,  # This is a REQUEST, not teaching
            "description": "Request for help (should NOT be teaching intent)"
        },
        {
            "message": "Em muốn chia sẻ cách thu thập số điện thoại hiệu quả: Đầu tiên phải giải thích mục đích rõ ràng...",
            "expected_intent": True,   # This is TEACHING
            "description": "Teaching content (should BE teaching intent)"
        },
        {
            "message": "Làm sao để tôi có thể cải thiện kỹ năng bán hàng?",
            "expected_intent": False,  # This is a QUESTION, not teaching
            "description": "Question/request (should NOT be teaching intent)"
        },
        {
            "message": "Tôi muốn hướng dẫn các bạn về kỹ thuật chốt đơn: Bước 1 là xây dựng niềm tin...",
            "expected_intent": True,   # This is TEACHING
            "description": "Teaching/instruction (should BE teaching intent)"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['description']}")
        print(f"Message: '{test_case['message'][:50]}...'")
        
        try:
            # Process the message
            result = await processor.process_incoming_message(
                message=test_case['message'],
                conversation_context="",
                user_id="test_user",
                thread_id=f"test_thread_{i}"
            )
            
            # Extract the teaching intent classification
            actual_intent = result.get("metadata", {}).get("has_teaching_intent", False)
            expected_intent = test_case["expected_intent"]
            
            # Check if classification is correct
            is_correct = actual_intent == expected_intent
            status = "✅ PASS" if is_correct else "❌ FAIL"
            
            print(f"Expected teaching intent: {expected_intent}")
            print(f"Actual teaching intent: {actual_intent}")
            print(f"Result: {status}")
            
            # Additional metadata
            similarity = result.get("metadata", {}).get("similarity_score", 0.0)
            response_strategy = result.get("metadata", {}).get("response_strategy", "unknown")
            should_save = result.get("metadata", {}).get("should_save_knowledge", False)
            
            print(f"Similarity score: {similarity:.3f}")
            print(f"Response strategy: {response_strategy}")
            print(f"Should save knowledge: {should_save}")
            
            results.append({
                "test_case": i,
                "description": test_case["description"],
                "expected": expected_intent,
                "actual": actual_intent,
                "correct": is_correct,
                "similarity": similarity,
                "strategy": response_strategy
            })
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                "test_case": i,
                "description": test_case["description"],
                "expected": expected_intent,
                "actual": None,
                "correct": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Teaching intent classification is working correctly.")
    else:
        print("⚠️  Some tests failed. The fix may need adjustment.")
        
        # Show failed tests
        failed_tests = [r for r in results if not r.get("correct", False)]
        for test in failed_tests:
            print(f"  - Test {test['test_case']}: {test['description']}")
            print(f"    Expected: {test['expected']}, Got: {test['actual']}")
    
    await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(test_teaching_intent_classification()) 