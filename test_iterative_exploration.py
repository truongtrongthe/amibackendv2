#!/usr/bin/env python3
"""
Test script for iterative knowledge exploration functionality.
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tool_learning import LearningProcessor

async def test_iterative_exploration():
    """Test the iterative knowledge exploration functionality."""
    print("🧪 Testing Iterative Knowledge Exploration")
    print("=" * 50)
    
    # Initialize the learning processor
    processor = LearningProcessor()
    await processor.initialize()
    
    # Test cases with different similarity expectations
    test_cases = [
        {
            "message": "Tôi muốn học về phân nhóm khách hàng",
            "expected_rounds": 2,
            "description": "Customer segmentation query (should find relevant knowledge)"
        },
        {
            "message": "Làm thế nào để cải thiện trải nghiệm khách hàng?",
            "expected_rounds": 3,
            "description": "Customer experience improvement (may need multiple rounds)"
        },
        {
            "message": "Quantum computing applications in blockchain",
            "expected_rounds": 3,
            "description": "Technical topic (likely needs full exploration)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['description']}")
        print(f"Query: '{test_case['message']}'")
        print("-" * 40)
        
        try:
            # Run iterative exploration
            result = await processor._iterative_knowledge_exploration(
                message=test_case["message"],
                conversation_context="",
                user_id="test_user",
                thread_id=None,
                max_rounds=3
            )
            
            # Extract results
            similarity = result.get("similarity", 0.0)
            rounds_completed = result.get("exploration_metadata", {}).get("rounds_completed", 0)
            total_items = result.get("exploration_metadata", {}).get("total_items_found", 0)
            final_similarity = result.get("exploration_metadata", {}).get("final_similarity", 0.0)
            rounds_data = result.get("exploration_metadata", {}).get("rounds_data", [])
            
            # Display results
            print(f"✅ Exploration completed!")
            print(f"   Final similarity: {similarity:.3f}")
            print(f"   Rounds completed: {rounds_completed}")
            print(f"   Total items found: {total_items}")
            print(f"   Strategy: {result.get('exploration_metadata', {}).get('exploration_strategy', 'unknown')}")
            
            # Show round-by-round breakdown
            if rounds_data:
                print(f"   Round breakdown:")
                for round_info in rounds_data:
                    print(f"     Round {round_info['round']} ({round_info['strategy']}): "
                          f"{round_info['similarity']:.3f} similarity, "
                          f"{round_info['result_count']} results")
            
            # Evaluate success
            if similarity >= 0.70:
                print(f"   🎯 HIGH SIMILARITY ACHIEVED! ({similarity:.3f} >= 0.70)")
            elif similarity >= 0.35:
                print(f"   ⚠️  Medium similarity ({similarity:.3f})")
            else:
                print(f"   ❌ Low similarity ({similarity:.3f})")
                
        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
    
    print(f"\n🏁 Testing completed!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_iterative_exploration()) 