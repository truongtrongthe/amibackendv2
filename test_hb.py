#!/usr/bin/env python3
"""
test_hb.py - Focused test script for query_graph_knowledge in hotbrain.py
"""

import os
import sys
import asyncio
import time
import logging
from typing import Dict, List, Any
import statistics

# Configure logging to see detailed info during tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the hotbrain module
try:
    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import hotbrain
except ImportError:
    raise ImportError("Could not import hotbrain.py. Make sure it's in the same directory.")

# Test constants
TEST_VERSION_ID = "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"  # The specific version ID to test

async def test_query_graph_knowledge():
    """Test the query_graph_knowledge function with 20 rounds of diverse queries"""
    logger.info("=== Testing Query Graph Knowledge with 20 Diverse Queries ===")
    logger.info(f"Using version ID: {TEST_VERSION_ID}")
    
    # Test queries covering different topics and languages
    # Include English, Vietnamese, and mixed queries to thoroughly test the system
    test_queries = [
        # English queries
        "I'm trying to understand person profile!",
        "How can I segment customers effectively?",
        "What are the different customer types?",
        "How to identify high-value customers?",
        "What metrics should I track for customer satisfaction?",
        
        # Vietnamese queries
        "Làm thế nào để phân nhóm khách hàng",  # How to segment customers
        "Khách hàng ở nhóm nào",                # Which group customers belong to
        "Có mấy nhóm khách hàng",               # How many customer groups
        "Tôi cần phân tích chân dung khách hàng", # I need to analyze customer profiles
        "Làm sao để tăng tỷ lệ giữ chân khách hàng", # How to increase customer retention
        
        # Complex or specialized queries
        "What's the relationship between customer lifetime value and acquisition cost?",
        "How to implement permission levels for different team members?",
        "What security settings should I configure for my team?",
        "How to integrate with existing CRM systems?",
        "What API endpoints are available for customer data?",
        
        # Additional 5 queries (mix of English, Vietnamese and technical)
        "How do I analyze customer purchasing patterns?",
        "Can you recommend loyalty program strategies?",
        "Làm thế nào để dự đoán hành vi khách hàng?", # How to predict customer behavior
        "Best practices for customer data governance?",
        "Các phương pháp phân tích dữ liệu khách hàng?" # Methods for customer data analysis
    ]
    
    # Check version status first
    try:
        import os
        from supabase import create_client
        
        # Initialize Supabase client to check version
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        version_response = supabase.table("brain_graph_version")\
            .select("brain_ids", "status")\
            .eq("id", TEST_VERSION_ID)\
            .execute()
            
        if not version_response.data:
            logger.warning(f"Version {TEST_VERSION_ID} not found. Test will continue but may not return results.")
        elif version_response.data[0]["status"] != "published":
            logger.warning(f"Version {TEST_VERSION_ID} is not published (status: {version_response.data[0]['status']}). Test will continue but may not return results.")
    except Exception as e:
        logger.error(f"Error checking version status: {e}")
    
    # Initialize performance tracking
    performance_stats = {
        "first_queries": [],
        "second_queries": [],
        "improvements": [],
        "result_counts": [],
        "top_scores": [],
        "zero_results_count": 0,
        "error_count": 0,
        "vietnamese_scores": [],
        "english_scores": [],
        "complex_scores": []
    }
    
    # Run each query and measure performance
    for i, query in enumerate(test_queries):
        # Determine query type for stats
        is_vietnamese = any(char in query for char in "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
        is_complex = i >= 10  # Last 10 queries are complex
        
        query_type = "Vietnamese" if is_vietnamese else ("Complex" if is_complex else "English")
        logger.info(f"\nQuery {i+1}: '{query}' ({query_type})")
        
        # First query attempt - typically a cache miss
        try:
            start_time = time.time()
            results = await hotbrain.query_graph_knowledge(
                version_id=TEST_VERSION_ID,
                query=query,
                top_k=5
            )
            first_query_time = time.time() - start_time
            performance_stats["first_queries"].append(first_query_time)
            
            # Process and display results
            if results:
                result_count = len(results)
                performance_stats["result_counts"].append(result_count)
                logger.info(f"Found {result_count} results")
                
                # Get top score
                top_score = results[0]["score"] if results else 0
                performance_stats["top_scores"].append(top_score)
                
                # Add to appropriate language category
                if is_vietnamese:
                    performance_stats["vietnamese_scores"].append(top_score)
                elif is_complex:
                    performance_stats["complex_scores"].append(top_score)
                else:
                    performance_stats["english_scores"].append(top_score)
                
                # Show top 3 results
                for j, result in enumerate(results[:3]):
                    # Safely truncate content which might be in various languages
                    content = result.get('raw', '')
                    if content:
                        # Safe truncation for multi-byte characters
                        try:
                            content = content[:50] + '...'
                        except:
                            # If there's an encoding issue, just show the score
                            content = '<content truncated>'
                    logger.info(f"  Result {j+1}: Score={result['score']:.4f}, Content: {content}")
            else:
                logger.info("No results found")
                performance_stats["result_counts"].append(0)
                performance_stats["zero_results_count"] += 1
                performance_stats["top_scores"].append(0)
                
        except Exception as e:
            logger.error(f"Error in first query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            first_query_time = 0
            performance_stats["error_count"] += 1
            performance_stats["result_counts"].append(0)
            performance_stats["top_scores"].append(0)
        
        # Second query attempt - should use cached results
        try:
            await asyncio.sleep(1)  # Small delay between queries
            start_time = time.time()
            _ = await hotbrain.query_graph_knowledge(
                version_id=TEST_VERSION_ID,
                query=query,
                top_k=5
            )
            second_query_time = time.time() - start_time
            performance_stats["second_queries"].append(second_query_time)
            
            # Calculate improvement if first query was successful
            if first_query_time > 0:
                improvement = ((first_query_time - second_query_time) / first_query_time) * 100
                performance_stats["improvements"].append(improvement)
                logger.info(f"First query time: {first_query_time:.4f}s")
                logger.info(f"Second query time: {second_query_time:.4f}s")
                logger.info(f"Cache improvement: {improvement:.1f}%")
            
        except Exception as e:
            logger.error(f"Error in second query: {e}")
            performance_stats["error_count"] += 1
        
        # Wait between queries to avoid rate limits
        if i < len(test_queries) - 1:
            await asyncio.sleep(1)
    
    # Calculate and report statistics
    logger.info("\n=== Query Performance Statistics ===")
    
    # Query times
    valid_first_queries = [t for t in performance_stats["first_queries"] if t > 0]
    if valid_first_queries:
        avg_first_time = statistics.mean(valid_first_queries)
        logger.info(f"Average first query time: {avg_first_time:.4f}s")
    
    valid_second_queries = [t for t in performance_stats["second_queries"] if t > 0]
    if valid_second_queries:
        avg_second_time = statistics.mean(valid_second_queries)
        logger.info(f"Average second query time: {avg_second_time:.4f}s")
    
    # Cache improvements
    valid_improvements = [i for i in performance_stats["improvements"] if i is not None]
    if valid_improvements:
        avg_improvement = statistics.mean(valid_improvements)
        logger.info(f"Average cache improvement: {avg_improvement:.1f}%")
        logger.info(f"Minimum cache improvement: {min(valid_improvements):.1f}%")
        logger.info(f"Maximum cache improvement: {max(valid_improvements):.1f}%")
    
    # Result statistics
    if performance_stats["result_counts"]:
        avg_results = statistics.mean(performance_stats["result_counts"])
        logger.info(f"Average results per query: {avg_results:.2f}")
        logger.info(f"Queries with zero results: {performance_stats['zero_results_count']} of {len(test_queries)}")
    
    # Score statistics
    valid_scores = [s for s in performance_stats["top_scores"] if s > 0]
    if valid_scores:
        avg_score = statistics.mean(valid_scores)
        logger.info(f"Average top result score: {avg_score:.4f}")
    
    # Language-specific scores
    if performance_stats["english_scores"]:
        avg_english = statistics.mean(performance_stats["english_scores"])
        logger.info(f"Average English query score: {avg_english:.4f}")
    
    if performance_stats["vietnamese_scores"]:
        avg_vietnamese = statistics.mean(performance_stats["vietnamese_scores"])
        logger.info(f"Average Vietnamese query score: {avg_vietnamese:.4f}")
    
    if performance_stats["complex_scores"]:
        avg_complex = statistics.mean(performance_stats["complex_scores"])
        logger.info(f"Average complex query score: {avg_complex:.4f}")
    
    # Error count
    logger.info(f"Total errors encountered: {performance_stats['error_count']}")
    
    logger.info("\nQuery graph knowledge test complete!")
    return True

async def run_tests():
    """Run the query graph knowledge test"""
    logger.info("Starting focused test for hotbrain.py query_graph_knowledge...")
    
    try:
        # Run only the query_graph_knowledge test
        await test_query_graph_knowledge()
        logger.info("\n✅ Testing completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Test complete.")

if __name__ == "__main__":
    """Run tests directly when script is executed"""
    asyncio.run(run_tests()) 