import asyncio
import brain_singleton
from utilities import logger
import time

async def test_brain_singleton():
    """
    Comprehensive test of the brain singleton functionality
    """
    print("\n=== Testing Brain Singleton ===")
    
    # 1. Get brain instance (async method)
    print("\n1. Testing get_brain (async):")
    brain_coroutine = brain_singleton.get_brain()
    brain = await brain_coroutine if asyncio.iscoroutine(brain_coroutine) else brain_coroutine
    print(f"  - Brain loaded: {brain is not None}")
    print(f"  - FAISS index: {'Present' if hasattr(brain, 'faiss_index') else 'Missing'}")
    if hasattr(brain, 'faiss_index'):
        print(f"  - Vector count: {brain.faiss_index.ntotal}")
    
    # 2. Test sync version of get_brain
    print("\n2. Testing get_brain_sync:")
    brain_sync = brain_singleton.get_brain_sync()
    print(f"  - Brain loaded: {brain_sync is not None}")
    print(f"  - FAISS index: {'Present' if hasattr(brain_sync, 'faiss_index') else 'Missing'}")
    if hasattr(brain_sync, 'faiss_index'):
        print(f"  - Vector count: {brain_sync.faiss_index.ntotal}")
    
    # 3. Test vector loading
    test_graph_version = "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"
    print(f"\n3. Testing load_brain_vectors with version {test_graph_version}:")
    start_time = time.time()
    load_success = await brain_singleton.load_brain_vectors(test_graph_version, force_delete=True)
    end_time = time.time()
    print(f"  - Load success: {load_success}")
    print(f"  - Loading time: {end_time - start_time:.2f} seconds")
    
    # 4. Test brain reset function
    print("\n4. Testing reset_brain_and_load_version:")
    reset_start = time.time()
    await brain_singleton.reset_brain_and_load_version(test_graph_version)
    reset_end = time.time()
    print(f"  - Reset time: {reset_end - reset_start:.2f} seconds")
    
    # Get updated brain state
    brain_coroutine = brain_singleton.get_brain()
    brain = await brain_coroutine if asyncio.iscoroutine(brain_coroutine) else brain_coroutine
    if hasattr(brain, 'faiss_index'):
        print(f"  - Vector count after reset: {brain.faiss_index.ntotal}")
    
    # 5. Test flick_out function
    print("\n5. Testing flick_out with a query:")
    test_query = "Phân nhóm khách hàng"
    flick_result = await brain_singleton.flick_out(test_query, test_graph_version)
    print(f"  - Query: '{test_query}'")
    print(f"  - Result count: {len(flick_result.get('results', []))}")
    if 'results' in flick_result and len(flick_result['results']) > 0:
        top_result = flick_result['results'][0]
        print(f"  - Top result similarity: {top_result.get('similarity', 0):.4f}")
        # Print first few words of top result metadata if available
        if 'metadata' in top_result and 'raw' in top_result['metadata']:
            raw_text = top_result['metadata']['raw']
            preview = raw_text[:100] + "..." if len(raw_text) > 100 else raw_text
            print(f"  - Content preview: {preview}")
    
    # 6. Test activate_brain_with_version
    print("\n6. Testing activate_brain_with_version:")
    activate_result = await brain_singleton.activate_brain_with_version(test_graph_version)
    print(f"  - Activation success: {activate_result.get('success', False)}")
    print(f"  - Graph version: {activate_result.get('graph_version_id', None)}")
    print(f"  - Vector count: {activate_result.get('vector_count', 0)}")
    
    print("\n=== Test Summary ===")
    print(f"All tests {'PASSED' if load_success and activate_result.get('success', False) else 'FAILED'}")
    
    return load_success and activate_result.get('success', False)

if __name__ == "__main__":
    asyncio.run(test_brain_singleton()) 