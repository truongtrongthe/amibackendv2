import asyncio
import brain_singleton
from utilities import logger
import time

async def test_brain_loading():
    print("\n--- Testing basic brain loading ---")
    
    # Get brain instance
    brain_coroutine = brain_singleton.get_brain()
    brain = await brain_coroutine if asyncio.iscoroutine(brain_coroutine) else brain_coroutine
    
    print(f"Brain loaded: {brain is not None}")
    
    if brain is not None:
        if hasattr(brain, 'faiss_index'):
            print(f"FAISS index has {brain.faiss_index.ntotal} vectors")
        else:
            print("Brain has no FAISS index")
    
    return brain is not None

async def test_vector_loading():
    print("\n--- Testing vector loading ---")
    
    # Use a test graph version ID
    test_graph_version = "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"
    print(f"Loading vectors for graph version: {test_graph_version}")
    
    try:
        # Try to load vectors
        start_time = time.time()
        success = await brain_singleton.load_brain_vectors(test_graph_version, force_delete=True)
        end_time = time.time()
        
        print(f"Vector loading {'succeeded' if success else 'failed'}")
        print(f"Loading took {end_time - start_time:.2f} seconds")
        
        # Get the brain instance to check the loaded vectors
        brain_coroutine = brain_singleton.get_brain()
        brain = await brain_coroutine if asyncio.iscoroutine(brain_coroutine) else brain_coroutine
        
        if brain and hasattr(brain, 'faiss_index'):
            print(f"FAISS index now has {brain.faiss_index.ntotal} vectors")
        
        return success
    except Exception as e:
        import traceback
        print(f"Error during vector loading: {e}")
        print(traceback.format_exc())
        return False

async def main():
    try:
        success1 = await test_brain_loading()
        success2 = await test_vector_loading()
        
        print("\n--- Test Results ---")
        print(f"Brain loading: {'SUCCESS' if success1 else 'FAILURE'}")
        print(f"Vector loading: {'SUCCESS' if success2 else 'FAILURE'}")
        print(f"Overall test: {'SUCCESS' if (success1 and success2) else 'FAILURE'}")
    except Exception as e:
        import traceback
        print(f"Error during test: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 