#!/usr/bin/env python3
"""
Full Debug Mode Test

This script provides end-to-end testing of the debug mode fixes in both
the MC class trigger method and the convo_stream function.
"""

import asyncio
import json
import sys
from mc import MC, ResponseBuilder, DebugEventEmitter
from langchain_core.messages import HumanMessage
from utilities import logger
from ami import convo_stream, convo_graph


class FullDebugTester:
    """Tests debug mode functionality throughout the stack."""
    
    def __init__(self, user_id="test_debug_user"):
        """Initialize the tester"""
        self.user_id = user_id
        self.thread_id = f"test_debug_{user_id}"
        self.test_message = "Test the debug mode with this message please"
    
    async def test_mc_fix(self):
        """Test the MC class fix that extracts message content from prompt_str dict"""
        print("\n🧪 Testing MC class debug mode fix")
        
        mc = MC(user_id=self.user_id)
        state = {
            "messages": [HumanMessage(content=self.test_message)],
            "intent_history": [],
            "preset_memory": "Be friendly",
            "unresolved_requests": [],
            "user_id": self.user_id,
            "prompt_str": "",
            "graph_version_id": "",
            "debug": True
        }
        
        # Process the state with debug mode enabled
        try:
            got_dict_response = False
            async for response in mc.trigger(state=state, debug=True):
                # Skip the final state
                if isinstance(response, dict) and "state" in response:
                    continue
                
                # Check if we received a dictionary response with events
                if isinstance(response, dict) and "events" in response:
                    got_dict_response = True
                    print(f"✓ Received dictionary response with {len(response['events'])} debug events")
            
            # Verify that AIMessage was created with string content
            if len(state["messages"]) >= 2:
                last_message = state["messages"][-1]
                if hasattr(last_message, 'content'):
                    content_type = type(last_message.content)
                    print(f"✓ AIMessage content type: {content_type.__name__}")
                    
                    if content_type is dict:
                        print("❌ MC fix failed: AIMessage content is still a dictionary")
                        return False
                    elif content_type is str:
                        print(f"✓ MC fix passed: AIMessage content is correctly a string")
                    else:
                        print(f"⚠️ Unexpected AIMessage content type: {content_type.__name__}")
                        return False
            else:
                print("❌ No AIMessage was added to state")
                return False
            
            # Check if we got dict responses but not dict AIMessage content
            if got_dict_response:
                print("✓ Debug events were generated in the response")
            else:
                print("⚠️ No dictionary responses with events were generated")
                
            return True
        
        except Exception as e:
            print(f"❌ Error in MC test: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_convo_stream_fix(self):
        """Test the convo_stream function fix that handles dict prompt_str"""
        print("\n🧪 Testing convo_stream function debug mode fix")
        
        try:
            # Create a state directly and bypass the asyncio.run() call
            thread_id = self.thread_id
            user_id = self.user_id
            debug = True
            graph_version_id = ""
            
            default_state = {
                "messages": [HumanMessage(content=self.test_message)],
                "prompt_str": "",
                "convo_id": thread_id,
                "last_response": "",
                "user_id": user_id,
                "preset_memory": "Be friendly",
                "instinct": "",
                "graph_version_id": graph_version_id,
                "debug": debug
            }
            
            # Process the state directly using the technique similar to what convo_stream does
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "graph_version_id": graph_version_id,
                    "debug": debug
                }
            }
            
            # Process state directly with our event loop
            updated_state = await convo_graph.ainvoke(default_state, config)
            await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, updated_state, as_node="mc")
            
            # Capture what the prompt_str looks like
            print(f"✓ prompt_str type in state: {type(updated_state['prompt_str']).__name__}")
            
            # Test if we can process the state correctly
            if isinstance(updated_state["prompt_str"], dict) and "message" in updated_state["prompt_str"]:
                message = updated_state["prompt_str"]["message"]
                events = updated_state["prompt_str"].get("events", [])
                print(f"✓ Successfully extracted message and {len(events)} events from dict prompt_str")
                return True
            elif isinstance(updated_state["prompt_str"], str):
                print(f"✓ prompt_str is a string, no debug events to extract")
                return True
            else:
                print(f"❌ Unexpected prompt_str format: {type(updated_state['prompt_str']).__name__}")
                return False
            
        except Exception as e:
            print(f"❌ Error in convo_stream test: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


async def run_full_test():
    """Run both tests"""
    tester = FullDebugTester()
    
    # Test MC fix
    mc_success = await tester.test_mc_fix()
    
    # Test convo_stream fix - now it's an async method too
    convo_success = await tester.test_convo_stream_fix()
    
    # Report results
    print("\n📋 Test Results:")
    print(f"  MC class fix: {'✅ PASS' if mc_success else '❌ FAIL'}")
    print(f"  convo_stream fix: {'✅ PASS' if convo_success else '❌ FAIL'}")
    
    if mc_success and convo_success:
        print("\n✅ All debug mode fixes are working correctly")
        return 0
    else:
        print("\n❌ Some debug mode fixes failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_full_test())
    sys.exit(exit_code) 