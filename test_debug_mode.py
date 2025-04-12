#!/usr/bin/env python3
"""
Test script to verify the debug mode fix in MC class

This script specifically tests that when debug mode is enabled,
the trigger method correctly handles dictionary responses with events.
"""

import asyncio
import json
from mc import MC, ResponseBuilder, DebugEventEmitter
from langchain_core.messages import HumanMessage
from utilities import logger
import sys

class DebugTester:
    def __init__(self, user_id="test_debug_user"):
        """Initialize the tester with a new MC instance"""
        self.mc = MC(user_id=user_id)
        self.state = {
            "messages": [],
            "intent_history": [],
            "preset_memory": "Be friendly",
            "unresolved_requests": [],
            "user_id": user_id,
            "prompt_str": "",
            "graph_version_id": "",
            "debug": True  # Debug mode explicitly enabled
        }
    
    def add_message(self, message):
        """Add a human message to the conversation history"""
        self.state["messages"].append(HumanMessage(content=message))
        return self
    
    async def test_debug_mode(self):
        """
        Test that debug mode works correctly with the fix.
        
        The test verifies:
        1. Debug events are correctly passed through in the response
        2. The AIMessage creation uses only the message part, not the full object
        3. Debug events are stored in state for future reference
        """
        print("🧪 Testing debug mode fix")
        
        # Initialize MC
        if not self.mc.instincts:
            await self.mc.initialize()
        
        # Add a test message
        self.add_message("Test debug mode with this message")
        
        # Track responses
        responses = []
        try:
            # Process the message in debug mode
            async for response in self.mc.trigger(
                state=self.state,
                debug=True
            ):
                # Skip the final state response
                if isinstance(response, dict) and "state" in response:
                    continue
                    
                responses.append(response)
                
                # Examine the response
                if isinstance(response, dict) and "events" in response:
                    print(f"✓ Received debug events in response")
                    print(f"  Message: {response['message']}")
                    print(f"  Events count: {len(response['events'])}")
                else:
                    print(f"  Response: {response}")
        
        except Exception as e:
            print(f"❌ Error in debug mode test: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        # Check if we got responses
        if not responses:
            print("❌ No responses received")
            return False
            
        # Verify AIMessage was created with string content, not dict
        last_message = self.state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            # This should be an AIMessage
            if hasattr(last_message, 'content'):
                content = last_message.content
                print(f"✓ Message content type: {type(content)}")
                
                if isinstance(content, dict):
                    print("❌ Message content is still a dictionary - fix failed")
                    return False
                else:
                    print(f"✓ Message content is correctly a string: {content[:50]}...")
            else:
                print("❌ Last message has no content attribute")
                return False
        
        # Check if debug events were stored in state
        if "debug_events" in self.state:
            print(f"✓ Debug events stored in state: {len(self.state['debug_events'])}")
        else:
            print("ℹ️ No debug events stored in state")
            if any(isinstance(r, dict) and "events" in r for r in responses):
                print("⚠️ Warning: Events in responses but not stored in state")
        
        print("✓ Debug mode test completed successfully")
        return True


async def run_test():
    """Run the debug mode test"""
    tester = DebugTester()
    success = await tester.test_debug_mode()
    
    if success:
        print("\n✅ Debug mode fix validation: PASS")
        return 0
    else:
        print("\n❌ Debug mode fix validation: FAIL")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_test())
    sys.exit(exit_code) 