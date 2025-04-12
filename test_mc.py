#!/usr/bin/env python3
"""
Test script for MC class _handle_request method

This script helps test how the _handle_request method processes different human messages.
"""

import argparse
import json
import asyncio
from typing import Dict, List
import sys

# Import the MC class and ResponseBuilder
from mc import MC, ResponseBuilder
from langchain_core.messages import HumanMessage, AIMessage
from utilities import logger

# Test class to examine _handle_request function
class MCTester:
    def __init__(self, user_id="test_user", convo_id="test_convo", graph_version_id=""):
        """Initialize the tester with a new MC instance"""
        self.mc = MC(user_id=user_id, convo_id=convo_id)
        self.graph_version_id = graph_version_id
        self.state = {
            "messages": [],
            "intent_history": [],
            "preset_memory": "Be friendly",
            "unresolved_requests": [],
            "convo_id": convo_id,
            "user_id": user_id,
            "prompt_str": "",
            "graph_version_id": graph_version_id,
        }
    
    def add_message(self, message, is_human=True):
        """Add a message to the conversation history"""
        if is_human:
            self.state["messages"].append(HumanMessage(content=message))
        else:
            self.state["messages"].append(AIMessage(content=message))
        return self
    
    def build_context(self):
        """Build context string from messages"""
        return "\n".join(
            f"User: {msg.content}" if isinstance(msg, HumanMessage) 
            else f"AI: {msg.content}" 
            for msg in self.state["messages"][-100:]
        )
    
    async def test_handle_request(self, message):
        """Test the _handle_request method with a specific message"""
        print(f"\n🔍 Testing _handle_request with message: '{message}'")
        
        # Initialize MC if needed
        if not self.mc.instincts:
            await self.mc.initialize()
        
        # Set up the builder and context
        builder = ResponseBuilder()
        context = self.build_context()
        
        # Collect responses
        responses = []
        try:
            async for response in self.mc._handle_request(
                message=message,
                user_id=self.mc.user_id,
                context=context,
                builder=builder,
                state=self.state,
                graph_version_id=self.graph_version_id
            ):
                responses.append(response)
                print(f"Response chunk: {response}")
        except Exception as e:
            print(f"Error during test: {str(e)}")
        
        return responses

async def run_tests(args):
    """Run tests with the provided arguments"""
    tester = MCTester(
        user_id=args.user_id,
        convo_id=args.convo_id,
        graph_version_id=args.graph_version_id
    )
    
    # Add conversation history if provided
    if args.history:
        for i, msg in enumerate(args.history):
            tester.add_message(msg, is_human=(i % 2 == 0))
    
    # Process the message
    results = await tester.test_handle_request(args.message)
    
    # Print final results
    print("\n==== TEST RESULTS ====")
    print(f"Message: '{args.message}'")
    print(f"Response: {''.join(results)}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "message": args.message,
                "response": results,
                "history": args.history
            }, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test MC._handle_request method')
    parser.add_argument('message', help='Message to test with the _handle_request method')
    parser.add_argument('--user-id', help='User ID to use', default='test_user')
    parser.add_argument('--convo-id', help='Conversation ID to use', default='test_convo')
    parser.add_argument('--graph-version-id', help='Graph Version ID to use', default='')
    parser.add_argument('--history', help='Conversation history messages', nargs='*', default=[])
    parser.add_argument('--output', help='Output file to save results', default='')
    
    args = parser.parse_args()
    asyncio.run(run_tests(args)) 