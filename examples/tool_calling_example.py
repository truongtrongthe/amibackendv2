#!/usr/bin/env python3
"""
Example script demonstrating the tool calling implementation.
"""

import os
import sys
import asyncio
import json
import traceback
from typing import Dict, Any, List

# Add the parent directory to sys.path to import modules from the project
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mc_tools import MCWithTools
from langchain_core.messages import HumanMessage, AIMessage
from utilities import logger


async def print_chunk(chunk: Any):
    """Pretty print a response chunk"""
    if isinstance(chunk, str):
        print(f"Response: {chunk}", end="", flush=True)
    elif isinstance(chunk, dict):
        if chunk.get("type") == "analysis":
            if chunk.get("complete", False):
                print("\n=== ANALYSIS COMPLETE ===\n")
            else:
                content = chunk.get("content", "")
                if isinstance(content, dict):
                    content = content.get("content", "")
                print(f"\rAnalysis: {content[:50]}..." + " " * 10, end="", flush=True)
        elif chunk.get("type") == "knowledge":
            content = chunk.get("content", [])
            if isinstance(content, list):
                print(f"\rKnowledge: {len(content)} results", end="", flush=True)
            else:
                print(f"\rKnowledge: Processing...", end="", flush=True)
        elif chunk.get("type") == "next_actions":
            if chunk.get("complete", False):
                print("\n=== NEXT ACTIONS COMPLETE ===\n")
            else:
                content = chunk.get("content", "")
                if isinstance(content, dict):
                    content = content.get("content", "")
                print(f"\rNext actions: {content[:50]}..." + " " * 10, end="", flush=True)
        elif "state" in chunk:
            # Don't print state updates
            pass
        elif "error" in chunk:
            print(f"\nERROR: {chunk.get('error')}")
        else:
            print(f"\rOther: {json.dumps(chunk)[:100]}...", end="", flush=True)


async def simulate_conversation(mc: MCWithTools, messages: List[str], graph_version_id: str = ""):
    """
    Simulate a conversation with the MC using tool calling.
    
    Args:
        mc: MCWithTools instance
        messages: List of user messages
        graph_version_id: ID of the knowledge graph to use
    """
    # Initialize the conversation state
    state = mc.state.copy()
    
    for i, message in enumerate(messages):
        print(f"\n\n--- User Message {i+1}: {message} ---\n")
        
        # Add the user message to the state
        state["messages"].append(HumanMessage(content=message))
        
        # Process the message
        result_chunks = []
        response_text = ""
        
        try:
            async for chunk in mc.trigger(state=state, graph_version_id=graph_version_id):
                # Print the chunk for debugging
                await print_chunk(chunk)
                
                # Store the chunk for later use
                result_chunks.append(chunk)
                
                # If this is a response text chunk, accumulate it
                if isinstance(chunk, str):
                    response_text += chunk
            
            print("\n\n--- AI Response ---\n")
            print(response_text)
            
            # The state is updated by the trigger method, so we don't need to do anything else
        except Exception as e:
            print(f"\n\nERROR in conversation: {str(e)}")
            print(traceback.format_exc())
            print("\nContinuing to next message...\n")


async def main():
    """Main function to run the example"""
    try:
        # Create an MCWithTools instance
        mc = MCWithTools(user_id="example_user", convo_id="example_conversation")
        
        # Initialize the MC
        await mc.initialize()
        
        # Sample conversation
        messages = [
            "Hello, how are you today?",
            "What can you help me with?",
            "Tell me about artificial intelligence.",
            "How does tool calling work?",
            "Thank you for the information!",
        ]
        
        # Simulate the conversation
        await simulate_conversation(mc, messages)
    
    except Exception as e:
        print(f"ERROR in main: {str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main()) 