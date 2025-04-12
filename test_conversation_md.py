#!/usr/bin/env python3
"""
Test script for simulating a conversation with the AI system

This script sends a series of human messages to the /havefun endpoint and records the responses.
It also generates a markdown file with the full conversation for better readability.
"""

import argparse
import json
import asyncio
import aiohttp
import sys
import re
from typing import List, Dict, Any
from datetime import datetime

# The messages to test
DEFAULT_MESSAGES = [
    "Ligit?",
    "Is that safe? Tablet?",
    "I like more than 8cm, is it sure that you will grow taller even your age more than 30?",
    "Good",
    "Too costly? No discount?",
    "And not today maybe some other mos..and the way of paying when I order always COD..",
    "Maybe next month. I'm doubt with the product",
    "I saw at the advertisement at hito now the people that using the other growth supplement before",
    "I'm hesitant because I will expense a big amount of money for the product that I'm not sure if this is true",
    "I don't know if your telling the truth 😞 about that product",
    "I will think first about it"
]

async def process_sse_response(response):
    """
    Process Server-Sent Events response
    
    Args:
        response: aiohttp response object
    
    Returns:
        List of parsed messages
    """
    messages = []
    buffer = ""
    
    async for line in response.content:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:].strip()
            try:
                parsed = json.loads(data)
                if 'message' in parsed:
                    messages.append(parsed['message'])
                    print(f"  > {parsed['message']}")
            except json.JSONDecodeError:
                print(f"Failed to parse: {data}")
    
    return messages

async def test_conversation(base_url: str, messages: List[str], user_id: str = "test_user", 
                          convo_id: str = None, graph_version_id: str = "", delay: int = 1):
    """
    Run a conversation test by sending messages to the /havefun endpoint
    
    Args:
        base_url: Base URL of the API
        messages: List of messages to send
        user_id: User ID for the conversation
        convo_id: Conversation ID (will be generated if None)
        graph_version_id: Graph version ID
        delay: Delay between messages in seconds
    """
    # Generate a conversation ID if none provided
    if not convo_id:
        convo_id = f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    print(f"Starting conversation test with user_id={user_id}, convo_id={convo_id}")
    
    # Keep track of the conversation history
    conversation = []
    
    # Create headers
    headers = {
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        for i, message in enumerate(messages):
            print(f"\n[{i+1}/{len(messages)}] Sending: '{message}'")
            
            # Prepare request payload
            payload = {
                "user_input": message,
                "user_id": user_id,
                "thread_id": convo_id,
                "graph_version_id": graph_version_id
            }
            
            # Send request to the havefun endpoint
            try:
                async with session.post(
                    f"{base_url}/havefun", 
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        print(f"Error: Received status code {response.status}")
                        try:
                            error_text = await response.text()
                            print(f"Error response: {error_text}")
                        except:
                            print("Could not read error response")
                        continue
                    
                    # Process SSE response
                    ai_messages = await process_sse_response(response)
                    ai_response = " ".join(ai_messages)
                    
                    # Add to conversation history
                    conversation.append({
                        "human": message,
                        "ai": ai_response
                    })
            except Exception as e:
                print(f"Error during request: {str(e)}")
            
            # Wait before sending the next message
            if i < len(messages) - 1:
                await asyncio.sleep(delay)
    
    return conversation

def save_markdown(conversation, filename, metadata=None):
    """
    Save the conversation to a markdown file
    
    Args:
        conversation: List of message exchanges
        filename: Output filename
        metadata: Optional metadata to include
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("# AI Conversation Test Results\n\n")
        
        # Write metadata if provided
        if metadata:
            f.write("## Test Metadata\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            for key, value in metadata.items():
                f.write(f"| {key} | {value} |\n")
            f.write("\n")
        
        # Write conversation
        f.write("## Conversation\n\n")
        for i, exchange in enumerate(conversation):
            f.write(f"### Exchange {i+1}\n\n")
            f.write(f"**Human:** {exchange['human']}\n\n")
            f.write(f"**AI:** {exchange['ai']}\n\n")
            f.write("---\n\n")
        
        # Write summary
        f.write(f"## Summary\n\n")
        f.write(f"Total exchanges: {len(conversation)}\n")
        f.write(f"Test completed at: {datetime.now().isoformat()}\n")

async def run_tests(args):
    """Run the tests with the provided arguments"""
    # Use provided messages or defaults
    messages = args.messages if args.messages else DEFAULT_MESSAGES
    
    # Run the conversation test
    conversation = await test_conversation(
        base_url=args.url,
        messages=messages,
        user_id=args.user_id,
        convo_id=args.convo_id,
        graph_version_id=args.graph_version_id,
        delay=args.delay
    )
    
    # Print summary
    print("\n==== CONVERSATION SUMMARY ====")
    print(f"Messages exchanged: {len(conversation)}")
    
    # Prepare metadata
    metadata = {
        "User ID": args.user_id,
        "Conversation ID": args.convo_id or f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "Graph Version ID": args.graph_version_id,
        "Timestamp": datetime.now().isoformat(),
        "API URL": args.url
    }
    
    # Save results to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "conversation": conversation,
                "metadata": metadata
            }, f, indent=2)
        print(f"Results saved to {args.output}")
    
    # Save results to markdown
    md_file = args.markdown_output or "conversation_results.md"
    save_markdown(conversation, md_file, metadata)
    print(f"Markdown saved to {md_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test conversation with the AI system')
    parser.add_argument('--url', help='Base URL of the API', default='http://localhost:5001')
    parser.add_argument('--user-id', help='User ID for the conversation', default='test_user')
    parser.add_argument('--convo-id', help='Conversation ID (will be generated if not provided)')
    parser.add_argument('--graph-version-id', help='Graph version ID', default='')
    parser.add_argument('--delay', help='Delay between messages in seconds', type=int, default=3)
    parser.add_argument('--messages', help='Messages to send (will use defaults if not provided)', nargs='*')
    parser.add_argument('--output', help='Output file to save results', default='conversation_results.json')
    parser.add_argument('--markdown-output', help='Markdown output file', default='conversation_results.md')
    
    args = parser.parse_args()
    asyncio.run(run_tests(args)) 