#!/usr/bin/env python3
"""
Example usage of the training_prep module for knowledge processing.
"""

import os
import io
import asyncio
from typing import List, Dict, Any

from werkzeug.datastructures import FileStorage
import training_prep


def create_mock_document(filename: str, content: bytes) -> FileStorage:
    """
    Create a mock FileStorage object for testing document processing.
    
    Args:
        filename: The name of the file with extension
        content: The binary content of the file
        
    Returns:
        A FileStorage object that simulates a file upload
    """
    stream = io.BytesIO(content)
    return FileStorage(
        stream=stream,
        filename=filename,
        content_type="application/pdf" if filename.endswith(".pdf") else "text/plain"
    )


async def example_chat_log() -> None:
    """Example of processing a chat log about product pricing."""
    print("\n=== Processing Chat Log Example ===")
    
    # Sample chat messages about product pricing
    chat_messages = [
        "Customer: I'm interested in your enterprise plan. What's the pricing?",
        "Agent: Our enterprise plan starts at $499/month for up to 10 users.",
        "Customer: Do you offer any discounts for annual payment?",
        "Agent: Yes, we offer a 15% discount for annual subscriptions.",
        "Customer: Great, and what about the features specific to enterprise?",
        "Agent: Enterprise includes priority support, advanced security, and custom integrations."
    ]
    
    # Process the chat log
    user_id = "user123"
    result = await training_prep.process_fragmented_input(
        messages=chat_messages,
        user_id=user_id,
        topic="Enterprise Pricing",
        tags=["pricing", "enterprise", "sales"]
    )
    
    if result:
        print(f"Successfully processed chat log for user {user_id}")
    else:
        print(f"Failed to process chat log for user {user_id}")


async def example_document_processing() -> None:
    """Example of processing a PDF document."""
    print("\n=== Processing Document Example ===")
    
    # Create a mock PDF document
    # Note: This would fail in a real environment since it's not a real PDF
    pdf_content = b"This is a mock PDF document about machine learning concepts."
    mock_pdf = create_mock_document("machine_learning_guide.pdf", pdf_content)
    
    # Process the document
    user_id = "user456"
    result = await training_prep.process_document(
        document=mock_pdf,
        user_id=user_id,
        tags=["machine learning", "guide", "tutorial"]
    )
    
    if result:
        print(f"Successfully processed document for user {user_id}")
    else:
        print(f"Failed to process document for user {user_id}") 
        print("Note: This failure is expected in a real environment since our mock isn't a real PDF")


async def example_unified_processing() -> None:
    """Example of using the unified interface for processing different input types."""
    print("\n=== Unified Processing Example ===")
    
    # Prepare inputs
    user_id = "user789"
    
    # Meeting notes as fragmented input
    meeting_notes = [
        "Project kickoff scheduled for next Monday",
        "Sarah will lead the frontend team",
        "Budget approval expected by end of week",
        "First milestone delivery in 3 weeks"
    ]
    
    # Mock document
    doc_content = b"Quarterly financial report with budget projections."
    mock_doc = create_mock_document("q2_financial_report.pdf", doc_content)
    
    # Process meeting notes using unified interface
    notes_result = await training_prep.unified_knowledge_processor(
        user_id=user_id,
        mode="fragmented",
        fragmented_inputs=meeting_notes,
        topic="Project Planning",
        tags=["meeting", "project"]
    )
    
    # Process document using unified interface
    doc_result = await training_prep.unified_knowledge_processor(
        user_id=user_id,
        mode="document",
        document_file=mock_doc,
        tags=["financial", "quarterly"]
    )
    
    # Print results
    print(f"Notes processing result: {notes_result['success']}")
    print(f"Notes processed type: {notes_result['processed_type']}")
    print(f"Notes message: {notes_result['message']}")
    
    print(f"Document processing result: {doc_result['success']}")
    print(f"Document processed type: {doc_result['processed_type']}")
    print(f"Document filename: {doc_result['filename']}")
    print(f"Document message: {doc_result['message']}")


async def example_batch_processing() -> None:
    """Example of batch processing multiple inputs in parallel."""
    print("\n=== Batch Processing Example ===")
    
    user_id = "user101"
    
    # Create multiple document mocks
    doc1 = create_mock_document("product_roadmap.pdf", b"Product roadmap for 2023")
    doc2 = create_mock_document("user_research.pdf", b"User research findings")
    
    # Create multiple fragment collections
    fragment_collections = [
        {
            "messages": [
                "The API performance improved by 25% after the latest update",
                "Response time decreased from 300ms to 225ms on average",
                "Error rate dropped from 2.3% to 0.8%"
            ],
            "topic": "API Performance",
            "tags": ["technical", "performance"]
        },
        {
            "messages": [
                "User satisfaction score increased to 4.5/5",
                "Most requested feature was dark mode",
                "89% of users would recommend the product"
            ],
            "topic": "User Feedback",
            "tags": ["feedback", "user experience"]
        }
    ]
    
    # Process everything in parallel
    result = await training_prep.process_multiple_inputs(
        user_id=user_id,
        documents=[doc1, doc2],
        fragment_collections=fragment_collections
    )
    
    # Print overall result
    print(f"Overall success: {result['success']}")
    print(f"Message: {result['message']}")
    
    # Print individual results
    print("Individual processing results:")
    for i, item_result in enumerate(result['results']):
        success = "✓" if item_result.get("success", False) else "✗"
        if item_result.get("processed_type") == "document":
            print(f"  {i+1}. Document: {item_result.get('filename')} - {success}")
        else:
            print(f"  {i+1}. Fragmented input: {item_result.get('topic')} - {success}")


async def run_examples():
    """Run all example functions."""
    await example_chat_log()
    await example_document_processing()
    await example_unified_processing()
    await example_batch_processing()


if __name__ == "__main__":
    asyncio.run(run_examples()) 