#!/usr/bin/env python3
"""
Training Preparation Module for Knowledge Processing

This module provides functions for processing various knowledge inputs,
including documents and fragmented text inputs, to prepare them for training.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union

from werkzeug.datastructures import FileStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Allowed file extensions for document processing
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}


async def process_document(
    document: FileStorage,
    user_id: str,
    tags: Optional[List[str]] = None
) -> bool:
    """
    Process a document file for knowledge extraction.
    
    Args:
        document: The document file object (pdf, txt, doc, docx)
        user_id: User identifier for tracking and logging
        tags: Optional list of tags to categorize the document
    
    Returns:
        Boolean indicating success or failure of the processing
    """
    if not document or not document.filename:
        logger.error(f"No valid document provided for user {user_id}")
        return False
    
    # Check if file extension is allowed
    extension = document.filename.rsplit('.', 1)[1].lower() if '.' in document.filename else ''
    if extension not in ALLOWED_EXTENSIONS:
        logger.error(f"Invalid file type {extension} for user {user_id}. Allowed types: {ALLOWED_EXTENSIONS}")
        return False
    
    # Log the processing attempt
    tag_str = ', '.join(tags) if tags else 'none'
    logger.info(f"Processing document {document.filename} for user {user_id} with tags: {tag_str}")
    
    try:
        # Simulating document processing with a delay
        # In a real implementation, this would involve actual document parsing and knowledge extraction
        await asyncio.sleep(1)  # Simulate processing time
        
        logger.info(f"Successfully processed document {document.filename} for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error processing document {document.filename} for user {user_id}: {str(e)}")
        return False


async def process_fragmented_input(
    messages: List[str],
    user_id: str,
    topic: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> bool:
    """
    Process a list of fragmented inputs (like chat messages or notes).
    
    Args:
        messages: List of text fragments to process
        user_id: User identifier for tracking and logging
        topic: Optional topic name for the fragmented input
        tags: Optional list of tags to categorize the input
    
    Returns:
        Boolean indicating success or failure of the processing
    """
    if not messages:
        logger.error(f"No messages provided for user {user_id}")
        return False
    
    topic_str = f" on topic '{topic}'" if topic else ""
    tag_str = ', '.join(tags) if tags else 'none'
    logger.info(f"Processing {len(messages)} fragments{topic_str} for user {user_id} with tags: {tag_str}")
    
    try:
        # Simulating fragmented input processing
        # In a real implementation, this would involve text processing and knowledge extraction
        await asyncio.sleep(0.5)  # Simulate processing time
        
        logger.info(f"Successfully processed {len(messages)} fragments for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error processing fragments for user {user_id}: {str(e)}")
        return False


async def batch_process_fragment_collections(
    user_id: str,
    collections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Process multiple collections of fragmented inputs concurrently.
    
    Args:
        user_id: User identifier for tracking and logging
        collections: List of dictionaries, each containing 'messages', optional 'topic', and optional 'tags'
    
    Returns:
        List of processing results for each collection
    """
    if not collections:
        logger.warning(f"No fragment collections provided for user {user_id}")
        return []
    
    logger.info(f"Batch processing {len(collections)} fragment collections for user {user_id}")
    
    # Create tasks for each collection
    tasks = []
    for collection in collections:
        messages = collection.get('messages', [])
        topic = collection.get('topic')
        tags = collection.get('tags')
        
        task = asyncio.create_task(
            process_fragmented_input(
                messages=messages,
                user_id=user_id,
                topic=topic,
                tags=tags
            )
        )
        tasks.append((task, collection))
    
    # Wait for all tasks to complete and collect results
    results = []
    for task, collection in tasks:
        try:
            success = await task
            results.append({
                "success": success,
                "processed_type": "fragmented",
                "topic": collection.get('topic', 'Untitled'),
                "message": "Processing completed successfully" if success else "Processing failed"
            })
        except Exception as e:
            logger.error(f"Error in batch processing for user {user_id}: {str(e)}")
            results.append({
                "success": False,
                "processed_type": "fragmented",
                "topic": collection.get('topic', 'Untitled'),
                "message": f"Exception occurred: {str(e)}"
            })
    
    return results


async def unified_knowledge_processor(
    user_id: str,
    mode: str,
    document_file: Optional[FileStorage] = None,
    fragmented_inputs: Optional[List[str]] = None,
    topic: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Unified interface to process either document files or fragmented inputs.
    
    Args:
        user_id: User identifier for tracking and logging
        mode: Processing mode - either 'document' or 'fragmented'
        document_file: Document file (required if mode is 'document')
        fragmented_inputs: List of text fragments (required if mode is 'fragmented')
        topic: Optional topic for fragmented inputs
        tags: Optional list of tags for categorization
    
    Returns:
        Dictionary containing processing results
    """
    result = {
        "success": False,
        "processed_type": mode,
        "message": ""
    }
    
    if mode not in ["document", "fragmented"]:
        result["message"] = f"Invalid mode: {mode}. Must be 'document' or 'fragmented'"
        return result
    
    try:
        if mode == "document":
            if not document_file:
                result["message"] = "No document file provided"
                return result
            
            result["filename"] = document_file.filename
            success = await process_document(
                document=document_file,
                user_id=user_id,
                tags=tags
            )
            
            result["success"] = success
            result["message"] = "Document processed successfully" if success else "Document processing failed"
            
        elif mode == "fragmented":
            if not fragmented_inputs:
                result["message"] = "No fragmented inputs provided"
                return result
            
            result["topic"] = topic or "Untitled"
            success = await process_fragmented_input(
                messages=fragmented_inputs,
                user_id=user_id,
                topic=topic,
                tags=tags
            )
            
            result["success"] = success
            result["message"] = "Fragmented inputs processed successfully" if success else "Fragmented input processing failed"
    
    except Exception as e:
        logger.error(f"Error in unified processor for user {user_id}: {str(e)}")
        result["success"] = False
        result["message"] = f"Exception occurred: {str(e)}"
    
    return result


async def process_multiple_inputs(
    user_id: str,
    documents: Optional[List[FileStorage]] = None,
    fragment_collections: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Process multiple documents and/or fragment collections in parallel.
    
    Args:
        user_id: User identifier for tracking and logging
        documents: Optional list of document files to process
        fragment_collections: Optional list of fragmented input collections
    
    Returns:
        Dictionary with summary of all processing operations
    """
    documents = documents or []
    fragment_collections = fragment_collections or []
    
    if not documents and not fragment_collections:
        return {
            "success": False,
            "message": "No inputs provided for processing",
            "results": []
        }
    
    logger.info(f"Processing multiple inputs for user {user_id}: {len(documents)} documents and {len(fragment_collections)} fragment collections")
    
    # Create tasks for document processing
    document_tasks = []
    for doc in documents:
        task = asyncio.create_task(process_document(
            document=doc,
            user_id=user_id
        ))
        document_tasks.append((task, doc))
    
    # Process fragment collections
    fragment_results = await batch_process_fragment_collections(
        user_id=user_id,
        collections=fragment_collections
    )
    
    # Collect document processing results
    document_results = []
    for task, doc in document_tasks:
        try:
            success = await task
            document_results.append({
                "success": success,
                "processed_type": "document",
                "filename": doc.filename,
                "message": "Processing completed successfully" if success else "Processing failed"
            })
        except Exception as e:
            logger.error(f"Error processing document {doc.filename} for user {user_id}: {str(e)}")
            document_results.append({
                "success": False,
                "processed_type": "document",
                "filename": doc.filename,
                "message": f"Exception occurred: {str(e)}"
            })
    
    # Combine all results
    all_results = document_results + fragment_results
    success_count = sum(1 for r in all_results if r.get("success", False))
    
    return {
        "success": success_count > 0,
        "message": f"Processed {success_count}/{len(all_results)} inputs successfully",
        "results": all_results
    } 