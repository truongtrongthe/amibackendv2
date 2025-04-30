import json
import re
import time
import asyncio
import traceback
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Tuple
from functools import lru_cache
from langchain_openai import ChatOpenAI
from utilities import logger
from brain_singleton import get_brain, set_graph_version, is_brain_loaded, load_brain_vectors, get_current_graph_version

brain = get_brain()

def extract_structured_data_from_raw(raw_text: str) -> Dict[str, str]:
    """
    Extract structured data from raw text using regex patterns.
    
    Args:
        raw_text: The raw text from a knowledge entry
        
    Returns:
        Dictionary with extracted fields (title, description, content, takeaways, etc.)
    """
    structured_data = {}
    
    # Extract title
    title_match = re.search(r'Title:\s*(.*?)(?:\n|$)', raw_text)
    if title_match:
        structured_data["title"] = title_match.group(1).strip()
    
    # Extract description
    desc_match = re.search(r'Description:\s*(.*?)(?:\nContent:|\nTakeaways:|\nDocument Summary:|\n\n|$)', raw_text, re.DOTALL)
    if desc_match:
        structured_data["description"] = desc_match.group(1).strip()
    
    # Extract content
    content_match = re.search(r'Content:\s*(.*?)(?:\nTakeaways:|\nDocument Summary:|\n\n|$)', raw_text, re.DOTALL)
    if content_match:
        structured_data["content"] = content_match.group(1).strip()
    
    # Extract application method from takeaways
    takeaways_match = re.search(r'Takeaways:\s*(.*?)(?:\nDocument Summary:|\n\n|$)', raw_text, re.DOTALL)
    if takeaways_match:
        takeaways_text = takeaways_match.group(1).strip()
        app_method_match = re.search(r'Application Method:\s*(.*?)(?:\n\n\d|\n\n$|$)', takeaways_text, re.DOTALL)
        if app_method_match:
            structured_data["application_method"] = app_method_match.group(1).strip()
        else:
            structured_data["takeaways"] = takeaways_text
    
    # Extract document summary
    summary_match = re.search(r'Document Summary:\s*(.*?)(?:\nCross-Cluster Connections:|\n\n|$)', raw_text, re.DOTALL)
    if summary_match:
        structured_data["document_summary"] = summary_match.group(1).strip()
    
    # Extract cross-cluster connections
    connections_match = re.search(r'Cross-Cluster Connections:\s*(.*?)(?:\n\n|$)', raw_text, re.DOTALL)
    if connections_match:
        structured_data["cross_cluster_connections"] = connections_match.group(1).strip()
    
    return structured_data

def optimize_knowledge_context(knowledge_entries: List[Dict], user_query: str, max_chars: int = 2500, target_classification: str = None) -> str:
    """
    Optimize knowledge context by prioritizing and formatting relevant knowledge
    entries for the prompt. Improved to extract only relevant segments and filter
    by classification when provided.
    
    Args:
        knowledge_entries: List of knowledge entry dictionaries
        user_query: Original user query for prioritization
        max_chars: Maximum characters allowed in the context
        target_classification: Optional classification to filter relevant segments (e.g., "Chán Nản")
        
    Returns:
        Formatted knowledge context string
    """
    if not knowledge_entries:
        return ""
    
    try:
        # Extract target classification from user query if not provided
        if not target_classification:
            # Look for common classification terms in Vietnamese
            classification_patterns = ["nhóm", "người dùng", "khách hàng"]
            for pattern in classification_patterns:
                match = re.search(f"{pattern} ([\\w\\s]+)", user_query, re.IGNORECASE)
                if match:
                    target_classification = match.group(1)
                    break
        
        # Step 1: Split entries into relevant segments
        segmented_entries = []
        
        for entry in knowledge_entries:
            raw_text = entry.get("raw", "")
            entry_id = entry.get("id", "unknown")
            similarity = entry.get("similarity", 0)
            query = entry.get("query", "")
            phase = entry.get("phase", "unknown")
            
            # Split text into paragraphs
            paragraphs = re.split(r'\n\n+', raw_text)
            
            # Process each paragraph as a potential segment
            for i, para in enumerate(paragraphs):
                # Skip empty paragraphs
                if not para.strip():
                    continue
                
                # Calculate relevance score
                relevance = similarity  # Start with base similarity
                
                # Check for target classification in segment
                if target_classification and target_classification.lower() in para.lower():
                    relevance += 0.3  # Big boost for segments containing the target classification
                
                # Check for query terms in segment
                query_terms = query.lower().split()
                if query_terms:
                    query_term_count = sum(1 for term in query_terms if term.lower() in para.lower())
                    relevance += 0.1 * (query_term_count / len(query_terms))
                
                # Check for specific section indicators
                if re.search(r'(Title:|Description:|Application:|Takeaways:)', para):
                    relevance += 0.15
                
                # Boost for first paragraph (often contains key information)
                if i == 0:
                    relevance += 0.05
                
                # Store the segment with its calculated relevance
                segmented_entries.append({
                    "text": para,
                    "relevance": relevance,
                    "id": f"{entry_id}_{i}",
                    "phase": phase
                })
        
        # Step 2: Sort segments by relevance
        sorted_segments = sorted(segmented_entries, key=lambda x: x["relevance"], reverse=True)
        
        # Step 3: Format segments in priority order until max_chars is reached
        formatted_context = ""
        added_segment_ids = set()
        
        for segment in sorted_segments:
            # Skip if already added (based on unique segment ID)
            if segment["id"] in added_segment_ids:
                continue
            
            # Format the segment text
            segment_text = segment["text"].strip()
            
            # Add phase indicator for clarity
            if segment["phase"] == "user_analysis":
                phase_marker = "[User Type Knowledge]"
            elif segment["phase"] == "technique_implementation":
                phase_marker = "[Technique Implementation]"
            elif segment["phase"] == "additional_knowledge":
                phase_marker = "[Additional Knowledge]"
            else:
                phase_marker = ""
            
            # Format the segment with phase marker if available
            formatted_segment = f"{phase_marker}\n{segment_text}" if phase_marker else segment_text
            
            # Check if adding this would exceed max chars
            if len(formatted_context) + len(formatted_segment) + 2 <= max_chars:  # +2 for newlines
                formatted_context += formatted_segment + "\n\n"
                added_segment_ids.add(segment["id"])
            else:
                # If we can't add more full segments, we're done
                break
        
        logger.info(f"Optimized knowledge context: {len(formatted_context)} chars from {len(segmented_entries)} segments")
        return formatted_context.strip()
    
    except Exception as e:
        logger.error(f"Error optimizing knowledge context: {str(e)}")
        # Fallback: just concatenate raw text of first few entries
        fallback_text = ""
        for entry in knowledge_entries[:2]:
            raw = entry.get("raw", "")
            if raw:
                fallback_text += raw[:300] + "...\n\n"  # Truncate each entry
        
        return fallback_text[:max_chars]

async def ensure_brain_loaded(graph_version_id: str = "") -> bool:
    """
    Ensure the brain is loaded and ready to use.
    
    Args:
        graph_version_id: Optional graph version ID
        
    Returns:
        True if brain is loaded successfully, False otherwise
    """
    try:
        # Set graph version if provided
        if graph_version_id:
            current = get_current_graph_version()
            if current != graph_version_id:
                set_graph_version(graph_version_id)
                logger.info(f"Set graph version to {graph_version_id}")
        
        # Check if brain is already loaded
        if is_brain_loaded():
            return True
        
        # Otherwise load the brain
        await load_brain_vectors()
        return is_brain_loaded()
        
    except Exception as e:
        logger.error(f"Error ensuring brain loaded: {str(e)}")
        return False

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    Simple heuristic-based detection for Vietnamese vs. English.
    
    Args:
        text: The text to analyze
        
    Returns:
        Language code ('vi' for Vietnamese, 'en' for English)
    """
    # Vietnamese-specific characters and words
    vietnamese_chars = set("ăâêôơưđ")
    vietnamese_words = [
        "không", "của", "và", "là", "được", "có", "tôi", "cho", "một", "để",
        "trong", "được", "người", "những", "nhưng", "với", "các", "mình", "này", "đã",
        "làm", "khi", "giúp", "từ", "cách", "như", "thể", "nếu", "vì", "tại"
    ]
    
    # Convert text to lowercase for better matching
    text_lower = text.lower()
    
    # Check for Vietnamese characters
    if any(char in vietnamese_chars for char in text_lower):
        return "vietnamese"
    
    # Check for Vietnamese words
    if any(word in text_lower for word in vietnamese_words):
        return "vietnamese"
    
    # Default to English
    return "english"

def process_knowledge_context(knowledge_entries: List[Dict], user_query: str) -> str:
    """
    Process knowledge entries into a formatted context for the LLM.
    
    Args:
        knowledge_entries: List of knowledge entry dictionaries
        user_query: The user's query for relevance scoring
        
    Returns:
        Formatted knowledge context string
    """
    if not knowledge_entries:
        return ""
    
    try:
        # Group entries by type for better organization
        entries_by_type = {
            "high_relevance": [],
            "medium_relevance": [],
            "low_relevance": []
        }
        
        # Calculate median similarity for threshold
        similarities = [entry.get("similarity", 0) for entry in knowledge_entries]
        median_similarity = sorted(similarities)[len(similarities) // 2] if similarities else 0
        
        # Create threshold values
        high_threshold = median_similarity + 0.1
        low_threshold = median_similarity - 0.1
        
        # Categorize entries by relevance
        for entry in knowledge_entries:
            similarity = entry.get("similarity", 0)
            if similarity >= high_threshold:
                entries_by_type["high_relevance"].append(entry)
            elif similarity <= low_threshold:
                entries_by_type["low_relevance"].append(entry)
            else:
                entries_by_type["medium_relevance"].append(entry)
        
        # Format with most relevant first, limited to 2500 chars total
        formatted_context = "KNOWLEDGE CONTEXT:\n\n"
        char_count = len(formatted_context)
        max_chars = 2500
        
        # Process high relevance
        for entry in entries_by_type["high_relevance"]:
            entry_text = format_knowledge_entry(entry)
            if char_count + len(entry_text) + 10 <= max_chars:
                formatted_context += f"[HIGH RELEVANCE]\n{entry_text}\n\n"
                char_count += len(entry_text) + 20  # Account for added tags
        
        # Process medium relevance
        for entry in entries_by_type["medium_relevance"]:
            entry_text = format_knowledge_entry(entry)
            if char_count + len(entry_text) + 10 <= max_chars:
                formatted_context += f"[MEDIUM RELEVANCE]\n{entry_text}\n\n"
                char_count += len(entry_text) + 22  # Account for added tags
        
        # Process low relevance (only if space allows)
        for entry in entries_by_type["low_relevance"]:
            entry_text = format_knowledge_entry(entry)
            if char_count + len(entry_text) + 10 <= max_chars:
                formatted_context += f"[ADDITIONAL INFO]\n{entry_text}\n\n"
                char_count += len(entry_text) + 21  # Account for added tags
        
        return formatted_context.strip()
        
    except Exception as e:
        logger.error(f"Error processing knowledge context: {str(e)}")
        # Fallback: just concatenate the first few entries
        fallback_text = "KNOWLEDGE CONTEXT:\n\n"
        for entry in knowledge_entries[:3]:
            fallback_text += format_knowledge_entry(entry) + "\n\n"
        
        # Ensure we don't exceed the maximum length
        if len(fallback_text) > 2500:
            fallback_text = fallback_text[:2497] + "..."
        
        return fallback_text

def format_knowledge_entry(entry: Dict) -> str:
    """
    Format a knowledge entry for inclusion in a prompt.
    
    Args:
        entry: Knowledge entry dictionary
        
    Returns:
        Formatted string representation of the knowledge entry
    """
    # Extract entry components
    entry_id = entry.get("id", "unknown")
    similarity = entry.get("similarity", 0)
    raw_text = entry.get("raw", "")
    structured = entry.get("structured", {})
    
    # Format with structured data if available
    if structured:
        parts = []
        
        # Add title if available
        if "title" in structured:
            parts.append(f"Title: {structured['title']}")
        
        # Add description if available
        if "description" in structured:
            parts.append(f"Description: {structured['description']}")
        
        # Add application method or takeaways if available
        if "application_method" in structured:
            parts.append(f"Application: {structured['application_method']}")
        elif "takeaways" in structured:
            parts.append(f"Takeaways: {structured['takeaways']}")
        
        # Fall back to raw text if structured extraction failed
        if not parts and raw_text:
            # Clean up raw text and truncate if needed
            cleaned_text = raw_text.strip()
            if len(cleaned_text) > 500:  # Limit to 500 chars
                cleaned_text = cleaned_text[:497] + "..."
            return cleaned_text
        
        return "\n".join(parts)
    
    # Fall back to raw text if no structured data
    if raw_text:
        # Clean up raw text and truncate if needed
        cleaned_text = raw_text.strip()
        if len(cleaned_text) > 500:  # Limit to 500 chars
            cleaned_text = cleaned_text[:497] + "..."
        return cleaned_text
    
    # Last resort if no content available
    return f"Entry {entry_id} (Relevance: {similarity:.2f})"

def extract_key_knowledge(knowledge_context: str, conversation_context: str) -> str:
    """
    Extract the most relevant parts from knowledge context based on the conversation.
    
    Args:
        knowledge_context: The full knowledge context text
        conversation_context: The conversation context to use for relevance
        
    Returns:
        Reduced knowledge context focusing on the most relevant parts
    """
    if not knowledge_context:
        return ""
    
    try:
        # Get key terms from the conversation (last user message)
        key_terms = []
        user_msg = ""
        
        for line in conversation_context.split('\n'):
            if line.startswith("User:"):
                user_msg = line[5:].strip()
                break
        
        # Extract potential key terms from user message
        if user_msg:
            # Remove common words and punctuation
            common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "and", "but", "or", "is", "are",
                           "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "of"}
            
            # Split by common delimiters and convert to lowercase
            words = re.split(r'[ \t\n,.?!:;()\[\]{}"]', user_msg.lower())
            words = [w for w in words if w and w not in common_words and len(w) > 2]
            
            # Take most frequent terms
            from collections import Counter
            term_counter = Counter(words)
            key_terms = [term for term, _ in term_counter.most_common(5)]
        
        # Split knowledge into sections/paragraphs
        sections = re.split(r'\n\n+', knowledge_context)
        
        # Score each section based on key terms
        scored_sections = []
        for section in sections:
            score = 1  # Base score
            
            # Score by key term presence
            for term in key_terms:
                if term.lower() in section.lower():
                    score += 3
            
            # Bonus for sections with "Title:", "Description:", etc.
            if re.search(r'(Title|Description|Application|Takeaways):', section):
                score += 5
            
            # Penalize very long sections
            if len(section) > 300:
                score -= 2
            
            scored_sections.append((section, score))
        
        # Sort by score and take top sections
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        top_sections = [section for section, _ in scored_sections[:3]]
        
        # Combine and return top sections
        reduced_context = "\n\n".join(top_sections)
        
        # Ensure we're within limit
        if len(reduced_context) > 1500:
            reduced_context = reduced_context[:1497] + "..."
        
        return reduced_context
        
    except Exception as e:
        logger.error(f"Error extracting key knowledge: {str(e)}")
        # Return truncated original context as fallback
        return knowledge_context[:1000] + "..."


def build_analyse_profile_query(user_profile: Dict) -> List[str]:
    """
    Build queries to analyze the user profile.
    
    Args:
        user_profile: The user profile dictionary

    Returns:
        A list of query strings for analyzing the user profile
    """
    # Extract key information from the user profile
    portrait = user_profile.get("portrait", "")
    logger.info(f"User profile portrait: {portrait[:500]}...")
    
    # Determine the language based on the portrait
    is_vietnamese = False
    if "tiếng Việt" in portrait.lower() or "vietnamese" in portrait.lower() or any(char in "ăâêôơưđ" for char in portrait):
        is_vietnamese = True
        language_hint = "Trả lời hoàn toàn bằng tiếng Việt. Tạo các câu truy vấn bằng tiếng Việt."
    else:
        language_hint = "Return the queries in English."
    
    # Default language detection if no profile is available
    if not portrait:
        is_vietnamese = True  # Default to Vietnamese for this application
    
    # Use LLM to generate queries to analyze the user profile
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Extract key user classification and generate queries based on this portrait: {portrait}

        Goal: First identify the user classification/group/type, then create queries that will find knowledge about:
        1. Understanding this user type's core characteristics
        2. Best approaches for this user type
        3. Specific next actions to take with this user type

        {language_hint}

        Your queries MUST follow this exact pattern:
        """

    # Add language-specific query patterns
    if is_vietnamese:
        prompt += """
        1. "Hiểu về người dùng nhóm [phân loại người dùng]"
        2. "Phương pháp tiếp cận nhóm [phân loại người dùng]"
        3. "Những việc cần làm với nhóm [phân loại người dùng]"

        IMPORTANT INSTRUCTIONS:
        - Extract ONLY the primary user classification (e.g., "Chán Nản", "Tự Tin", "Lo Lắng", etc.)
        - Use the EXACT SAME classification term in all three queries
        - Keep the query patterns exactly as shown, only replacing [phân loại người dùng]
        - ENSURE ALL QUERIES ARE COMPLETELY IN VIETNAMESE
        """
    else:
        prompt += """
        1. "Understanding [user classification] users"
        2. "Best approaches for [user classification] users"
        3. "What to do next with [user classification] users"

        IMPORTANT INSTRUCTIONS:
        - Extract ONLY the primary user classification (e.g., "Discouraged", "Confident", "Anxious", etc.)
        - Use the EXACT SAME classification term in all three queries
        - Keep the query patterns exactly as shown, only replacing [user classification]
        - ENSURE ALL QUERIES ARE COMPLETELY IN ENGLISH
        """
    response = llm.invoke(prompt)
    
    # Extract the response content
    content = response.content if hasattr(response, 'content') else str(response)
    
    # Split by newlines and extract queries from numbered items
    queries = []
    for line in content.split('\n'):
        if re.match(r'^\s*\d+[\.\)]', line):  # Line starts with a number followed by . or )
            query = re.sub(r'^\s*\d+[\.\)]\s*', '', line).strip()
            # Remove markdown formatting (** at beginning and end) and quotes
            query = re.sub(r'^\*\*(.*)\*\*$', r'\1', query)
            query = re.sub(r'^["\'](.*)["\']$', r'\1', query)
            if query:
                queries.append(query)
    
    # If no queries were found, use fallback queries based on detected language
    if not queries:
        segment = user_profile.get("segment", {}).get("category", "general")
        if is_vietnamese:
            queries = [
                f"Hiểu về người dùng nhóm {segment}",
                f"Phương pháp tiếp cận nhóm {segment}",
                f"Những việc cần làm với nhóm {segment}"
            ]
        else:
            queries = [
                f"Understanding {segment} users",
                f"Best approaches for {segment} users",
                f"What to do next with {segment} users"
            ]
    
    logger.info(f"Generated {len(queries)} analysis queries: {queries}")
    return queries
