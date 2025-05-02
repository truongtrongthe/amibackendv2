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

def extract_structured_data_from_raw(raw_text: str) -> Dict[str, Any]:
    """
    Extract structured data from raw text using enhanced regex patterns.
    Captures full descriptions and detailed application methods with steps.
    
    Args:
        raw_text: The raw text from a knowledge entry
        
    Returns:
        Dictionary with extracted fields (title, description, content, takeaways, application_method with steps, etc.)
    """
    structured_data = {}
    
    logger.info(f"EXTRACTING STRUCTURED DATA FROM RAW: {raw_text}")    
    # Extract title
    title_match = re.search(r'Title:\s*(.*?)(?:\n|$)', raw_text)
    if title_match:
        structured_data["title"] = title_match.group(1).strip()
    
    # Extract full description (not just until the next section marker)
    desc_match = re.search(r'Description:\s*(.*?)(?=\nContent:|\nTakeaways:|\nDocument Summary:|\n\n\w+:)', raw_text, re.DOTALL)
    if desc_match:
        structured_data["description"] = desc_match.group(1).strip()
    
    # Extract content
    content_match = re.search(r'Content:\s*(.*?)(?=\nTakeaways:|\nDocument Summary:|\n\n\w+:)', raw_text, re.DOTALL)
    if content_match:
        structured_data["content"] = content_match.group(1).strip()
    
    # Extract takeaways and application method with detailed steps
    takeaways_match = re.search(r'Takeaways:\s*(.*?)(?=\nDocument Summary:|\n\n\w+:|$)', raw_text, re.DOTALL)
    if takeaways_match:
        takeaways_text = takeaways_match.group(1).strip()
        
        # Extract application method title
        app_method_title_match = re.search(r'Application Method:\s*(.*?)(?:\n\n\d|\n\n$|$)', takeaways_text)
        
        if app_method_title_match:
            app_method_title = app_method_title_match.group(1).strip()
            
            # Initialize application method structure
            app_method = {
                "title": app_method_title,
                "steps": []
            }
            
            # Find all steps in the application method section
            steps_pattern = r'\d+\.\s*\*\*([^*]+)\*\*\s*(.*?)(?=\n\d+\.\s*\*\*|\n\nDocument Summary:|$)'
            steps_matches = re.finditer(steps_pattern, takeaways_text, re.DOTALL)
            
            for step_match in steps_matches:
                step_title = step_match.group(1).strip()
                step_content = step_match.group(2).strip()
                
                # Extract sub-steps or bullet points if present
                sub_steps = []
                sub_step_pattern = r'\s*\*\*Bước\s+(\d+)\*\*:\s*(.*?)(?=\s*\*\*Bước|\n\n|$)'
                sub_step_matches = re.finditer(sub_step_pattern, step_content, re.DOTALL)
                
                for sub_match in sub_step_matches:
                    sub_step_num = sub_match.group(1)
                    sub_step_content = sub_match.group(2).strip()
                    sub_steps.append({
                        "number": int(sub_step_num),
                        "content": sub_step_content
                    })
                
                # If no formal sub-steps, look for bullet points
                if not sub_steps:
                    bullet_pattern = r'[-•]\s*(.*?)(?=\n\s*[-•]|\n\n|$)'
                    bullet_matches = re.finditer(bullet_pattern, step_content, re.DOTALL)
                    for bullet_match in bullet_matches:
                        sub_steps.append({
                            "bullet": bullet_match.group(1).strip()
                        })
                
                # Add the step to the application method
                step_obj = {
                    "title": step_title,
                    "content": step_content
                }
                
                if sub_steps:
                    step_obj["sub_steps"] = sub_steps
                
                app_method["steps"].append(step_obj)
            
            # Store the structured application method
            structured_data["application_method"] = app_method
            
            # Also store the raw application method text for backward compatibility
            structured_data["application_method_raw"] = takeaways_text
        else:
            # No application method found, use takeaways as is
            structured_data["takeaways"] = takeaways_text
    
    # Extract document summary
    summary_match = re.search(r'Document Summary:\s*(.*?)(?=\nCross-Cluster Connections:|\n\n\w+:|$)', raw_text, re.DOTALL)
    if summary_match:
        structured_data["document_summary"] = summary_match.group(1).strip()
    
    # Extract cross-cluster connections
    connections_match = re.search(r'Cross-Cluster Connections:\s*(.*?)(?=\n\n\w+:|$)', raw_text, re.DOTALL)
    if connections_match:
        structured_data["cross_cluster_connections"] = connections_match.group(1).strip()
        
    logger.info(f"STRUCTURED DATA: {structured_data}")
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

def prepare_knowledge(knowledge_entries: List[Dict], user_query: str, max_chars: int = 10000, target_classification: str = None) -> str:
    """
    Prepare knowledge context by extracting relevant segments, understanding cross-cluster connections,
    and creating a unified, comprehensive instruction set using LLM.
    
    Args:
        knowledge_entries: List of knowledge entry dictionaries, each item is formatted by extract_structured_data_from_raw
        user_query: Original user query for prioritization
        max_chars: Maximum characters allowed in the context
        target_classification: Optional classification to filter relevant segments (e.g., "Chán Nản")
        
    Returns:
        Formatted knowledge context string with comprehensive instructions
    """
    if not knowledge_entries:
        return ""
    
    try:
        # Step 1: Extract structured data and organize by relevance
        structured_entries = []
        cross_connections = []
        
        for entry in knowledge_entries:
            raw_text = entry.get("raw", "")
            entry_id = entry.get("id", "unknown")
            similarity = entry.get("similarity", 0)
            query = entry.get("query", "")
            phase = entry.get("phase", "unknown")
            
            # Extract structured data
            structured = entry.get("structured", {})
            if not structured and raw_text:
                # Only extract structured data if it doesn't already exist
                logger.info(f"No structured data found for entry {entry_id}, extracting from raw text")
                structured = extract_structured_data_from_raw(raw_text)
                
            # Log the structured data for debugging
            if structured:
                logger.info(f"Using structured data for entry {entry_id}: {list(structured.keys())}")
            else:
                logger.info(f"No structured data available for entry {entry_id}")
            
            # Extract cross-cluster connections if available
            if "cross_cluster_connections" in structured:
                connections = structured["cross_cluster_connections"]
                cross_connections.append({
                    "id": entry_id,
                    "connections": connections,
                    "title": structured.get("title", "Untitled Entry")
                })
            
            # Calculate relevance score
            relevance = similarity  # Start with base similarity
            
            # Boost entries containing target classification
            if target_classification and (
                (target_classification.lower() in raw_text.lower()) or
                (structured.get("title", "").lower() and target_classification.lower() in structured["title"].lower())
            ):
                relevance += 0.3
            
            # Boost for entries matching query terms
            query_terms = [term for term in query.lower().split() if len(term) > 3]
            if query_terms:
                term_matches = sum(1 for term in query_terms if term in raw_text.lower())
                relevance += 0.1 * (term_matches / len(query_terms))
            
            # Boost for entries with application methods
            if "application_method" in structured:
                relevance += 0.25
            
            # Store the structured entry with relevance
            structured_entries.append({
                "id": entry_id,
                "structured": structured,
                "raw": raw_text,
                "relevance": relevance,
                "phase": phase
            })
        
        # Sort entries by relevance
        sorted_entries = sorted(structured_entries, key=lambda x: x["relevance"], reverse=True)
        
        # Step 2: Prepare input for LLM synthesis
        # Take top most relevant entries (limit to preserve token count)
        top_entries = sorted_entries[:3]
        
        # Prepare knowledge context for LLM
        knowledge_input = ""
        
        # Format the top entries with detailed structured data
        for i, entry in enumerate(top_entries):
            structured = entry["structured"]
            knowledge_input += f"KNOWLEDGE ENTRY {i+1}:\n"
            
            if "title" in structured:
                knowledge_input += f"Title: {structured['title']}\n\n"
            
            if "description" in structured:
                knowledge_input += f"Description: {structured['description']}\n\n"
            
            # Add content if available
            if "content" in structured:
                content_preview = structured['content'][:800] + "..." if len(structured['content']) > 800 else structured['content']
                knowledge_input += f"Content: {content_preview}\n\n"
            
            # Format application method with detailed structure if available
            if "application_method" in structured and isinstance(structured["application_method"], dict):
                app_method = structured["application_method"]
                knowledge_input += f"APPLICATION METHOD: {app_method.get('title', 'No Title')}\n\n"
                
                # Process each step with its structure
                steps = app_method.get("steps", [])
                for j, step in enumerate(steps):
                    step_title = step.get("title", f"Step {j+1}")
                    knowledge_input += f"Step {j+1}: {step_title}\n"
                    
                    # Include step content
                    if "content" in step:
                        step_content = step["content"]
                        knowledge_input += f"{step_content}\n"
                    
                    # Format sub-steps if available
                    if "sub_steps" in step and step["sub_steps"]:
                        knowledge_input += "Detailed steps:\n"
                        for sub_step in step["sub_steps"]:
                            if "number" in sub_step and "content" in sub_step:
                                knowledge_input += f"- Sub-step {sub_step['number']}: {sub_step['content']}\n"
                            elif "bullet" in sub_step:
                                knowledge_input += f"- {sub_step['bullet']}\n"
                    
                    knowledge_input += "\n"
            # Fall back to raw application method or takeaways if structured format not available
            elif "application_method_raw" in structured:
                knowledge_input += f"Application Method: {structured['application_method_raw']}\n\n"
            elif "takeaways" in structured:
                knowledge_input += f"Takeaways: {structured['takeaways']}\n\n"
            
            # Add document summary if available
            if "document_summary" in structured:
                summary = structured["document_summary"]
                knowledge_input += f"Summary: {summary}\n\n"
            
            knowledge_input += "----\n\n"
        
        # Add cross-cluster connections
        if cross_connections:
            knowledge_input += "CROSS-CLUSTER CONNECTIONS:\n"
            for connection in cross_connections:
                knowledge_input += f"- {connection['title']} connects to: {connection['connections']}\n"
            knowledge_input += "\n----\n\n"
        
        logger.info(f"SYNTHESIZE STEP 2: Knowledge input before LLM: {knowledge_input}")
        # Step 3: Use LLM to synthesize knowledge into unified instructions
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        # Detect language for prompt
        language = detect_language(user_query)
        lang_prompt = "in Vietnamese" if language == "vietnamese" else "in English"
        
        # Create a more structured prompt that encourages the LLM to preserve the step-by-step format
        synthesis_prompt = f"""
        Based on the following knowledge entries, create a comprehensive set of instructions and information {lang_prompt} for addressing the user's needs.
        
        USER QUERY: {user_query}
        USER CLASSIFICATION: {target_classification if target_classification else "Unknown"}
        
        {knowledge_input}
        
        Create a unified, comprehensive set of instructions that:
        1. Begins with a clear explanation of what the user is dealing with
        2. Highlights specific techniques and approaches mentioned in the knowledge entries
        3. Provides a STEP-BY-STEP action plan with clear, numbered steps
        4. Preserves the detailed sub-steps from the original knowledge where available
        5. Connects related concepts across entries when relevant
        6. Uses a clear structure with headings, bullet points, and numbered lists
        
        FORMAT REQUIREMENTS:
        - Use numbered steps for the main action plan (1., 2., 3., etc.)
        - Preserve sub-steps with bullet points or sub-numbering (e.g., 1.1, 1.2, etc.)
        - Use bold text for important concepts
        - Create clear section headings
        - Include application techniques with explicit "how to" instructions
        
        Focus especially on application methods and specific instructions from the knowledge. 
        If entries have connections between them, explain how these concepts relate to each other.
        
        IMPORTANT: Do not fabricate information. Only use what is provided in the knowledge entries.
        """
        
        logger.info("Using LLM to synthesize knowledge into comprehensive instructions")
        response = llm.invoke(synthesis_prompt)
        synthesized_knowledge = response.content if hasattr(response, 'content') else str(response)
        
        # Ensure we don't exceed max characters
        if len(synthesized_knowledge) > max_chars:
            synthesized_knowledge = synthesized_knowledge[:max_chars-3] + "..."
        
        logger.info(f"Successfully synthesized {len(synthesized_knowledge)} chars of knowledge using LLM")
        return synthesized_knowledge
    
    except Exception as e:
        logger.error(f"Error in prepare_knowledge: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Enhanced fallback: try to preserve structured format even in fallback mode
        fallback_text = ""
        try:
            for entry in knowledge_entries[:2]:
                structured = entry.get("structured", {})
                if structured and "title" in structured:
                    fallback_text += f"# {structured['title']}\n\n"
                    
                    if "description" in structured:
                        fallback_text += f"{structured['description']}\n\n"
                    
                    # Include application method with steps if available
                    if "application_method" in structured and isinstance(structured["application_method"], dict):
                        app_method = structured["application_method"]
                        fallback_text += f"## {app_method.get('title', 'Application Method')}\n\n"
                        
                        # Include steps
                        steps = app_method.get("steps", [])
                        for i, step in enumerate(steps):
                            fallback_text += f"{i+1}. **{step.get('title', f'Step {i+1}')}**\n"
                            if "content" in step:
                                fallback_text += f"   {step['content']}\n\n"
                    elif "application_method_raw" in structured:
                        fallback_text += f"## Application Method\n{structured['application_method_raw']}\n\n"
                    elif "takeaways" in structured:
                        fallback_text += f"## Key Takeaways\n{structured['takeaways']}\n\n"
                    
                    fallback_text += "---\n\n"
                else:
                    raw = entry.get("raw", "")
                    if raw:
                        # Try to extract at least the title
                        title_match = re.search(r'Title:\s*(.*?)(?:\n|$)', raw)
                        title = title_match.group(1).strip() if title_match else "Knowledge Entry"
                        fallback_text += f"# {title}\n\n"
                        fallback_text += raw[:300] + "...\n\n---\n\n"
        except Exception as fallback_error:
            logger.error(f"Error in fallback text generation: {str(fallback_error)}")
            # Ultimate fallback - just return raw text
            for entry in knowledge_entries[:2]:
                fallback_text += entry.get("raw", "")[:500] + "\n\n"
        
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
    Enhanced detection for Vietnamese vs. English with more robust heuristics.
    
    Args:
        text: The text to analyze
        
    Returns:
        Language code ('vietnamese' or 'english')
    """
    if not text or len(text.strip()) < 2:
        # Default to Vietnamese for the application context
        return "vietnamese"
    
    # Convert text to lowercase for better matching
    text_lower = text.strip().lower()
    
    # 1. Check for Vietnamese-specific characters (highest confidence indicator)
    vietnamese_chars = set("ăâêôơưđáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    vi_char_count = sum(1 for char in text_lower if char in vietnamese_chars)
    
    # If we have multiple Vietnamese characters, it's very likely Vietnamese
    if vi_char_count > 1:
        logger.info(f"Detected Vietnamese by specific characters (found {vi_char_count})")
        return "vietnamese"
    
    # 2. Check for common Vietnamese words (high confidence)
    vietnamese_words = [
        "không", "của", "và", "là", "được", "có", "tôi", "cho", "một", "để",
        "trong", "người", "những", "nhưng", "với", "các", "mình", "này", "đã",
        "làm", "khi", "giúp", "từ", "cách", "như", "thể", "nếu", "vì", "tại",
        "quá", "rất", "thì", "phải", "nhiều", "cũng", "sẽ", "đang", "nên", "chỉ",
        "trên", "bị", "theo", "còn", "đến", "tình", "anh", "em", "bạn", "chúng",
        "hoặc", "mà", "gì", "năm", "ngày", "đã", "đây", "khoảng", "lúc", "mới",
        "nhất", "phút", "quan hệ", "xuất tinh", "sớm", "chậm", "lâu", "nhanh", "đủ"
    ]
    
    # Split text into words for matching
    words = re.findall(r'\b\w+\b', text_lower)
    vi_word_count = sum(1 for word in words if word in vietnamese_words)
    
    # If we have multiple Vietnamese words, it's likely Vietnamese
    if vi_word_count > 0:
        vi_word_ratio = vi_word_count / len(words) if words else 0
        if vi_word_ratio > 0.15 or vi_word_count >= 2:  # More than 15% of words are Vietnamese or at least 2 Vietnamese words
            logger.info(f"Detected Vietnamese by common words (found {vi_word_count} words, ratio {vi_word_ratio:.2f})")
            return "vietnamese"
    
    # 3. Check for common Vietnamese phrases/patterns
    vietnamese_patterns = [
        r'\b(tôi|mình|em|anh)\s+\w+',  # "tôi là", "mình muốn", etc.
        r'\b\w+\s+(rồi|chưa|á|ạ|nhé)\b',  # Words ending with conversational particles
        r'\b(làm|mua|xem|đi|biết)\s+(sao|thế|vậy)\b',  # Question patterns
        r'\b(chào|cám ơn|cảm ơn|xin)\s+\w+',  # Greeting patterns
        r'\bquá\s+\w+',  # "quá + adjective" pattern
        r'\b\w+\s+quá\b',  # "adjective + quá" pattern
        r'\bđã\s+\w+\s+chưa',  # "đã ... chưa" question pattern
        r'\bcó\s+(thể|phải|nên)',  # "có thể/phải/nên" modal patterns
        r'\bkhông\s+(thể|phải|được)',  # "không thể/phải/được" patterns
        r'\b(bị|đang|phải|không|có|đã)\s+\w+',  # Common verb prefixes
    ]
    
    for pattern in vietnamese_patterns:
        if re.search(pattern, text_lower):
            logger.info(f"Detected Vietnamese by phrase pattern: {pattern}")
            return "vietnamese"
    
    # 4. Check for English patterns before defaulting
    english_patterns = [
        r'\b(i|you|he|she|we|they)\s+\w+',  # English pronouns followed by verbs
        r'\b(is|are|was|were|have|has|had|do|does|did)\s+\w+',  # English auxiliary verbs
        r'\b(the|a|an)\s+\w+',  # English articles
        r'\b(will|would|should|could|can|may|might)\s+\w+',  # English modal verbs
        r'\b(have|has)\s+been\b',  # Perfect continuous tense markers
        r'\b(in|on|at|for|with|by|from|to)\s+\w+',  # Common English prepositions
    ]
    
    for pattern in english_patterns:
        if re.search(pattern, text_lower):
            logger.info(f"Detected English by phrase pattern: {pattern}")
            return "english"
    
    # 5. Statistical approach as last resort
    # Count Vietnamese characters, vowels, endings
    vi_indicators = sum(1 for c in text_lower if c in "ăâêôơưđáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    # Count English-specific patterns (th, wh, gh, etc.)
    en_indicators = len(re.findall(r'\b(th|wh|gh|kn|ph|qu|sh)\w+', text_lower))
    
    if vi_indicators > en_indicators:
        logger.info(f"Detected Vietnamese by character statistics ({vi_indicators} vs {en_indicators})")
        return "vietnamese"
    if en_indicators > 0:
        logger.info(f"Detected English by character statistics ({en_indicators} vs {vi_indicators})")
        return "english"
    
    # Default to Vietnamese for this specific application context
    logger.info("Defaulting to Vietnamese (application context)")
    return "vietnamese"

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
