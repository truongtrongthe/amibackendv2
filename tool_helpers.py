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

# Track the currently loaded graph version to avoid redundant reloading
_current_loaded_version = None

# Keep track of the last load time to prevent thrashing
_last_load_time = 0
MIN_LOAD_INTERVAL = 30  # seconds

# Track load status to avoid duplicate loading
_load_in_progress = False

def extract_structured_data_from_raw(raw_text: str) -> Dict[str, Any]:
    """
    DEPRECATED: This function is kept for backward compatibility but no longer used in the main pipeline.
    
    Extract structured data from raw text using enhanced regex patterns.
    Captures full descriptions and detailed application methods with steps.
    
    Args:
        raw_text: The raw text from a knowledge entry
        
    Returns:
        Dictionary with extracted fields (title, description, content, takeaways, application_method with steps, etc.)
    """
    # Add warning log
    logger.warning("Using deprecated extract_structured_data_from_raw function. Consider using prepare_knowledge directly.")
    
    structured_data = {}
        
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
    return structured_data

def prepare_knowledge(knowledge_entries: List[Dict], user_query: str, max_chars: int = 10000, target_classification: str = None, preserve_exact_terminology: bool = False) -> str:
    """
    Prepare knowledge context by directly processing raw knowledge text and creating a unified,
    comprehensive instruction set. This function skips the rigid extraction step.
    
    Args:
        knowledge_entries: List of knowledge entry dictionaries
        user_query: Original user query for prioritization
        max_chars: Maximum characters allowed in the context
        target_classification: Optional classification to filter relevant segments (e.g., "Chán Nản")
        preserve_exact_terminology: If True, instructs LLM to preserve exact classification terms without translation or transformation
        
    Returns:
        Formatted knowledge context string with comprehensive instructions
    """
    if not knowledge_entries:
        return ""
    
    try:
        # Step 1: Calculate relevance and organize entries
        ranked_entries = []
        #explore the knowledge entries aka vectors  
        for entry in knowledge_entries:
            raw_text = entry.get("raw", "")
            entry_id = entry.get("id", "unknown")
            similarity = entry.get("similarity", 0)
            query = entry.get("query", "")
            phase = entry.get("phase", "unknown")
            
            # Skip if no raw text
            if not raw_text:
                continue
                
            # Calculate relevance score - more straightforward now
            relevance = similarity  # Base score is similarity
            
            # Boost entries containing target classification if provided
            if target_classification and target_classification.lower() in raw_text.lower():
                relevance += 0.3
                logger.info(f"Boosting entry {entry_id} for containing classification '{target_classification}'")
            
            # Boost entries containing key terms from user query
            query_terms = [term for term in user_query.lower().split() if len(term) > 3]
            if query_terms:
                term_matches = sum(1 for term in query_terms if term in raw_text.lower())
                relevance += 0.1 * (term_matches / len(query_terms))
            
            # Boost entries with application methods
            if "Application Method" in raw_text or "Steps:" in raw_text:
                relevance += 0.25
                logger.info(f"Boosting entry {entry_id} for containing application methods")
            
            # Store for ranking
            ranked_entries.append({
                "id": entry_id,
                "raw": raw_text,
                "relevance": relevance,
                "phase": phase
            })
        
        # Sort entries by relevance
        sorted_entries = sorted(ranked_entries, key=lambda x: x["relevance"], reverse=True)
        
        # Step 2: Prepare raw knowledge entries for LLM synthesis
        # Take top most relevant entries (limit to preserve token count)
        top_entries = sorted_entries[:5]
        logger.info(f"Selected top {len(top_entries)} entries for knowledge context")
        
        # Format raw entries with minimal processing
        knowledge_input = ""
        for i, entry in enumerate(top_entries):
            raw_text = entry["raw"]
            
            # Extract only title and application methods
            title = ""
            application_methods = []
            
            # Extract title
            title_match = re.search(r'Title:\s*(.*?)(?:\n|$)', raw_text)
            if title_match:
                title = title_match.group(1).strip()
            
            # Extract takeaways section - simpler approach
            takeaways_match = re.search(r'Takeaways:(.*?)(?=Document Summary:|Cross-Cluster Connections:|$)', raw_text, re.DOTALL)
            if takeaways_match:
                takeaways_text = takeaways_match.group(1).strip()
                
                # Capture all distinct application methods by splitting on 'Application method:' or similar patterns
                method_blocks = re.split(r'(?i)Application method:', takeaways_text)
                for block in method_blocks[1:]:  # Skip the first block as it is before the first method
                    method_text = block.strip()
                    if method_text:
                        application_methods.append(method_text)
                
                # If no split by 'Application method:', check for a single method or fallback
                if len(method_blocks) <= 1 and "application method" in takeaways_text.lower():
                    application_methods.append(takeaways_text)
            
            #logger.info(f"Application methods found: {application_methods}")
            
            # Format the extracted content
            formatted_content = f"KNOWLEDGE ENTRY {i+1}:\n"
            if title:
                formatted_content += f"Title: {title}\n\n"
            
            if application_methods:
                formatted_content += "HOW TO APPLY:\n"
                for idx, method in enumerate(application_methods):
                    # Simply include the entire method text with minimal processing
                    formatted_content += f"Method {idx+1}:\n{method}\n\n"
            else:
                # Fallback to using some raw text if no application methods found
                content_preview = raw_text.split("Takeaways:", 1)[0].strip()
                if len(content_preview) > 300:
                    content_preview = content_preview[:300] + "..."
                formatted_content += f"Content: {content_preview}\n\n"
            
            knowledge_input += formatted_content + "----\n\n"
        
        # Step 3: Use LLM to synthesize knowledge into unified instructions
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0.05)
        
        # Detect language for prompt
        language = detect_language(user_query)
        lang_prompt = "in Vietnamese" if language == "vietnamese" else "in English"
        
        # Create a more targeted prompt for knowledge synthesis
        preservation_instruction = """
        CRITICAL INSTRUCTION: You MUST PRESERVE THE EXACT TERMINOLOGY of ALL classification categories.
        - VERBATIM COPY classification names with EXACT CHARACTERS, SPACES, and FORMATTING
        - DO NOT translate classification names (e.g., keep "Nhóm Chán Nản" exactly as is, not as "Discouraged Group" or similar)
        - DO NOT reinterpret classification categories with equivalent terms
        - DO NOT reorganize or merge classification categories
        - DO NOT ADD new classifications or categories that aren't explicitly in the knowledge
        - COPY-PASTE examples from the knowledge EXACTLY as they appear, especially example phrases like "Tôi bị xuất sớm"
        - Maintain the original hierarchy and relationships between classifications
        
        WHEN CREATING THE CLASSIFICATION FRAMEWORK:
        - Place the classifications in a titled section at the top of your response
        - Include EXACT EXAMPLES from the knowledge on how to apply each classification
        - Include explicit categorization criteria for each classification with VERBATIM QUOTES when possible
        """ if preserve_exact_terminology else ""
        
        synthesis_prompt = f"""
        Synthesize these structured knowledge entries {lang_prompt} to address the user's needs, creating a concise narrative for analysis and implementation.
        
        USER QUERY: {user_query}
        {f"USER CLASSIFICATION: {target_classification}" if target_classification else ""}
        {preservation_instruction}
        
        {knowledge_input}
        
        CRITICAL PRIORITIES:
        1. KNOWLEDGE NARRATIVE: Create a concise introduction connecting knowledge titles
          - Summarize ALL knowledge titles found without extra spacing
          - Show how these knowledge areas solve the user's problem
          - Keep this section brief (50-75 words maximum)
        
        2. CLASSIFICATIONS: Present the complete customer classification framework
          - Use "## Customer Classification Framework" heading
          - PRESERVE EXACT classification terminology (e.g., "Nhóm Chán Nản", "Nhóm Tự Tin")
          - Include ALL identifying criteria for each classification (behaviors, phrases, thresholds)
          - Format each classification with bold markers: **Nhóm Chán Nản**
          - Include EXACT verbatim examples from knowledge (e.g., "Tôi bị xuất sớm")
        
        3. APPLICATION METHODS: Organize and present ALL methods without consolidation
          - Group by classification (e.g., "## Methods for Nhóm Chán Nản")
          - If a method applies to multiple classifications or is general, include it under a "## General Application Methods" section
          - PRESERVE EACH DISTINCT APPLICATION METHOD NAME EXACTLY AS FOUND IN SOURCE
          - List all methods separately with their original headings
          - For each method, include the complete title exactly as written (e.g., "Đặt câu hỏi với mỗi nhóm khách hàng")
          - NEVER combine or merge different methods even if they seem related
          - INCLUDE ALL NUMBERED STEPS exactly as they appear in the original knowledge
          - DO NOT truncate steps or limit them to any predetermined number
          - Keep ALL specific instructions, scripts, and examples within their original method
          - PRESERVE original numbered steps format with bold titles
          - DO NOT RENAME OR REWORD method titles under any circumstances
        
        4. IMPLEMENTATION GUIDE: Create practical implementation summary
          - For the user classification, provide step-by-step implementation guide
          - Reference each distinct application method by its exact name
          - Include concrete language and phrases to use (verbatim from knowledge)
          - Ensure ALL critical steps are preserved, including steps for collecting demographic information like age
          - Include steps from general methods if they are relevant to building a complete customer portrait
        
        FORMAT (use this exact structure with minimal spacing):
        ## Knowledge Found
        [Brief narrative connecting knowledge titles - 80-100 words]
        
        ## Customer Classification Framework
        **[Classification 1]**: [Criteria with EXACT terminology - include verbatim examples]
        **[Classification 2]**: [Criteria with EXACT terminology - include verbatim examples]
        
        ## Application Methods for [Primary Classification]
        ### [EXACT Method 1 Name - PRESERVE COMPLETE ORIGINAL TITLE]
        **What**: [Objective of this method]
        **How**:
        1. [Step 1 with EXACT wording]
        2. [Step 2 with EXACT wording]
        ... [Include ALL additional steps exactly as numbered in original]
        N. [Last step with EXACT wording]
        
        ### [EXACT Method 2 Name - PRESERVE COMPLETE ORIGINAL TITLE]
        **What**: [Objective of this method]
        **How**:
        1. [Step 1 with EXACT wording]
        2. [Step 2 with EXACT wording]
        ... [Include ALL additional steps exactly as numbered in original]
        N. [Last step with EXACT wording]
        
        ## General Application Methods (if applicable)
        ### [EXACT Method Name for General Use - PRESERVE COMPLETE ORIGINAL TITLE]
        **What**: [Objective of this method]
        **How**:
        1. [Step 1 with EXACT wording]
        2. [Step 2 with EXACT wording]
        ... [Include ALL additional steps exactly as numbered in original]
        N. [Last step with EXACT wording]
        
        ## Implementation Guide
        1. [First implementation step referencing specific method by exact name]
        2. [Second implementation step with concrete examples]
        ... [Include additional steps as necessary to cover all relevant methods, including general ones for demographic data collection]
        
        IMPORTANT: 
        - NEVER translate classification terms
        - PRESERVE ALL application methods as SEPARATE, DISTINCT entries with their COMPLETE original titles
        - DO NOT CONSOLIDATE methods - each original method must appear as its own section
        - INCLUDE EVERY SINGLE STEP from each application method - do not omit any steps regardless of how many there are
        - PRESERVE THE EXACT NUMBERING SCHEME from the original knowledge
        - DO NOT limit methods to a specific number of steps - include all steps from the original
        - MAINTAIN exact wording, especially for classification names and method titles
        - DO NOT merge or simplify application methods under any circumstances
        - ENSURE all distinct methods like "Đặt câu hỏi với mỗi nhóm khách hàng", "Đánh vào tâm lý với mỗi nhóm khách hàng", etc. are listed separately
        - INCLUDE methods for collecting demographic information (e.g., age) even if they are not specific to a classification
        """
        
        logger.info("Using LLM to synthesize knowledge into comprehensive instructions")
        response = llm.invoke(synthesis_prompt,temperature=0.05)
        synthesized_knowledge = response.content if hasattr(response, 'content') else str(response)
        
        # Ensure we don't exceed max characters
        if len(synthesized_knowledge) > max_chars:
            synthesized_knowledge = synthesized_knowledge[:max_chars-3] + "..."
        
        return synthesized_knowledge
    
    except Exception as e:
        logger.error(f"Error in prepare_knowledge: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback: Return simple concatenation of knowledge entries
        fallback_content = ""
        for entry in knowledge_entries[:2]:  # Limit to 2 entries for brevity
            raw_text = entry.get("raw", "")
            if raw_text:
                fallback_content += f"--- KNOWLEDGE ---\n{raw_text[:1000]}\n\n"
        
        if len(fallback_content) > max_chars:
            fallback_content = fallback_content[:max_chars-3] + "..."
            
        return fallback_content

async def ensure_brain_loaded(graph_version_id: str) -> bool:
    """
    Ensure the brain is loaded with the correct graph version.
    
    Args:
        graph_version_id: The graph version ID to load into the brain
        
    Returns:
        bool: True if brain is loaded and ready, False otherwise
    """
    global _current_loaded_version, _last_load_time, _load_in_progress
    
    # Skip loading if no graph_version_id provided
    if not graph_version_id:
        logger.warning("No graph version ID provided, cannot load brain")
        return False
    
    try:
        # Avoid duplicate loading
        if _load_in_progress:
            logger.info(f"Brain loading already in progress, waiting for completion")
            await asyncio.sleep(0.5)  # Brief sleep to allow other tasks to progress
            return _current_loaded_version == graph_version_id
        
        # Get the current version from the brain
        current_version = get_current_graph_version()
        
        # Check if we need to load a new version
        if current_version == graph_version_id:
            # Already loaded with the correct version
            return True
        
        # If we've loaded recently, wait to avoid thrashing
        current_time = time.time()
        if current_time - _last_load_time < MIN_LOAD_INTERVAL:
            # If too recent, log but still proceed
            load_age = current_time - _last_load_time
            logger.info(f"Recent brain load ({load_age:.2f}s ago), proceeding but may be redundant")
        
        # Mark loading as in progress
        _load_in_progress = True
        
        try:
            # Get the brain instance with this version
            logger.info(f"Loading brain with graph version {graph_version_id}")
            brain_coroutine = get_brain(graph_version_id)
            
            # FIX: Check if the result is a coroutine and await it if needed
            brain = await brain_coroutine if asyncio.iscoroutine(brain_coroutine) else brain_coroutine
            
            if brain is None:
                logger.error("Failed to get brain instance")
                return False
            
            # Update tracking variables
            _current_loaded_version = graph_version_id
            _last_load_time = time.time()
            
            # Wait a moment for loading to complete in the background
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading brain with graph version {graph_version_id}: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            _load_in_progress = False
            
    except Exception as e:
        logger.error(f"Unexpected error in ensure_brain_loaded: {e}")
        logger.error(traceback.format_exc())
        return False

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    Enhanced detection for Vietnamese vs. English with more robust heuristics.
    
    Args:
        text: The text to analyze
        
    Returns:
        String indicating the detected language ("vietnamese" or "english")
    """
    try:
        # Vietnamese-specific characters
        vn_chars = set("ăâêôơưđáàảãạắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ")
        
        # Common Vietnamese words
        vn_words = [
            "anh", "tôi", "bạn", "bị", "của", "và", "là", "được", "có", "cho", 
            "một", "để", "trong", "người", "những", "không", "với", "các", "mình", 
            "này", "đã", "khi", "từ", "cách", "như", "thể", "nếu", "vì", "tại"
        ]
        
        # Clean text
        cleaned_text = text.lower().strip()
        
        # Check for Vietnamese characters (strong indicator)
        if any(char in vn_chars for char in cleaned_text):
            return "vietnamese"
        
        # Check for common Vietnamese words
        words = set(re.findall(r'\b\w+\b', cleaned_text))
        vn_word_matches = sum(1 for word in words if word in vn_words)
        
        # If multiple Vietnamese words are found
        if vn_word_matches >= 2:
            return "vietnamese"
            
        # Check for Vietnamese phrases (additional check)
        vn_phrases = ["tôi muốn", "anh bị", "em bị", "chúng tôi", "tôi cần", "xin chào"]
        if any(phrase in cleaned_text for phrase in vn_phrases):
            return "vietnamese"
            
        # Default to English if not detected as Vietnamese
        return "english"
    except Exception as e:
        logger.warning(f"Error in language detection: {str(e)}")
        return "english"  # Default to English on error

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
    
    # First, look for specifically bolded classifications in the portrait
    # These are the explicit classifications from prepare_knowledge
    bolded_classifications = re.findall(r'\*\*(.*?)\*\*', portrait)
    primary_classification = None
    
    if bolded_classifications:
        # Use the first bolded classification found
        primary_classification = bolded_classifications[0].strip()
        logger.info(f"Found bolded classification in portrait: {primary_classification}")
    
    # Determine the language based on the portrait
    is_vietnamese = False
    if "tiếng Việt" in portrait.lower() or "vietnamese" in portrait.lower() or any(char in "ăâêôơưđ" for char in portrait):
        is_vietnamese = True
        language_hint = f"Trả lời hoàn toàn bằng tiếng Việt. Tạo các câu truy vấn bằng tiếng Việt."
    else:
        language_hint = "Return the queries in English."
    
    # Default language detection if no profile is available
    if not portrait:
        is_vietnamese = True  # Default to Vietnamese for this application
    
    # If we already have a classification from bold markers, generate queries directly
    if primary_classification:
        if is_vietnamese:
            queries = [
                f"Hiểu về người dùng nhóm {primary_classification}",
                f"Phương pháp tiếp cận nhóm {primary_classification}",
                f"Những việc cần làm với nhóm {primary_classification}"
            ]
            logger.info(f"Generated queries using bolded classification: {primary_classification}")
            return queries
    
    # If no bolded classification was found, use LLM to extract it
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # If we have a primary classification from bold markers, instruct LLM to use it
    if primary_classification:
        instruction = f"Use ONLY this classification: {primary_classification}"
    else:
        instruction = "Extract ONLY the primary user classification (e.g., \"Chán Nản\", \"Tự Tin\", \"Lo Lắng\", etc.)"
    
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
        prompt += f"""
        1. "Hiểu về người dùng nhóm [phân loại người dùng]"
        2. "Phương pháp tiếp cận nhóm [phân loại người dùng]"
        3. "Những việc cần làm với nhóm [phân loại người dùng]"

        IMPORTANT INSTRUCTIONS:
        - {instruction}
        - Use the EXACT SAME classification term in all three queries
        - Keep the query patterns exactly as shown, only replacing [phân loại người dùng]
        - ENSURE ALL QUERIES ARE COMPLETELY IN VIETNAMESE
        """
    else:
        prompt += f"""
        1. "Understanding [user classification] users"
        2. "Best approaches for [user classification] users"
        3. "What to do next with [user classification] users"

        IMPORTANT INSTRUCTIONS:
        - {instruction}
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
        segment = primary_classification or user_profile.get("segment", {}).get("category", "general")
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
