"""
Knowledge-enhanced user profiling system.
This module provides functions for building comprehensive user profiles based on conversation context
and knowledge retrieved from the brain.
"""

import re
import time
import asyncio
import copy
import json
from typing import List, Dict, Any, Optional
from utilities import logger
from brain_singleton import get_brain
from tool_helpers import (
    build_user_profile,
    extract_structured_data_from_raw,
    detect_language,
    ensure_brain_loaded
)
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize the LangChain ChatOpenAI model
LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    streaming=False,
    response_format={"type": "json_object"}
)

def detect_vietnamese_language(text: str) -> bool:
    """
    More accurate detection of Vietnamese language in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Boolean indicating if text is likely Vietnamese
    """
    # Vietnamese-specific characters
    vn_chars = set("ăâêôơưđáàảãạắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ")
    
    # Common Vietnamese words (expanded list)
    vn_words = [
        "anh", "tôi", "bạn", "bị", "của", "và", "là", "được", "có", "cho", 
        "một", "để", "trong", "người", "những", "không", "với", "các", "mình", 
        "này", "đã", "khi", "từ", "cách", "như", "thể", "nếu", "vì", "tại",
        "xuất", "sớm", "rối", "loạn", "cương", "dương", "sinh", "lý", "khó", "khăn"
    ]
    
    # Check for Vietnamese characters (strongest indicator)
    text_lower = text.lower()
    if any(char in vn_chars for char in text):
        return True
    
    # Check for common Vietnamese words (more comprehensive)
    words = re.findall(r'\b\w+\b', text_lower)
    vn_word_count = sum(1 for word in words if word in vn_words)
    
    # Lowered threshold and check for common Vietnamese patterns
    if vn_word_count >= 1 or "anh bị" in text_lower or "em bị" in text_lower:
        return True
        
    return False


def extract_profile_rules_from_knowledge(knowledge_entries: List[Dict]) -> Dict:
    """
    Extract profiling rules and patterns from knowledge entries.
    
    Args:
        knowledge_entries: List of knowledge entry dictionaries
        
    Returns:
        Dictionary of extracted rules organized by profile dimension
    """
    rules = {
        "identity": [],
        "segment": [],
        "desires": [],
        "communication": [],
        "emotional_state": []
    }
    
    for entry in knowledge_entries:
        # Get all text fields that might contain rules
        raw_text = entry.get("raw", "")
        structured = entry.get("structured", {})
        title = structured.get("title", "")
        content = structured.get("content", "")
        
        # Combine text for more effective searching
        combined_text = f"{title} {content} {raw_text}"
        
        # Extract customer segments
        segment_patterns = [
            r'segment[s]?\s*(?:called|named|:)?\s*["\']?([^"\'\.]+)["\']?',
            r'user type[s]?\s*(?:called|named|:)?\s*["\']?([^"\'\.]+)["\']?',
            r'customer profile[s]?\s*(?:called|named|:)?\s*["\']?([^"\'\.]+)["\']?',
            r'(?:chán nản|tự tin|chưa rõ tâm lý)[^\.]*(?:segment|group|type)'
        ]
        
        for pattern in segment_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                if match.strip() and len(match.strip()) > 3:  # Basic filtering
                    rules["segment"].append(match.strip())
        
        # Extract sexual health indicators
        health_patterns = [
            r'sexual health indicator[s]?[:\s]+([^\.]+)',
            r'expertise level[s]?[:\s]+([^\.]+)',
            r'knowledge level[s]?[:\s]+([^\.]+)'
        ]
        
        for pattern in health_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                if match.strip():
                    rules["identity"].append(match.strip())
        
        # Extract desire detection methods
        desire_patterns = [
            r'(?:implicit|explicit) desire[s]?[:\s]+([^\.]+)',
            r'user need[s]?[:\s]+([^\.]+)',
            r'(?:desire|need) indicator[s]?[:\s]+([^\.]+)'
        ]
        
        for pattern in desire_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                if match.strip():
                    rules["desires"].append(match.strip())
        
        # Extract communication style indicators
        comm_patterns = [
            r'communication style[s]?[:\s]+([^\.]+)',
            r'language preference[s]?[:\s]+([^\.]+)',
            r'tone adjustment[s]?[:\s]+([^\.]+)'
        ]
        
        for pattern in comm_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                if match.strip():
                    rules["communication"].append(match.strip())
        
        # Extract emotional state indicators
        emotion_patterns = [
            r'emotion(?:al)? indicator[s]?[:\s]+([^\.]+)',
            r'emotional state[s]?[:\s]+([^\.]+)',
            r'urgency level[s]?[:\s]+([^\.]+)'
        ]
        
        for pattern in emotion_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                if match.strip():
                    rules["emotional_state"].append(match.strip())
    
    # Remove duplicates and clean up
    for category in rules:
        rules[category] = list(set(rules[category]))
        
    return rules


def generate_profiling_queries() -> Dict[str, List[str]]:
    """
    Generate specialized queries for fetching profile-related knowledge.
    Args:
        conversation_context: The full conversation history
        last_user_message: The most recent user message
        is_health_topic: Whether this is a health-related topic
        is_sexual_health_topic: Whether this is specifically about sexual health
        
    Returns:
        Dictionary of query categories and their corresponding queries
    """

    
    # Generate profile-specific queries
    profiling_queries = {
        "identity_detection": [
            "cách xây dựng chân dung khách hàng",
        ],
        "segment_detection": [
            "customer segmentation criteria",
            "Làm thế nào để phân nhóm khách hàng"
        ]
       
    }
    
    return profiling_queries


async def llm_based_profiling(conversation_context: str, profiling_knowledge: List[Dict] = None) -> Dict:
    """
    Advanced LLM-based profiling that utilizes knowledge retrieval to inform its analysis.
    This approach relies on LLM reasoning rather than embeddings or rules.
    
    Args:
        conversation_context: The full conversation text
        profiling_knowledge: Optional list of retrieved knowledge entries to inform profiling
        
    Returns:
        Dictionary with classified profile elements
    """
    try:
        # Prepare knowledge context if available
        knowledge_context = ""
        segment_info = []
        profiling_techniques = []
        emotional_state_indicators = []
        communication_styles = []
        
        if profiling_knowledge and len(profiling_knowledge) > 0:
            knowledge_context = "Use the following knowledge to inform your analysis:\n\n"
            
            for entry in profiling_knowledge:
                category = entry.get("category", "")
                raw_text = entry.get("raw", "")
                structured = entry.get("structured", {})
                
                # Extract the most relevant parts from the knowledge
                title = structured.get("title", "")
                content = structured.get("content", "")
                description = structured.get("description", "")
                
                # Combine text for analysis
                combined_text = f"{title} {content} {description} {raw_text}".lower()
                
                # Extract relevant information based on category
                if "segment" in category.lower():
                    # Extract potential customer segments
                    if "chán nản" in combined_text or "frustrated" in combined_text:
                        segment_info.append('"Chán Nản" (frustrated with performance)')
                    if "tự tin" in combined_text or "confident" in combined_text:
                        segment_info.append('"Tự Tin" (confident but seeking improvement)')
                    if "chưa rõ" in combined_text or "unclear" in combined_text:
                        segment_info.append('"Chưa Rõ Tâm Lý" (unclear psychological state)')
                
                # Extract profiling techniques
                if "identity" in category.lower():
                    profiling_techniques.append("Identity Assessment Techniques")
                if "desire" in category.lower():
                    profiling_techniques.append("Desire Detection Methods")
                if "emotion" in category.lower() or "state" in category.lower():
                    profiling_techniques.append("Emotional State Analysis")
                    # Extract emotional state indicators
                    emotions = re.findall(r'emotional indicators:?\s*([^\.]+)', combined_text, re.IGNORECASE)
                    if emotions:
                        emotional_state_indicators.extend([e.strip() for e in emotions[0].split(',')])
                if "communication" in category.lower():
                    profiling_techniques.append("Communication Style Analysis")
                    # Extract communication styles
                    styles = re.findall(r'communication styles:?\s*([^\.]+)', combined_text, re.IGNORECASE)
                    if styles:
                        communication_styles.extend([s.strip() for s in styles[0].split(',')])
                
                knowledge_context += f"--- {category.upper()} KNOWLEDGE ---\n"
                if title:
                    knowledge_context += f"Title: {title}\n"
                if content:
                    knowledge_context += f"Content: {content}\n"
                if description and len(description) < 300:  # Keep context manageable
                    knowledge_context += f"Description: {description}\n"
                knowledge_context += "\n"
        
        # Define the profiling prompt for LLM analysis
        prompt = f"""
        You are a Vietnamese sexual health expert who specializes in customer psychology.
        
        Analyze the following conversation to create a detailed psychological profile.
        
        {knowledge_context}
        
        CONVERSATION:
        {conversation_context}
        
        Based on the conversation and the provided knowledge, create a detailed user profile with the following elements:
        """
        
        # Add dynamic profiling instructions based on retrieved knowledge
        # 1. Customer Segment Analysis
        prompt += "\n1. Customer Segment Analysis:"
        if segment_info:
            prompt += f"\n   - Identify which segment the user belongs to from these options: {', '.join(set(segment_info))}."
        else:
            prompt += "\n   - Identify which segment the user belongs to: \"Chán Nản\" (frustrated with performance), \"Tự Tin\" (confident but seeking improvement), or \"Chưa Rõ Tâm Lý\" (unclear psychological state)."
        prompt += """
           - Provide evidence from the conversation supporting this classification.
           - Assign a confidence score (0-10) for this classification.
        """
        
        # 2. Desires and Needs Assessment
        prompt += """
        2. Desires and Needs Assessment:
           - List 2-3 explicit desires the user has directly stated.
           - Infer 2-3 implicit desires based on the conversation.
           - Rank these desires by priority (high, medium, low).
           - Base these on actual evidence from the conversation.
        """
        
        # 3. Emotional State Evaluation
        prompt += "\n3. Emotional State Evaluation:"
        if emotional_state_indicators:
            prompt += f"\n   - Identify the primary emotions the user is experiencing (consider these indicators: {', '.join(set(emotional_state_indicators))})."
        else:
            prompt += "\n   - Identify the primary emotions the user is experiencing (e.g., embarrassment, anxiety, frustration, confusion, hope)."
        prompt += """
           - Rate the urgency level of the user's emotional state (high, medium, low).
           - Support with specific language indicators from the conversation.
        """
        
        # 4. Communication Preferences
        prompt += "\n4. Communication Preferences:"
        if communication_styles:
            prompt += f"\n   - Determine the user's preferred communication style (consider: {', '.join(set(communication_styles))})."
        else:
            prompt += "\n   - Determine the user's preferred communication style (e.g., direct, empathetic, technical, simple)."
        prompt += """
           - Identify language preferences (Vietnamese, English, or mixed).
           - Recommend tone adjustments that would better engage this user.
        """
        
        # 5. Identity Assessment
        prompt += """
        5. Identity Assessment:
           - Determine language proficiency and preference.
           - Assess the user's knowledge level about sexual health (beginner, intermediate, advanced).
        """
        
        # Response format
        prompt += """
        Format your response as a JSON object with the following structure:
        {
            "segment": {
                "category": "",
                "evidence": "",
                "confidence": 0
            },
            "desires": {
                "explicit": [],
                "implicit": [],
                "priorities": {}
            },
            "emotional_state": {
                "current": [],
                "urgency": "",
                "evidence": ""
            },
            "communication": {
                "style_preferences": [],
                "language_preferences": [],
                "recommended_tone": ""
            },
            "identity": {
                "language": "",
                "expertise": ""
            }
        }
        
        Only return valid JSON without any additional text or explanation.
        """
        
        # Use the LangChain ChatOpenAI model
        try:
            # Properly use the LangChain ChatOpenAI model
            messages = [
                SystemMessage(content="You are an expert psychologist specializing in Vietnamese sexual health customer profiling."),
                HumanMessage(content=prompt)
            ]
            
            # Invoke the model
            response = await LLM.ainvoke(messages)
            
            # Extract the content from the response
            content = response.content
            
            # Parse the JSON
            try:
                profile_result = json.loads(content)
                return profile_result
            except json.JSONDecodeError as json_error:
                logger.error(f"Failed to parse LLM response as JSON: {str(json_error)}")
                logger.error(f"Raw response: {content}")
                return {}
                
        except Exception as llm_error:
            logger.error(f"Error invoking LangChain LLM: {str(llm_error)}")
            return {}
                
    except Exception as e:
        logger.error(f"Error in LLM-based profiling: {str(e)}")
        return {}

async def build_knowledge_enhanced_profile(conversation_context: str, last_user_message: str, graph_version_id: str) -> Dict:
    """
    Build a comprehensive user profile using knowledge from the brain and LLM analysis.
    
    Args:
        conversation_context: The full conversation history
        last_user_message: The most recent user message
        graph_version_id: The version ID of the knowledge graph to query
        
    Returns:
        Enhanced user profile with knowledge-backed insights
    """
    try:
        # Step 1: Initialize a basic profile structure
        base_profile = {
            "identity": {
                "language": "vi" if detect_vietnamese_language(last_user_message) else "en",
                "expertise": "beginner"
            },
            "segment": {
                "category": "general",
                "evidence": "",
                "confidence": 0
            },
            "desires": {
                "explicit": [],
                "implicit": [],
                "priority": "normal"
            },
            "communication": {
                "style_preferences": [],
                "tone_preferences": [],
                "language_preferences": []
            },
            "emotional_state": {
                "current": [],
                "urgency": "normal",
                "evidence": ""
            },
            "query_characteristics": {
                "type": "general",
                "complexity": "simple"
            },
            "meta": {
                "profile_version": "llm_knowledge_enhanced_v1",
                "enhancement_timestamp": time.time(),
                "detected_topics": []
            }
        }
        
                
        # Step 3: Generate knowledge queries for profiling areas
        profiling_queries = generate_profiling_queries()
        
        # Step 4: Retrieve profiling knowledge from brain
        profile_enhancement_data = []
        
        # Ensure brain is loaded
        brain_loaded = await ensure_brain_loaded(graph_version_id)
        if not brain_loaded:
            logger.warning("Brain loading failed, proceeding with LLM-only profiling")
            
            # Use LLM profiling without knowledge
            llm_profile = await llm_based_profiling(conversation_context)
            
            # Merge the LLM profile with our base profile structure
            if llm_profile:
                # Apply segment information
                if "segment" in llm_profile and "category" in llm_profile["segment"]:
                    base_profile["segment"]["category"] = llm_profile["segment"]["category"].lower()
                    base_profile["segment"]["evidence"] = llm_profile["segment"].get("evidence", "")
                    base_profile["segment"]["confidence"] = llm_profile["segment"].get("confidence", 0)
                
                # Apply desires information
                if "desires" in llm_profile:
                    if "explicit" in llm_profile["desires"]:
                        base_profile["desires"]["explicit"] = llm_profile["desires"]["explicit"]
                    if "implicit" in llm_profile["desires"]:
                        base_profile["desires"]["implicit"] = llm_profile["desires"]["implicit"]
                    
                    # Check if we have priority information
                    priorities = llm_profile["desires"].get("priorities", {})
                    if priorities and len(priorities) > 0:
                        # Find the highest priority
                        high_priority_items = [k for k, v in priorities.items() if v.lower() == "high"]
                        if high_priority_items:
                            base_profile["desires"]["priority"] = "high"
                
                # Apply emotional state information
                if "emotional_state" in llm_profile:
                    if "current" in llm_profile["emotional_state"]:
                        base_profile["emotional_state"]["current"] = llm_profile["emotional_state"]["current"]
                    if "urgency" in llm_profile["emotional_state"]:
                        base_profile["emotional_state"]["urgency"] = llm_profile["emotional_state"]["urgency"].lower()
                    if "evidence" in llm_profile["emotional_state"]:
                        base_profile["emotional_state"]["evidence"] = llm_profile["emotional_state"]["evidence"]
                
                # Apply communication preferences
                if "communication" in llm_profile:
                    if "style_preferences" in llm_profile["communication"]:
                        base_profile["communication"]["style_preferences"] = llm_profile["communication"]["style_preferences"]
                    if "language_preferences" in llm_profile["communication"]:
                        base_profile["communication"]["language_preferences"] = llm_profile["communication"]["language_preferences"]
                    if "recommended_tone" in llm_profile["communication"]:
                        base_profile["communication"]["tone_preferences"] = [llm_profile["communication"]["recommended_tone"]]
                
                # Apply identity information
                if "identity" in llm_profile:
                    if "language" in llm_profile["identity"]:
                        base_profile["identity"]["language"] = llm_profile["identity"]["language"].lower()
                    if "expertise" in llm_profile["identity"]:
                        base_profile["identity"]["expertise"] = llm_profile["identity"]["expertise"].lower()
            
            base_profile["meta"]["profile_method"] = "llm_only"
            return base_profile
        
        # Step 5: Get brain instance and fetch knowledge
        brain = get_brain()
        QUERY_TIMEOUT = 5  # 5 second timeout per category
        
        for category, queries in profiling_queries.items():
            try:
                # Use async with timeout to prevent blocking
                async with asyncio.timeout(QUERY_TIMEOUT):
                    # Use only the first query per category for speed
                    if queries:
                        primary_query = queries[0]
                        results = await brain.get_similar_vectors_by_text(primary_query, top_k=1)
                        
                        for vector_id, vector, metadata, similarity in results:
                            raw_text = metadata.get("raw", "")
                            structured_data = extract_structured_data_from_raw(raw_text)
                            
                            entry = {
                                "id": vector_id,
                                "category": category,
                                "query": primary_query,
                                "raw": raw_text,
                                "structured": structured_data,
                                "similarity": float(similarity)
                            }
                            profile_enhancement_data.append(entry)
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching profiling knowledge for {category}")
            except Exception as e:
                logger.error(f"Error fetching profiling knowledge for {category}: {str(e)}")
        
        # Step 6: Extract profile rules from knowledge entries
        profile_rules = extract_profile_rules_from_knowledge(profile_enhancement_data)
        
        # Log extracted rules for debugging
        logger.debug(f"Extracted profiling rules: {json.dumps(profile_rules, indent=2)}")
        
        # Step 7: Use the LLM to generate a profile with the retrieved knowledge
        llm_profile = await llm_based_profiling(conversation_context, profile_enhancement_data)
        
        # Step 8: Create the final enhanced profile
        final_profile = base_profile
        
        # Merge the LLM profile with our base profile structure
        if llm_profile:
            # Apply segment information
            if "segment" in llm_profile and "category" in llm_profile["segment"]:
                final_profile["segment"]["category"] = llm_profile["segment"]["category"].lower()
                final_profile["segment"]["evidence"] = llm_profile["segment"].get("evidence", "")
                final_profile["segment"]["confidence"] = llm_profile["segment"].get("confidence", 0)
            
            # Apply desires information
            if "desires" in llm_profile:
                if "explicit" in llm_profile["desires"]:
                    final_profile["desires"]["explicit"] = llm_profile["desires"]["explicit"]
                if "implicit" in llm_profile["desires"]:
                    final_profile["desires"]["implicit"] = llm_profile["desires"]["implicit"]
                
                # Check if we have priority information
                priorities = llm_profile["desires"].get("priorities", {})
                if priorities and len(priorities) > 0:
                    # Find the highest priority
                    high_priority_items = [k for k, v in priorities.items() if v.lower() == "high"]
                    if high_priority_items:
                        final_profile["desires"]["priority"] = "high"
            
            # Apply emotional state information
            if "emotional_state" in llm_profile:
                if "current" in llm_profile["emotional_state"]:
                    final_profile["emotional_state"]["current"] = llm_profile["emotional_state"]["current"]
                if "urgency" in llm_profile["emotional_state"]:
                    final_profile["emotional_state"]["urgency"] = llm_profile["emotional_state"]["urgency"].lower()
                if "evidence" in llm_profile["emotional_state"]:
                    final_profile["emotional_state"]["evidence"] = llm_profile["emotional_state"]["evidence"]
            
            # Apply communication preferences
            if "communication" in llm_profile:
                if "style_preferences" in llm_profile["communication"]:
                    final_profile["communication"]["style_preferences"] = llm_profile["communication"]["style_preferences"]
                if "language_preferences" in llm_profile["communication"]:
                    final_profile["communication"]["language_preferences"] = llm_profile["communication"]["language_preferences"]
                if "recommended_tone" in llm_profile["communication"]:
                    final_profile["communication"]["tone_preferences"] = [llm_profile["communication"]["recommended_tone"]]
            
            # Apply identity information
            if "identity" in llm_profile:
                if "language" in llm_profile["identity"]:
                    final_profile["identity"]["language"] = llm_profile["identity"]["language"].lower()
                if "expertise" in llm_profile["identity"]:
                    final_profile["identity"]["expertise"] = llm_profile["identity"]["expertise"].lower()
        
        # Step 9: Finalize metadata
        final_profile["meta"]["knowledge_sources"] = len(profile_enhancement_data)
        final_profile["meta"]["profile_method"] = "llm_knowledge_enhanced"
        
        # Add profile rules to metadata
        final_profile["meta"]["profile_rules"] = profile_rules
        
        # Return the enhanced profile
        return final_profile
        
    except Exception as e:
        logger.error(f"Error in knowledge-enhanced profiling: {str(e)}")
        
        # Create fallback profile
        fallback_profile = {
            "identity": {
                "language": detect_vietnamese_language(last_user_message) and "vi" or "en",
                "expertise": "beginner"
            },
            "segment": {
                "category": "general",
                "evidence": "",
                "confidence": 0
            },
            "desires": {
                "explicit": [],
                "implicit": [],
                "priority": "normal"
            },
            "communication": {
                "style_preferences": [],
                "tone_preferences": [],
                "language_preferences": []
            },
            "emotional_state": {
                "current": [],
                "urgency": "normal",
                "evidence": ""
            },
            "query_characteristics": {
                "type": "general",
                "complexity": "simple"
            },
            "meta": {
                "profile_version": "fallback_v1",
                "fallback_reason": str(e),
                "profile_rules": {
                    "identity": [],
                    "segment": [],
                    "desires": [],
                    "communication": [],
                    "emotional_state": []
                }
            }
        }
        
        # Basic fallback detection
        if "xuất tinh sớm" in last_user_message.lower() or "xuất sớm" in last_user_message.lower():
            fallback_profile["segment"]["category"] = "chán nản"
            fallback_profile["emotional_state"]["current"].append("embarrassed")
            fallback_profile["desires"]["implicit"].append("cải thiện thời gian quan hệ")
        
        try:
            # Try emergency LLM profiling without knowledge
            emergency_profile = await llm_based_profiling(last_user_message)
            if emergency_profile:
                # Just take what we can get in an emergency
                return {**fallback_profile, **emergency_profile}
        except:
            pass
            
        return fallback_profile

def format_enhanced_profile_for_cot(user_profile: Dict) -> str:
    """
    Format the enhanced user profile for inclusion in the Chain of Thought prompt.
    
    Args:
        user_profile: The enhanced user profile
        
    Returns:
        Formatted profile summary string
    """
    # Format the enhanced user profile for the prompt with new fields
    profile_summary = f"""
    USER PROFILE:
    - Identity: {user_profile['identity']['expertise']} level user, language: {user_profile['identity']['language']}
    - Segment: {user_profile['segment']['category']}"""

    # Add segment evidence if available
    if user_profile.get("segment", {}).get("evidence"):
        profile_summary += f"\n    - Segment Evidence: {user_profile['segment']['evidence'][:100]}..."

    # Add communication preferences
    profile_summary += "\n    - Communication Preferences:"
    if user_profile.get("communication", {}).get("style_preferences"):
        styles = ", ".join(user_profile["communication"]["style_preferences"])
        profile_summary += f" Styles: {styles};"
    
    if user_profile.get("communication", {}).get("tone_preferences"):
        tones = ", ".join(user_profile["communication"]["tone_preferences"])
        profile_summary += f" Tones: {tones};"
        
    if user_profile.get("communication", {}).get("language_preferences"):
        langs = ", ".join(user_profile["communication"]["language_preferences"])
        profile_summary += f" Languages: {langs}"

    # Add emotional state
    profile_summary += "\n    - Emotional State:"
    if user_profile.get("emotional_state", {}).get("current"):
        emotions = ", ".join(user_profile["emotional_state"]["current"])
        profile_summary += f" {emotions},"
    profile_summary += f" urgency: {user_profile['emotional_state']['urgency']}"

    # Add desires
    if user_profile.get("desires"):
        explicit_desires = ', '.join(user_profile["desires"]["explicit"][:2]) if user_profile["desires"]["explicit"] else "none explicit"
        implicit_desires = ', '.join(user_profile["desires"]["implicit"][:2]) if user_profile["desires"]["implicit"] else "none detected"
        profile_summary += f"\n    - User Desires: Explicit: {explicit_desires}; Implicit: {implicit_desires}; Priority: {user_profile['desires']['priority']}"
    
    # Add profile rules if present in metadata
    if user_profile.get("meta", {}).get("profile_rules"):
        profile_summary += "\n    - Profile Knowledge Rules:"
        rules = user_profile["meta"]["profile_rules"]
        
        # Add most important rules for each dimension
        for dimension, rule_list in rules.items():
            if rule_list and len(rule_list) > 0:
                # Include at most 2 rules per dimension
                sample_rules = rule_list[:2]
                profile_summary += f"\n      * {dimension.capitalize()}: {', '.join(sample_rules)}"
    
    return profile_summary 