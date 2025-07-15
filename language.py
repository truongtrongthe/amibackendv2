"""
Language Detection Module

Lightweight module for detecting user language and providing response guidance.
Extracted from mc.py to avoid importing unnecessary dependencies.
"""

import asyncio
import json
import re
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Create LLM instance for language detection
# Using gpt-4o-mini for fast and cost-effective language detection
try:
    LANGUAGE_DETECTION_LLM = ChatOpenAI(model="gpt-4o-mini", streaming=False, temperature=0.1)
    logger.info("Language detection LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize language detection LLM: {e}")
    LANGUAGE_DETECTION_LLM = None


async def detect_language_with_llm(text: str, llm=None) -> Dict[str, Any]:
    """
    Use LLM to detect language and provide appropriate response guidance
    
    Args:
        text: The text to analyze for language detection
        llm: Optional LLM instance to use (defaults to LANGUAGE_DETECTION_LLM)
        
    Returns:
        Dict containing language info:
        - language: Language name in English
        - code: ISO 639-1 two-letter code
        - confidence: Confidence score 0.0-1.0
        - responseGuidance: Instructions for responding in that language
    """
    if llm is None:
        llm = LANGUAGE_DETECTION_LLM
    
    if llm is None:
        logger.warning("No LLM available for language detection")
        return {
            "language": "English",
            "code": "en",
            "confidence": 0.5,
            "responseGuidance": "Respond in a neutral, professional tone"
        }
    
    # For very short inputs, give LLM more context
    if len(text.strip()) < 10:
        context_prompt = (
            f"This is a very short text: '{text}'\n"
            f"Based on this limited sample, identify the most likely language.\n"
            f"Consider common greetings, questions, or expressions that might indicate the language.\n"
            f"Return your answer in this JSON format:\n"
            f"{{\n"
            f"  \"language\": \"[language name in English]\",\n"
            f"  \"code\": \"[ISO 639-1 two-letter code]\",\n"
            f"  \"confidence\": [0-1 value],\n"
            f"  \"responseGuidance\": \"[Brief guidance on responding appropriately in this language]\"\n"
            f"}}"
        )
    else:
        context_prompt = (
            f"Identify the language of this text: '{text}'\n"
            f"Analyze the text carefully, considering vocabulary, grammar, script, and cultural markers.\n"
            f"Return your answer in this JSON format:\n"
            f"{{\n"
            f"  \"language\": \"[language name in English]\",\n"
            f"  \"code\": \"[ISO 639-1 two-letter code]\",\n"
            f"  \"confidence\": [0-1 value],\n"
            f"  \"responseGuidance\": \"[Brief guidance on responding appropriately in this language]\"\n"
            f"}}"
        )
    
    try:
        response = await llm.ainvoke(context_prompt) if asyncio.iscoroutinefunction(llm.invoke) else llm.invoke(context_prompt)
        response_text = getattr(response, 'content', response).strip()
        
        # Extract JSON from response (handling cases where LLM adds extra text)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            lang_data = json.loads(json_str)
            
            # Validate required fields
            if all(k in lang_data for k in ["language", "code", "confidence", "responseGuidance"]):
                logger.info(f"Language detected: {lang_data['language']} ({lang_data['code']}) with confidence {lang_data['confidence']:.2f}")
                return lang_data
            
        # If we get here, something went wrong with the JSON
        logger.warning(f"Language detection returned invalid format: {response_text[:100]}...")
        return {
            "language": "English",
            "code": "en",
            "confidence": 0.5,
            "responseGuidance": "Respond in a neutral, professional tone"
        }
        
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}")
        # Fallback to English on any error
        return {
            "language": "English",
            "code": "en",
            "confidence": 0.5,
            "responseGuidance": "Respond in a neutral, professional tone"
        }


def get_language_code_mapping() -> Dict[str, str]:
    """
    Get mapping of language names to ISO codes
    
    Returns:
        Dict mapping language names (lowercase) to ISO 639-1 codes
    """
    return {
        "english": "en",
        "vietnamese": "vi",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "thai": "th",
        "indonesian": "id",
        "malay": "ms",
        "tagalog": "tl"
    }


def normalize_language_info(lang_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and validate language information
    
    Args:
        lang_info: Raw language info from detection
        
    Returns:
        Normalized language info with consistent format
    """
    language_mapping = get_language_code_mapping()
    
    # Normalize language name
    detected_language = lang_info.get("language", "English")
    language_code = lang_info.get("code", "en")
    
    # If code is missing, try to infer from language name
    if not language_code or language_code == "unknown":
        language_code = language_mapping.get(detected_language.lower(), "en")
    
    # If language name is missing, try to infer from code
    if not detected_language or detected_language == "Unknown":
        reverse_mapping = {v: k.title() for k, v in language_mapping.items()}
        detected_language = reverse_mapping.get(language_code.lower(), "English")
    
    return {
        "language": detected_language,
        "code": language_code.lower(),
        "confidence": float(lang_info.get("confidence", 0.5)),
        "responseGuidance": lang_info.get("responseGuidance", "Respond in a neutral, professional tone")
    }


# For backward compatibility, create a simple LanguageDetector class
class LanguageDetector:
    """Simple language detector class for compatibility with existing code"""
    
    def __init__(self):
        self.llm = LANGUAGE_DETECTION_LLM
        
    async def detect_language_with_llm(self, text: str, llm=None) -> Dict[str, Any]:
        """Detect language using LLM - wrapper for the standalone function"""
        return await detect_language_with_llm(text, llm or self.llm) 