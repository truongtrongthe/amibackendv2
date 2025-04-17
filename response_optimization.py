import re
import random
import nltk
from typing import Dict, List, Optional
from utilities import logger

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Enhanced language detection function with Vietnamese pronoun awareness
def detect_language(text: str) -> str:
    """Detect if text is likely non-English based on character patterns."""
    # Check for Vietnamese-specific characters
    vietnamese_chars = ['ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 
                    'á', 'à', 'ả', 'ã', 'ạ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
                    'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ',
                    'ế', 'ề', 'ể', 'ễ', 'ệ', 'í', 'ì', 'ỉ', 'ĩ', 'ị',
                    'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ',
                    'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ú', 'ù', 'ủ', 'ũ', 'ụ',
                    'ứ', 'ừ', 'ử', 'ữ', 'ự']
    
    # Basic check for presence of Vietnamese characters
    if any(char in text.lower() for char in vietnamese_chars):
        return "vi"
    
    # Common Vietnamese words check
    vietnamese_words = ['của', 'và', 'các', 'có', 'trong', 'không', 'với', 'là', 'để',
                        'người', 'những', 'được', 'trên', 'phải', 'nhiều', 'em', 'anh', 'chị', 
                        'cô', 'bác', 'chú', 'ông', 'bà', 'cháu', 'mình', 'bạn']
    
    if any(f" {word} " in f" {text.lower()} " for word in vietnamese_words):
        return "vi"
    
    # Check for Vietnamese pronoun patterns
    pronoun_patterns = [r'\bem\b', r'\banh\b', r'\bchị\b', r'\bcô\b', r'\bbác\b', r'\bchú\b']
    if any(re.search(pattern, text.lower()) for pattern in pronoun_patterns):
        return "vi"
    
    return "en"

# Detect Vietnamese relationship from text
def detect_vietnamese_relationship(text: str, context: Dict = None) -> Dict:
    """
    Detect the relationship dynamic in Vietnamese text based on pronouns.
    
    Returns a dictionary with:
    - speaker_role: How the speaker refers to themselves (anh, em, chị, etc.)
    - listener_role: How the listener is addressed
    - formality: formal/informal/neutral assessment
    """
    result = {
        "speaker_role": None,
        "listener_role": None, 
        "formality": "neutral"
    }
    
    # Common Vietnamese pronouns
    pronouns = {
        "formal": ["ông", "bà", "cô", "chú", "bác", "thầy", "cô"],
        "informal_older": ["anh", "chị"],
        "informal_younger": ["em", "cháu"],
        "neutral": ["bạn", "quý vị", "mọi người"]
    }
    
    # Check for direct usage of pronouns
    text_lower = text.lower()
    
    # First word detection - in Vietnamese, first-person pronouns often start sentences
    words = text_lower.split()
    if words and len(words) > 0:
        first_word = words[0].strip()
        
        # Common pattern: sentences that start with "anh"/"em" are usually self-referential
        if first_word == "anh":
            result["speaker_role"] = "anh"
            result["listener_role"] = "em"
            result["formality"] = "informal"
            logger.info(f"[RELATIONSHIP] Detected 'anh' as first word - likely self-reference")
            return result
            
        elif first_word == "em":
            result["speaker_role"] = "em"
            result["listener_role"] = "anh"
            result["formality"] = "informal"
            logger.info(f"[RELATIONSHIP] Detected 'em' as first word - likely self-reference")
            return result
            
        elif first_word == "chị":
            result["speaker_role"] = "chị"
            result["listener_role"] = "em"
            result["formality"] = "informal"
            return result
    
    # Look for phrases that strongly indicate relationships
    if "anh muốn" in text_lower or "anh cần" in text_lower or "anh bị" in text_lower or "anh đang" in text_lower:
        result["speaker_role"] = "anh"
        result["listener_role"] = "em"
        result["formality"] = "informal"
        logger.info(f"[RELATIONSHIP] Detected 'anh' with action verb - likely self-reference")
        return result
        
    elif "em muốn" in text_lower or "em cần" in text_lower or "em bị" in text_lower or "em đang" in text_lower:
        result["speaker_role"] = "em"
        result["listener_role"] = "anh"
        result["formality"] = "informal"
        logger.info(f"[RELATIONSHIP] Detected 'em' with action verb - likely self-reference")
        return result
    
    # Check for vocative usage (direct address)
    if re.search(r'(ơi|à|nhé)\s*$', text_lower) or re.search(r'^(chào|xin chào|hi|hello)\s+', text_lower):
        # Check what pronoun appears near vocative markers
        for pronoun in ["anh", "em", "chị", "bạn"]:
            pattern = fr'(chào|xin chào|hi|hello)\s+{pronoun}\b|\b{pronoun}\s+(ơi|à|nhé)'
            if re.search(pattern, text_lower):
                # If directly addressing the other person
                if pronoun == "anh":
                    result["speaker_role"] = "em"
                    result["listener_role"] = "anh"
                elif pronoun == "em":
                    result["speaker_role"] = "anh"
                    result["listener_role"] = "em"
                elif pronoun == "chị":
                    result["speaker_role"] = "em"
                    result["listener_role"] = "chị"
                result["formality"] = "informal"
                logger.info(f"[RELATIONSHIP] Detected vocative usage of '{pronoun}'")
                return result
    
    # Check against conversation history if available
    if context and "messages" in context:
        previous_relationship = None
        
        # Check if we've already established a relationship pattern
        for msg in context.get("messages", []):
            if isinstance(msg, dict) and "content" in msg:
                msg_content = msg.get("content", "").lower()
                
                # Common greeting phrases that establish relationships
                if "chào em" in msg_content or "em ơi" in msg_content:
                    previous_relationship = {"speaker_role": "anh", "listener_role": "em"}
                    break
                    
                elif "chào anh" in msg_content or "anh ơi" in msg_content:
                    previous_relationship = {"speaker_role": "em", "listener_role": "anh"}
                    break
        
        if previous_relationship:
            # Return the previously established relationship
            result["speaker_role"] = previous_relationship["speaker_role"]
            result["listener_role"] = previous_relationship["listener_role"]
            result["formality"] = "informal"
            logger.info(f"[RELATIONSHIP] Using previously established relationship pattern from conversation history")
            return result
    
    # Still no clear relationship, try pronoun frequency analysis
    pronoun_counts = {}
    for pronoun in ["anh", "em", "chị", "bạn"]:
        count = len(re.findall(fr'\b{pronoun}\b', text_lower))
        pronoun_counts[pronoun] = count
    
    # If "anh" appears more frequently, it's likely the speaker refers to themselves as "anh"
    if pronoun_counts.get("anh", 0) > 0 and pronoun_counts.get("anh", 0) >= pronoun_counts.get("em", 0):
        result["speaker_role"] = "anh"
        result["listener_role"] = "em"
        result["formality"] = "informal"
        logger.info(f"[RELATIONSHIP] Frequency analysis suggests 'anh' as speaker role")
        return result
        
    # If "em" appears more frequently, it's likely the speaker refers to themselves as "em"
    elif pronoun_counts.get("em", 0) > 0:
        result["speaker_role"] = "em"
        result["listener_role"] = "anh"
        result["formality"] = "informal"
        logger.info(f"[RELATIONSHIP] Frequency analysis suggests 'em' as speaker role")
        return result
    
    # Default to neutral if nothing else matches
    result["speaker_role"] = "neutral"
    result["listener_role"] = "neutral"
    result["formality"] = "neutral"
    
    logger.info(f"[RELATIONSHIP] No clear relationship detected, using neutral pronouns")
    return result

class ResponseFilter:
    """Filter and restructure responses to ensure they follow natural conversation patterns."""
    
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|greetings|xin chào|chào|hola|howdy)',
        r'^good (morning|afternoon|evening|day)',
        r'^(welcome|nice to meet you)',
        r'^(how are you|how\'s it going|what\'s up)',
    ]
    
    def __init__(self):
        # Compile greeting patterns
        self.greeting_regex = re.compile('|'.join(self.GREETING_PATTERNS), re.IGNORECASE)
        self.conversation_turn_count = 0
    
    def increment_turn(self):
        """Increment the conversation turn counter."""
        self.conversation_turn_count += 1
        
    def should_include_greeting(self) -> bool:
        """Determine if a greeting should be included based on conversation state."""
        # Only include greeting in the first turn
        return self.conversation_turn_count == 0
    
    def remove_greeting(self, text: str) -> str:
        """Remove greeting patterns from the beginning of a response."""
        if not text:
            return text
            
        # Check if the response starts with a greeting
        match = self.greeting_regex.match(text)
        if not match:
            return text
            
        # Find the end of the greeting (usually marked by punctuation or newline)
        greeting_end = match.end()
        
        # Look for the end of the sentence containing the greeting
        for i in range(greeting_end, min(greeting_end + 100, len(text))):
            if i >= len(text):
                break
            if text[i] in ['.', '!', '?', '\n']:
                greeting_end = i + 1
                break
        
        # Remove the greeting and trim whitespace
        filtered_text = text[greeting_end:].lstrip()
        
        # If we've emptied the text, return the original
        if not filtered_text:
            return text
            
        # Capitalize the first letter if needed
        if filtered_text and filtered_text[0].islower():
            filtered_text = filtered_text[0].upper() + filtered_text[1:]
            
        return filtered_text
    
    def apply_filters(self, text: str, include_greeting: Optional[bool] = None) -> str:
        """Apply all response filters based on conversation state."""
        if include_greeting is None:
            include_greeting = self.should_include_greeting()
            
        if not include_greeting:
            text = self.remove_greeting(text)
            
        return text


class ResponseStructure:
    """Defines structure templates for different conversation contexts and facilitates selection."""
    
    def __init__(self):
        # Define standard structure templates
        self.templates = {
            "first_contact": {
                "template": "introduction|core_response|follow_up",
                "description": "For first interactions with new users",
                "conditions": {
                    "is_first_message": 2.0,
                    "message_count_low": 1.0
                },
                "transitions": {
                    "introduction_to_core": ["first", "to start with", "let me", "I'd like to"],
                    "core_to_follow_up": ["also", "additionally", "furthermore", "by the way"]
                },
                "transitions_vi": {
                    "introduction_to_core": ["trước tiên", "đầu tiên", "để bắt đầu", "mình muốn"],
                    "core_to_follow_up": ["ngoài ra", "thêm nữa", "bên cạnh đó", "mặt khác"]
                }
            },
            "information_delivery": {
                "template": "context_acknowledgment|information|clarification",
                "description": "For providing factual information or answering questions",
                "conditions": {
                    "has_question": 1.5,
                    "knowledge_found": 1.0,
                    "looking_for_information": 1.0
                },
                "transitions": {
                    "context_to_information": ["to answer your question", "here's what I found", "the information you need is"],
                    "information_to_clarification": ["to clarify", "in other words", "to put it simply", "this means that"]
                },
                "transitions_vi": {
                    "context_to_information": ["để trả lời câu hỏi của bạn", "đây là thông tin mình tìm được", "thông tin bạn cần là"],
                    "information_to_clarification": ["để làm rõ hơn", "nói cách khác", "nói đơn giản là", "điều này có nghĩa là"]
                }
            },
            "empathetic_response": {
                "template": "empathy|understanding|assistance",
                "description": "For responding to emotional or personal situations",
                "conditions": {
                    "emotional_content": 1.5,
                    "personal_situation": 1.0,
                    "seeking_comfort": 1.0
                },
                "transitions": {
                    "empathy_to_understanding": ["I understand that", "this sounds like", "I can see how"],
                    "understanding_to_assistance": ["to help with this", "what might work is", "one approach is"]
                }
            },
            "follow_up": {
                "template": "previous_context|development|next_step",
                "description": "For continuing an ongoing conversation thread",
                "conditions": {
                    "conversation_continuing": 1.0,
                    "has_previous_context": 1.5,
                    "message_count_high": 1.0
                },
                "transitions": {
                    "previous_to_development": ["building on that", "continuing from", "following up on"],
                    "development_to_next": ["moving forward", "the next step", "from here"]
                }
            },
            "direct_response": {
                "template": "core_response",
                "description": "For straightforward responses needing no additional structure",
                "conditions": {
                    "simple_query": 1.0,
                    "short_expected_answer": 1.5,
                    "directive_request": 1.0
                },
                "transitions": {}
            }
        }
        
        # Define section-specific patterns
        self.section_patterns = {
            "introduction": [
                "Let me help with {topic}",
                "Regarding {topic}",
                "About your question on {topic}"
            ],
            "context_acknowledgment": [
                "I understand you're asking about {topic}",
                "You're looking for information on {topic}",
                "Regarding your question about {topic}"
            ],
            "empathy": [
                "I understand how {emotion} that can be",
                "That sounds {emotion}",
                "I can see why you'd feel {emotion} about that"
            ],
            "core_response": [
                "{content}",
                "Here's what you need to know: {content}",
                "{content}"
            ]
        }
    
    def score_template_for_context(self, template_name: str, context_analysis: Dict) -> float:
        """Score how appropriate a template is for the current context."""
        template = self.templates.get(template_name)
        if not template:
            return 0.0
            
        score = 0.0
        for condition, weight in template["conditions"].items():
            # Check various conditions based on context analysis
            if condition == "is_first_message" and context_analysis.get("message_count", 0) <= 1:
                score += weight
            elif condition == "message_count_low" and context_analysis.get("message_count", 0) < 5:
                score += weight
            elif condition == "message_count_high" and context_analysis.get("message_count", 0) > 5:
                score += weight
            elif condition == "has_question" and context_analysis.get("question_detected", False):
                score += weight
            elif condition == "knowledge_found" and context_analysis.get("knowledge_found", False):
                score += weight
            elif condition == "emotional_content" and context_analysis.get("emotional_content", 0) > 0.5:
                score += weight
            elif condition == "conversation_continuing" and context_analysis.get("message_count", 0) > 1:
                score += weight
            elif condition == "simple_query" and len(context_analysis.get("latest_message", "")) < 100:
                score += weight
            elif condition == "directive_request" and any(w in context_analysis.get("latest_message", "").lower() 
                                                       for w in ["can you", "could you", "please", "help me"]):
                score += weight
                
        return score
    
    def select_best_template(self, context_analysis: Dict) -> Dict:
        """Select the most appropriate template for the current context."""
        scores = {}
        for template_name in self.templates:
            scores[template_name] = self.score_template_for_context(template_name, context_analysis)
            
        # Default to direct_response if no clear winner or low scores
        if not scores or max(scores.values()) < 0.5:
            return self.templates.get("direct_response")
            
        best_template_name = max(scores, key=scores.get)
        return self.templates.get(best_template_name)
    
    def get_section_pattern(self, section_name: str, context: Dict) -> str:
        """Get a pattern for a specific section, appropriate to the context."""
        patterns = self.section_patterns.get(section_name, ["{content}"])
        
        # If no patterns available, return default
        if not patterns:
            return "{content}"
            
        # Select pattern based on context (for now, randomly)
        # In the future, this could be more sophisticated
        return random.choice(patterns)
    
    def get_transition_phrase(self, from_section: str, to_section: str, template: Dict, language: str = "en") -> str:
        """Get an appropriate transition phrase between sections."""
        # Check if there's a specific language-specific transition defined
        transition_key = f"{from_section}_to_{to_section}"
        
        # Use language-specific transitions if available
        if language == "vi" and "transitions_vi" in template:
            transitions = template.get("transitions_vi", {}).get(transition_key, [])
        else:
            transitions = template.get("transitions", {}).get(transition_key, [])
        
        # If no specific transition, use generic transitions based on language
        if not transitions:
            if language == "vi":
                generic_transitions = [
                    "ngoài ra", "thêm nữa", "bên cạnh đó", "tiếp theo", 
                    "sau đó", "kế tiếp", "cuối cùng", "một điều nữa"
                ]
            else:
                generic_transitions = [
                    "also", "additionally", "moreover", "furthermore",
                    "next", "then", "following that", "subsequently",
                    "on another note", "regarding", "as for"
                ]
            return random.choice(generic_transitions)
            
        return random.choice(transitions)


class ResponseProcessor:
    """Processes raw LLM responses to match desired response structures."""
    
    def __init__(self):
        self.response_structure = ResponseStructure()
        self.sentence_tokenizer = nltk.sent_tokenize
    
    def analyze_context(self, state: Dict, message: str) -> Dict:
        """Analyze the conversation context to inform structure selection."""
        messages = state.get("messages", [])
        
        analysis = {
            "message_count": len(messages),
            "latest_message": message,
            "question_detected": "?" in message,
            "knowledge_found": False,  # Will be populated from knowledge search results
            "emotional_content": 0.0,  # Will be inferred from message content
        }
        
        # Check for emotional content indicators
        emotional_terms = {
            "positive": ["happy", "excited", "glad", "great", "nice", "love", "like", "thanks", "appreciate"],
            "negative": ["sad", "upset", "angry", "disappointed", "frustrated", "annoyed", "worried", "concerned"]
        }
        
        # Simple emotion detection - can be enhanced with more sophisticated analysis
        message_lower = message.lower()
        emotional_score = 0.0
        
        # Check for emotional terms
        for term in emotional_terms["positive"] + emotional_terms["negative"]:
            if term in message_lower:
                emotional_score += 0.2
                
        # Check for exclamation marks (potential enthusiasm or frustration)
        if "!" in message:
            emotional_score += 0.1 * message.count("!")
        
        analysis["emotional_content"] = min(1.0, emotional_score)  # Cap at 1.0
        
        # Detect if message is seeking information
        info_seeking_phrases = ["what is", "how to", "can you explain", "tell me about", "i need to know", "do you know"]
        if any(phrase in message_lower for phrase in info_seeking_phrases):
            analysis["looking_for_information"] = True
        else:
            analysis["looking_for_information"] = False
            
        return analysis
    
    def update_context_with_knowledge(self, context_analysis: Dict, knowledge_found: bool) -> Dict:
        """Update the context analysis with knowledge search results."""
        context_analysis["knowledge_found"] = knowledge_found
        return context_analysis
    
    def segment_response(self, response: str) -> List[str]:
        """Segment a response into sentences for restructuring."""
        # Use NLTK for proper sentence tokenization
        sentences = self.sentence_tokenizer(response)
        return sentences
    
    def categorize_sentences(self, sentences: List[str]) -> Dict[str, List[str]]:
        """Categorize sentences into different section types."""
        categorized = {
            "greeting": [],
            "context_acknowledgment": [],
            "information": [],
            "question": [],
            "suggestion": [],
            "core_response": [],
            "clarification": [],
            "empathy": [],
            "closing": []
        }
        
        # Simple categorization based on sentence content and structure
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for greeting patterns
            if re.match(r'^(hi|hello|hey|welcome|greetings|chào)', sentence_lower):
                categorized["greeting"].append(sentence)
                continue
                
            # Check for questions
            if "?" in sentence:
                categorized["question"].append(sentence)
                continue
                
            # Check for context acknowledgment
            if any(phrase in sentence_lower for phrase in ["you asked", "your question", "you mentioned", "regarding your"]):
                categorized["context_acknowledgment"].append(sentence)
                continue
                
            # Check for empathetic statements
            if any(phrase in sentence_lower for phrase in ["understand", "sorry to hear", "that must be", "i know how"]):
                categorized["empathy"].append(sentence)
                continue
                
            # Check for suggestions
            if any(phrase in sentence_lower for phrase in ["suggest", "recommend", "try", "consider", "option", "alternative"]):
                categorized["suggestion"].append(sentence)
                continue
                
            # Check for clarifications
            if any(phrase in sentence_lower for phrase in ["in other words", "to clarify", "this means", "in simple terms"]):
                categorized["clarification"].append(sentence)
                continue
                
            # Check for closing statements
            if any(phrase in sentence_lower for phrase in ["hope this helps", "let me know", "anything else", "feel free to"]):
                categorized["closing"].append(sentence)
                continue
                
            # Default category for other sentences
            categorized["core_response"].append(sentence)
        
        return categorized
    
    def structure_response(self, raw_response: str, template: Dict, context: Dict) -> str:
        """Structure a response according to the selected template."""
        # Detect language of the response
        language = detect_language(raw_response)
        logger.info(f"[RESPONSE_STRUCTURE] Detected language: {language}")
        
        # For non-English responses, we'll be more conservative with restructuring
        if language != "en":
            # For non-English, just ensure proper sentence breaks without heavy restructuring
            sentences = self.segment_response(raw_response)
            return " ".join(sentences)
        
        # Standard response structuring for English
        # Segment the response into sentences
        sentences = self.segment_response(raw_response)
        
        # If we have no sentences, return the raw response
        if not sentences:
            return raw_response
            
        # Categorize sentences into different types
        categorized = self.categorize_sentences(sentences)
        
        # Get the structure template
        structure_parts = template["template"].split("|")
        
        # Assemble the structured response
        structured_response = []
        for i, section in enumerate(structure_parts):
            # Select sentences for this section
            section_sentences = categorized.get(section, [])
            
            # If no sentences available for this section, try to use core_response
            if not section_sentences and section != "core_response":
                if categorized.get("core_response"):
                    # Take one sentence from core_response
                    section_sentences = [categorized["core_response"].pop(0)]
            
            # If we still have no sentences, skip this section
            if not section_sentences:
                continue
                
            # Add a transition if this is not the first section
            if i > 0 and structured_response:
                prev_section = structure_parts[i-1]
                transition = self.response_structure.get_transition_phrase(prev_section, section, template, language)
                
                # Only add transition if it's meaningful and doesn't already exist in the text
                if transition and not any(transition.lower() in s.lower() for s in section_sentences):
                    # Ensure proper capitalization and spacing
                    first_char = section_sentences[0][0]
                    if first_char.isupper():
                        # If sentence starts with capital, add transition with comma
                        transition_cap = transition[0].upper() + transition[1:]
                        section_sentences[0] = f"{transition_cap}, {section_sentences[0][0].lower()}{section_sentences[0][1:]}"
                    else:
                        # Otherwise just add the transition
                        section_sentences[0] = f"{transition} {section_sentences[0]}"
            
            # Add the sentences to the response
            structured_response.extend(section_sentences)
        
        # If we have leftover core_response sentences, add them if the response is short
        if len(structured_response) < 3 and categorized.get("core_response"):
            structured_response.extend(categorized["core_response"])
        
        # Join everything together
        result = " ".join(structured_response)
        
        return result
    
    def enhance_linguistic_patterns(self, text: str, context: Dict) -> str:
        """Apply linguistic pattern enhancement to the response."""
        # Detect language
        language = detect_language(text)
        
        # Skip linguistic enhancements for non-English text
        if language != "en":
            return text
            
        # For English text, apply enhancements
        sentences = self.segment_response(text)
        
        if not sentences:
            return text
            
        # Convert some statements to questions for variety if appropriate
        if len(sentences) > 3 and not any("?" in s for s in sentences) and random.random() > 0.7:
            # Find a statement that could be converted to a question
            for i, sentence in enumerate(sentences):
                if i < len(sentences) - 1 and len(sentence) > 15 and not "?" in sentence:
                    # Simple conversion of a statement to a question
                    if sentence.lower().startswith("you can"):
                        sentences[i] = sentence.replace("You can", "Have you considered")
                        sentences[i] = sentences[i][:-1] + "?"
                        break
                    elif sentence.lower().startswith("it is"):
                        sentences[i] = sentence.replace("It is", "Isn't it")
                        sentences[i] = sentences[i][:-1] + "?"
                        break
        
        # Add varied sentence lengths
        is_long_response = len(sentences) > 5
        if is_long_response:
            # Add some shorter sentences for rhythm
            for i in range(1, len(sentences)):
                if len(sentences[i]) > 100 and random.random() > 0.7:
                    # Simplify a long sentence
                    parts = re.split(r'[,;]', sentences[i])
                    if len(parts) > 1:
                        sentences[i] = parts[0] + "."
        
        return " ".join(sentences)
    
    def process_response(self, raw_response: str, state: Dict, message: str, knowledge_found: bool = False) -> str:
        """Process a raw LLM response to optimize its structure and linguistic patterns."""
        # Detect language
        language = detect_language(raw_response)
        logger.info(f"[RESPONSE_PROCESSOR] Detected language: {language}")
        
        # For Vietnamese responses, apply cultural-specific processing
        if language == "vi":
            # Detect relationship dynamic
            relationship = detect_vietnamese_relationship(message, state)
            logger.info(f"[RESPONSE_PROCESSOR] Vietnamese relationship: {relationship}")
            
            # Apply pronoun consistency - this is a simple fix for the most common issue
            sentences = self.segment_response(raw_response)
            
            # Process Vietnamese text to ensure natural pronoun usage
            speaker_role = relationship["speaker_role"]
            listener_role = relationship["listener_role"]
            processed_sentences = []
            
            # Only proceed with special processing if we have a clear relationship
            if listener_role != "neutral":
                for i, sentence in enumerate(sentences):
                    # Fix spacing after punctuation but before new sentences
                    sentence = re.sub(r'([.!?])([A-Za-zÀ-ỹ])', r'\1 \2', sentence)
                    
                    # Add greeting if first sentence and doesn't already have one
                    if i == 0 and not any(greeting in sentence.lower() for greeting in ["chào", "xin chào", "hi"]):
                        # Different greeting based on detected relationship
                        if listener_role == "anh":
                            sentence = f"Chào anh! {sentence}"
                        elif listener_role == "em":
                            sentence = f"Chào em! {sentence}"
                        elif listener_role == "chị":
                            sentence = f"Chào chị! {sentence}"
                    
                    # Handle specific cases for different relationship dynamics
                    if listener_role == "anh":
                        # Replace generic "bạn" with "anh"
                        sentence = re.sub(r'\bbạn\b', 'anh', sentence)
                        
                        # Make sure AI refers to itself as "em"
                        sentence = re.sub(r'\btôi\b', 'em', sentence)
                        
                        # Change possessive forms
                        sentence = re.sub(r'\bcủa bạn\b', 'của anh', sentence)
                        sentence = re.sub(r'\bcủa tôi\b', 'của em', sentence)
                        
                    elif listener_role == "em":
                        # Replace generic "bạn" with "em"
                        sentence = re.sub(r'\bbạn\b', 'em', sentence)
                        
                        # Make sure AI refers to itself as "anh" or at least consistently
                        sentence = re.sub(r'\btôi\b', speaker_role if speaker_role != "neutral" else "tôi", sentence)
                        
                        # Change possessive forms
                        sentence = re.sub(r'\bcủa bạn\b', 'của em', sentence)
                        
                    # Add to processed sentences
                    processed_sentences.append(sentence)
                
                # Join with proper spacing
                processed_text = " ".join(processed_sentences)
                
                # Ensure proper spacing around punctuation again (in case we missed any)
                processed_text = re.sub(r'([.!?])([A-Za-zÀ-ỹ])', r'\1 \2', processed_text)
                
                # Remove any doubled up spaces
                processed_text = re.sub(r' +', ' ', processed_text)
                
                logger.info(f"[RESPONSE_PROCESSOR] Adapted Vietnamese response with {listener_role}/{speaker_role} relationship")
                return processed_text
            else:
                # If no clear relationship detected, just fix spacing/formatting without changing pronouns
                processed_text = " ".join(sentences)
                processed_text = re.sub(r'([.!?])([A-Za-zÀ-ỹ])', r'\1 \2', processed_text)
                processed_text = re.sub(r' +', ' ', processed_text)
                
                logger.info(f"[RESPONSE_PROCESSOR] No clear relationship detected, only fixed formatting")
                return processed_text
        
        # For English, apply standard processing
        # Analyze the conversation context
        context_analysis = self.analyze_context(state, message)
        
        # Update with knowledge search results
        context_analysis = self.update_context_with_knowledge(context_analysis, knowledge_found)
        
        # Select the best template for this context
        template = self.response_structure.select_best_template(context_analysis)
        logger.info(f"[RESPONSE_STRUCTURE] Selected template: {template['description']}")
        
        # Apply the template to structure the response
        structured_response = self.structure_response(raw_response, template, context_analysis)
        
        # Log the changes for analysis
        if len(raw_response) > 100:
            raw_preview = raw_response[:100] + "..."
            structured_preview = structured_response[:100] + "..."
            logger.info(f"[RESPONSE_STRUCTURE] Raw: {raw_preview}")
            logger.info(f"[RESPONSE_STRUCTURE] Structured: {structured_preview}")
        else:
            logger.info(f"[RESPONSE_STRUCTURE] Raw: {raw_response}")
            logger.info(f"[RESPONSE_STRUCTURE] Structured: {structured_response}")
        
        # Enhance linguistic patterns
        enhanced_response = self.enhance_linguistic_patterns(structured_response, context_analysis)
        
        return enhanced_response 