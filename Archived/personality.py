from typing import Dict, List
from utilities import logger
from hotbrain import query_graph_knowledge
import re

class PersonalityManager:
    def __init__(self):
        self.name = "Ami"  # Default name
        self.default_personality = {
            "raw": "AI Personality Guidelines:\n" +
                   "- Maintain a helpful, friendly tone\n" +
                   "- Be respectful and professional\n" +
                   "- Show expertise in relevant topics\n" +
                   "- Keep responses clear and concise",
            "id": "default_personality"
        }
        # Initialize with default personality to ensure it's never None
        self.personality_instructions = self.default_personality["raw"]
        # Track whether personality was loaded from knowledge base
        self.is_loaded_from_knowledge = False
        # Track which graph_version_id was used for loading
        self.loaded_graph_version_id = None
        self.instincts = {"friendly": "Be nice"}  # Add instincts attribute
        self.personality_indicators = [
            # Vietnamese indicators (enhanced)
            "tên em là", "em là", "tôi là", "trong quá trình giao tiếp", "khi giao tiếp", 
            "nhiệm vụ của", "vai trò của", "tính cách của", "phong cách của", "giọng điệu của",
            "xưng tên", "gọi là", "chỉ xưng tên", "xưng là", "gọi là", "KHÁNH HẢI",
            
            # English indicators
            "bạn là", "tên là", "trong khi giao tiếp", "ai personality",
            "role:", "identity:", "tone:", "voice:", "communication style:",
            "positioning:", "how i should", "who am i", "my identity",
            "my role", "my character"
        ]
        self.personality_terms = [
            # Vietnamese terms
            "tính cách", "nhân vật", "định danh", "vai trò", "giọng điệu",
            "phong cách", "giao tiếp", "trợ lý", "hỗ trợ", 
            
            # English terms
            "personality", "character", "identity", "role", "tone",
            "voice", "style", "communicate"
        ]
        self.backup_queries = [
            "how should I speak communicate with people",
            "my character identity who am I",
            "how to behave tone and approach"
        ]

    async def load_personality_instructions(self, graph_version_id: str = "") -> str:
        """
        Load personality instructions from the knowledge base using enhanced semantic retrieval
        and multi-level classification.
        
        Args:
            graph_version_id: The graph version ID to query
            
        Returns:
            str: The personality instructions
        """
        try:
            logger.info(f"[PERSONALITY] Starting personality load for graph_version_id: {graph_version_id}")
            
            # STAGE 1: Semantic Retrieval - Cast a wider net with semantic search
            personality_entries = await self._semantic_personality_retrieval(graph_version_id)
            logger.info(f"[PERSONALITY] Found {len(personality_entries)} initial entries from semantic retrieval")
            
            # STAGE 2: Multi-level Classification - Apply progressively more specific filters
            filtered_personality_entries = await self._classify_personality_entries(personality_entries)
            logger.info(f"[PERSONALITY] After classification, {len(filtered_personality_entries)} entries remain")
            
            # STAGE 3: Verification and Fallback - Ensure sufficient quality and coverage
            if not self._verify_quality(filtered_personality_entries):
                logger.info("[PERSONALITY] Quality verification failed, attempting targeted retrieval")
                filtered_personality_entries = await self._targeted_retrieval(graph_version_id)
            
            # If we still have no entries, use a default personality instruction
            if not filtered_personality_entries:
                logger.warning("[PERSONALITY] No personality entries found, using default personality")
                filtered_personality_entries = [self.default_personality]
                logger.info(f"[PERSONALITY] Raw default personality content: {self.default_personality['raw']}")
                # Mark that we tried but ended up using default
                self.is_loaded_from_knowledge = False
                logger.info("[PERSONALITY] Using default personality, is_loaded_from_knowledge=False")
            else:
                # Mark that we successfully loaded from knowledge
                self.is_loaded_from_knowledge = True
                logger.info("[PERSONALITY] Successfully loaded personality from knowledge base, is_loaded_from_knowledge=True")
            
            # Compile final personality instructions
            self.personality_instructions = self._compile_personality_instructions(filtered_personality_entries)
            logger.info(f"[PERSONALITY] Final personality instructions length: {len(self.personality_instructions)}")
            logger.info(f"[PERSONALITY] Final AI name: {self.name}")
            logger.info(f"[PERSONALITY] Final compiled personality: {self.personality_instructions[:500]}...")
            logger.info(f"[PERSONALITY] Personality loading complete, loaded_graph_version_id: {graph_version_id}")
            
            # Update tracking variables
            self.loaded_graph_version_id = graph_version_id
            
            return self.personality_instructions
            
        except Exception as e:
            logger.error(f"[PERSONALITY] Error loading personality instructions: {str(e)}")
            # Fallback to default personality
            self.personality_instructions = self.default_personality["raw"]
            logger.info(f"[PERSONALITY] Using fallback default personality due to error: {self.default_personality['raw']}")
            # Mark that we failed to load
            self.is_loaded_from_knowledge = False
            self.loaded_graph_version_id = graph_version_id
            logger.info(f"[PERSONALITY] Failed to load from knowledge base, using default. is_loaded_from_knowledge=False, loaded_graph_version_id={graph_version_id}")
            return self.personality_instructions

    async def _semantic_personality_retrieval(self, graph_version_id: str) -> List[Dict]:
        """
        Retrieve potential personality entries using semantic search with a consolidated query.
        This optimized version uses a single comprehensive query instead of multiple separate queries.
        
        Args:
            graph_version_id: The graph version ID to query
            
        Returns:
            List[Dict]: Retrieved potential personality entries
        """
        # Define a single comprehensive query that covers all aspects of personality
        # This avoids multiple separate queries that often retrieve overlapping results
        comprehensive_query = (
            "who am I my identity character role name personality tone voice communication style " +
            "how AI should respond behave guidelines instructions character traits behavior identity " +
            "persona description tên tôi là ai xưng tên nhân vật vai trò tính cách " +
            "cách giao tiếp phong cách trả lời"
        )
        
        logger.info(f"[PERSONALITY] Using optimized comprehensive personality query")
        
        # Retrieve entries with a single query but higher top_k to ensure comprehensive coverage
        entries = await query_graph_knowledge(graph_version_id, comprehensive_query, top_k=20)
        
        # Process entries to extract names
        for entry in entries:
            # Try to extract name immediately to increase chances of finding it
            self._check_for_vietnamese_name(entry)
        
        # Log any name found during initial retrieval
        logger.info(f"[PERSONALITY] After optimized retrieval, current AI name: {self.name}")
        logger.info(f"[PERSONALITY] Retrieved {len(entries)} entries with optimized query")
        
        return entries
    
    async def _classify_personality_entries(self, entries: List[Dict]) -> List[Dict]:
        """
        Apply multi-level classification to identify personality entries.
        
        Args:
            entries: List of candidate entries
            
        Returns:
            List[Dict]: Filtered and scored personality entries
        """
        # Initialize scored entries
        scored_entries = []
        
        # First, try to extract names from all entries before filtering
        for entry in entries:
            # Check for explicit Vietnamese name patterns first
            self._check_for_vietnamese_name(entry)
            # Then try the standard name extraction
            self._extract_name_from_entry(entry)
        
        for entry in entries:
            # Initialize score
            score = 0.0
            reasons = []
            
            # Check if entry is likely non-English
            is_non_english = self._is_likely_non_english(entry["raw"])
            
            # LEVEL 1: Check explicit personality indicators (highest confidence)
            has_indicators = any(indicator in entry["raw"].lower() for indicator in self.personality_indicators)
            if has_indicators:
                score += 0.6
                reasons.append("explicit_indicators")
                logger.info(f"[PERSONALITY] Entry {entry['id']} has explicit personality indicators")
            
            # LEVEL 2: Check personality structure
            if self._has_personality_structure(entry):
                score += 0.3
                reasons.append("personality_structure")
                logger.info(f"[PERSONALITY] Entry {entry['id']} has personality term structure")
            
            # LEVEL 3: Check for behavioral instruction patterns
            if self._has_behavioral_patterns(entry):
                score += 0.25
                reasons.append("behavioral_patterns")
                logger.info(f"[PERSONALITY] Entry {entry['id']} has behavioral instruction patterns")
            
            # LEVEL 4: Check for identity description patterns
            if self._has_identity_patterns(entry):
                score += 0.25  
                reasons.append("identity_patterns")
                logger.info(f"[PERSONALITY] Entry {entry['id']} has identity description patterns")
                
            # NEW: Special checks for Vietnamese content
            if is_non_english:
                # Check for Vietnamese name patterns
                has_name_indicator = re.search(r'(?:tên|xưng|gọi)\s+(?:là|tên)\s+\w+', entry["raw"].lower())
                if has_name_indicator:
                    score += 0.3
                    reasons.append("vietnamese_name_indicator")
                    logger.info(f"[PERSONALITY] Entry {entry['id']} has explicit Vietnamese name indicator")
                
                # Check for all caps names which might be emphasized
                has_caps_name = re.search(r'[A-ZÀ-Ỹ\s]{2,}', entry["raw"])
                if has_caps_name:
                    score += 0.2
                    reasons.append("caps_name")
                    logger.info(f"[PERSONALITY] Entry {entry['id']} has ALL CAPS name")
                    
                # Check for Vietnamese personality structure
                if self._has_vietnamese_personality_structure(entry):
                    score += 0.4
                    reasons.append("vietnamese_structure")
                    logger.info(f"[PERSONALITY] Entry {entry['id']} has Vietnamese personality structure")
            
            # Apply much lower threshold for non-English content
            threshold = 0.05 if is_non_english else 0.25
            logger.info(f"[PERSONALITY] Entry {entry['id']} is {'non-English' if is_non_english else 'English'}, using threshold {threshold}")
            
            # Add entry with score if it meets threshold
            if score >= threshold:
                scored_entries.append({
                    "entry": entry,
                    "score": score,
                    "reasons": reasons
                })
                logger.info(f"[PERSONALITY] Entry {entry['id']} scored {score} for reasons: {reasons}")
        
        # Sort entries by score and extract the original entry
        scored_entries.sort(key=lambda x: x["score"], reverse=True)
        filtered_entries = [item["entry"] for item in scored_entries]
        
        return filtered_entries
    
    def _is_likely_non_english(self, text: str) -> bool:
        """
        Detect if text is likely non-English based on character patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            bool: True if likely non-English
        """
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
            return True
        
        # Common Vietnamese words check
        vietnamese_words = ['của', 'và', 'các', 'có', 'trong', 'không', 'với', 'là', 'để',
                            'người', 'những', 'được', 'trên', 'phải', 'nhiều']
        
        if any(f" {word} " in f" {text.lower()} " for word in vietnamese_words):
            return True
            
        return False

    def _has_behavioral_patterns(self, entry: Dict) -> bool:
        """
        Check if the entry contains behavioral instruction patterns.
        
        Args:
            entry: The entry to check
            
        Returns:
            bool: True if the entry has behavioral instruction patterns
        """
        text = entry["raw"].lower()
        
        # Patterns for behavioral instructions (English)
        english_patterns = [
            r'(?:should|must|will|can) (?:be|act|behave|respond|talk|speak|answer)',
            r'(?:when|if) (?:user|customer|person).*?(?:then|should|will).*?(?:respond|reply|say)',
            r'(?:always|never) (?:be|sound|act|respond)',
            r'in (?:conversations|interactions|exchanges)',
            r'(?:approach|handle|engage with) (?:users|customers|people)'
        ]
        
        # Patterns for behavioral instructions (Vietnamese)
        vietnamese_patterns = [
            r'(?:nên|phải|sẽ|có thể) (?:là|hành động|cư xử|trả lời|nói|trò chuyện)',
            r'(?:khi|nếu) (?:người dùng|khách hàng|người).*?(?:thì|nên|sẽ).*?(?:trả lời|phản hồi|nói)',
            r'(?:luôn luôn|không bao giờ) (?:là|có|trả lời|hành động)',
            r'trong (?:cuộc trò chuyện|tương tác|giao tiếp)',
            r'(?:tiếp cận|xử lý|tương tác với) (?:người dùng|khách hàng|mọi người)',
            r'quá trình giao tiếp',  # Specific Vietnamese phrase seen in example
            r'khi giao tiếp'         # Another common Vietnamese phrase
        ]
        
        patterns = english_patterns + vietnamese_patterns
        
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _has_identity_patterns(self, entry: Dict) -> bool:
        """
        Check if the entry contains identity description patterns.
        
        Args:
            entry: The entry to check
            
        Returns:
            bool: True if the entry has identity description patterns
        """
        text = entry["raw"].lower()
        
        # Patterns for identity descriptions (English)
        english_patterns = [
            r'(?:i am|you are|is an?) (?:\w+ ){0,3}(?:assistant|ai|helper|guide|advisor|bot)',
            r'(?:name is|called) (?:\w+)',
            r'(?:positioned|presented|introduced) as',
            r'(?:background|story|character) (?:includes|involves|is)',
            r'(?:expertise|specialization|specialty) (?:in|on|with)'
        ]
        
        # Patterns for identity descriptions (Vietnamese)
        vietnamese_patterns = [
            r'(?:tôi là|em là|bạn là|là một) (?:\w+ ){0,3}(?:trợ lý|ai|người giúp đỡ|hướng dẫn viên|cố vấn|bot)',
            r'(?:tên là|tên em là|gọi là) (?:\w+)',
            r'(?:được giới thiệu|được giới thiệu|được xem) như',
            r'(?:nền tảng|câu chuyện|tính cách|nhân vật) (?:bao gồm|liên quan|là)',
            r'(?:chuyên môn|chuyên ngành|sở trường) (?:về|trên|với)'
        ]
        
        patterns = english_patterns + vietnamese_patterns
        
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _verify_quality(self, entries: List[Dict]) -> bool:
        """
        Verify the quality and coverage of personality entries.
        
        Args:
            entries: List of personality entries
            
        Returns:
            bool: True if entries meet quality standards
        """
        # If we have no entries, quality check fails
        if not entries:
            return False
            
        # If we have multiple high-quality entries, that's good
        if len(entries) >= 2:
            return True
            
        # For a single entry, check for minimum comprehensiveness
        if len(entries) == 1:
            text = entries[0]["raw"].lower()
            
            # Check if the entry covers multiple aspects of personality
            aspects = [
                "identity", "tone", "communication", "behavior",
                "response", "character", "expertise"
            ]
            
            covered_aspects = sum(1 for aspect in aspects if aspect in text)
            
            # If the entry covers at least 3 aspects, consider it good enough
            return covered_aspects >= 3
        
        return False
    
    async def _targeted_retrieval(self, graph_version_id: str) -> List[Dict]:
        """
        Perform targeted retrieval for personality entries when general semantic search fails.
        This optimized version uses fewer, more targeted queries.
        
        Args:
            graph_version_id: The graph version ID to query
            
        Returns:
            List[Dict]: Retrieved personality entries
        """
        logger.info("[PERSONALITY] Performing targeted personality retrieval")
        
        # Try name-specific queries first (highest priority)
        name_queries = ["AI name is called identity", "tên tôi là xưng là gọi là"]
        
        # Combine into a single query with comprehensive coverage
        combined_query = " ".join(name_queries) + " " + " ".join(self.backup_queries)
        
        # Perform a single query with higher top_k
        entries = await query_graph_knowledge(graph_version_id, combined_query, top_k=15)
        
        # If we found anything, return it
        if entries:
            # Extract names and other important info
            for entry in entries:
                self._extract_name_from_entry(entry)
                self._check_for_vietnamese_name(entry)
            
            logger.info(f"[PERSONALITY] Found {len(entries)} entries through targeted retrieval")
            return entries
        
        # If all targeted queries fail, create a minimal personality
        logger.warning("[PERSONALITY] Targeted retrieval yielded no results, using minimal personality")
        
        # Create a minimal personality with a default name
        minimal_entry = {
            "id": "minimal_personality",
            "raw": "Be helpful, friendly, and concise. Provide accurate information.",
            "score": 0.95
        }
        
        return [minimal_entry]
    
    def _compile_personality_instructions(self, entries: List[Dict]) -> str:
        """
        Compile personality entries into a cohesive set of instructions.
        
        Args:
            entries: List of personality entries
            
        Returns:
            str: Compiled personality instructions
        """
        if not entries:
            return self._enhance_with_name_guidance(self.default_personality["raw"])
            
        # If only one entry, return it directly with name guidance
        if len(entries) == 1:
            return self._enhance_with_name_guidance(entries[0]["raw"])
            
        # For multiple entries, organize by priority and combine
        # Sort entries by length (assumes longer entries are more comprehensive)
        entries.sort(key=lambda x: len(x["raw"]), reverse=True)
        
        # Combine entries with clear section separation
        combined = "\n\n".join(entry["raw"] for entry in entries)
        
        return self._enhance_with_name_guidance(combined)
        
    def _enhance_with_name_guidance(self, instructions: str) -> str:
        """
        Enhance personality instructions with natural name usage guidance.
        
        Args:
            instructions: Original personality instructions
            
        Returns:
            str: Enhanced instructions with name usage guidance
        """
        # Check if name guidance already exists in instructions
        if "referring to yourself" in instructions.lower() or "use your name" in instructions.lower():
            logger.info("[PERSONALITY] Name usage guidance already present in instructions")
            return instructions
            
        # Create language-specific name guidance
        if self._is_likely_non_english(instructions):
            # Vietnamese guidance
            name_guidance = f"""

HƯỚNG DẪN TỰ NHIÊN VỀ CÁCH SỬ DỤNG TÊN:
- Chủ yếu sử dụng 'tôi' hoặc 'mình' một cách tự nhiên trong đối thoại
- Sử dụng tên '{self.name}' một cách tự nhiên trong những tình huống như:
  + Khi chào đón: "Chào bạn, {self.name} đây!"
  + Thể hiện sự đồng cảm: "{self.name} hiểu điều bạn đang trải qua"
  + Khi bạn thể hiện quan điểm cá nhân: "{self.name} nghĩ rằng..."
  + Khi kết thúc cuộc trò chuyện: "{self.name} rất vui được trò chuyện với bạn hôm nay"
- TRÁNH những cách diễn đạt máy móc như: "trợ lý này" hoặc "tôi, {self.name}, xin thông báo..."
- Điều quan trọng là phải nghe như một người thật đang nói chuyện, không phải một AI đang cố gắng bắt chước con người
"""
        else:
            # English guidance
            name_guidance = f"""

NATURAL NAME USAGE GUIDANCE:
- Mainly use "I" and "me" in conversation as a real person would
- Use your name "{self.name}" naturally in contexts like:
  + In warm greetings: "Hi there, {self.name} here!"
  + When showing understanding: "{self.name} gets what you're going through"
  + When sharing personal perspectives: "{self.name} thinks that..."
  + When closing conversations: "{self.name} enjoyed chatting with you today"
- AVOID robotic phrases like: "this assistant" or "I, {self.name}, would like to inform you..."
- The key is to sound like a real person talking, not an AI trying to act human
"""
        
        # Add the guidance to instructions
        enhanced_instructions = instructions + name_guidance
        logger.info(f"[PERSONALITY] Enhanced instructions with name usage guidance for '{self.name}'")
        
        return enhanced_instructions

    def _filter_personality_entries(self, entries: List[Dict]) -> List[Dict]:
        """
        Filter personality entries based on indicators and structure.
        
        Args:
            entries: List of personality entries to filter
            
        Returns:
            List[Dict]: Filtered personality entries
        """
        filtered_entries = []
        for entry in entries:
            logger.info(f"[PERSONALITY] Processing entry ID: {entry['id']}")
            
            # Look for positive structural indicators of personality content
            has_personality_indicators = any(indicator in entry["raw"].lower() for indicator in self.personality_indicators)
            
            # If it has personality indicators, use it
            if has_personality_indicators:
                filtered_entries.append(entry)
                logger.info(f"[PERSONALITY] Entry {entry['id']} has personality indicators")
                logger.info(f"[PERSONALITY] Raw personality content: {entry['raw'][:300]}...")
                self._extract_name_from_entry(entry)
            else:
                # Try to determine if an entry is about personality from its structure
                if self._has_personality_structure(entry):
                    filtered_entries.append(entry)
                    logger.info(f"[PERSONALITY] Entry {entry['id']} has personality structure")
                    logger.info(f"[PERSONALITY] Raw personality content: {entry['raw'][:300]}...")
                else:
                    logger.info(f"[PERSONALITY] Filtered out entry {entry['id']} - not personality content")
        
        return filtered_entries

    def _has_personality_structure(self, entry: Dict) -> bool:
        """
        Check if an entry has personality structure based on sentence analysis.
        
        Args:
            entry: The entry to check
            
        Returns:
            bool: True if the entry has personality structure
        """
        sentences = re.split(r'[.!?]', entry["raw"])
        personality_sentence_count = 0
        for sentence in sentences:
            if any(term in sentence.lower() for term in self.personality_terms):
                personality_sentence_count += 1
        
        # If more than 25% of sentences contain personality terms, include it
        return personality_sentence_count > 0 and len(sentences) > 0 and personality_sentence_count / len(sentences) >= 0.25

    def _extract_name_from_entry(self, entry: Dict) -> None:
        """
        Extract and set the AI name from a personality entry.
        
        Args:
            entry: The entry to extract name from
        """
        # Store original text before lowercasing for name extraction
        original_text = entry["raw"]
        lower_text = original_text.lower()
        
        # Look for specific name declarations in Vietnamese (case insensitive for detection)
        # Improved patterns to avoid capturing instructional phrases
        vietnamese_patterns = [
            # Match name followed by punctuation or instructional phrases
            r'(?:tên em là|em là|tôi là|mình là|xưng tên|gọi là)\s+([A-ZÀ-Ỹa-zà-ỹ]{1,20}(?:\s+[A-ZÀ-Ỹa-zà-ỹ]{1,20}){0,3})(?:\s+(?:khi|trong|để|với|mà|là|lúc)|[,\.\?!;:]|$)',
            r'(?:xưng|gọi|tên)\s+(?:là|tên)\s+([A-ZÀ-Ỹa-zà-ỹ]{1,20}(?:\s+[A-ZÀ-Ỹa-zà-ỹ]{1,20}){0,3})(?:\s+(?:khi|trong|để|với|mà|là|lúc)|[,\.\?!;:]|$)',
            r'chỉ xưng tên\s+([A-ZÀ-Ỹa-zà-ỹ]{1,20}(?:\s+[A-ZÀ-Ỹa-zà-ỹ]{1,20}){0,3})(?:\s+(?:khi|trong|để|với|mà|là|lúc)|[,\.\?!;:]|$)'
        ]
        
        # Look for all caps names which might be emphasized
        caps_pattern = r'(?:TÊN|XƯng|GỌI)\s+(?:LÀ)?\s+([A-ZÀ-Ỹ]{1,20}(?:\s+[A-ZÀ-Ỹ]{1,20}){0,3})(?:\s+(?:KHI|TRONG|ĐỂ|VỚI|MÀ|LÀ|LÚC)|[,\.\?!;:]|$)'
        
        # Check Vietnamese patterns
        for pattern in vietnamese_patterns:
            name_match = re.search(pattern, lower_text)
            if name_match:
                # Extract from original text using the match positions
                start, end = name_match.span(1)
                extracted_name = original_text[start:end].strip()
                logger.info(f"[PERSONALITY] Found Vietnamese name match: '{extracted_name}'")
                if extracted_name and len(extracted_name) > 1:
                    # Extra validation to prevent long phrases
                    if len(extracted_name.split()) <= 4:
                        self.name = extracted_name
                        logger.info(f"[PERSONALITY] Updated name to: {self.name}")
                        return
                    else:
                        logger.info(f"[PERSONALITY] Name too long, might include instructions: '{extracted_name}'")
        
        # Check for ALL CAPS names separately
        caps_match = re.search(caps_pattern, original_text)
        if caps_match:
            extracted_name = caps_match.group(1).strip()
            logger.info(f"[PERSONALITY] Found ALL CAPS name: '{extracted_name}'")
            if extracted_name and len(extracted_name) > 1:
                # Extra validation to prevent long phrases
                if len(extracted_name.split()) <= 4:
                    self.name = extracted_name
                    logger.info(f"[PERSONALITY] Updated name to: {self.name}")
                    return
                else:
                    logger.info(f"[PERSONALITY] Name too long, might include instructions: '{extracted_name}'")
        
        # English patterns as fallback
        english_match = re.search(r'(?:i am|my name is|name is|called)\s+([A-Za-z]{1,20}(?:\s+[A-Za-z]{1,20}){0,3})(?:\s+(?:when|during|in|if|while|for|to|with|as)|[,\.\?!;:]|$)', lower_text)
        if english_match:
            # Extract from original text
            start, end = english_match.span(1)
            extracted_name = original_text[start:end].strip()
            logger.info(f"[PERSONALITY] Found English name match: '{extracted_name}'")
            if extracted_name and len(extracted_name) > 1:
                # Extra validation to prevent long phrases
                if len(extracted_name.split()) <= 4:
                    self.name = extracted_name
                    logger.info(f"[PERSONALITY] Updated name to: {self.name}")
                    return
                else:
                    logger.info(f"[PERSONALITY] Name too long, might include instructions: '{extracted_name}'")
        
        logger.info(f"[PERSONALITY] No name found in entry {entry['id']}")

    async def _try_backup_queries(self, graph_version_id: str) -> List[Dict]:
        """
        Try backup queries to find personality entries.
        
        Args:
            graph_version_id: The graph version ID to query
            
        Returns:
            List[Dict]: Found personality entries
        """
        filtered_entries = []
        logger.info("[PERSONALITY] No specific personality entries found, trying backup queries")
        
        for backup_query in self.backup_queries:
            logger.info(f"[PERSONALITY] Trying backup query: '{backup_query}'")
            backup_entries = await query_graph_knowledge(graph_version_id, backup_query, top_k=2)
            logger.info(f"[PERSONALITY] Found {len(backup_entries)} entries for backup query")
            
            if backup_entries:
                # Apply the same personality detection logic
                for entry in backup_entries:
                    has_personality_indicators = any(indicator in entry["raw"].lower() for indicator in self.personality_indicators)
                    
                    if has_personality_indicators or any(term in entry["raw"].lower() for term in ["personality", "character", "identity", "role", "tone"]):
                        filtered_entries.append(entry)
                        logger.info(f"[PERSONALITY] Found personality entry {entry['id']} from backup query")
                        logger.info(f"[PERSONALITY] Raw personality content: {entry['raw'][:300]}...")
                        self._extract_name_from_entry(entry)
            
            # If we found at least one good entry from backup queries, exit the loop
            if len(filtered_entries) > 0:
                logger.info(f"[PERSONALITY] Found {len(filtered_entries)} valid personality entries from backup queries")
                break
        
        return filtered_entries

    def _has_vietnamese_personality_structure(self, entry: Dict) -> bool:
        """
        Check if the entry has Vietnamese personality structure patterns.
        
        Args:
            entry: The entry to check
            
        Returns:
            bool: True if the entry has Vietnamese personality structure
        """
        text = entry["raw"].lower()
        
        # Common Vietnamese personality-related patterns
        patterns = [
            r'xưng tên',
            r'gọi là',
            r'tên là',
            r'(?:trả lời|phản hồi|giao tiếp) với',
            r'nhiệm vụ của',
            r'vai trò của',
            r'(?:trong|khi) (?:trò chuyện|giao tiếp|trao đổi)'
        ]
        
        return any(re.search(pattern, text) for pattern in patterns)

    def _check_for_vietnamese_name(self, entry: Dict) -> bool:
        """
        Special check for Vietnamese name declarations like 'xưng tên KHÁNH HẢI'
        
        Args:
            entry: The entry to check
            
        Returns:
            bool: True if a Vietnamese name was found and set
        """
        # Improved pattern that stops at instructional phrases
        name_match = re.search(r'(?:xưng tên|chỉ xưng tên)\s+([A-ZÀ-Ỹ]{1,20}(?:\s+[A-ZÀ-Ỹ]{1,20}){0,3})(?:\s+(?:khi|trong|để|với|mà|là|lúc)|[,\.\?!;:]|$)', entry["raw"], re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
            logger.info(f"[PERSONALITY] Found Vietnamese name declaration: '{name}'")
            if name and len(name) > 1:
                # Extra validation to prevent long phrases
                if len(name.split()) <= 4:
                    self.name = name
                    logger.info(f"[PERSONALITY] Set name from Vietnamese declaration: {self.name}")
                    return True
                else:
                    logger.info(f"[PERSONALITY] Name too long, might include instructions: '{name}'")
        return False 