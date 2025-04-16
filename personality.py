from typing import Dict, List
from utilities import logger
from database import query_graph_knowledge
import re

class PersonalityManager:
    def __init__(self):
        self.name = "Ami"  # Default name
        self.personality_instructions = None
        self.instincts = {"friendly": "Be nice"}  # Add instincts attribute
        self.personality_indicators = [
            "bạn là",
            "tên là",
            "trong khi giao tiếp",
            "ai personality",
            "role:",
            "identity:",
            "tone:",
            "voice:",
            "communication style:",
            "positioning:",
            "how i should",
            "who am i",
            "my identity",
            "my role",
            "my character"
        ]
        self.personality_terms = [
            "personality",
            "character",
            "identity",
            "role",
            "tone",
            "voice",
            "style",
            "communicate"
        ]
        self.backup_queries = [
            "how should I speak communicate with people",
            "my character identity who am I",
            "how to behave tone and approach"
        ]
        self.default_personality = {
            "raw": "AI Personality Guidelines:\n" +
                   "- Maintain a helpful, friendly tone\n" +
                   "- Be respectful and professional\n" +
                   "- Show expertise in relevant topics\n" +
                   "- Keep responses clear and concise",
            "id": "default_personality"
        }

    async def load_personality_instructions(self, graph_version_id: str = "") -> str:
        """
        Load personality instructions from the knowledge base and extract identity information.
        
        Args:
            graph_version_id: The graph version ID to query
            
        Returns:
            str: The personality instructions
        """
        try:
            logger.info(f"[PERSONALITY] Starting personality load for graph_version_id: {graph_version_id}")
            # Get personality instructions from knowledge base
            personality_query = "who am I how should I behave my identity character role"
            personality_entries = await query_graph_knowledge(graph_version_id, personality_query, top_k=3)
            logger.info(f"[PERSONALITY] Found {len(personality_entries)} initial entries")
            
            # Detect proper personality entries using structure indicators
            filtered_personality_entries = self._filter_personality_entries(personality_entries)
            
            # If we didn't find enough specific personality entries, try with different queries
            if len(filtered_personality_entries) < 1:
                filtered_personality_entries = await self._try_backup_queries(graph_version_id)
            
            # If we still have no entries, use a default personality instruction
            if not filtered_personality_entries:
                logger.warning("[PERSONALITY] No personality entries found, using default personality")
                filtered_personality_entries = [self.default_personality]
            
            # Store the personality instructions
            self.personality_instructions = "\n\n".join(entry["raw"] for entry in filtered_personality_entries)
            logger.info(f"[PERSONALITY] Final personality instructions length: {len(self.personality_instructions)}")
            logger.info(f"[PERSONALITY] Final AI name: {self.name}")
            logger.info(f"[PERSONALITY] Personality loading complete")
            
            return self.personality_instructions
            
        except Exception as e:
            logger.error(f"[PERSONALITY] Error loading personality instructions: {str(e)}")
            # Fallback to default personality
            self.personality_instructions = self.default_personality["raw"]
            return self.personality_instructions

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
                self._extract_name_from_entry(entry)
            else:
                # Try to determine if an entry is about personality from its structure
                if self._has_personality_structure(entry):
                    filtered_entries.append(entry)
                    logger.info(f"[PERSONALITY] Entry {entry['id']} has personality structure")
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
        name_match = re.search(r'(?:bạn là|tên là|i am|my name is)\s+([^,.!?]+)', entry["raw"].lower())
        if name_match:
            extracted_name = name_match.group(1).strip()
            logger.info(f"[PERSONALITY] Found name match: '{extracted_name}'")
            if extracted_name and len(extracted_name) > 2:  # Basic validation
                self.name = extracted_name
                logger.info(f"[PERSONALITY] Updated name to: {self.name}")
            else:
                logger.info(f"[PERSONALITY] Extracted name too short or invalid: '{extracted_name}'")
        else:
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
                        self._extract_name_from_entry(entry)
            
            # If we found at least one good entry from backup queries, exit the loop
            if len(filtered_entries) > 0:
                logger.info(f"[PERSONALITY] Found {len(filtered_entries)} valid personality entries from backup queries")
                break
        
        return filtered_entries 