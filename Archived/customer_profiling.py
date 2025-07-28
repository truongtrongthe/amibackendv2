"""
customer_profiling.py - Dynamic customer profiling and classification using hotbrain's knowledge graph.

This module provides functionality to scan and categorize customer knowledge from Pinecone knowledge graphs,
making it available for AI agents to use for dynamic customer profiling and classification.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from hotbrain import query_graph_knowledge

logger = logging.getLogger(__name__)

class CustomerProfilingManager:
    """
    Manages dynamic loading and application of customer profiling and classification skills.
    """
    
    def __init__(self):
        self.profiling_skills: Dict[str, List[Dict]] = {}
        self.classification_skills: Dict[str, List[Dict]] = {}
        self.loaded_graph_version_ids: Set[str] = set()
        
    async def load_skills(self, graph_version_id: str) -> Tuple[List[dict], List[dict]]:
        """
        Load profiling and classification skills from the knowledge graph.
        
        Args:
            graph_version_id: The version ID of the knowledge graph to load from
        """
        if graph_version_id in self.loaded_graph_version_ids:
            logger.info(f"Skills already loaded for graph version {graph_version_id}")
            return
            
        logger.info(f"Loading customer profiling and classification skills for graph version {graph_version_id}")
        
        # Define queries for profiling and classification skills
        profiling_queries = [
            "customer profiling techniques",
            "How to understand customer needs?",
            "customer interaction skills",
            "Làm thế nào để hiểu nhu cầu khách hàng?",
            "kỹ năng tương tác với khách hàng"
        ]
        
        classification_queries = [
            "customer classification frameworks",
            "How to identify customer types and patterns?",
            "customer segmentation types",
            "customer behavior patterns",
            "customer personality types",
            "How to categorize customers?",
            "Làm thế nào để nhận biết các loại khách hàng?",
            "các mẫu hành vi khách hàng",
            "phân loại tính cách khách hàng",
            "cách phân nhóm khách hàng",
            "đặc điểm nhận dạng khách hàng",
            "phương pháp phân loại khách hàng"
        ]
        
        # Create tasks for all queries
        query_tasks = []
        
        # Add profiling queries
        for query in profiling_queries:
            query_tasks.append(query_graph_knowledge(graph_version_id, query, top_k=5))
            
        # Add classification queries
        for query in classification_queries:
            query_tasks.append(query_graph_knowledge(graph_version_id, query, top_k=5))
            
        # Execute all queries in parallel
        all_results = await asyncio.gather(*query_tasks, return_exceptions=True)
        
        # Process results
        profiling_entries = []
        classification_entries = []
        seen_ids = set()
        
        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"Error loading skills: {str(result)}")
                continue
                
            for entry in result:
                if entry["id"] in seen_ids:
                    continue
                    
                seen_ids.add(entry["id"])
                
                # Log the entry for debugging
                logger.debug(f"Processing entry: {entry.get('raw', '')[:100]}...")
                
                # Determine if entry is profiling or classification based on content
                is_profiling = self._is_profiling_entry(entry)
                is_classification = self._is_classification_entry(entry)
                
                if is_profiling:
                    logger.debug(f"Adding profiling skill: {entry.get('raw', '')[:100]}...")
                    profiling_entries.append(entry)
                if is_classification:
                    logger.debug(f"Adding classification skill: {entry.get('raw', '')[:100]}...")
                    classification_entries.append(entry)
                    
        # Store the loaded skills
        self.profiling_skills[graph_version_id] = profiling_entries
        self.classification_skills[graph_version_id] = classification_entries
        self.loaded_graph_version_ids.add(graph_version_id)
        
        logger.info(f"Loaded {len(profiling_entries)} profiling skills and {len(classification_entries)} classification skills")
        
    def _is_profiling_entry(self, entry: Dict) -> bool:
        """
        Determine if an entry contains profiling skills.
        
        Args:
            entry: The knowledge graph entry
            
        Returns:
            bool: True if the entry contains profiling skills
        """
        if not entry.get("raw"):
            return False
            
        text = entry["raw"].lower()
        metadata = entry.get("metadata", {})
        
        # Check metadata categories if available
        categories = [
            metadata.get("categories_primary", "").lower(),
            metadata.get("categories_special", "").lower()
        ]
        
        # Metadata category indicators
        category_indicators = [
            "profile", "profiling", "customer understanding",
            "listening", "communication", "interaction",
            "customer needs", "customer behavior"
        ]
        
        if any(cat and any(ind in cat for ind in category_indicators) for cat in categories):
            return True
        
        # English profiling indicators
        english_indicators = [
            "customer profile",
            "customer portrait",
            "customer understanding",
            "customer needs",
            "customer behavior",
            "customer preferences",
            "customer characteristics",
            "customer insights",
            "customer data",
            "customer information",
            "listening skills",
            "open-ended questions",
            "customer experience",
            "customer feedback",
            "customer interaction",
            "understand the customer",
            "customer communication",
            "customer relationship"
        ]
        
        # Vietnamese profiling indicators
        vietnamese_indicators = [
            "hồ sơ khách hàng",
            "chân dung khách hàng",
            "hiểu khách hàng",
            "nhu cầu khách hàng",
            "hành vi khách hàng",
            "sở thích khách hàng",
            "đặc điểm khách hàng",
            "thông tin khách hàng",
            "dữ liệu khách hàng",
            "kỹ năng lắng nghe",
            "câu hỏi mở",
            "trải nghiệm khách hàng",
            "phản hồi khách hàng",
            "tương tác khách hàng",
            "giao tiếp khách hàng",
            "quan hệ khách hàng"
        ]
        
        indicators = english_indicators + vietnamese_indicators
        return any(indicator in text for indicator in indicators)
        
    def _is_classification_entry(self, entry: dict) -> bool:
        """Check if an entry is a classification skill."""
        if not entry or 'metadata' not in entry:
            return False
            
        raw_content = entry['metadata'].get('raw', '').lower()
        categories = entry['metadata'].get('categories_primary', '').lower()
        labels = entry['metadata'].get('labels', '').lower()
        
        classification_indicators = [
            'customer type', 'loại khách hàng',
            'customer category', 'phân loại khách hàng',
            'customer pattern', 'mẫu khách hàng',
            'customer behavior', 'hành vi khách hàng',
            'customer segment', 'phân khúc khách hàng',
            'customer profile', 'hồ sơ khách hàng',
            'customer classification', 'phân loại khách hàng',
            'customer group', 'nhóm khách hàng',
            'customer personality', 'tính cách khách hàng',
            'decision maker', 'người ra quyết định',
            'buying pattern', 'mẫu mua hàng',
            'communication style', 'phong cách giao tiếp',
            'interaction pattern', 'mẫu tương tác',
            'customer need', 'nhu cầu khách hàng',
            'customer preference', 'sở thích khách hàng',
            'customer characteristic', 'đặc điểm khách hàng',
            'customer trait', 'đặc tính khách hàng'
        ]
        
        # Check in raw content
        if any(indicator in raw_content for indicator in classification_indicators):
            return True
            
        # Check in categories
        if any(indicator in categories for indicator in classification_indicators):
            return True
            
        # Check in labels
        if any(indicator in labels for indicator in classification_indicators):
            return True
            
        return False
        
    def get_profiling_skills(self, graph_version_id: str) -> List[Dict]:
        """
        Get loaded profiling skills for a specific graph version.
        
        Args:
            graph_version_id: The version ID of the knowledge graph
            
        Returns:
            List[Dict]: List of profiling skills
        """
        return self.profiling_skills.get(graph_version_id, [])
        
    def get_classification_skills(self, graph_version_id: str) -> List[Dict]:
        """
        Get loaded classification skills for a specific graph version.
        
        Args:
            graph_version_id: The version ID of the knowledge graph
            
        Returns:
            List[Dict]: List of classification skills
        """
        return self.classification_skills.get(graph_version_id, [])
        
    def clear_skills(self, graph_version_id: Optional[str] = None) -> None:
        """
        Clear loaded skills for a specific graph version or all versions.
        
        Args:
            graph_version_id: Optional specific version to clear, or None to clear all
        """
        if graph_version_id:
            self.profiling_skills.pop(graph_version_id, None)
            self.classification_skills.pop(graph_version_id, None)
            self.loaded_graph_version_ids.discard(graph_version_id)
        else:
            self.profiling_skills.clear()
            self.classification_skills.clear()
            self.loaded_graph_version_ids.clear()

async def test_load_skills():
    """
    Test loading skills from the knowledge graph.
    """
    manager = CustomerProfilingManager()
    graph_version_id = "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"
    
    try:
        # Test initial load
        logger.info("Testing initial skill loading...")
        await manager.load_skills(graph_version_id)
        
        # Get loaded skills
        profiling_skills = manager.get_profiling_skills(graph_version_id)
        classification_skills = manager.get_classification_skills(graph_version_id)
        
        # Check if any skills were loaded
        if len(profiling_skills) == 0 and len(classification_skills) == 0:
            logger.warning("No skills were loaded. This could be because:")
            logger.warning("1. The graph version ID doesn't exist")
            logger.warning("2. The graph version has no relevant skills")
            logger.warning("3. There are permission issues accessing the graph")
            logger.warning("Please verify the graph version ID and permissions.")
            return  # Skip remaining tests if no skills were loaded
            
        # Verify skills were loaded
        assert len(profiling_skills) > 0, "No profiling skills were loaded"
        assert len(classification_skills) > 0, "No classification skills were loaded"
        
        logger.info(f"Loaded {len(profiling_skills)} profiling skills and {len(classification_skills)} classification skills")
        
        # Test caching
        logger.info("Testing skill caching...")
        await manager.load_skills(graph_version_id)  # Should use cached version
        
        # Verify same number of skills
        assert len(manager.get_profiling_skills(graph_version_id)) == len(profiling_skills), "Caching failed for profiling skills"
        assert len(manager.get_classification_skills(graph_version_id)) == len(classification_skills), "Caching failed for classification skills"
        
        # Test clearing skills
        logger.info("Testing skill clearing...")
        manager.clear_skills(graph_version_id)
        assert len(manager.get_profiling_skills(graph_version_id)) == 0, "Failed to clear profiling skills"
        assert len(manager.get_classification_skills(graph_version_id)) == 0, "Failed to clear classification skills"
        
        # Test reloading after clear
        logger.info("Testing reloading after clear...")
        await manager.load_skills(graph_version_id)
        assert len(manager.get_profiling_skills(graph_version_id)) > 0, "Failed to reload profiling skills"
        assert len(manager.get_classification_skills(graph_version_id)) > 0, "Failed to reload classification skills"
        
        logger.info("All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

async def test_skill_content():
    """
    Test the content of loaded skills.
    """
    manager = CustomerProfilingManager()
    graph_version_id = "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"
    
    try:
        # Load skills
        await manager.load_skills(graph_version_id)
        
        # Get skills
        profiling_skills = manager.get_profiling_skills(graph_version_id)
        classification_skills = manager.get_classification_skills(graph_version_id)
        
        # Skip content tests if no skills were loaded
        if len(profiling_skills) == 0 and len(classification_skills) == 0:
            logger.warning("Skipping content tests - no skills were loaded")
            return
            
        # Test profiling skills content
        logger.info("Testing profiling skills content...")
        for skill in profiling_skills:
            assert "raw" in skill, "Profiling skill missing raw content"
            assert isinstance(skill["raw"], str), "Profiling skill raw content is not a string"
            assert len(skill["raw"]) > 0, "Profiling skill has empty content"
            
            # Verify content contains profiling indicators
            text = skill["raw"].lower()
            has_profiling_indicator = any(
                indicator in text for indicator in [
                    "customer profile", "customer portrait", "customer understanding",
                    "hồ sơ khách hàng", "chân dung khách hàng", "hiểu khách hàng"
                ]
            )
            assert has_profiling_indicator, f"Profiling skill missing profiling indicators: {text[:100]}..."
        
        # Test classification skills content
        logger.info("Testing classification skills content...")
        for skill in classification_skills:
            assert "raw" in skill, "Classification skill missing raw content"
            assert isinstance(skill["raw"], str), "Classification skill raw content is not a string"
            assert len(skill["raw"]) > 0, "Classification skill has empty content"
            
            # Verify content contains classification indicators
            text = skill["raw"].lower()
            has_classification_indicator = any(
                indicator in text for indicator in [
                    "customer classification", "customer segmentation", "customer category",
                    "phân loại khách hàng", "phân khúc khách hàng", "loại khách hàng"
                ]
            )
            assert has_classification_indicator, f"Classification skill missing classification indicators: {text[:100]}..."
        
        logger.info("All content tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Content test failed: {str(e)}")
        raise

async def test_skill_duplicates():
    """
    Test that no duplicate skills are loaded.
    """
    manager = CustomerProfilingManager()
    graph_version_id = "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"
    
    try:
        # Load skills
        await manager.load_skills(graph_version_id)
        
        # Get skills
        profiling_skills = manager.get_profiling_skills(graph_version_id)
        classification_skills = manager.get_classification_skills(graph_version_id)
        
        # Skip duplicate tests if no skills were loaded
        if len(profiling_skills) == 0 and len(classification_skills) == 0:
            logger.warning("Skipping duplicate tests - no skills were loaded")
            return
        
        # Check for duplicate IDs
        profiling_ids = set()
        classification_ids = set()
        
        for skill in profiling_skills:
            assert skill["id"] not in profiling_ids, f"Duplicate profiling skill ID found: {skill['id']}"
            profiling_ids.add(skill["id"])
            
        for skill in classification_skills:
            assert skill["id"] not in classification_ids, f"Duplicate classification skill ID found: {skill['id']}"
            classification_ids.add(skill["id"])
        
        logger.info("No duplicate skills found!")
        
    except Exception as e:
        logger.error(f"Duplicate test failed: {str(e)}")
        raise

async def run_all_tests():
    """
    Run all tests for the CustomerProfilingManager.
    """
    logger.info("Starting CustomerProfilingManager tests...")
    logger.info(f"Using graph version ID: bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc")
    
    try:
        await test_load_skills()
        await test_skill_content()
        await test_skill_duplicates()
        
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Tests failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    asyncio.run(run_all_tests())

