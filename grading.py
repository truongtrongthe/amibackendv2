import asyncio
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

# Import the COTProcessor from tool6
from tool6 import CoTProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COTProcessorGrader:
    """A grading system to validate COTProcessor knowledge loading and compilation."""
    
    def __init__(self):
        self.cot_processor = None
        self.validation_results = {}
        
    async def initialize_cot_processor(self, graph_version_id: str = None) -> bool:
        """Initialize the COTProcessor and trigger knowledge compilation."""
        try:
            logger.info("Initializing COTProcessor for grading...")
            self.cot_processor = CoTProcessor()
            
            # This will load basic knowledge and compile comprehensive versions
            await self.cot_processor.initialize(graph_version_id)
            
            logger.info("COTProcessor initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize COTProcessor: {str(e)}")
            return False
    
    async def manually_trigger_compilation(self) -> Dict[str, Any]:
        """Manually trigger the _compile_self_awareness method for testing."""
        if not self.cot_processor:
            raise ValueError("COTProcessor not initialized. Call initialize_cot_processor first.")
        
        try:
            logger.info("Manually triggering _compile_self_awareness...")
            compilation_result = await self.cot_processor._compile_self_awareness()
            logger.info(f"Manual compilation result: {compilation_result}")
            return compilation_result
            
        except Exception as e:
            logger.error(f"Manual compilation failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def validate_knowledge_loading(self) -> Dict[str, Any]:
        """Validate if all knowledge bases are properly loaded."""
        if not self.cot_processor:
            return {"error": "COTProcessor not initialized"}
        
        validation_results = {
            "basic_knowledge": {},
            "comprehensive_knowledge": {},
            "overall_status": "unknown"
        }
        
        # Check basic knowledge bases
        basic_knowledge_checks = {
            "profiling_skills": self.cot_processor.profiling_skills,
            "communication_skills": self.cot_processor.communication_skills,
            "ai_business_objectives": self.cot_processor.ai_business_objectives
        }
        
        for knowledge_type, knowledge_data in basic_knowledge_checks.items():
            validation_results["basic_knowledge"][knowledge_type] = self._validate_knowledge_structure(knowledge_data)
        
        # Check comprehensive knowledge bases
        comprehensive_knowledge_checks = {
            "comprehensive_profiling_skills": self.cot_processor.comprehensive_profiling_skills,
            "comprehensive_communication_skills": self.cot_processor.comprehensive_communication_skills,
            "comprehensive_business_objectives": self.cot_processor.comprehensive_business_objectives
        }
        
        for knowledge_type, knowledge_data in comprehensive_knowledge_checks.items():
            validation_results["comprehensive_knowledge"][knowledge_type] = self._validate_knowledge_structure(knowledge_data)
        
        # Determine overall status
        all_basic_valid = all(result["is_valid"] for result in validation_results["basic_knowledge"].values())
        all_comprehensive_valid = all(result["is_valid"] for result in validation_results["comprehensive_knowledge"].values())
        
        if all_basic_valid and all_comprehensive_valid:
            validation_results["overall_status"] = "success"
        elif all_basic_valid:
            validation_results["overall_status"] = "basic_only"
        else:
            validation_results["overall_status"] = "failed"
        
        self.validation_results = validation_results
        return validation_results
    
    def _validate_knowledge_structure(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the structure and content of a knowledge base."""
        validation = {
            "is_valid": False,
            "has_knowledge_context": False,
            "has_metadata": False,
            "content_length": 0,
            "issues": []
        }
        
        if not isinstance(knowledge_data, dict):
            validation["issues"].append("Knowledge data is not a dictionary")
            return validation
        
        # Check for knowledge_context
        if "knowledge_context" in knowledge_data:
            validation["has_knowledge_context"] = True
            knowledge_context = knowledge_data["knowledge_context"]
            
            if isinstance(knowledge_context, str):
                validation["content_length"] = len(knowledge_context)
                
                # Check if it's not just loading message
                if knowledge_context == "Loading...":
                    validation["issues"].append("Knowledge is still loading")
                elif knowledge_context == "No knowledge available.":
                    validation["issues"].append("No knowledge was loaded")
                elif knowledge_context == "Compilation failed.":
                    validation["issues"].append("Knowledge compilation failed")
                elif len(knowledge_context.strip()) < 50:
                    validation["issues"].append("Knowledge content seems too short")
            else:
                validation["issues"].append("Knowledge context is not a string")
        else:
            validation["issues"].append("Missing knowledge_context field")
        
        # Check for metadata
        if "metadata" in knowledge_data:
            validation["has_metadata"] = True
            metadata = knowledge_data["metadata"]
            
            if isinstance(metadata, dict):
                # Check for common metadata fields
                if "timestamp" not in metadata:
                    validation["issues"].append("Missing timestamp in metadata")
                if "entry_count" in metadata and metadata["entry_count"] == 0:
                    validation["issues"].append("No entries found in knowledge")
            else:
                validation["issues"].append("Metadata is not a dictionary")
        else:
            validation["issues"].append("Missing metadata field")
        
        # Determine if valid
        validation["is_valid"] = (
            validation["has_knowledge_context"] and 
            validation["has_metadata"] and 
            len(validation["issues"]) == 0
        )
        
        return validation
    
    def get_comprehensive_profiling_skills(self) -> Dict[str, Any]:
        """External access to comprehensive_profiling_skills."""
        if not self.cot_processor:
            return {"error": "COTProcessor not initialized"}
        
        return self.cot_processor.comprehensive_profiling_skills
    
    def get_comprehensive_communication_skills(self) -> Dict[str, Any]:
        """External access to comprehensive_communication_skills."""
        if not self.cot_processor:
            return {"error": "COTProcessor not initialized"}
        
        return self.cot_processor.comprehensive_communication_skills
    
    def get_comprehensive_business_objectives(self) -> Dict[str, Any]:
        """External access to comprehensive_business_objectives."""
        if not self.cot_processor:
            return {"error": "COTProcessor not initialized"}
        
        return self.cot_processor.comprehensive_business_objectives
    
    def print_knowledge_summary(self):
        """Print a summary of all loaded knowledge."""
        if not self.cot_processor:
            print("COTProcessor not initialized")
            return
        
        print("\n" + "="*80)
        print("COT PROCESSOR KNOWLEDGE SUMMARY")
        print("="*80)
        
        # Basic Knowledge
        print("\n📚 BASIC KNOWLEDGE BASES:")
        basic_knowledge = {
            "Profiling Skills": self.cot_processor.profiling_skills,
            "Communication Skills": self.cot_processor.communication_skills,
            "Business Objectives": self.cot_processor.ai_business_objectives
        }
        
        for name, knowledge in basic_knowledge.items():
            content_length = len(knowledge.get("knowledge_context", ""))
            entry_count = knowledge.get("metadata", {}).get("entry_count", "N/A")
            print(f"  {name}: {content_length} chars, {entry_count} entries")
        
        # Comprehensive Knowledge
        print("\n🧠 COMPREHENSIVE KNOWLEDGE BASES:")
        comprehensive_knowledge = {
            "Comprehensive Profiling": self.cot_processor.comprehensive_profiling_skills,
            "Comprehensive Communication": self.cot_processor.comprehensive_communication_skills,
            "Comprehensive Business": self.cot_processor.comprehensive_business_objectives
        }
        
        for name, knowledge in comprehensive_knowledge.items():
            content_length = len(knowledge.get("knowledge_context", ""))
            timestamp = knowledge.get("metadata", {}).get("timestamp", "N/A")
            print(f"  {name}: {content_length} chars, compiled at {timestamp}")
        
        print("\n" + "="*80)
    
    

# Usage functions for easy testing
async def grade_cot_processor(graph_version_id: str = None) -> COTProcessorGrader:
    """Main function to grade a COTProcessor instance."""
    grader = COTProcessorGrader()
    
    print("🔍 Starting COTProcessor grading...")
    
    # Initialize
    success = await grader.initialize_cot_processor(graph_version_id)
    if not success:
        print("❌ Failed to initialize COTProcessor")
        return grader
    
    print("✅ COTProcessor initialized successfully")
    
    # Print summary
    grader.print_knowledge_summary()
    
    return grader


async def quick_test():
    """Quick test function to demonstrate usage."""
    print("🚀 Running quick COTProcessor test...")
    
    grader = await grade_cot_processor()
    
    # Access comprehensive skills externally
    print("\n🎯 Accessing comprehensive skills externally:")
    
    profiling_skills = grader.get_comprehensive_profiling_skills()
    profiling_length = len(profiling_skills.get("knowledge_context", ""))
    print(f"Comprehensive Profiling Skills: {profiling_length} characters")
    
    communication_skills = grader.get_comprehensive_communication_skills()
    communication_length = len(communication_skills.get("knowledge_context", ""))
    print(f"Comprehensive Communication Skills: {communication_length} characters")
    
    business_objectives = grader.get_comprehensive_business_objectives()
    business_length = len(business_objectives.get("knowledge_context", ""))
    print(f"Comprehensive Business Objectives: {business_length} characters")
    
    
    return grader


if __name__ == "__main__":
    # Run the quick test
    asyncio.run(quick_test()) 