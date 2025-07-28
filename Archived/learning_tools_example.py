"""
Interactive Learning Tools - Example Usage

This example demonstrates how the learning tools transform the monolithic
hidden learning process into explicit, interactive, human-guided steps.
"""

import asyncio
import json
from typing import Dict, Any
from learning_tools import (
    LearningSearchTool, LearningAnalysisTool, HumanLearningTool,
    KnowledgePreviewTool, KnowledgeSaveTool, LearningToolsFactory
)

# Example learning scenarios
EXAMPLE_SCENARIOS = {
    "teaching_intent": {
        "user_message": "Bạn nên sử dụng async/await trong Python khi làm việc với I/O operations vì nó giúp tránh blocking và cải thiện performance.",
        "context": "User is explaining Python async programming concepts",
        "expected_analysis": "HIGH learning value - clear teaching intent"
    },
    "new_information": {
        "user_message": "Just learned that React 18 introduced automatic batching for better performance. This means setState calls are automatically batched even in timeouts and promises.",
        "context": "User sharing recent technical discovery",
        "expected_analysis": "MEDIUM learning value - new technical information"
    },
    "procedural_knowledge": {
        "user_message": "To deploy Docker containers to production: 1) Build the image 2) Tag it properly 3) Push to registry 4) Update deployment config 5) Roll out changes",
        "context": "User providing step-by-step process",
        "expected_analysis": "HIGH learning value - procedural knowledge"
    },
    "low_value": {
        "user_message": "Thanks for the help!",
        "context": "Simple acknowledgment",
        "expected_analysis": "LOW learning value - social interaction"
    }
}


class LearningToolsDemo:
    """Demonstration of the interactive learning tools"""
    
    def __init__(self):
        self.user_id = "demo_user"
        self.org_id = "demo_org"
        
        # Initialize learning tools
        self.search_tool = LearningSearchTool(self.user_id, self.org_id)
        self.analysis_tool = LearningAnalysisTool(self.user_id, self.org_id)
        self.human_tool = HumanLearningTool(self.user_id, self.org_id)
        self.preview_tool = KnowledgePreviewTool(self.user_id, self.org_id)
        self.save_tool = KnowledgeSaveTool(self.user_id, self.org_id)
    
    def demonstrate_step_by_step_learning(self, scenario_name: str):
        """
        Demonstrate the step-by-step learning process
        
        This shows how LLM would use the tools for interactive learning
        """
        print(f"\n{'='*60}")
        print(f"📚 LEARNING DEMONSTRATION: {scenario_name.upper()}")
        print(f"{'='*60}")
        
        scenario = EXAMPLE_SCENARIOS[scenario_name]
        user_message = scenario["user_message"]
        context = scenario["context"]
        
        print(f"\n👤 User Message: {user_message}")
        print(f"📝 Context: {context}")
        print(f"🎯 Expected: {scenario['expected_analysis']}")
        
        # Step 1: Search for existing knowledge
        print(f"\n🔍 STEP 1: Search for existing knowledge")
        print("=" * 50)
        search_results = self.search_tool.search_learning_context(
            query=user_message,
            search_depth="basic"
        )
        print(search_results)
        
        # Step 2: Analyze learning opportunity
        print(f"\n🧠 STEP 2: Analyze learning opportunity")
        print("=" * 50)
        analysis_results = self.analysis_tool.analyze_learning_opportunity(
            user_message=user_message,
            conversation_context=context,
            search_results=search_results
        )
        print(analysis_results)
        
        # Step 3: Determine next action based on analysis
        print(f"\n⚖️ STEP 3: Determine next action")
        print("=" * 50)
        
        # Extract recommendation from analysis
        if "REQUEST_HUMAN_DECISION" in analysis_results:
            self._demonstrate_human_decision_flow(user_message, analysis_results)
        elif "LEARN_NEW" in analysis_results:
            self._demonstrate_direct_learning_flow(user_message, analysis_results)
        elif "SKIP_LEARNING" in analysis_results:
            print("🚫 Analysis recommends skipping learning - continuing conversation normally")
        else:
            print("📋 Analysis suggests maybe learning - would need human confirmation")
        
        print(f"\n✅ Learning demonstration complete for {scenario_name}")
    
    def _demonstrate_human_decision_flow(self, user_message: str, analysis_results: str):
        """Demonstrate the human decision flow"""
        print("🤝 Human decision required - requesting user input")
        
        # Step 3a: Request human decision
        decision_response = self.human_tool.request_learning_decision(
            decision_type="save_new",
            context=f"User taught: {user_message[:100]}...",
            options=["save_new", "update_existing", "skip_learning"],
            additional_info="This appears to be valuable teaching content"
        )
        print(decision_response)
        
        # Extract decision ID
        decision_id = self._extract_decision_id(decision_response)
        if decision_id:
            print(f"\n📋 Decision ID: {decision_id}")
            print("🎯 In real scenario, frontend would show decision UI")
            print("👆 Human would click one of the options")
            
            # Simulate human decision (in real usage, this comes from frontend)
            self._simulate_human_decision(decision_id, "save_new", user_message)
        
    def _demonstrate_direct_learning_flow(self, user_message: str, analysis_results: str):
        """Demonstrate direct learning without human intervention"""
        print("✅ High confidence learning - proceeding directly")
        
        # Step 3a: Preview what would be saved
        ai_response = f"Thank you for sharing that information about {user_message[:50]}..."
        preview_results = self.preview_tool.preview_knowledge_save(
            user_message=user_message,
            ai_response=ai_response,
            save_format="conversation"
        )
        print("\n🔍 Knowledge Preview:")
        print(preview_results)
        
        # Step 3b: Save knowledge
        save_results = self.save_tool.save_knowledge(
            content=f"User taught: {user_message}",
            title="User Teaching Session",
            categories=["teaching_intent", "high_confidence"],
            thread_id="demo_thread"
        )
        print("\n💾 Save Results:")
        print(save_results)
    
    def _simulate_human_decision(self, decision_id: str, choice: str, user_message: str):
        """Simulate human making a decision"""
        print(f"\n🤖 Simulating human choice: {choice}")
        
        # Import decision completion function
        from learning_tools import complete_learning_decision
        
        # Complete the decision
        result = complete_learning_decision(decision_id, choice)
        print(f"✅ Decision completed: {result}")
        
        if choice == "save_new":
            # Continue with preview and save
            print("\n📋 Human chose to save - continuing with preview...")
            ai_response = f"Thank you for that valuable information about {user_message[:50]}..."
            
            preview_results = self.preview_tool.preview_knowledge_save(
                user_message=user_message,
                ai_response=ai_response,
                save_format="conversation"
            )
            print(preview_results)
            
            # Save with decision reference
            save_results = self.save_tool.save_knowledge(
                content=f"User taught: {user_message}",
                title="Human-Approved Teaching Session",
                categories=["teaching_intent", "human_approved"],
                thread_id="demo_thread",
                decision_id=decision_id
            )
            print(f"\n💾 Save Results:")
            print(save_results)
        
        elif choice == "skip_learning":
            print("🚫 Human chose to skip learning - continuing conversation normally")
        
        else:
            print(f"📝 Human chose: {choice} - would handle accordingly")
    
    def _extract_decision_id(self, decision_response: str) -> str:
        """Extract decision ID from response"""
        import re
        match = re.search(r'Decision ID: (\w+)', decision_response)
        return match.group(1) if match else None
    
    def demonstrate_all_scenarios(self):
        """Run demonstration for all scenarios"""
        print("\n🎭 INTERACTIVE LEARNING TOOLS DEMONSTRATION")
        print("=" * 60)
        print("This demo shows how the monolithic learning logic has been")
        print("transformed into explicit, interactive, human-guided steps.")
        print("\nKey Benefits:")
        print("• ✅ Explicit learning steps (no hidden logic)")
        print("• 🤝 Human control at decision points")
        print("• 👀 Transparent process (every step visible)")
        print("• 🎯 Frontend integration ready")
        print("• 🔧 Granular control over learning")
        
        for scenario_name in EXAMPLE_SCENARIOS:
            self.demonstrate_step_by_step_learning(scenario_name)
        
        print("\n🏁 ALL DEMONSTRATIONS COMPLETE")
        print("=" * 60)
        print("The learning process is now:")
        print("1. 🔍 Explicit (LLM calls specific tools)")
        print("2. 🤝 Interactive (Human decisions at key points)")
        print("3. 👀 Transparent (Every step visible)")
        print("4. 🎯 Controllable (Granular tool-level control)")
        print("5. 🔧 Extensible (Easy to add new learning logic)")
    
    def demonstrate_tool_definitions(self):
        """Show how LLM sees the learning tools"""
        print("\n🛠️ TOOL DEFINITIONS FOR LLM")
        print("=" * 50)
        
        tool_definitions = LearningToolsFactory.get_tool_definitions()
        
        for i, tool_def in enumerate(tool_definitions, 1):
            print(f"\n{i}. {tool_def['name']}")
            print(f"   Description: {tool_def['description']}")
            print(f"   Parameters: {list(tool_def['parameters']['properties'].keys())}")
        
        print(f"\n📊 Total Learning Tools: {len(tool_definitions)}")
        print("🎯 LLM can call these tools explicitly when learning opportunities arise")


def main():
    """Run the learning tools demonstration"""
    demo = LearningToolsDemo()
    
    # Show tool definitions
    demo.demonstrate_tool_definitions()
    
    # Run all scenario demonstrations
    demo.demonstrate_all_scenarios()
    
    print("\n🎉 PHASE 1 COMPLETE!")
    print("=" * 60)
    print("✅ Learning tools created and integrated")
    print("✅ Executive tool updated to include learning tools")
    print("✅ API endpoints added for human decisions")
    print("✅ System prompts updated for learning awareness")
    print("✅ Comprehensive examples and documentation")
    print("\n🚀 Ready for Phase 2: Frontend Integration")


if __name__ == "__main__":
    main() 