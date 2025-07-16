#!/usr/bin/env python3
"""
Test script for Cursor-style request handling feature
Demonstrates intent classification, tool orchestration, and progressive enhancement
"""

import asyncio
import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime

class CursorStyleRequestDemo:
    """Demo class for testing Cursor-style request handling"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/tool/llm"
        
    def test_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Define test scenarios for different intent types"""
        return {
            "learning_intent": {
                "user_query": "Our company uses microservices architecture with Docker containers and Kubernetes orchestration. We deploy to AWS using CI/CD pipelines.",
                "expected_intent": "learning",
                "expected_tools": ["search_learning_context", "analyze_learning_opportunity"],
                "description": "User sharing company technical stack information"
            },
            "problem_solving_intent": {
                "user_query": "My React app is crashing with 'Cannot read property of undefined' error when users click the login button. The error happens intermittently.",
                "expected_intent": "problem_solving", 
                "expected_tools": ["search", "context"],
                "description": "User reporting a specific technical problem"
            },
            "general_chat_intent": {
                "user_query": "What's the difference between REST and GraphQL APIs?",
                "expected_intent": "general_chat",
                "expected_tools": ["search"],
                "description": "General knowledge question"
            },
            "task_execution_intent": {
                "user_query": "Create a Python function that validates email addresses using regex and returns True/False",
                "expected_intent": "task_execution",
                "expected_tools": ["context", "search"],
                "description": "User requesting specific task completion"
            },
            "high_complexity_problem": {
                "user_query": "I need to implement real-time chat with WebSockets, handle user authentication, manage message history, and scale to 10,000 concurrent users. How should I architect this?",
                "expected_intent": "problem_solving",
                "expected_tools": ["search", "context", "learning_search"],
                "description": "Complex architectural problem"
            }
        }
    
    def make_request(self, scenario: Dict[str, Any], cursor_mode: bool = True) -> Dict[str, Any]:
        """Make a request to the LLM endpoint"""
        payload = {
            "llm_provider": "anthropic",
            "user_query": scenario["user_query"],
            "system_prompt": "You are a helpful technical assistant with expertise in software development.",
            "cursor_mode": cursor_mode,
            "enable_intent_classification": True,
            "enable_request_analysis": True,
            "enable_tools": True,
            "org_id": "demo_org",
            "user_id": "demo_user"
        }
        
        print(f"\n{'='*80}")
        print(f"🎯 Testing: {scenario['description']}")
        print(f"{'='*80}")
        print(f"Query: {scenario['user_query']}")
        print(f"Expected Intent: {scenario['expected_intent']}")
        print(f"Expected Tools: {scenario['expected_tools']}")
        print(f"Cursor Mode: {cursor_mode}")
        print(f"{'='*80}")
        
        try:
            # Make the request
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'},
                stream=True,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
            
            # Parse SSE stream
            events = []
            content_buffer = ""
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        events.append(data)
                        
                        # Process different event types
                        if data.get('type') == 'analysis_start':
                            print(f"🔍 {data.get('content', '')}")
                        
                        elif data.get('type') == 'analysis_complete':
                            analysis = data.get('analysis', {})
                            orchestration = data.get('orchestration_plan', {})
                            
                            print(f"📊 Intent Analysis:")
                            print(f"   Intent: {analysis.get('intent', 'unknown')}")
                            print(f"   Confidence: {analysis.get('confidence', 0):.2f}")
                            print(f"   Complexity: {analysis.get('complexity', 'unknown')}")
                            print(f"   Suggested Tools: {analysis.get('suggested_tools', [])}")
                            print(f"   Reasoning: {analysis.get('reasoning', '')}")
                            
                            print(f"🛠️  Tool Orchestration:")
                            print(f"   Strategy: {orchestration.get('strategy', 'unknown')}")
                            print(f"   Primary Tools: {orchestration.get('primary_tools', [])}")
                            print(f"   Secondary Tools: {orchestration.get('secondary_tools', [])}")
                            print(f"   Reasoning: {orchestration.get('reasoning', '')}")
                        
                        elif data.get('type') == 'thought':
                            # NEW: Handle Cursor-style thoughts
                            thought_type = data.get('thought_type', 'unknown')
                            timestamp = data.get('timestamp', '')
                            content = data.get('content', '')
                            
                            # Color code thoughts by type
                            if thought_type == 'understanding':
                                print(f"💭 [UNDERSTANDING] {content}")
                            elif thought_type == 'analysis':
                                print(f"🔍 [ANALYSIS] {content}")
                            elif thought_type == 'tool_selection':
                                print(f"🛠️  [TOOL SELECTION] {content}")
                            elif thought_type == 'strategy':
                                print(f"📋 [STRATEGY] {content}")
                            elif thought_type == 'execution':
                                print(f"🚀 [EXECUTION] {content}")
                            elif thought_type == 'tool_execution':
                                tool_name = data.get('tool_name', 'unknown')
                                print(f"⚙️  [TOOL: {tool_name}] {content}")
                            elif thought_type == 'response_generation':
                                print(f"✍️  [RESPONSE] {content}")
                            else:
                                print(f"💡 [THOUGHT] {content}")
                        
                        elif data.get('type') == 'tool_orchestration':
                            print(f"🔧 {data.get('content', '')}")
                            print(f"   Planned Tools: {data.get('tools_planned', [])}")
                        
                        elif data.get('type') == 'tools_loaded':
                            print(f"📚 {data.get('content', '')}")
                        
                        elif data.get('type') == 'response_chunk':
                            content_buffer += data.get('content', '')
                            # Don't print chunks to avoid spam, just collect them
                        
                        elif data.get('type') == 'response_complete':
                            print(f"✅ Response Complete")
                            print(f"📝 Final Response:")
                            print(f"   {content_buffer[:200]}..." if len(content_buffer) > 200 else content_buffer)
                        
                        elif data.get('type') == 'error':
                            print(f"❌ Error: {data.get('content', '')}")
                            
                    except json.JSONDecodeError:
                        continue
            
            return {
                "success": True,
                "events": events,
                "final_content": content_buffer,
                "event_count": len(events)
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {str(e)}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def validate_intent_classification(self, events: List[Dict[str, Any]], expected_intent: str) -> bool:
        """Validate that intent classification worked correctly"""
        for event in events:
            if event.get('type') == 'analysis_complete':
                analysis = event.get('analysis', {})
                actual_intent = analysis.get('intent', '')
                confidence = analysis.get('confidence', 0)
                
                if actual_intent == expected_intent:
                    print(f"✅ Intent classification PASSED: {actual_intent} (confidence: {confidence:.2f})")
                    return True
                else:
                    print(f"❌ Intent classification FAILED: expected {expected_intent}, got {actual_intent} (confidence: {confidence:.2f})")
                    return False
        
        print(f"❌ Intent classification FAILED: no analysis_complete event found")
        return False
    
    def validate_tool_orchestration(self, events: List[Dict[str, Any]], expected_tools: List[str]) -> bool:
        """Validate that tool orchestration included expected tools"""
        for event in events:
            if event.get('type') == 'analysis_complete':
                orchestration = event.get('orchestration_plan', {})
                primary_tools = orchestration.get('primary_tools', [])
                secondary_tools = orchestration.get('secondary_tools', [])
                all_tools = primary_tools + secondary_tools
                
                # Check if at least one expected tool is in the orchestration
                found_tools = [tool for tool in expected_tools if tool in all_tools]
                if found_tools:
                    print(f"✅ Tool orchestration PASSED: found {found_tools} in {all_tools}")
                    return True
                else:
                    print(f"❌ Tool orchestration FAILED: expected {expected_tools}, got {all_tools}")
                    return False
        
        print(f"❌ Tool orchestration FAILED: no analysis_complete event found")
        return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("🚀 Starting Cursor-Style Request Handling Test Suite")
        print("=" * 80)
        
        scenarios = self.test_scenarios()
        results = {}
        
        for scenario_name, scenario in scenarios.items():
            print(f"\n🧪 Testing Scenario: {scenario_name}")
            
            # Test with cursor mode enabled
            result = self.make_request(scenario, cursor_mode=True)
            
            if result["success"]:
                # Validate intent classification
                intent_ok = self.validate_intent_classification(
                    result["events"], 
                    scenario["expected_intent"]
                )
                
                # Validate tool orchestration
                tools_ok = self.validate_tool_orchestration(
                    result["events"], 
                    scenario["expected_tools"]
                )
                
                results[scenario_name] = {
                    "success": True,
                    "intent_classification": intent_ok,
                    "tool_orchestration": tools_ok,
                    "event_count": result["event_count"],
                    "response_length": len(result["final_content"])
                }
            else:
                results[scenario_name] = {
                    "success": False,
                    "error": result["error"]
                }
        
        # Print summary
        self.print_test_summary(results)
        return results
    
    def print_test_summary(self, results: Dict[str, Dict[str, Any]]):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r["success"])
        intent_successes = sum(1 for r in results.values() if r.get("intent_classification", False))
        tool_successes = sum(1 for r in results.values() if r.get("tool_orchestration", False))
        
        print(f"Total Test Scenarios: {total_tests}")
        print(f"Successful Requests: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"Intent Classification Accuracy: {intent_successes}/{total_tests} ({intent_successes/total_tests*100:.1f}%)")
        print(f"Tool Orchestration Accuracy: {tool_successes}/{total_tests} ({tool_successes/total_tests*100:.1f}%)")
        
        print("\n📋 Detailed Results:")
        for scenario, result in results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {scenario}: {status}")
            
            if result["success"]:
                intent_status = "✅" if result.get("intent_classification", False) else "❌"
                tools_status = "✅" if result.get("tool_orchestration", False) else "❌"
                print(f"    Intent: {intent_status} | Tools: {tools_status} | Events: {result['event_count']} | Response: {result['response_length']} chars")
            else:
                print(f"    Error: {result['error']}")
        
        print("\n" + "=" * 80)
        
        # Overall assessment
        if successful_tests == total_tests and intent_successes >= total_tests * 0.8:
            print("🎉 OVERALL ASSESSMENT: EXCELLENT")
            print("   All tests passed with high accuracy!")
        elif successful_tests >= total_tests * 0.8:
            print("🎯 OVERALL ASSESSMENT: GOOD")
            print("   Most tests passed, some fine-tuning needed.")
        else:
            print("⚠️  OVERALL ASSESSMENT: NEEDS IMPROVEMENT")
            print("   Several tests failed, review implementation.")
    
    def run_single_test(self, scenario_name: str = "learning_intent"):
        """Run a single test scenario for debugging"""
        scenarios = self.test_scenarios()
        
        if scenario_name not in scenarios:
            print(f"❌ Scenario '{scenario_name}' not found")
            print(f"Available scenarios: {list(scenarios.keys())}")
            return
        
        scenario = scenarios[scenario_name]
        result = self.make_request(scenario, cursor_mode=True)
        
        if result["success"]:
            print(f"\n✅ Test completed successfully")
            print(f"Events received: {result['event_count']}")
            print(f"Response length: {len(result['final_content'])} characters")
        else:
            print(f"\n❌ Test failed: {result['error']}")


def main():
    """Main function to run the test suite"""
    demo = CursorStyleRequestDemo()
    
    # Run comprehensive test suite
    print("🎯 Cursor-Style Request Handling Test Suite")
    print("This will test intent classification, tool orchestration, and progressive enhancement")
    print("=" * 80)
    
    # Option 1: Run all tests
    demo.run_comprehensive_test()
    
    # Option 2: Run single test (uncomment to use)
    # demo.run_single_test("learning_intent")


if __name__ == "__main__":
    main() 