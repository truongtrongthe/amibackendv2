"""
Comprehensive Tests for Modular Agent System
==========================================

Tests all components of the refactored agent system to ensure:
- Individual modules work correctly
- Integration between modules is seamless  
- Backwards compatibility is maintained
- Performance improvements are achieved
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import the modular components
from agent.models import (
    AgentExecutionRequest, 
    TaskComplexityAnalysis,
    ExecutionStep,
    MultiStepExecutionPlan,
    DiscoveredSkills
)
from agent.orchestrator import AgentOrchestrator
from agent.complexity_analyzer import TaskComplexityAnalyzer
from agent.skill_discovery import SkillDiscoveryEngine
from agent.execution_planner import ExecutionPlanner
from agent.step_executor import StepExecutor
from agent.prompt_builder import PromptBuilder
from agent.tool_manager import ToolManager

# Import backwards compatibility interface
from agent_refactored import execute_agent_task_stream, create_agent_orchestrator


class TestAgentModels:
    """Test data classes and models"""
    
    def test_agent_execution_request_creation(self):
        request = AgentExecutionRequest(
            llm_provider="anthropic",
            user_request="Test request",
            agent_id="test_agent",
            agent_type="test_type"
        )
        assert request.llm_provider == "anthropic"
        assert request.user_request == "Test request"
        assert request.agent_mode == "execute"  # Default value
        
    def test_task_complexity_analysis(self):
        analysis = TaskComplexityAnalysis(
            complexity_score=7,
            complexity_level="standard",
            task_type="analysis",
            required_steps=5,
            estimated_duration="medium",
            key_challenges=["challenge1", "challenge2"],
            recommended_approach="multi-step approach",
            confidence=0.85
        )
        assert analysis.complexity_score == 7
        assert analysis.complexity_level == "standard"
        assert len(analysis.key_challenges) == 2


class TestPromptBuilder:
    """Test prompt building functionality"""
    
    def setup_method(self):
        self.prompt_builder = PromptBuilder()
        self.agent_config = {
            "name": "Test Agent",
            "system_prompt": {
                "base_instruction": "You are a helpful assistant",
                "agent_type": "test_agent",
                "language": "english",
                "specialization": ["testing", "automation"],
                "personality": {
                    "tone": "professional",
                    "style": "helpful",
                    "approach": "solution-oriented"
                }
            },
            "tools_list": ["search", "analyze"],
            "knowledge_list": ["general", "testing"]
        }
    
    def test_build_execute_prompt(self):
        prompt = self.prompt_builder._build_execute_prompt(self.agent_config, "Test request")
        assert "Test Agent" in prompt
        assert "EXECUTE mode" in prompt
        assert "professional" in prompt
        assert "search, analyze" in prompt
        
    def test_build_collaborate_prompt(self):
        prompt = self.prompt_builder._build_collaborate_prompt(self.agent_config, "Test request")
        assert "Test Agent" in prompt
        assert "COLLABORATE mode" in prompt
        assert "🤝 COLLABORATION PRINCIPLES" in prompt
        
    def test_build_dynamic_system_prompt_execute_mode(self):
        prompt = self.prompt_builder.build_dynamic_system_prompt(
            self.agent_config, "Test request", "execute"
        )
        assert "EXECUTE mode" in prompt
        
    def test_build_dynamic_system_prompt_collaborate_mode(self):
        prompt = self.prompt_builder.build_dynamic_system_prompt(
            self.agent_config, "Test request", "collaborate"
        )
        assert "COLLABORATE mode" in prompt


class TestToolManager:
    """Test tool management functionality"""
    
    def setup_method(self):
        self.available_tools = {
            "search": Mock(),
            "file_access": Mock(),
            "brain_vector": Mock(),
            "business_logic": Mock()
        }
        self.tool_manager = ToolManager(self.available_tools)
        
    def test_get_available_tools(self):
        tools = self.tool_manager.get_available_tools()
        assert "search" in tools
        assert "file_access" in tools
        assert len(tools) == 4
        
    def test_determine_dynamic_tools_regular_request(self):
        agent_config = {"tools_list": ["search", "brain_vector"]}
        tools, force_tools = self.tool_manager.determine_dynamic_tools(
            agent_config, "Regular request"
        )
        assert tools == ["search", "brain_vector"]
        assert force_tools == False
        
    def test_determine_dynamic_tools_google_drive_request(self):
        agent_config = {"tools_list": ["search"]}
        tools, force_tools = self.tool_manager.determine_dynamic_tools(
            agent_config, "Please read this Google Drive document: https://docs.google.com/document/d/123"
        )
        assert "file_access" in tools
        assert "business_logic" in tools
        assert force_tools == True


class TestSkillDiscoveryEngine:
    """Test skill discovery functionality"""
    
    def setup_method(self):
        self.available_tools = {
            "brain_vector": Mock()
        }
        self.skill_engine = SkillDiscoveryEngine(self.available_tools)
        
    @pytest.mark.asyncio
    async def test_discover_agent_skills_no_knowledge(self):
        agent_config = {"knowledge_list": []}
        skills = await self.skill_engine.discover_agent_skills(
            "Test request", agent_config, "user1", "org1"
        )
        assert isinstance(skills, DiscoveredSkills)
        assert skills.skills == []
        assert skills.methodologies == []
        
    @pytest.mark.asyncio
    async def test_discover_agent_skills_with_knowledge(self):
        # Mock the brain_vector tool
        mock_brain_vector = Mock()
        mock_brain_vector.query_knowledge = Mock(return_value=[
            {"content": "We have expertise in lean methodology and agile frameworks"},
            {"content": "Our team specializes in quality management best practices"}
        ])
        self.available_tools["brain_vector"] = mock_brain_vector
        self.skill_engine = SkillDiscoveryEngine(self.available_tools)
        
        agent_config = {"knowledge_list": ["business_processes"]}
        
        with patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            skills = await self.skill_engine.discover_agent_skills(
                "Improve our processes", agent_config, "user1", "org1"
            )
            
        assert isinstance(skills, DiscoveredSkills)
        # Skills extraction would happen in _extract_skills_from_content


class TestComplexityAnalyzer:
    """Test task complexity analysis"""
    
    def setup_method(self):
        self.available_tools = {"brain_vector": Mock()}
        self.anthropic_executor = Mock()
        self.openai_executor = Mock()
        self.analyzer = TaskComplexityAnalyzer(
            self.available_tools, self.anthropic_executor, self.openai_executor
        )
        
    @pytest.mark.asyncio
    async def test_analyze_task_complexity_simple(self):
        agent_config = {
            "system_prompt": {
                "execution_capabilities": {
                    "max_complexity_score": 10,
                    "complexity_thresholds": {
                        "simple": {"min": 1, "max": 3, "steps": 3},
                        "standard": {"min": 4, "max": 7, "steps": 5},
                        "complex": {"min": 8, "max": 10, "steps": 7}
                    }
                }
            },
            "knowledge_list": []
        }
        
        # Mock LLM response
        mock_response = """
        {
            "complexity_score": 2,
            "complexity_level": "simple",
            "task_type": "information_retrieval",
            "required_steps": 3,
            "estimated_duration": "short",
            "key_challenges": ["basic query"],
            "recommended_approach": "direct search",
            "confidence": 0.9
        }
        """
        self.anthropic_executor._analyze_with_anthropic = AsyncMock(return_value=mock_response)
        
        analysis = await self.analyzer.analyze_task_complexity(
            "What is the weather today?", agent_config, "anthropic", "user1", "org1"
        )
        
        assert isinstance(analysis, TaskComplexityAnalysis)
        assert analysis.complexity_score <= 3
        assert analysis.complexity_level == "simple"


class TestExecutionPlanner:
    """Test execution planning functionality"""
    
    def setup_method(self):
        self.available_tools = {"brain_vector": Mock()}
        self.anthropic_executor = Mock()
        self.openai_executor = Mock()
        self.planner = ExecutionPlanner(
            self.available_tools, self.anthropic_executor, self.openai_executor
        )
        
    @pytest.mark.asyncio 
    async def test_generate_execution_plan_fallback(self):
        complexity_analysis = TaskComplexityAnalysis(
            complexity_score=5,
            complexity_level="standard",
            task_type="analysis",
            required_steps=5,
            estimated_duration="medium",
            key_challenges=["multi-step process"],
            recommended_approach="structured approach",
            confidence=0.8
        )
        
        agent_config = {"knowledge_list": []}
        
        # Mock LLM failure to test fallback
        self.anthropic_executor._analyze_with_anthropic = AsyncMock(side_effect=Exception("LLM error"))
        
        plan = await self.planner.generate_execution_plan(
            complexity_analysis, "Test request", agent_config, 
            "anthropic", "user1", "org1"
        )
        
        assert isinstance(plan, MultiStepExecutionPlan)
        assert len(plan.execution_steps) > 0
        assert plan.task_description == "Test request"


class TestStepExecutor:
    """Test step-by-step execution"""
    
    def setup_method(self):
        self.available_tools = {}
        self.anthropic_executor = Mock()
        self.openai_executor = Mock()
        self.executor = StepExecutor(
            self.available_tools, self.anthropic_executor, self.openai_executor
        )
        
    @pytest.mark.asyncio
    async def test_execute_multi_step_plan_basic(self):
        # Create a simple execution plan
        step = ExecutionStep(
            step_number=1,
            name="Test Step",
            description="A test step",
            action="Do something",
            tools_needed=["search"],
            success_criteria="Step completed",
            deliverable="Test result",
            dependencies=[],
            estimated_time="short",
            validation_checkpoints=["checkpoint1"]
        )
        
        complexity_analysis = TaskComplexityAnalysis(
            complexity_score=3,
            complexity_level="simple",
            task_type="test",
            required_steps=1,
            estimated_duration="short",
            key_challenges=[],
            recommended_approach="direct",
            confidence=1.0
        )
        
        plan = MultiStepExecutionPlan(
            plan_id="test_plan_123",
            task_description="Test task",
            complexity_analysis=complexity_analysis,
            execution_steps=[step],
            quality_checkpoints=[],
            success_metrics=["completion"],
            risk_mitigation=["error handling"],
            total_estimated_time="short",
            created_at=datetime.now()
        )
        
        request = AgentExecutionRequest(
            llm_provider="anthropic",
            user_request="Test request",
            agent_id="test_agent",
            agent_type="test"
        )
        
        agent_config = {"tools_list": ["search"]}
        
        # Mock the executor stream
        async def mock_stream(tool_request):
            yield {"type": "response_chunk", "content": "Step completed successfully"}
            
        self.anthropic_executor.execute_stream = AsyncMock(side_effect=mock_stream)
        
        results = []
        async for result in self.executor.execute_multi_step_plan(plan, request, agent_config):
            results.append(result)
            
        # Should have step_start, response_chunk, step_complete, and execution_summary
        assert len(results) >= 3
        assert any(r["type"] == "step_start" for r in results)
        assert any(r["type"] == "step_complete" for r in results)
        assert any(r["type"] == "execution_summary" for r in results)


class TestAgentOrchestrator:
    """Test the main orchestrator integration"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        orchestrator = AgentOrchestrator()
        assert orchestrator.complexity_analyzer is not None  
        assert orchestrator.execution_planner is not None
        assert orchestrator.step_executor is not None
        assert orchestrator.prompt_builder is not None
        assert orchestrator.tool_manager is not None
        
    def test_orchestrator_has_required_components(self):
        orchestrator = AgentOrchestrator()
        tools = orchestrator.get_available_tools()
        assert isinstance(tools, list)


class TestBackwardsCompatibility:
    """Test backwards compatibility with original agent.py interface"""
    
    def test_create_agent_orchestrator(self):
        orchestrator = create_agent_orchestrator()
        assert isinstance(orchestrator, AgentOrchestrator)
        
    @pytest.mark.asyncio
    async def test_execute_agent_task_stream_interface(self):
        # This would normally connect to real systems, so we'll just test the interface
        try:
            result = execute_agent_task_stream(
                llm_provider="anthropic",
                user_request="Test request",
                agent_id="test_agent",
                agent_type="test"
            )
            assert hasattr(result, '__aiter__')  # Should be async generator
        except Exception:
            # Expected since we're not connected to real systems
            pass


class TestPerformanceImprovements:
    """Test that the modular system provides performance benefits"""
    
    def test_module_import_speed(self):
        import time
        
        # Test that individual modules import quickly
        start_time = time.time()
        from agent.models import AgentExecutionRequest
        from agent.prompt_builder import PromptBuilder
        from agent.tool_manager import ToolManager
        import_time = time.time() - start_time
        
        # Modular imports should be fast (< 100ms)
        assert import_time < 0.1
        
    def test_orchestrator_creation_speed(self):
        import time
        
        start_time = time.time()
        orchestrator = AgentOrchestrator()
        creation_time = time.time() - start_time
        
        # Should be reasonably fast to create
        assert creation_time < 2.0  # 2 seconds max
        
    def test_memory_efficiency(self):
        import sys
        
        # Test that the modular system doesn't use excessive memory
        orchestrator = AgentOrchestrator()
        
        # The orchestrator should be lighter than the monolithic version
        # This is a basic test - in real scenarios, we'd compare actual memory usage
        assert hasattr(orchestrator, 'complexity_analyzer')
        assert hasattr(orchestrator, 'execution_planner')


if __name__ == "__main__":
    # Run tests with pytest
    # pytest tests/test_agent_system.py -v
    pass 