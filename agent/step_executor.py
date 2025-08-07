"""
Step Executor Module
===================

Handles step-by-step execution of multi-step plans with progress tracking.
"""

import logging
from typing import Dict, Any, AsyncGenerator, List
from datetime import datetime
from .models import MultiStepExecutionPlan, AgentExecutionRequest, ExecutionStep, StepExecutionResult

logger = logging.getLogger(__name__)


class StepExecutor:
    """Executes multi-step plans with step-by-step progress tracking"""
    
    def __init__(self, available_tools: Dict[str, Any], anthropic_executor, openai_executor):
        self.available_tools = available_tools
        self.anthropic_executor = anthropic_executor
        self.openai_executor = openai_executor
    
    async def execute_multi_step_plan(self, execution_plan: MultiStepExecutionPlan, 
                                    request: AgentExecutionRequest, 
                                    agent_config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute multi-step plan with step-by-step progress tracking
        
        Args:
            execution_plan: The multi-step execution plan
            request: Original execution request
            agent_config: Agent configuration
            
        Yields:
            Dict containing step execution results and progress updates
        """
        
        step_results = {}
        
        for step in execution_plan.execution_steps:
            step_start_time = datetime.now()
            
            # Enhanced logging for backend tracing
            logger.info(f"🔄 [STEP {step.step_number}/{len(execution_plan.execution_steps)}] Starting: {step.name}")
            logger.info(f"   📋 Description: {step.description[:100]}...")
            logger.info(f"   🛠️  Tools needed: {', '.join(step.tools_needed)}")
            logger.info(f"   ⏱️  Estimated time: {step.estimated_time}")
            logger.info(f"   🔗 Dependencies: {step.dependencies}")

            # Yield step start notification
            yield {
                "type": "step_start",
                "content": f"🔄 Step {step.step_number}/{len(execution_plan.execution_steps)}: {step.name}",
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "step_info": {
                    "step_number": step.step_number,
                    "name": step.name,
                    "description": step.description,
                    "estimated_time": step.estimated_time,
                    "tools_needed": step.tools_needed,
                    "dependencies": step.dependencies
                }
            }
            
            try:
                # Check dependencies with enhanced logging
                if step.dependencies:
                    logger.info(f"   🔍 Checking dependencies: {step.dependencies}")
                    for dep_step in step.dependencies:
                        if dep_step not in step_results:
                            logger.error(f"   ❌ Dependency Step {dep_step} not found in results")
                        elif step_results[dep_step].status != "completed":
                            logger.error(f"   ❌ Dependency Step {dep_step} status: {step_results[dep_step].status}")
                        else:
                            logger.info(f"   ✅ Dependency Step {dep_step} completed successfully")
                    
                    for dep_step in step.dependencies:
                        if dep_step not in step_results or step_results[dep_step].status != "completed":
                            error_msg = f"❌ Step {step.step_number} dependency not met: Step {dep_step} must complete first"
                            logger.error(f"   {error_msg}")
                            yield {
                                "type": "error",
                                "content": error_msg,
                                "provider": request.llm_provider,
                                "agent_id": request.agent_id,
                                "step_number": step.step_number
                            }
                            return
                else:
                    logger.info(f"   ✅ No dependencies to check")
                
                # Build step-specific prompt with logging
                logger.info(f"   🏗️  Building step execution prompt...")
                step_prompt = self._build_step_execution_prompt(
                    step, execution_plan, step_results, agent_config, request.user_request
                )
                logger.info(f"   📝 Prompt built successfully ({len(step_prompt)} chars)")

                # Execute step using tool request
                logger.info(f"   🚀 Creating tool request for step execution...")
                step_tool_request = self._create_step_tool_request(
                    request, agent_config, step_prompt, step.tools_needed
                )
                logger.info(f"   🛠️  Tool request created with {len(step.tools_needed)} tools")
                
                # Execute and collect results
                step_response_chunks = []
                tool_calls_made = []
                
                if request.llm_provider.lower() == "anthropic":
                    async for chunk in self.anthropic_executor.execute_stream(step_tool_request):
                        # Track tool calls
                        if chunk.get("type") == "tool_execution":
                            tool_calls_made.append(chunk.get("tool_name"))
                        
                        # Forward chunk with step context
                        chunk["step_number"] = step.step_number
                        chunk["step_name"] = step.name
                        yield chunk
                        
                        # Collect response chunks
                        if chunk.get("type") == "response_chunk":
                            step_response_chunks.append(chunk.get("content", ""))
                else:
                    async for chunk in self.openai_executor.execute_stream(step_tool_request):
                        # Track tool calls
                        if chunk.get("type") == "tool_execution":
                            tool_calls_made.append(chunk.get("tool_name"))
                        
                        # Forward chunk with step context
                        chunk["step_number"] = step.step_number
                        chunk["step_name"] = step.name
                        yield chunk
                        
                        # Collect response chunks
                        if chunk.get("type") == "response_chunk":
                            step_response_chunks.append(chunk.get("content", ""))
                
                # Combine response with logging
                step_response = "".join(step_response_chunks)
                step_execution_time = (datetime.now() - step_start_time).total_seconds()
                
                logger.info(f"   📊 Step execution completed:")
                logger.info(f"      ⏱️  Execution time: {step_execution_time:.2f}s")
                logger.info(f"      🔧 Tools used: {list(set(tool_calls_made))}")
                logger.info(f"      📄 Response length: {len(step_response)} chars")
                logger.info(f"      📦 Chunks received: {len(step_response_chunks)}")

                # Validate step success criteria
                success_criteria_met = len(step_response.strip()) > 0  # Basic validation: non-empty response
                step_status = "completed" if success_criteria_met else "failed"
                
                if not success_criteria_met:
                    logger.warning(f"   ⚠️  Step {step.step_number} completed but failed success criteria:")
                    logger.warning(f"      📄 Response empty: {len(step_response)} chars")
                    logger.warning(f"      📦 No chunks received: {len(step_response_chunks)} chunks")
                    logger.warning(f"      🔧 Tools used: {list(set(tool_calls_made))}")
                
                # Create step result
                step_result = StepExecutionResult(
                    step_number=step.step_number,
                    status=step_status,
                    deliverable={
                        "response": step_response,
                        "summary": f"Step {step.step_number} {step_status}: {step.name}"
                    },
                    tools_used=list(set(tool_calls_made)),
                    execution_time=step_execution_time,
                    success_criteria_met=success_criteria_met,
                    validation_results=["Response content validation"],
                    next_actions=[] if success_criteria_met else ["Retry step", "Check LLM response flow"]
                )
                
                step_results[step.step_number] = step_result
                
                # Yield step completion or failure based on actual status
                completion_icon = "✅" if step_status == "completed" else "⚠️"
                yield {
                    "type": "step_complete" if step_status == "completed" else "step_error",
                    "content": f"{completion_icon} Step {step.step_number} {step_status}: {step.name} ({step_execution_time:.1f}s)",
                    "provider": request.llm_provider,
                    "agent_id": request.agent_id,
                    "step_result": {
                        "step_number": step.step_number,
                        "name": step.name,
                        "status": step_status,
                        "execution_time": step_execution_time,
                        "tools_used": step_result.tools_used,
                        "success_criteria_met": step_result.success_criteria_met,
                        "response_length": len(step_response),
                        "chunks_received": len(step_response_chunks)
                    }
                }
                
                # Check for quality checkpoints
                for checkpoint in execution_plan.quality_checkpoints:
                    if checkpoint.get("trigger_after_step") == step.step_number:
                        yield {
                            "type": "checkpoint",
                            "content": f"🔍 Quality Checkpoint: {checkpoint['checkpoint_name']}",
                            "provider": request.llm_provider,
                            "agent_id": request.agent_id,
                            "checkpoint": checkpoint
                        }
                
            except Exception as e:
                step_execution_time = (datetime.now() - step_start_time).total_seconds()
                
                # Enhanced error logging for debugging
                logger.error(f"   ❌ Step {step.step_number} EXCEPTION occurred:")
                logger.error(f"      🐛 Error type: {type(e).__name__}")
                logger.error(f"      📝 Error message: {str(e)}")
                logger.error(f"      ⏱️  Failed after: {step_execution_time:.2f}s")
                logger.error(f"      📍 Error location: Step execution during LLM call")
                
                # Log stack trace for debugging
                import traceback
                logger.error(f"      🔍 Stack trace: {traceback.format_exc()}")
                
                # Create failed step result
                step_result = StepExecutionResult(
                    step_number=step.step_number,
                    status="failed",
                    deliverable={"error": str(e)},
                    tools_used=[],
                    execution_time=step_execution_time,
                    success_criteria_met=False,
                    validation_results=[],
                    next_actions=["Retry step", "Skip to next step", "Abort execution"],
                    error_details=str(e)
                )
                
                step_results[step.step_number] = step_result
                
                yield {
                    "type": "step_error",
                    "content": f"❌ Step {step.step_number} failed: {step.name} - {str(e)}",
                    "provider": request.llm_provider,
                    "agent_id": request.agent_id,
                    "step_result": {
                        "step_number": step.step_number,
                        "name": step.name,
                        "status": "failed",
                        "error": str(e),
                        "execution_time": step_execution_time
                    }
                }
                
                # For now, continue with next step (could add retry logic here)
                continue
        
        # Multi-step execution summary
        completed_steps = sum(1 for result in step_results.values() if result.status == "completed")
        failed_steps = sum(1 for result in step_results.values() if result.status == "failed")
        total_execution_time = sum(result.execution_time for result in step_results.values())
        
        yield {
            "type": "execution_summary",
            "content": f"📋 Multi-step execution summary: {completed_steps}/{len(execution_plan.execution_steps)} steps completed, {failed_steps} failed, {total_execution_time:.1f}s total",
            "provider": request.llm_provider,
            "agent_id": request.agent_id,
            "summary": {
                "total_steps": len(execution_plan.execution_steps),
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "total_execution_time": total_execution_time,
                "success_rate": completed_steps / len(execution_plan.execution_steps) if execution_plan.execution_steps else 0
            }
        }
    
    def _build_step_execution_prompt(self, step: ExecutionStep, execution_plan: MultiStepExecutionPlan,
                                   step_results: Dict[int, StepExecutionResult], agent_config: Dict[str, Any],
                                   original_request: str) -> str:
        """Build step-specific execution prompt"""
        
        # Get previous step context
        previous_context = ""
        if step_results:
            previous_context = "\nPREVIOUS STEPS COMPLETED:\n"
            for step_num, result in step_results.items():
                if result.status == "completed":
                    previous_context += f"- Step {step_num}: {result.deliverable.get('summary', 'Completed')}\n"
        
        step_prompt = f"""
            You are executing Step {step.step_number} of a {len(execution_plan.execution_steps)}-step plan.

            ORIGINAL REQUEST: "{original_request}"

            CURRENT STEP:
            - Step Number: {step.step_number}/{len(execution_plan.execution_steps)}
            - Name: {step.name}
            - Description: {step.description}
            - Action: {step.action}
            - Success Criteria: {step.success_criteria}
            - Expected Deliverable: {step.deliverable}

            AVAILABLE TOOLS FOR THIS STEP: {', '.join(step.tools_needed)}

            {previous_context}

            EXECUTION INSTRUCTIONS:
            1. Focus specifically on completing this step's objectives
            2. Use the available tools as needed to accomplish the step action
            3. Ensure your output meets the success criteria
            4. Provide the deliverable as described

            Complete this step now:
            """
        
        return step_prompt
    
    def _create_step_tool_request(self, request: AgentExecutionRequest, agent_config: Dict[str, Any], 
                                step_prompt: str, step_tools: List[str], blueprint_tool_configs: Dict[str, Any] = None):
        """Create tool request for a specific step"""
        from exec_tool import ToolExecutionRequest
        
        # Determine tools for this step
        tools_whitelist = step_tools if step_tools else None
        force_tools = bool(step_tools)  # Force tools if specific tools are needed
        
        return ToolExecutionRequest(
            llm_provider=request.llm_provider,
            user_query=request.user_request,
            system_prompt=step_prompt,
            model=request.model,
            model_params=request.model_params,
            tools_config=blueprint_tool_configs or {},  # Pass blueprint tool configurations
            org_id=request.org_id,
            user_id=request.user_id,
            enable_tools=True,
            force_tools=force_tools,
            tools_whitelist=tools_whitelist,
            conversation_history=None,  # Don't pass history for individual steps
            max_history_messages=0,
            max_history_tokens=0,
            # Simplified settings for step execution
            enable_deep_reasoning=False,
            reasoning_depth="light",
            enable_intent_classification=False,
            enable_request_analysis=False,
            cursor_mode=False
        ) 