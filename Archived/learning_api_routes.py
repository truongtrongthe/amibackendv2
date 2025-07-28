#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Learning API Routes for AMI Backend

This module provides FastAPI endpoints for the interactive learning system including:
- Getting pending learning decisions
- Submitting human learning decisions
- Cleaning up expired decisions

Built by: The Fusion Lab
Date: January 2025
"""

import traceback
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import logging
logger = logging.getLogger(__name__)

# Create FastAPI router
router = APIRouter(prefix="/api/learning", tags=["learning"])

# Define request/response models
class LearningDecisionResponse(BaseModel):
    id: str = Field(..., description="Decision ID")
    type: str = Field(..., description="Decision type")
    context: str = Field(..., description="Decision context")
    options: List[str] = Field(..., description="Available options")
    additional_info: str = Field(default="", description="Additional information")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: str = Field(..., description="Expiration timestamp")

class GetLearningDecisionsResponse(BaseModel):
    success: bool = Field(..., description="Success status")
    decisions: List[LearningDecisionResponse] = Field(..., description="List of pending decisions")
    count: int = Field(..., description="Number of decisions")

class SubmitLearningDecisionRequest(BaseModel):
    decision_id: str = Field(..., description="Decision ID to submit")
    human_choice: str = Field(..., description="Human choice for the decision")

class SubmitLearningDecisionResponse(BaseModel):
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")
    decision: Optional[Dict[str, Any]] = Field(None, description="Decision details")

class CleanupDecisionsResponse(BaseModel):
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Cleanup message")

# Learning Decision Endpoints
@router.get("/decisions", response_model=GetLearningDecisionsResponse)
async def get_learning_decisions(user_id: Optional[str] = Query(None, description="User ID to filter decisions")):
    """Get pending learning decisions for a user"""
    try:
        # Import the learning tools functionality
        from learning_tools import get_pending_decisions
        
        # Get decisions for the user
        decisions = get_pending_decisions(user_id)
        
        # Format for frontend
        formatted_decisions = []
        for decision_id, decision in decisions.items():
            if decision.get('status') == 'PENDING':
                formatted_decisions.append(LearningDecisionResponse(
                    id=decision_id,
                    type=decision.get('type', ''),
                    context=decision.get('context', ''),
                    options=decision.get('options', []),
                    additional_info=decision.get('additional_info', ''),
                    created_at=decision.get('created_at', ''),
                    expires_at=decision.get('expires_at', '')
                ))
        
        return GetLearningDecisionsResponse(
            success=True,
            decisions=formatted_decisions,
            count=len(formatted_decisions)
        )
        
    except Exception as e:
        logger.error(f"Error getting learning decisions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decision", response_model=SubmitLearningDecisionResponse)
async def submit_learning_decision(request: SubmitLearningDecisionRequest):
    """Submit a human decision for a learning request"""
    try:
        if not request.decision_id or not request.human_choice:
            raise HTTPException(status_code=400, detail="Missing decision_id or human_choice")
        
        # Import the learning tools functionality
        from learning_tools import complete_learning_decision
        
        # Complete the decision
        result = await complete_learning_decision(request.decision_id, request.human_choice)
        
        if result['success']:
            return SubmitLearningDecisionResponse(
                success=True,
                message='Decision submitted successfully',
                decision=result['decision']
            )
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting learning decision: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decisions/cleanup", response_model=CleanupDecisionsResponse)
async def cleanup_learning_decisions():
    """Clean up expired learning decisions"""
    try:
        # Import the learning tools functionality
        from learning_tools import cleanup_expired_decisions
        
        # Clean up expired decisions
        cleaned_count = cleanup_expired_decisions()
        
        return CleanupDecisionsResponse(
            success=True,
            message=f'Cleaned up {cleaned_count} expired decisions'
        )
        
    except Exception as e:
        logger.error(f"Error cleaning up learning decisions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def learning_health_check():
    """Health check for learning system"""
    try:
        # Test that learning tools can be imported
        from learning_tools import LearningToolsFactory
        
        # Get tool definitions to verify tools are working
        tool_definitions = LearningToolsFactory.get_tool_definitions()
        
        return {
            "status": "healthy",
            "learning_tools_available": len(tool_definitions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Learning health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Learning system unhealthy: {str(e)}") 