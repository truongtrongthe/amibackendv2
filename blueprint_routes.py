"""
Blueprint routes for direct blueprint access
Complements the nested blueprint routes in org_agent.py
"""

from fastapi import APIRouter, Depends, HTTPException
from orgdb import get_blueprint, get_user_role_in_organization, get_agent
from org_agent import blueprint_to_response, AgentBlueprintResponse
from login import get_current_user
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-blueprints", tags=["agent-blueprints"])


@router.get("/{blueprint_id}", response_model=AgentBlueprintResponse)
async def get_blueprint_by_id(
    blueprint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a blueprint directly by blueprint ID (convenience endpoint)"""
    try:
        # Get blueprint first
        blueprint = get_blueprint(blueprint_id)
        if not blueprint:
            raise HTTPException(status_code=404, detail="Blueprint not found")
        
        # Get the associated agent to check permissions
        agent = get_agent(blueprint.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Associated agent not found")
        
        # Check if user has permission to view this blueprint
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to view this blueprint")
        
        return blueprint_to_response(blueprint)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting blueprint {blueprint_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get blueprint")