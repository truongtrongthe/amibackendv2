from supabase import create_client, Client
from typing import List, Dict, Optional
import os
import json
from datetime import datetime, timezone
from utilities import logger
from uuid import UUID

# Initialize Supabase client
# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(
    spb_url,
    spb_key
)


class Organization:
    def __init__(self, id: str, org_id: int, name: str, description: Optional[str], 
                 email: Optional[str], phone: Optional[str], address: Optional[str],
                 created_date: datetime):
        self.id = id  # UUID
        self.org_id = org_id  # INT
        self.name = name
        self.description = description
        self.email = email
        self.phone = phone
        self.address = address
        self.created_date = created_date

class Brain:
    def __init__(self, id: str, brain_id: int, org_id: str, name: str, status: str, bank_name: str, 
                summary: Optional[str], created_date: datetime):
        self.id = id  # UUID
        self.brain_id = brain_id  # INT
        self.org_id = org_id  # UUID
        self.name = name
        self.status = status
        self.bank_name = bank_name
        self.summary = summary
        self.created_date = created_date

class Agent:
    def __init__(self, id: str, agent_id: int, org_id: str, name: str, description: Optional[str],
                 status: str, created_by: str, created_date: datetime, updated_date: datetime,
                 current_blueprint_id: Optional[str] = None):
        self.id = id  # UUID
        self.agent_id = agent_id  # INT
        self.org_id = org_id  # UUID
        self.name = name
        self.description = description
        self.status = status  # 'active', 'deactive', 'delete'
        self.created_by = created_by  # User ID
        self.created_date = created_date
        self.updated_date = updated_date
        self.current_blueprint_id = current_blueprint_id  # UUID to current blueprint

class AgentBlueprint:
    def __init__(self, id: str, agent_id: str, version: int, agent_blueprint: dict,
                 created_date: datetime, created_by: str, conversation_id: Optional[str] = None,
                 compiled_system_prompt: Optional[str] = None, compiled_at: Optional[datetime] = None,
                 compiled_by: Optional[str] = None, compilation_status: str = 'draft',
                 implementation_todos: Optional[list] = None, todos_completion_status: str = 'not_generated',
                 todos_generated_at: Optional[datetime] = None, todos_generated_by: Optional[str] = None,
                 todos_completed_at: Optional[datetime] = None, todos_completed_by: Optional[str] = None):
        self.id = id  # UUID
        self.agent_id = agent_id  # UUID
        self.version = version  # INT
        self.agent_blueprint = agent_blueprint  # JSONB field
        self.created_date = created_date
        self.created_by = created_by  # User ID
        self.conversation_id = conversation_id  # Optional conversation link
        self.compiled_system_prompt = compiled_system_prompt  # Compiled prompt for LLM
        self.compiled_at = compiled_at  # When it was compiled
        self.compiled_by = compiled_by  # Who compiled it
        self.compilation_status = compilation_status  # draft, todos_pending, ready_for_compilation, compiled, failed
        
        # Todo-related fields
        self.implementation_todos = implementation_todos or []  # List of implementation todos
        self.todos_completion_status = todos_completion_status  # not_generated, generated, in_progress, completed
        self.todos_generated_at = todos_generated_at  # When Ami generated todos
        self.todos_generated_by = todos_generated_by  # Who triggered todo generation
        self.todos_completed_at = todos_completed_at  # When all todos were completed
        self.todos_completed_by = todos_completed_by  # Who marked todos as completed

def create_organization(name: str, description: Optional[str] = None, 
                        email: Optional[str] = None, phone: Optional[str] = None, 
                        address: Optional[str] = None) -> Organization:
    """
    Create a new organization with contact information and created_date
    
    Args:
        name: Organization name
        description: Optional description
        email: Optional contact email
        phone: Optional contact phone
        address: Optional physical address
    
    Returns:
        Organization object
    """
    data = {
        "name": name,
        "description": description,
        "email": email,
        "phone": phone,
        "address": address,
        "created_date": datetime.now(timezone.utc).isoformat()
    }
    
    response = supabase.table("organization").insert(data).execute()
    
    if response.data:
        org_data = response.data[0]
        return Organization(
            id=org_data["id"],
            org_id=org_data["org_id"],
            name=org_data["name"],
            description=org_data["description"],
            email=org_data["email"],
            phone=org_data["phone"],
            address=org_data["address"],
            created_date=datetime.fromisoformat(org_data["created_date"].replace("Z", "+00:00"))
        )
    raise Exception("Failed to create organization")

def update_organization(id: str, name: str, description: Optional[str] = None,
                        email: Optional[str] = None, phone: Optional[str] = None,
                        address: Optional[str] = None) -> Organization:
    """
    Update an organization's information
    
    Args:
        id: UUID of the organization
        name: Organization name
        description: Optional description
        email: Optional contact email
        phone: Optional contact phone
        address: Optional physical address
    
    Returns:
        Updated Organization object
    """
    update_data = {
        "name": name,
        "description": description
    }
    
    # Only include fields in update if they're provided
    if email is not None:
        update_data["email"] = email
    if phone is not None:
        update_data["phone"] = phone
    if address is not None:
        update_data["address"] = address
    
    response = supabase.table("organization")\
        .update(update_data)\
        .eq("id", id)\
        .execute()
    
    if response.data and len(response.data) > 0:
        org_data = response.data[0]
        return Organization(
            id=org_data["id"],
            org_id=org_data["org_id"],
            name=org_data["name"],
            description=org_data["description"],
            email=org_data["email"],
            phone=org_data["phone"],
            address=org_data["address"],
            created_date=datetime.fromisoformat(org_data["created_date"].replace("Z", "+00:00"))
        )
    raise Exception("Failed to update organization or organization not found")

def get_organization(org_id: str) -> Optional[Organization]:
    """
    Fetch details of a specific organization using the UUID id
    """
    response = supabase.table("organization")\
        .select("*")\
        .eq("id", org_id)\
        .execute()
    
    if response.data and len(response.data) > 0:
        org_data = response.data[0]
        return Organization(
            id=org_data["id"],
            org_id=org_data["org_id"],
            name=org_data["name"],
            description=org_data["description"],
            email=org_data.get("email"),
            phone=org_data.get("phone"),
            address=org_data.get("address"),
            created_date=datetime.fromisoformat(org_data["created_date"].replace("Z", "+00:00"))
        )
    return None

def create_brain(org_id: str, user_id: str, name: str,summary: str) -> Brain:
    """
    Create a new brain record with default values including created_date
    """
    org_check = supabase.table("organization")\
        .select("id")\
        .eq("id", org_id)\
        .execute()
    
    if not org_check.data:
        raise ValueError(f"Organization with id {org_id} does not exist")

    logger.info(f"summary at create brain={summary}")
    temp_bank_name = f"temp_wisdom_bank_{org_id}"
    data = {
        "org_id": org_id,
        "name": name,
        "status": "training",
        "bank_name": temp_bank_name,
        "summary": summary,
        "created_date": datetime.now(timezone.utc).isoformat()
    }
    
    response = supabase.table("brain").insert(data).execute()
    
    if response.data:
        brain_data = response.data[0]
        final_bank_name = f"wisdom_bank_{org_id}_{brain_data['brain_id']}"
        
        update_response = supabase.table("brain")\
            .update({"bank_name": final_bank_name})\
            .eq("id", brain_data["id"])\
            .execute()
        
        if update_response.data:
            brain_data = update_response.data[0]
            return Brain(
                id=brain_data["id"],
                brain_id=brain_data["brain_id"],
                org_id=brain_data["org_id"],
                name=brain_data["name"],
                status=brain_data["status"],
                bank_name=brain_data["bank_name"],
                summary=brain_data["summary"],
                created_date=datetime.fromisoformat(brain_data["created_date"].replace("Z", "+00:00"))
            )
    raise Exception("Failed to create brain")

def get_brains(org_id: str) -> List[Brain]:
    """
    Fetch all brains for a given organization using UUID org_id
    """
    try:
        logger.info(f"Fetching brains for org_id: {org_id}")
        
        # Validate org_id format
        try:
            UUID(org_id)
        except ValueError:
            logger.error(f"Invalid org_id format: {org_id}")
            raise ValueError("Invalid org_id format - must be a valid UUID")
        
        # Set a timeout for the query
        response = supabase.table("brain")\
            .select("*")\
            .eq("org_id", org_id)\
            .execute()
        
        brains = []
        if response.data:
            for brain_data in response.data:
                try:
                    # Ensure all required fields are present
                    required_fields = ["id", "brain_id", "org_id", "name", "status", "bank_name"]
                    if not all(field in brain_data for field in required_fields):
                        logger.warning(f"Missing required fields in brain data: {brain_data}")
                        continue
                        
                    brains.append(Brain(
                        id=brain_data["id"],
                        brain_id=brain_data["brain_id"],
                        org_id=brain_data["org_id"],
                        name=brain_data["name"],
                        status=brain_data["status"],
                        bank_name=brain_data["bank_name"],
                        summary=brain_data.get("summary"),  # Optional field
                        created_date=datetime.fromisoformat(brain_data.get("created_date", "").replace("Z", "+00:00"))
                    ))
                except Exception as e:
                    logger.error(f"Error processing brain data: {e}")
                    continue
        
        logger.info(f"Successfully fetched {len(brains)} brains")
        return brains
        
    except Exception as e:
        logger.error(f"Error in get_brains: {str(e)}")
        raise

def get_brain_details(brain_id: str) -> Optional[Brain]:
    """
    Fetch details of a specific brain using the UUID id
    """
    response = supabase.table("brain")\
        .select("*")\
        .eq("id", brain_id)\
        .execute()
    
    if response.data and len(response.data) > 0:
        brain_data = response.data[0]
        return Brain(
            id=brain_data["id"],
            brain_id=brain_data["brain_id"],
            org_id=brain_data["org_id"],
            name=brain_data["name"],
            status=brain_data["status"],
            bank_name=brain_data["bank_name"],
            summary=brain_data["summary"],
            created_date=datetime.fromisoformat(brain_data["created_date"].replace("Z", "+00:00"))
        )
    return None

def update_brain(id: str, new_name: str, new_status: str) -> Brain:
    """
    Update a brain's name and status using the UUID id
    """
    valid_statuses = {"training", "published", "closed"}
    if new_status not in valid_statuses:
        raise ValueError(f"Status must be one of {valid_statuses}")

    update_data = {
        "name": new_name,
        "status": new_status
    }
    
    response = supabase.table("brain")\
        .update(update_data)\
        .eq("id", id)\
        .execute()
    
    if response.data and len(response.data) > 0:
        brain_data = response.data[0]
        return Brain(
            id=brain_data["id"],
            brain_id=brain_data["brain_id"],
            org_id=brain_data["org_id"],
            name=brain_data["name"],
            status=brain_data["status"],
            bank_name=brain_data["bank_name"],
            summary=brain_data["summary"],
            created_date=datetime.fromisoformat(brain_data["created_date"].replace("Z", "+00:00"))
        )
    raise Exception("Failed to update brain or brain not found")

def find_organization_by_name(name: str) -> Optional[Organization]:
    """
    Find organization by name (case-insensitive)
    
    Args:
        name: Organization name to search for
    
    Returns:
        Organization object if found, None otherwise
    """
    try:
        response = supabase.table("organization")\
            .select("*")\
            .ilike("name", name)\
            .execute()
        
        if response.data and len(response.data) > 0:
            org_data = response.data[0]
            return Organization(
                id=org_data["id"],
                org_id=org_data["org_id"],
                name=org_data["name"],
                description=org_data["description"],
                email=org_data.get("email"),
                phone=org_data.get("phone"),
                address=org_data.get("address"),
                created_date=datetime.fromisoformat(org_data["created_date"].replace("Z", "+00:00"))
            )
        return None
    except Exception as e:
        logger.error(f"Error finding organization by name: {str(e)}")
        return None

def search_organizations(query: str, limit: int = 10) -> List[Organization]:
    """
    Search organizations by name (case-insensitive partial match)
    
    Args:
        query: Search query
        limit: Maximum number of results
    
    Returns:
        List of matching Organization objects
    """
    try:
        response = supabase.table("organization")\
            .select("*")\
            .ilike("name", f"%{query}%")\
            .limit(limit)\
            .execute()
        
        organizations = []
        if response.data:
            for org_data in response.data:
                organizations.append(Organization(
                    id=org_data["id"],
                    org_id=org_data["org_id"],
                    name=org_data["name"],
                    description=org_data["description"],
                    email=org_data.get("email"),
                    phone=org_data.get("phone"),
                    address=org_data.get("address"),
                    created_date=datetime.fromisoformat(org_data["created_date"].replace("Z", "+00:00"))
                ))
        return organizations
    except Exception as e:
        logger.error(f"Error searching organizations: {str(e)}")
        return []

def add_user_to_organization(user_id: str, org_id: str, role: str = "member") -> bool:
    """
    Add user as member of organization
    
    Args:
        user_id: User ID from users table
        org_id: Organization UUID
        role: User role in organization (owner, admin, member)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate organization exists
        org = get_organization(org_id)
        if not org:
            raise ValueError(f"Organization with id {org_id} does not exist")
        
        # Check if user is already a member
        existing = supabase.table("user_organizations")\
            .select("*")\
            .eq("user_id", user_id)\
            .eq("org_id", org_id)\
            .execute()
        
        if existing.data:
            logger.info(f"User {user_id} is already a member of organization {org_id}")
            return True
        
        # Add user to organization
        data = {
            "user_id": user_id,
            "org_id": org_id,
            "role": role,
            "joined_at": datetime.now(timezone.utc).isoformat()
        }
        
        response = supabase.table("user_organizations").insert(data).execute()
        return bool(response.data)
    
    except Exception as e:
        logger.error(f"Error adding user to organization: {str(e)}")
        return False

def get_user_organization(user_id: str) -> Optional[Organization]:
    """
    Get the organization that a user belongs to
    
    Args:
        user_id: User ID from users table
    
    Returns:
        Organization object if user belongs to one, None otherwise
    """
    try:
        response = supabase.table("user_organizations")\
            .select("org_id, role")\
            .eq("user_id", user_id)\
            .execute()
        
        if response.data and len(response.data) > 0:
            org_id = response.data[0]["org_id"]
            return get_organization(org_id)
        
        return None
    except Exception as e:
        logger.error(f"Error getting user organization: {str(e)}")
        return None

def get_user_role_in_organization(user_id: str, org_id: str) -> Optional[str]:
    """
    Get user's role in a specific organization
    
    Args:
        user_id: User ID from users table
        org_id: Organization UUID
    
    Returns:
        Role string if user is member, None otherwise
    """
    try:
        response = supabase.table("user_organizations")\
            .select("role")\
            .eq("user_id", user_id)\
            .eq("org_id", org_id)\
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]["role"]
        
        return None
    except Exception as e:
        logger.error(f"Error getting user role in organization: {str(e)}")
        return None

def remove_user_from_organization(user_id: str, org_id: str) -> bool:
    """
    Remove user from organization
    
    Args:
        user_id: User ID from users table
        org_id: Organization UUID
    
    Returns:
        True if successful, False otherwise
    """
    try:
        response = supabase.table("user_organizations")\
            .delete()\
            .eq("user_id", user_id)\
            .eq("org_id", org_id)\
            .execute()
        return True
    except Exception as e:
        logger.error(f"Error removing user from organization: {str(e)}")
        return False

def get_user_owned_organizations(user_id: str) -> List[Organization]:
    """
    Get all organizations where the user is an owner
    
    Args:
        user_id: User ID from users table
    
    Returns:
        List of Organization objects where user is owner
    """
    try:
        response = supabase.table("user_organizations")\
            .select("org_id")\
            .eq("user_id", user_id)\
            .eq("role", "owner")\
            .execute()
        
        organizations = []
        if response.data:
            for user_org in response.data:
                org = get_organization(user_org["org_id"])
                if org:
                    organizations.append(org)
        
        return organizations
    except Exception as e:
        logger.error(f"Error getting user owned organizations: {str(e)}")
        return []

def get_user_organizations(user_id: str) -> List[tuple]:
    """
    Get all organizations a user belongs to with their roles
    
    Args:
        user_id: User ID from users table
    
    Returns:
        List of tuples (Organization, role)
    """
    try:
        response = supabase.table("user_organizations")\
            .select("org_id, role")\
            .eq("user_id", user_id)\
            .execute()
        
        organizations = []
        if response.data:
            for user_org in response.data:
                org = get_organization(user_org["org_id"])
                if org:
                    organizations.append((org, user_org["role"]))
        
        return organizations
    except Exception as e:
        logger.error(f"Error getting user organizations: {str(e)}")
        return []

def get_organization_members(org_id: str) -> List[Dict]:
    """
    Get all users and their roles for a specific organization
    
    Args:
        org_id: Organization UUID
    
    Returns:
        List of dictionaries containing user info and role
    """
    try:
        # Get all user-organization relationships for this org
        response = supabase.table("user_organizations")\
            .select("user_id, role, joined_at")\
            .eq("org_id", org_id)\
            .execute()
        
        members = []
        if response.data:
            # Get user details for each member
            user_ids = [member["user_id"] for member in response.data]
            
            # Fetch user details in batch
            users_response = supabase.table("users")\
                .select("id, email, name, phone, avatar, provider, email_verified")\
                .in_("id", user_ids)\
                .execute()
            
            # Create a lookup dict for users
            users_dict = {user["id"]: user for user in users_response.data} if users_response.data else {}
            
            # Combine user details with role information
            for member in response.data:
                user_id = member["user_id"]
                if user_id in users_dict:
                    user = users_dict[user_id]
                    members.append({
                        "user_id": user["id"],
                        "email": user["email"],
                        "name": user["name"],
                        "phone": user.get("phone"),
                        "avatar": user.get("avatar"),
                        "provider": user.get("provider", "email"),
                        "email_verified": user.get("email_verified", False),
                        "role": member["role"],
                        "joined_at": member["joined_at"]
                    })
        
        return members
    except Exception as e:
        logger.error(f"Error getting organization members: {str(e)}")
        return []

# Agent CRUD Functions

def create_agent(org_id: str, created_by: str, name: str, description: Optional[str] = None) -> Agent:
    """
    Create a new agent for an organization (without blueprint)
    
    Args:
        org_id: Organization UUID
        created_by: User ID who created the agent
        name: Agent name
        description: Optional description
    
    Returns:
        Agent object
    """
    try:
        # Validate organization exists
        org = get_organization(org_id)
        if not org:
            raise ValueError(f"Organization with id {org_id} does not exist")
        
        data = {
            "org_id": org_id,
            "created_by": created_by,
            "name": name,
            "description": description,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        response = supabase.table("org_agents").insert(data).execute()
        
        if response.data:
            agent_data = response.data[0]
            return Agent(
                id=agent_data["id"],
                agent_id=agent_data["agent_id"],
                org_id=agent_data["org_id"],
                name=agent_data["name"],
                description=agent_data.get("description"),
                status=agent_data["status"],
                created_by=agent_data["created_by"],
                created_date=datetime.fromisoformat(agent_data["created_at"].replace("Z", "+00:00")),
                updated_date=datetime.fromisoformat(agent_data["updated_at"].replace("Z", "+00:00")),
                current_blueprint_id=agent_data.get("current_blueprint_id")
            )
        raise Exception("Failed to create agent")
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise

def create_agent_with_blueprint(org_id: str, created_by: str, name: str, 
                               blueprint_data: dict, description: Optional[str] = None,
                               conversation_id: Optional[str] = None) -> tuple[Agent, AgentBlueprint]:
    """
    Create a new agent with its initial blueprint
    
    Args:
        org_id: Organization UUID
        created_by: User ID who created the agent
        name: Agent name
        blueprint_data: Complete blueprint JSON
        description: Optional description
        conversation_id: Optional conversation that created this
    
    Returns:
        Tuple of (Agent, AgentBlueprint)
    """
    try:
        # Create the agent first
        agent = create_agent(org_id, created_by, name, description)
        
        # Create the blueprint
        blueprint = create_blueprint(agent.id, blueprint_data, created_by, conversation_id)
        
        # Update agent with blueprint reference
        updated_agent = update_agent_current_blueprint(agent.id, blueprint.id)
        
        return updated_agent, blueprint
        
    except Exception as e:
        logger.error(f"Error creating agent with blueprint: {str(e)}")
        raise

def get_agents(org_id: str, status: str = "active") -> List[Agent]:
    """
    Get all agents for an organization with optional status filter
    
    Args:
        org_id: Organization UUID
        status: Status filter ('active', 'deactive', 'delete', or None for all)
    
    Returns:
        List of Agent objects
    """
    try:
        # Validate org_id format
        try:
            UUID(org_id)
        except ValueError:
            logger.error(f"Invalid org_id format: {org_id}")
            raise ValueError("Invalid org_id format - must be a valid UUID")
        
        query = supabase.table("org_agents").select("*").eq("org_id", org_id)
        
        if status:
            query = query.eq("status", status)
        
        response = query.execute()
        
        agents = []
        if response.data:
            for agent_data in response.data:
                try:
                    agents.append(Agent(
                        id=agent_data["id"],
                        agent_id=agent_data["agent_id"],
                        org_id=agent_data["org_id"],
                        name=agent_data["name"],
                        description=agent_data.get("description"),
                        status=agent_data["status"],
                        created_by=agent_data["created_by"],
                        created_date=datetime.fromisoformat(agent_data["created_at"].replace("Z", "+00:00")),
                        updated_date=datetime.fromisoformat(agent_data["updated_at"].replace("Z", "+00:00")),
                        current_blueprint_id=agent_data.get("current_blueprint_id")
                    ))
                except Exception as e:
                    logger.error(f"Error processing agent data: {e}")
                    continue
        
        logger.info(f"Successfully fetched {len(agents)} agents for org_id: {org_id}")
        return agents
        
    except Exception as e:
        logger.error(f"Error in get_agents: {str(e)}")
        raise

def get_agent(agent_id: str) -> Optional[Agent]:
    """
    Get a specific agent by ID
    
    Args:
        agent_id: Agent UUID
    
    Returns:
        Agent object if found, None otherwise
    """
    try:
        response = supabase.table("org_agents").select("*").eq("id", agent_id).execute()
        
        if response.data and len(response.data) > 0:
            agent_data = response.data[0]
            return Agent(
                id=agent_data["id"],
                agent_id=agent_data["agent_id"],
                org_id=agent_data["org_id"],
                name=agent_data["name"],
                description=agent_data.get("description"),
                status=agent_data["status"],
                created_by=agent_data["created_by"],
                created_date=datetime.fromisoformat(agent_data["created_at"].replace("Z", "+00:00")),
                updated_date=datetime.fromisoformat(agent_data["updated_at"].replace("Z", "+00:00")),
                current_blueprint_id=agent_data.get("current_blueprint_id")
            )
        return None
        
    except Exception as e:
        logger.error(f"Error getting agent: {str(e)}")
        return None

def update_agent(agent_id: str, name: Optional[str] = None, description: Optional[str] = None,
                 status: Optional[str] = None) -> Optional[Agent]:
    """
    Update an agent's basic information (not blueprint data)
    
    Args:
        agent_id: Agent UUID
        name: New name
        description: New description
        status: New status
    
    Returns:
        Updated Agent object if successful, None otherwise
    """
    try:
        update_data = {}
        
        if name is not None:
            update_data["name"] = name
        if description is not None:
            update_data["description"] = description
        if status is not None:
            if status not in ["active", "deactive", "delete"]:
                raise ValueError("Status must be 'active', 'deactive', or 'delete'")
            update_data["status"] = status
        
        if not update_data:
            raise ValueError("No fields to update")
        
        update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        response = supabase.table("org_agents").update(update_data).eq("id", agent_id).execute()
        
        if response.data and len(response.data) > 0:
            agent_data = response.data[0]
            return Agent(
                id=agent_data["id"],
                agent_id=agent_data["agent_id"],
                org_id=agent_data["org_id"],
                name=agent_data["name"],
                description=agent_data.get("description"),
                status=agent_data["status"],
                created_by=agent_data["created_by"],
                created_date=datetime.fromisoformat(agent_data["created_at"].replace("Z", "+00:00")),
                updated_date=datetime.fromisoformat(agent_data["updated_at"].replace("Z", "+00:00")),
                current_blueprint_id=agent_data.get("current_blueprint_id")
            )
        return None
        
    except Exception as e:
        logger.error(f"Error updating agent: {str(e)}")
        raise

def update_agent_current_blueprint(agent_id: str, blueprint_id: str) -> Optional[Agent]:
    """
    Update an agent's current blueprint reference
    
    Args:
        agent_id: Agent UUID
        blueprint_id: Blueprint UUID
    
    Returns:
        Updated Agent object if successful, None otherwise
    """
    try:
        update_data = {
            "current_blueprint_id": blueprint_id,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        response = supabase.table("org_agents").update(update_data).eq("id", agent_id).execute()
        
        if response.data and len(response.data) > 0:
            agent_data = response.data[0]
            return Agent(
                id=agent_data["id"],
                agent_id=agent_data["agent_id"],
                org_id=agent_data["org_id"],
                name=agent_data["name"],
                description=agent_data.get("description"),
                status=agent_data["status"],
                created_by=agent_data["created_by"],
                created_date=datetime.fromisoformat(agent_data["created_at"].replace("Z", "+00:00")),
                updated_date=datetime.fromisoformat(agent_data["updated_at"].replace("Z", "+00:00")),
                current_blueprint_id=agent_data.get("current_blueprint_id")
            )
        return None
        
    except Exception as e:
        logger.error(f"Error updating agent current blueprint: {str(e)}")
        raise

def delete_agent(agent_id: str) -> bool:
    """
    Soft delete an agent by setting status to 'delete'
    
    Args:
        agent_id: Agent UUID
    
    Returns:
        True if successful, False otherwise
    """
    try:
        response = supabase.table("org_agents").update({
            "status": "delete",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", agent_id).execute()
        
        return bool(response.data)
        
    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        return False

def search_agents(org_id: str, query: str, limit: int = 10) -> List[Agent]:
    """
    Search agents by name within an organization
    
    Args:
        org_id: Organization UUID
        query: Search query
        limit: Maximum number of results
    
    Returns:
        List of matching Agent objects
    """
    try:
        response = supabase.table("org_agents")\
            .select("*")\
            .eq("org_id", org_id)\
            .neq("status", "delete")\
            .ilike("name", f"%{query}%")\
            .limit(limit)\
            .execute()
        
        agents = []
        if response.data:
            for agent_data in response.data:
                agents.append(Agent(
                    id=agent_data["id"],
                    agent_id=agent_data["agent_id"],
                    org_id=agent_data["org_id"],
                    name=agent_data["name"],
                    description=agent_data.get("description"),
                    system_prompt=agent_data.get("system_prompt", {}),
                    tools_list=agent_data.get("tools_list", []),
                    knowledge_list=agent_data.get("knowledge_list", []),
                    status=agent_data["status"],
                    created_by=agent_data["created_by"],
                    created_date=datetime.fromisoformat(agent_data["created_at"].replace("Z", "+00:00")),
                    updated_date=datetime.fromisoformat(agent_data["updated_at"].replace("Z", "+00:00"))
                ))
        return agents
        
    except Exception as e:
        logger.error(f"Error searching agents: {str(e)}")
        return []

# Agent Blueprint CRUD Functions

def create_blueprint(agent_id: str, blueprint_data: dict, created_by: str, 
                    conversation_id: Optional[str] = None) -> AgentBlueprint:
    """
    Create a new blueprint version for an agent
    
    Args:
        agent_id: Agent UUID
        blueprint_data: Complete blueprint JSON
        created_by: User ID who created the blueprint
        conversation_id: Optional conversation that created this
    
    Returns:
        AgentBlueprint object
    """
    try:
        # Validate agent exists
        agent = get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent with id {agent_id} does not exist")
        
        # Get next version number
        version_response = supabase.table("agent_blueprints")\
            .select("version")\
            .eq("agent_id", agent_id)\
            .order("version", desc=True)\
            .limit(1)\
            .execute()
        
        next_version = 1
        if version_response.data:
            next_version = version_response.data[0]["version"] + 1
        
        data = {
            "agent_id": agent_id,
            "version": next_version,
            "agent_blueprint": blueprint_data,
            "created_by": created_by,
            "conversation_id": conversation_id,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        response = supabase.table("agent_blueprints").insert(data).execute()
        
        if response.data:
            blueprint_data = response.data[0]
            return AgentBlueprint(
                id=blueprint_data["id"],
                agent_id=blueprint_data["agent_id"],
                version=blueprint_data["version"],
                agent_blueprint=blueprint_data["agent_blueprint"],
                created_date=datetime.fromisoformat(blueprint_data["created_at"].replace("Z", "+00:00")),
                created_by=blueprint_data["created_by"],
                conversation_id=blueprint_data.get("conversation_id"),
                compiled_system_prompt=blueprint_data.get("compiled_system_prompt"),
                compiled_at=datetime.fromisoformat(blueprint_data["compiled_at"].replace("Z", "+00:00")) if blueprint_data.get("compiled_at") else None,
                compiled_by=blueprint_data.get("compiled_by"),
                compilation_status=blueprint_data.get("compilation_status", "draft"),
                # Todo-related fields
                implementation_todos=blueprint_data.get("implementation_todos", []),
                todos_completion_status=blueprint_data.get("todos_completion_status", "not_generated"),
                todos_generated_at=datetime.fromisoformat(blueprint_data["todos_generated_at"].replace("Z", "+00:00")) if blueprint_data.get("todos_generated_at") else None,
                todos_generated_by=blueprint_data.get("todos_generated_by"),
                todos_completed_at=datetime.fromisoformat(blueprint_data["todos_completed_at"].replace("Z", "+00:00")) if blueprint_data.get("todos_completed_at") else None,
                todos_completed_by=blueprint_data.get("todos_completed_by")
            )
        raise Exception("Failed to create blueprint")
        
    except Exception as e:
        logger.error(f"Error creating blueprint: {str(e)}")
        raise

def get_blueprint(blueprint_id: str) -> Optional[AgentBlueprint]:
    """
    Get a specific blueprint by ID
    
    Args:
        blueprint_id: Blueprint UUID
    
    Returns:
        AgentBlueprint object if found, None otherwise
    """
    try:
        response = supabase.table("agent_blueprints").select("*").eq("id", blueprint_id).execute()
        
        if response.data and len(response.data) > 0:
            blueprint_data = response.data[0]
            return AgentBlueprint(
                id=blueprint_data["id"],
                agent_id=blueprint_data["agent_id"],
                version=blueprint_data["version"],
                agent_blueprint=blueprint_data["agent_blueprint"],
                created_date=datetime.fromisoformat(blueprint_data["created_at"].replace("Z", "+00:00")),
                created_by=blueprint_data["created_by"],
                conversation_id=blueprint_data.get("conversation_id"),
                compiled_system_prompt=blueprint_data.get("compiled_system_prompt"),
                compiled_at=datetime.fromisoformat(blueprint_data["compiled_at"].replace("Z", "+00:00")) if blueprint_data.get("compiled_at") else None,
                compiled_by=blueprint_data.get("compiled_by"),
                compilation_status=blueprint_data.get("compilation_status", "draft"),
                # Todo-related fields
                implementation_todos=blueprint_data.get("implementation_todos", []),
                todos_completion_status=blueprint_data.get("todos_completion_status", "not_generated"),
                todos_generated_at=datetime.fromisoformat(blueprint_data["todos_generated_at"].replace("Z", "+00:00")) if blueprint_data.get("todos_generated_at") else None,
                todos_generated_by=blueprint_data.get("todos_generated_by"),
                todos_completed_at=datetime.fromisoformat(blueprint_data["todos_completed_at"].replace("Z", "+00:00")) if blueprint_data.get("todos_completed_at") else None,
                todos_completed_by=blueprint_data.get("todos_completed_by")
            )
        return None
        
    except Exception as e:
        logger.error(f"Error getting blueprint: {str(e)}")
        return None

def get_agent_blueprints(agent_id: str, limit: int = 10) -> List[AgentBlueprint]:
    """
    Get all blueprint versions for an agent
    
    Args:
        agent_id: Agent UUID
        limit: Maximum number of results
    
    Returns:
        List of AgentBlueprint objects ordered by version desc
    """
    try:
        response = supabase.table("agent_blueprints")\
            .select("*")\
            .eq("agent_id", agent_id)\
            .order("version", desc=True)\
            .limit(limit)\
            .execute()
        
        blueprints = []
        if response.data:
            for blueprint_data in response.data:
                blueprints.append(AgentBlueprint(
                    id=blueprint_data["id"],
                    agent_id=blueprint_data["agent_id"],
                    version=blueprint_data["version"],
                    agent_blueprint=blueprint_data["agent_blueprint"],
                    created_date=datetime.fromisoformat(blueprint_data["created_at"].replace("Z", "+00:00")),
                    created_by=blueprint_data["created_by"],
                    conversation_id=blueprint_data.get("conversation_id"),
                    compiled_system_prompt=blueprint_data.get("compiled_system_prompt"),
                    compiled_at=datetime.fromisoformat(blueprint_data["compiled_at"].replace("Z", "+00:00")) if blueprint_data.get("compiled_at") else None,
                    compiled_by=blueprint_data.get("compiled_by"),
                    compilation_status=blueprint_data.get("compilation_status", "draft")
                ))
        
        return blueprints
        
    except Exception as e:
        logger.error(f"Error getting agent blueprints: {str(e)}")
        return []

def get_current_blueprint(agent_id: str) -> Optional[AgentBlueprint]:
    """
    Get the current active blueprint for an agent
    
    Args:
        agent_id: Agent UUID
    
    Returns:
        AgentBlueprint object if found, None otherwise
    """
    try:
        agent = get_agent(agent_id)
        if not agent or not agent.current_blueprint_id:
            return None
        
        return get_blueprint(agent.current_blueprint_id)
        
    except Exception as e:
        logger.error(f"Error getting current blueprint: {str(e)}")
        return None

def activate_blueprint(agent_id: str, blueprint_id: str) -> bool:
    """
    Set a blueprint as the current active blueprint for an agent
    
    Args:
        agent_id: Agent UUID
        blueprint_id: Blueprint UUID
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate blueprint exists and belongs to the agent
        blueprint = get_blueprint(blueprint_id)
        if not blueprint or blueprint.agent_id != agent_id:
            raise ValueError(f"Blueprint {blueprint_id} does not exist or does not belong to agent {agent_id}")
        
        updated_agent = update_agent_current_blueprint(agent_id, blueprint_id)
        return updated_agent is not None
        
    except Exception as e:
        logger.error(f"Error activating blueprint: {str(e)}")
        return False

def get_agent_with_current_blueprint(agent_id: str) -> Optional[tuple[Agent, AgentBlueprint]]:
    """
    Get an agent with its current blueprint
    
    Args:
        agent_id: Agent UUID
    
    Returns:
        Tuple of (Agent, AgentBlueprint) if found, None otherwise
    """
    try:
        agent = get_agent(agent_id)
        if not agent:
            return None
        
        blueprint = get_current_blueprint(agent_id)
        if not blueprint:
            return None
        
        return agent, blueprint
        
    except Exception as e:
        logger.error(f"Error getting agent with current blueprint: {str(e)}")
        return None

# Blueprint Compilation Functions

def compile_blueprint(blueprint_id: str, compiled_by: str) -> Optional[AgentBlueprint]:
    """
    Compile a blueprint into a system prompt
    Note: Can only compile if all implementation todos are completed
    
    Args:
        blueprint_id: Blueprint UUID
        compiled_by: User ID who initiated the compilation
    
    Returns:
        Updated AgentBlueprint object if successful, None otherwise
    """
    try:
        # Get the blueprint
        blueprint = get_blueprint(blueprint_id)
        if not blueprint:
            raise ValueError(f"Blueprint with id {blueprint_id} does not exist")
        
        # Check if todos are completed (if any exist)
        if blueprint.implementation_todos and len(blueprint.implementation_todos) > 0:
            if blueprint.todos_completion_status != "completed":
                raise ValueError(f"Cannot compile blueprint: implementation todos are not completed. Status: {blueprint.todos_completion_status}")
        
        # Check compilation status
        if blueprint.compilation_status not in ["draft", "ready_for_compilation"]:
            raise ValueError(f"Blueprint cannot be compiled in status: {blueprint.compilation_status}")
        
        # Get collected inputs from todos
        collected_inputs = get_all_collected_inputs(blueprint_id)
        
        # Generate system prompt from blueprint with collected inputs
        compiled_prompt = generate_system_prompt_from_blueprint(blueprint.agent_blueprint, collected_inputs)
        
        # Add tool instructions and knowledge context from completed todos
        compiled_prompt = _enhance_prompt_with_todo_insights(compiled_prompt, blueprint_id)
        
        # Update blueprint with compiled prompt
        response = supabase.table("agent_blueprints").update({
            "compiled_system_prompt": compiled_prompt,
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "compiled_by": compiled_by,
            "compilation_status": "compiled"
        }).eq("id", blueprint_id).execute()
        
        if response.data and len(response.data) > 0:
            blueprint_data = response.data[0]
            return AgentBlueprint(
                id=blueprint_data["id"],
                agent_id=blueprint_data["agent_id"],
                version=blueprint_data["version"],
                agent_blueprint=blueprint_data["agent_blueprint"],
                created_date=datetime.fromisoformat(blueprint_data["created_at"].replace("Z", "+00:00")),
                created_by=blueprint_data["created_by"],
                conversation_id=blueprint_data.get("conversation_id"),
                compiled_system_prompt=blueprint_data.get("compiled_system_prompt"),
                compiled_at=datetime.fromisoformat(blueprint_data["compiled_at"].replace("Z", "+00:00")) if blueprint_data.get("compiled_at") else None,
                compiled_by=blueprint_data.get("compiled_by"),
                compilation_status=blueprint_data.get("compilation_status", "draft")
            )
        return None
        
    except Exception as e:
        logger.error(f"Error compiling blueprint: {str(e)}")
        # Mark compilation as failed
        try:
            supabase.table("agent_blueprints").update({
                "compiled_at": datetime.now(timezone.utc).isoformat(),
                "compiled_by": compiled_by,
                "compilation_status": "failed"
            }).eq("id", blueprint_id).execute()
        except:
            pass
        raise

def generate_system_prompt_from_blueprint(blueprint_data: dict, collected_inputs: dict = None) -> str:
    """
    Generate a comprehensive system prompt from blueprint sections with collected inputs
    
    Args:
        blueprint_data: The agent blueprint JSON
        collected_inputs: Dict of collected inputs from todos (optional)
    
    Returns:
        Compiled system prompt string
    """
    try:
        prompt_parts = []
        
        # 1. Identity section - Core role and personality
        if "identity" in blueprint_data:
            identity = blueprint_data["identity"]
            prompt_parts.append("# AGENT IDENTITY")
            if "name" in identity:
                prompt_parts.append(f"You are {identity['name']}.")
            if "role" in identity:
                prompt_parts.append(f"Your role is: {identity['role']}")
            if "personality" in identity:
                prompt_parts.append(f"Your personality: {identity['personality']}")
            if "primary_purpose" in identity:
                prompt_parts.append(f"Your primary purpose: {identity['primary_purpose']}")
            prompt_parts.append("")
        
        # 2. Capabilities section - What you can do and how
        if "capabilities" in blueprint_data:
            capabilities = blueprint_data["capabilities"]
            prompt_parts.append("# CAPABILITIES")
            prompt_parts.append("You are designed to handle the following tasks:")
            
            # Handle both old format (list) and new format (dict with tasks)
            tasks = capabilities.get("tasks", capabilities) if isinstance(capabilities, dict) else capabilities
            
            for i, capability in enumerate(tasks, 1):
                if isinstance(capability, dict):
                    task = capability.get("task", "")
                    description = capability.get("description", "")
                    prompt_parts.append(f"{i}. {task}: {description}")
                    if "examples" in capability:
                        examples = capability["examples"]
                        if examples:
                            prompt_parts.append(f"   Examples: {', '.join(examples[:3])}")  # Limit to 3 examples
                else:
                    prompt_parts.append(f"{i}. {capability}")
            prompt_parts.append("")
        
        # 3. Tools section - Available tools and when to use them
        if "capabilities" in blueprint_data and "tools" in blueprint_data["capabilities"]:
            tools = blueprint_data["capabilities"]["tools"]
            if tools:
                prompt_parts.append("# AVAILABLE TOOLS")
                prompt_parts.append("You have access to the following tools:")
                for tool in tools:
                    if isinstance(tool, dict):
                        name = tool.get("name", tool.get("tool", ""))
                        purpose = tool.get("purpose", tool.get("description", ""))
                        triggers = tool.get("triggers", [])
                        prompt_parts.append(f"- {name}: {purpose}")
                        if triggers:
                            prompt_parts.append(f"  Use when: {', '.join(triggers)}")
                    else:
                        prompt_parts.append(f"- {tool}")
            prompt_parts.append("")
        
        # 4. Knowledge sources - Where to get information
        knowledge_sources = None
        if "capabilities" in blueprint_data and "knowledge_sources" in blueprint_data["capabilities"]:
            knowledge_sources = blueprint_data["capabilities"]["knowledge_sources"]
        elif "knowledge_sources" in blueprint_data:
            knowledge_sources = blueprint_data["knowledge_sources"]
            
        if knowledge_sources:
            prompt_parts.append("# KNOWLEDGE SOURCES")
            prompt_parts.append("Your knowledge comes from:")
            for source in knowledge_sources:
                if isinstance(source, dict):
                    source_name = source.get("source", source.get("domain", ""))
                    source_type = source.get("type", "")
                    description = source.get("description", "")
                    update_freq = source.get("update_frequency", "")
                    
                    if source_name:
                        if source_type:
                            prompt_parts.append(f"- {source_name} ({source_type})")
                        else:
                            prompt_parts.append(f"- {source_name}")
                        
                        if description:
                            prompt_parts.append(f"  {description}")
                    if update_freq:
                        prompt_parts.append(f"  Updated: {update_freq}")
                else:
                    prompt_parts.append(f"- {source}")
            prompt_parts.append("")
        
        # 5. Monitoring and fallback behavior
        if "monitoring" in blueprint_data:
            monitoring = blueprint_data["monitoring"]
            prompt_parts.append("# BEHAVIOR GUIDELINES")
            if "reporting" in monitoring:
                prompt_parts.append(f"Reporting: {monitoring['reporting']}")
            if "fallbacks" in monitoring:
                prompt_parts.append(f"When uncertain: {monitoring['fallbacks']}")
            prompt_parts.append("")
        
        # 6. Workflow - How to process requests
        workflow_steps = None
        if "workflow" in blueprint_data and "steps" in blueprint_data["workflow"]:
            workflow_steps = blueprint_data["workflow"]["steps"]
        elif "workflow_steps" in blueprint_data:
            workflow_steps = blueprint_data["workflow_steps"]
            
        if workflow_steps and workflow_steps != ["To be defined during collaboration"] and workflow_steps != ["To be refined during collaboration"]:
            prompt_parts.append("# WORKFLOW")
            prompt_parts.append("Follow this process when handling requests:")
            for i, step in enumerate(workflow_steps, 1):
                if step and step.strip() and "to be" not in step.lower():
                    prompt_parts.append(f"{i}. {step}")
            prompt_parts.append("")
        
        # 7. Test scenarios for consistency
        if "test_scenarios" in blueprint_data:
            scenarios = blueprint_data["test_scenarios"]
            if scenarios:
                prompt_parts.append("# RESPONSE EXAMPLES")
                prompt_parts.append("For consistency, here are example interactions:")
                for scenario in scenarios[:3]:  # Limit to 3 examples
                    if isinstance(scenario, dict) and "question" in scenario and "expected" in scenario:
                        prompt_parts.append(f'Q: "{scenario["question"]}"')
                        prompt_parts.append(f'A: "{scenario["expected"]}"')
                        prompt_parts.append("")
        
        # 8. Collected inputs, credentials, and tool instructions (if any)
        if collected_inputs and any(collected_inputs.values()):
            prompt_parts.append("# INTEGRATION CONFIGURATIONS")
            prompt_parts.append("You have access to the following configured integrations:")
            
            # Add integration configurations with more specific details
            if collected_inputs.get("integrations"):
                for todo_id, inputs in collected_inputs["integrations"].items():
                    if inputs:
                        # Try to extract service name from todo_id or inputs
                        service_name = inputs.get("service_name", todo_id.replace("_", " ").title())
                        prompt_parts.append(f"\n## {service_name}")
                        
                        # Add configuration details
                        for key, value in inputs.items():
                            if key == "service_name":
                                continue
                            elif "password" not in key.lower() and "secret" not in key.lower() and "key" not in key.lower() and "token" not in key.lower():
                                # Show non-sensitive configuration
                                if key == "folder_name":
                                    prompt_parts.append(f"**Folder Access:** {value}")
                                elif key == "file_types":
                                    prompt_parts.append(f"**File Types:** {value}")
                                elif key == "access_level":
                                    prompt_parts.append(f"**Access Level:** {value}")
                                elif key == "channel_name":
                                    prompt_parts.append(f"**Channel:** {value}")
                                elif key == "email_address":
                                    prompt_parts.append(f"**Email:** {value}")
                            else:
                                prompt_parts.append(f"**{key.replace('_', ' ').title()}:** {value}")
                        else:
                                # Hide sensitive information but confirm it's configured
                                prompt_parts.append(f"**{key.replace('_', ' ').title()}:** [SECURELY CONFIGURED]")
            
            # Add tool configurations
            if collected_inputs.get("tools"):
                for todo_id, inputs in collected_inputs["tools"].items():
                    if inputs:
                        tool_name = inputs.get("tool_name", todo_id.replace("_", " ").title())
                        prompt_parts.append(f"\n## {tool_name}")
                        for key, value in inputs.items():
                            if key == "tool_name":
                                continue
                            elif "password" not in key.lower() and "secret" not in key.lower() and "key" not in key.lower() and "token" not in key.lower():
                                prompt_parts.append(f"**{key.replace('_', ' ').title()}:** {value}")
                            else:
                                prompt_parts.append(f"**{key.replace('_', ' ').title()}:** [SECURELY CONFIGURED]")
            
            # Add general configurations
            if collected_inputs.get("configurations"):
                for todo_id, inputs in collected_inputs["configurations"].items():
                    if inputs:
                        config_name = inputs.get("config_name", todo_id.replace("_", " ").title())
                        prompt_parts.append(f"\n## {config_name}")
                        for key, value in inputs.items():
                            if key != "config_name":
                                prompt_parts.append(f"**{key.replace('_', ' ').title()}:** {value}")
            
            prompt_parts.append("")
        
        # Note: Tool instructions and knowledge context will be added during compilation
        # when blueprint_id is available
        
        # 11. General instructions
        prompt_parts.append("# GENERAL INSTRUCTIONS")
        prompt_parts.append("- Always stay in character based on your identity and personality")
        prompt_parts.append("- Use your capabilities to help users effectively")
        prompt_parts.append("- When uncertain, follow your monitoring guidelines")
        prompt_parts.append("- Be helpful, accurate, and consistent with your role")
        if collected_inputs and any(collected_inputs.values()):
            prompt_parts.append("- Use your configured integrations and tools as needed")
            prompt_parts.append("- All credentials and API keys have been securely configured")
        
        # Join all parts
        compiled_prompt = "\n".join(prompt_parts)
        
        # Validate length (warn if too long for typical LLM context)
        if len(compiled_prompt) > 4000:  # Rough token estimate
            logger.warning(f"Generated system prompt is quite long ({len(compiled_prompt)} chars). Consider simplifying the blueprint.")
        
        return compiled_prompt
        
    except Exception as e:
        logger.error(f"Error generating system prompt from blueprint: {str(e)}")
        raise ValueError(f"Failed to generate system prompt: {str(e)}")

def get_blueprint_compilation_status(blueprint_id: str) -> Optional[dict]:
    """
    Get compilation status for a blueprint
    
    Args:
        blueprint_id: Blueprint UUID
    
    Returns:
        Dict with compilation info if found, None otherwise
    """
    try:
        response = supabase.table("agent_blueprints")\
            .select("compilation_status, compiled_at, compiled_by")\
            .eq("id", blueprint_id)\
            .execute()
        
        if response.data and len(response.data) > 0:
            data = response.data[0]
            return {
                "status": data["compilation_status"],
                "compiled_at": data.get("compiled_at"),
                "compiled_by": data.get("compiled_by")
            }
        return None
        
    except Exception as e:
        logger.error(f"Error getting blueprint compilation status: {str(e)}")
        return None

def get_compiled_blueprints_for_agent(agent_id: str, limit: int = 10) -> List[AgentBlueprint]:
    """
    Get only compiled blueprint versions for an agent
    
    Args:
        agent_id: Agent UUID
        limit: Maximum number of results
    
    Returns:
        List of compiled AgentBlueprint objects
    """
    try:
        response = supabase.table("agent_blueprints")\
            .select("*")\
            .eq("agent_id", agent_id)\
            .eq("compilation_status", "compiled")\
            .order("version", desc=True)\
            .limit(limit)\
            .execute()
        
        blueprints = []
        if response.data:
            for blueprint_data in response.data:
                blueprints.append(AgentBlueprint(
                    id=blueprint_data["id"],
                    agent_id=blueprint_data["agent_id"],
                    version=blueprint_data["version"],
                    agent_blueprint=blueprint_data["agent_blueprint"],
                    created_date=datetime.fromisoformat(blueprint_data["created_at"].replace("Z", "+00:00")),
                    created_by=blueprint_data["created_by"],
                    conversation_id=blueprint_data.get("conversation_id"),
                    compiled_system_prompt=blueprint_data.get("compiled_system_prompt"),
                    compiled_at=datetime.fromisoformat(blueprint_data["compiled_at"].replace("Z", "+00:00")) if blueprint_data.get("compiled_at") else None,
                    compiled_by=blueprint_data.get("compiled_by"),
                    compilation_status=blueprint_data.get("compilation_status", "draft")
                ))
        
        return blueprints
        
    except Exception as e:
        logger.error(f"Error getting compiled blueprints: {str(e)}")
        return []


# Blueprint Implementation Todos Functions

def generate_implementation_todos(blueprint_id: str, generated_by: str) -> Optional[AgentBlueprint]:
    """
    Generate implementation todos for a blueprint using Ami's reasoning
    
    Args:
        blueprint_id: Blueprint UUID
        generated_by: User ID who triggered todo generation
    
    Returns:
        Updated AgentBlueprint with todos, None if failed
    """
    try:
        # Get the blueprint
        blueprint = get_blueprint(blueprint_id)
        if not blueprint:
            raise ValueError(f"Blueprint with id {blueprint_id} does not exist")
        
        # Analyze blueprint and generate todos using LLM
        todos = _analyze_blueprint_and_generate_todos(blueprint.agent_blueprint)
        
        # Update blueprint with todos
        update_data = {
            "implementation_todos": todos,
            "todos_completion_status": "generated",
            "todos_generated_at": datetime.now(timezone.utc).isoformat(),
            "todos_generated_by": generated_by,
            "compilation_status": "todos_pending"
        }
        
        response = supabase.table("agent_blueprints")\
            .update(update_data)\
            .eq("id", blueprint_id)\
            .execute()
        
        if response.data and len(response.data) > 0:
            # Return updated blueprint
            return get_blueprint(blueprint_id)
        
        raise Exception("Failed to update blueprint with todos")
        
    except Exception as e:
        logger.error(f"Error generating todos for blueprint: {str(e)}")
        return None

def _analyze_blueprint_and_generate_todos(blueprint_data: dict) -> list:
    """
    Analyze blueprint and generate implementation todos using Ami's LLM reasoning
    This replaces hardcoded logic with intelligent analysis of what needs to be done
    """
    try:
        # Use Ami's LLM reasoning to analyze blueprint and generate todos
        return _generate_todos_with_ami_reasoning(blueprint_data)
    except Exception as e:
        logger.error(f"LLM-powered todo generation failed, falling back to basic analysis: {str(e)}")
        return _generate_basic_todos_fallback(blueprint_data)

def _generate_todos_with_ami_reasoning(blueprint_data: dict) -> list:
    """
    Use Ami's LLM reasoning to analyze blueprint and generate intelligent todos
    """
    import json  # Import at function level to avoid scope issues
    
    # Create comprehensive analysis prompt - think like a thorough project manager
    analysis_prompt = f"""
You are Ami, an experienced project manager who helps people set up their AI agents properly. You've seen many agents fail because important details were missed during setup. Your job is to review this agent blueprint and create a checklist of everything that needs to be configured before the agent can work.

AGENT BLUEPRINT TO REVIEW:
{json.dumps(blueprint_data, indent=2)}

**YOUR MISSION:**
Look at this blueprint like a careful project manager who wants everything to work perfectly. Think about what specific information and setup steps are missing that could cause problems later.

**WHAT TO LOOK FOR:**

🔍 **1. MISSING CONNECTION DETAILS**
- For any app or service mentioned (Google Drive, Slack, email, etc.), what specific details are missing?
- Where exactly should the agent connect? Which accounts, folders, or channels?
- What login information does the agent need to access these services?

🔍 **2. MISSING LOGIN INFORMATION**
- What usernames, passwords, or access codes does the agent need?
- Should the agent have permission to just read information, or also make changes?
- Are there specific accounts that need to be set up first?

🔍 **3. MISSING FILE AND DATA DETAILS**
- If the agent needs to read files or information, what exactly should it look for?
- What types of files should it work with? Where are these files located?
- How often should the agent check for new information?

🔍 **4. MISSING WORK RULES**
- What should the agent do in different situations?
- Are there specific steps or approval processes it should follow?
- What should the agent NOT do? What are its limits?

🔍 **5. MISSING TESTING STEPS**
- How can we make sure the agent is working correctly?
- What situations should we test before the agent starts working?
- What should happen if something goes wrong?

**EXAMPLES OF GOOD THINKING:**
- Blueprint mentions "Google Drive" → Ask: "Which specific folder should I access? What's the exact folder name or path? What types of files should I look for? Do I need read-only or editing access?"
- Blueprint mentions "send notifications" → Ask: "Where should I send notifications? Which Slack channel or email addresses? What should the message look like? When should I send them?"
- Blueprint mentions "read reports" → Ask: "What reports specifically? Where are they stored? What format are they in? How do I know which reports are new?"

**FOR EACH MISSING PIECE, CREATE A SIMPLE TODO:**
- Use friendly, non-technical language that anyone can understand
- Explain why this information is needed in simple terms
- Ask for specific details, not vague descriptions
- Make it clear what the person needs to provide

Respond with this EXACT JSON format:
{{
  "reasoning": "I've reviewed your agent blueprint and found several important details that need to be specified before your agent can work properly. Each of these items represents information that's missing and could cause your agent to fail or not work as expected...",
  "todos": [
    {{
      "id": "todo_1",
      "title": "Specify Which Google Drive Folder to Use",
      "description": "Your agent needs to access Google Drive, but I need to know exactly which folder to look in. Without this specific information, your agent won't know where to find the files it needs to work with.",
      "category": "setup",
      "priority": "high", 
      "status": "pending",
      "why_needed": "Your agent needs to know exactly where to look for files so it can find the right information and won't get confused",
      "input_required": {{
        "type": "setup_info",
        "fields": [
          {{"name": "folder_name", "type": "text", "required": true, "description": "What's the exact name of the Google Drive folder? (for example: 'Company Reports' or 'Sales Data')"}},
          {{"name": "file_types", "type": "text", "required": true, "description": "What types of files should I look for? (for example: 'Excel spreadsheets', 'PDF documents', 'Word documents')"}},
          {{"name": "access_level", "type": "select", "options": ["Just read files", "Read and edit files", "Full access"], "required": true, "description": "What should your agent be allowed to do with these files?"}},
          {{"name": "google_account", "type": "text", "required": false, "description": "Which Google account should I use? (you can leave this blank to use your main account)"}}
        ]
      }}
    }}
  ]
}}

**IMPORTANT:** Generate 4-8 todos based on what's missing in the blueprint. Focus on the most common tools and services (Google Workspace, Microsoft Office, Slack, email, databases, file storage, etc.). Use simple language that non-technical people can easily understand.
"""

    try:
        # Use simple direct LLM call to avoid re-initialization issues
        import asyncio
        import re
        import anthropic
        import os
        
        # Identify important terms that might need web research for better todo generation
        terms_needing_research = _identify_important_terms(blueprint_data)
        
        # Log analysis results
        if terms_needing_research:
            logger.info(f"🔍 Important terms detected for web research: {terms_needing_research}")
            logger.info(f"   Will enable web search to research: {', '.join(terms_needing_research)}")
        else:
            logger.info("🔍 No important terms detected - web search may not be needed")
        
        # Enhanced prompt that instructs Claude to use web search for important terms
        enhanced_prompt = analysis_prompt
        if terms_needing_research:
            enhanced_prompt += f"""

**RESEARCH REQUIREMENT:**
I've detected these important business terms/concepts in the blueprint: {', '.join(terms_needing_research)}

Please use web search to research any terms you're not familiar with to understand their:
- Business context and industry meaning
- Technical requirements and implementation details
- API documentation and integration methods (if applicable)
- Common setup steps and configuration requirements
- Typical credentials, access keys, or prerequisites needed

Use this research to generate more accurate and specific todos that reflect proper understanding of the domain.
"""

        # Direct Anthropic API call with native web search enabled
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Enable web search tool for researching important terms
        tools = [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5  # Limit searches during todo generation
            # No domain restrictions - Claude can research ANY business software!
            # This enables research of Vietnamese software like misa.vn, 1office.vn, etc.
        }]
        
        # API call with web search enabled
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            messages=[{"role": "user", "content": enhanced_prompt}],
            tools=tools  # Enable native web search
        )
        
        # Log web search usage and results
        web_search_requests = 0
        search_results_found = []
        
        if hasattr(response, 'usage') and response.usage:
            server_tool_use = getattr(response.usage, 'server_tool_use', None)
            if server_tool_use:
                web_search_requests = getattr(server_tool_use, 'web_search_requests', 0)
        
        # Extract and log web search results from response content
        for content_block in response.content:
            if hasattr(content_block, 'type'):
                if content_block.type == 'web_search_tool_result':
                    # Found web search results
                    if hasattr(content_block, 'content') and content_block.content:
                        for result in content_block.content:
                            if hasattr(result, 'url') and hasattr(result, 'title'):
                                search_results_found.append({
                                    'url': result.url,
                                    'title': result.title,
                                    'page_age': getattr(result, 'page_age', 'unknown')
                                })
        
        # Log web search activity
        if web_search_requests > 0:
            logger.info(f"🌐 Web search activity during todo generation:")
            logger.info(f"   🔍 Search requests made: {web_search_requests}")
            logger.info(f"   📄 Results found: {len(search_results_found)}")
            if search_results_found:
                logger.info(f"   📋 Sources researched:")
                for i, result in enumerate(search_results_found[:5], 1):  # Log first 5 results
                    logger.info(f"      {i}. {result['title']}")
                    logger.info(f"         URL: {result['url']}")
                    logger.info(f"         Age: {result['page_age']}")
            logger.info(f"   💰 Estimated cost: ${web_search_requests * 0.01:.3f} (at $10/1000 searches)")
        else:
            logger.info("🌐 No web searches performed - all terms were recognized")
        
        response_text = response.content[0].text
        
        # Parse the structured response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            
            # Log Ami's reasoning and web search impact
            reasoning = data.get('reasoning', 'No reasoning provided')
            logger.info(f"📝 Ami's todo generation reasoning: {reasoning}")
            
            # Return the generated todos
            todos = data.get('todos', [])
            
            # Add metadata to each todo
            from datetime import datetime, timezone
            for i, todo in enumerate(todos):
                todo['created_at'] = datetime.now(timezone.utc).isoformat()
                todo['collected_inputs'] = {}
                if 'id' not in todo:
                    todo['id'] = f"todo_{i+1}"
            
            logger.info(f"✅ Todo generation complete:")
            logger.info(f"   📋 Generated {len(todos)} intelligent todos for blueprint")
            if web_search_requests > 0:
                logger.info(f"   🌐 Enhanced with {web_search_requests} web searches")
            logger.info(f"   🎯 Todos created with {'web-researched' if web_search_requests > 0 else 'existing'} knowledge")
            return todos
        else:
            raise ValueError("Could not parse Ami's response as JSON")
            
    except Exception as e:
        logger.error(f"Ami LLM reasoning failed: {str(e)}")
        raise

def _identify_important_terms(blueprint_data: dict) -> list:
    """
    Use LLM to intelligently identify important business terms, concepts, processes, 
    or entities that need web research for better understanding.
    Much broader and more useful than just "unknown tools".
    """
    import anthropic
    import os
    import json
    
    # Collect all relevant text from blueprint
    
    text_fields = []
    
    # Add purpose and identity fields
    if 'identity' in blueprint_data:
        text_fields.append(blueprint_data['identity'].get('purpose', ''))
        text_fields.append(blueprint_data['identity'].get('name', ''))
    
    # Add conversation requirements (from user input)
    if 'conversation_requirements' in blueprint_data:
        text_fields.append(blueprint_data['conversation_requirements'].get('concept', ''))
        text_fields.append(blueprint_data['conversation_requirements'].get('purpose', ''))
        text_fields.extend(blueprint_data['conversation_requirements'].get('key_tasks', []))
    
    # Add capabilities and tasks
    if 'capabilities' in blueprint_data:
        if 'tasks' in blueprint_data['capabilities']:
            for task in blueprint_data['capabilities']['tasks']:
                if isinstance(task, dict):
                    text_fields.append(task.get('task', ''))
                    text_fields.append(task.get('description', ''))
                else:
                    text_fields.append(str(task))
    
    # Add integrations
    if 'capabilities' in blueprint_data and 'integrations' in blueprint_data['capabilities']:
        integrations = blueprint_data['capabilities']['integrations']
        for integration in integrations:
            if isinstance(integration, dict):
                text_fields.append(integration.get('tool', ''))
                text_fields.append(integration.get('purpose', ''))
            else:
                text_fields.append(str(integration))
    
    # Combine all text for analysis
    combined_text = ' '.join([str(t) for t in text_fields if t])
    
    if not combined_text.strip():
        logger.info("🔍 No text content found for tool analysis")
        return []
    
    # Use LLM to identify unknown tools/services
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        analysis_prompt = f"""Analyze this agent blueprint text and identify important business terms, concepts, processes, software, or entities that would benefit from web research to better understand the domain:

BLUEPRINT TEXT:
{combined_text}

Please identify terms that are:
1. **Business software/tools** (like "MISA software", "1Office.vn", "Oceanbank")
2. **Industry processes** (like "order reconciliation", "inventory management", "KYC compliance")
3. **Business concepts** (like "dropshipping", "multi-tenant architecture", "real-time analytics") 
4. **Domain-specific terms** (like "fulfillment", "SKU management", "payment gateway")
5. **Company/product names** that aren't widely known
6. **Technical concepts** specific to the business context

IGNORE very common terms like: Google Drive, Microsoft Office, basic web concepts, etc.

Return ONLY a JSON array of important terms that need research:
["term1", "term2", "term3"]

If no important terms are found, return: []"""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        # Parse the LLM response
        response_text = response.content[0].text.strip()
        
        # Extract JSON array from response
        try:
            # Try to parse as JSON directly
            important_terms = json.loads(response_text)
            if isinstance(important_terms, list):
                logger.info(f"🧠 LLM detected important terms: {important_terms}")
                logger.info(f"   Analyzed text: {combined_text[:100]}...")
                return important_terms
            else:
                logger.warning(f"LLM returned non-list: {important_terms}")
                return []
        except json.JSONDecodeError:
            # Try to extract JSON array from response
            import re
            json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
            if json_match:
                try:
                    important_terms = json.loads(f"[{json_match.group(1)}]")
                    logger.info(f"🧠 LLM detected important terms (extracted): {important_terms}")
                    return important_terms
                except json.JSONDecodeError:
                    pass
            
            logger.warning(f"Could not parse LLM response: {response_text}")
            return []
    
    except Exception as e:
        logger.warning(f"LLM important terms detection failed: {e}")
        # Fallback to empty list - web search won't be triggered but todos will still generate
        return []
    


# REMOVED: _research_tools_requirements_sync function
# This function has been replaced with Anthropic's native web search capabilities
# in the _generate_todos_with_ami_reasoning function above

# REMOVED: _perform_web_search_sync function  
# This function has been replaced with Anthropic's native web search capabilities
# which are automatically used when the web_search tool is enabled in the API call

def _generate_basic_todos_fallback(blueprint_data: dict) -> list:
    """
    Fallback basic todo generation if LLM reasoning fails
    Creates simple, generic todos without intelligent analysis
    """
    todos = []
    todo_id_counter = 1
    from datetime import datetime, timezone
    
    # Basic fallback todos - generic and simple
    base_todos = [
        {
            "title": "Review Agent Configuration",
            "description": "Review the agent blueprint and identify any required integrations or credentials",
            "category": "setup",
            "priority": "high"
        },
        {
            "title": "Configure Required Tools",
            "description": "Set up and test any tools that the agent needs to function properly",
            "category": "tool_configuration", 
            "priority": "medium"
        },
        {
            "title": "Validate Agent Setup",
            "description": "Test the agent configuration to ensure it works as expected",
            "category": "testing",
            "priority": "medium"
        }
    ]
    
    # Create todos with metadata
    for base_todo in base_todos:
        todos.append({
            "id": f"todo_{todo_id_counter}",
            "title": base_todo["title"],
            "description": base_todo["description"],
            "category": base_todo["category"],
            "priority": base_todo["priority"],
            "estimated_effort": "1-2 hours",
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collected_inputs": {},
            "input_required": {
                "type": "manual_configuration",
                "fields": [
                    {"name": "notes", "type": "text", "required": False, "description": "Any notes or configuration details"}
                ]
            }
        })
        todo_id_counter += 1
    
    logger.warning("Using basic fallback todos - LLM-powered analysis unavailable")
    return todos

def update_todo_status(blueprint_id: str, todo_id: str, new_status: str, updated_by: str, collected_inputs: dict = None) -> bool:
    """
    Update the status of a specific todo and optionally store collected inputs
    
    Args:
        blueprint_id: Blueprint UUID
        todo_id: Todo ID within the blueprint
        new_status: New status (pending, in_progress, completed, cancelled)
        updated_by: User ID who updated the todo
        collected_inputs: Dict of collected inputs for this todo (optional)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        blueprint = get_blueprint(blueprint_id)
        if not blueprint:
            return False
        
        # Update the specific todo in the list
        updated_todos = []
        todo_found = False
        
        for todo in blueprint.implementation_todos:
            if todo.get("id") == todo_id:
                todo["status"] = new_status
                todo["updated_at"] = datetime.now(timezone.utc).isoformat()
                todo["updated_by"] = updated_by
                
                # Store collected inputs if provided
                if collected_inputs is not None:
                    todo["collected_inputs"] = collected_inputs
                    
                todo_found = True
            updated_todos.append(todo)
        
        if not todo_found:
            return False
        
        # Update the blueprint with new todos
        response = supabase.table("agent_blueprints")\
            .update({"implementation_todos": updated_todos})\
            .eq("id", blueprint_id)\
            .execute()
        
        return response.data and len(response.data) > 0
        
    except Exception as e:
        logger.error(f"Error updating todo status: {str(e)}")
        return False

def validate_todo_inputs(blueprint_id: str, todo_id: str, provided_inputs: dict) -> dict:
    """
    Validate that provided inputs meet the requirements for a todo
    
    Args:
        blueprint_id: Blueprint UUID
        todo_id: Todo ID within the blueprint  
        provided_inputs: Dict of inputs provided by human
    
    Returns:
        Dict with 'valid': bool and 'errors': list of error messages
    """
    try:
        blueprint = get_blueprint(blueprint_id)
        if not blueprint:
            return {"valid": False, "errors": ["Blueprint not found"]}
        
        # Find the specific todo
        target_todo = None
        for todo in blueprint.implementation_todos:
            if todo.get("id") == todo_id:
                target_todo = todo
                break
        
        if not target_todo:
            return {"valid": False, "errors": ["Todo not found"]}
        
        input_requirements = target_todo.get("input_required", {})
        if not input_requirements:
            return {"valid": True, "errors": []}  # No inputs required
        
        errors = []
        required_fields = input_requirements.get("fields", [])
        
        # Check each required field
        for field in required_fields:
            field_name = field.get("name")
            is_required = field.get("required", False)
            field_type = field.get("type", "string")
            
            if is_required and field_name not in provided_inputs:
                errors.append(f"Required field '{field_name}' is missing")
            elif field_name in provided_inputs:
                value = provided_inputs[field_name]
                
                # Basic type validation
                if field_type == "number" and not isinstance(value, (int, float)):
                    try:
                        float(value)  # Try to convert
                    except (ValueError, TypeError):
                        errors.append(f"Field '{field_name}' must be a number")
                elif field_type == "url" and value:
                    if not (value.startswith("http://") or value.startswith("https://")):
                        errors.append(f"Field '{field_name}' must be a valid URL")
        
        return {"valid": len(errors) == 0, "errors": errors}
        
    except Exception as e:
        logger.error(f"Error validating todo inputs: {str(e)}")
        return {"valid": False, "errors": [f"Validation error: {str(e)}"]}

def get_all_collected_inputs(blueprint_id: str) -> dict:
    """
    Get all collected inputs from completed todos for compilation
    
    Args:
        blueprint_id: Blueprint UUID
    
    Returns:
        Dict with all collected inputs organized by category
    """
    try:
        blueprint = get_blueprint(blueprint_id)
        if not blueprint:
            return {}
        
        all_inputs = {
            "integrations": {},
            "tools": {},
            "credentials": {},
            "configurations": {}
        }
        
        for todo in blueprint.implementation_todos:
            if todo.get("status") == "completed" and todo.get("collected_inputs"):
                category = todo.get("category", "configurations")
                todo_id = todo.get("id")
                
                if category == "integration":
                    all_inputs["integrations"][todo_id] = todo["collected_inputs"]
                elif category == "tool_configuration":
                    all_inputs["tools"][todo_id] = todo["collected_inputs"]
                else:
                    all_inputs["configurations"][todo_id] = todo["collected_inputs"]
        
        return all_inputs
        
    except Exception as e:
        logger.error(f"Error getting collected inputs: {str(e)}")
        return {}

def _get_tool_instructions_from_todos(blueprint_id: str) -> list:
    """
    Extract tool usage instructions from completed todos
    
    Args:
        blueprint_id: Blueprint UUID
    
    Returns:
        List of tool instruction objects
    """
    try:
        blueprint = get_blueprint(blueprint_id)
        if not blueprint:
            return []
        
        instructions = []
        for todo in blueprint.implementation_todos:
            if todo.get("status") == "completed" and todo.get("tool_instructions"):
                tool_instruction = todo["tool_instructions"]
                if all(key in tool_instruction for key in ["tool_name", "how_to_call", "when_to_use", "expected_output"]):
                    instructions.append(tool_instruction)
        
        return instructions
        
    except Exception as e:
        logger.error(f"Error getting tool instructions from todos: {str(e)}")
        return []

def _get_knowledge_context_from_todos(blueprint_id: str) -> list:
    """
    Extract domain knowledge context from completed todos
    
    Args:
        blueprint_id: Blueprint UUID
    
    Returns:
        List of knowledge context objects
    """
    try:
        blueprint = get_blueprint(blueprint_id)
        if not blueprint:
            return []
        
        knowledge_contexts = []
        for todo in blueprint.implementation_todos:
            if todo.get("status") == "completed" and todo.get("knowledge_to_collect"):
                knowledge = todo["knowledge_to_collect"]
                if knowledge.get("type") and knowledge.get("description"):
                    knowledge_contexts.append(knowledge)
        
        return knowledge_contexts
        
    except Exception as e:
        logger.error(f"Error getting knowledge context from todos: {str(e)}")
        return []

def _enhance_prompt_with_todo_insights(base_prompt: str, blueprint_id: str) -> str:
    """
    Enhance the base system prompt with tool instructions and knowledge context from completed todos
    
    Args:
        base_prompt: The base system prompt generated from blueprint
        blueprint_id: Blueprint UUID to get todo insights from
    
    Returns:
        Enhanced system prompt with todo insights
    """
    try:
        prompt_parts = [base_prompt]
        
        # Add tool usage instructions from completed todos
        tool_instructions = _get_tool_instructions_from_todos(blueprint_id)
        if tool_instructions:
            prompt_parts.append("\n# TOOL USAGE INSTRUCTIONS")
            prompt_parts.append("Here are specific instructions for using your configured tools:")
            
            for instruction in tool_instructions:
                prompt_parts.append(f"\n## {instruction['tool_name']}")
                prompt_parts.append(f"**How to call:** {instruction['how_to_call']}")
                prompt_parts.append(f"**When to use:** {instruction['when_to_use']}")
                prompt_parts.append(f"**Expected output:** {instruction['expected_output']}")
        
        # Add knowledge context from completed todos
        knowledge_context = _get_knowledge_context_from_todos(blueprint_id)
        if knowledge_context:
            prompt_parts.append("\n# DOMAIN KNOWLEDGE")
            prompt_parts.append("You have access to the following domain-specific knowledge:")
            
            for knowledge in knowledge_context:
                prompt_parts.append(f"\n## {knowledge['type'].replace('_', ' ').title()}")
                prompt_parts.append(f"{knowledge['description']}")
                if knowledge.get('examples'):
                    prompt_parts.append("**Examples:**")
                    for example in knowledge['examples']:
                        prompt_parts.append(f"- {example}")
        
        # Join all parts
        enhanced_prompt = "\n".join(prompt_parts)
        
        return enhanced_prompt
        
    except Exception as e:
        logger.error(f"Error enhancing prompt with todo insights: {str(e)}")
        return base_prompt  # Return base prompt if enhancement fails

def get_blueprint_todos(blueprint_id: str) -> Optional[dict]:
    """
    Get todos for a blueprint with completion statistics
    
    Args:
        blueprint_id: Blueprint UUID
    
    Returns:
        Dict with todos and stats, None if blueprint not found
    """
    try:
        blueprint = get_blueprint(blueprint_id)
        if not blueprint:
            return None
        
        todos = blueprint.implementation_todos
        
        # Calculate completion statistics
        total_todos = len(todos)
        completed_todos = len([t for t in todos if t.get("status") == "completed"])
        in_progress_todos = len([t for t in todos if t.get("status") == "in_progress"])
        pending_todos = len([t for t in todos if t.get("status") == "pending"])
        
        return {
            "blueprint_id": blueprint_id,
            "todos": todos,
            "completion_status": blueprint.todos_completion_status,
            "statistics": {
                "total": total_todos,
                "completed": completed_todos,
                "in_progress": in_progress_todos,
                "pending": pending_todos,
                "completion_percentage": round((completed_todos / total_todos * 100) if total_todos > 0 else 0, 1)
            },
            "generated_at": blueprint.todos_generated_at,
            "generated_by": blueprint.todos_generated_by
        }
        
    except Exception as e:
        logger.error(f"Error getting blueprint todos: {str(e)}")
        return None

def check_todos_completion_and_update_status(blueprint_id: str) -> bool:
    """
    Check if all todos are completed and update blueprint status accordingly
    
    Args:
        blueprint_id: Blueprint UUID
    
    Returns:
        True if all todos are completed, False otherwise
    """
    try:
        todos_info = get_blueprint_todos(blueprint_id)
        if not todos_info:
            return False
        
        stats = todos_info["statistics"]
        all_completed = stats["completed"] == stats["total"] and stats["total"] > 0
        
        if all_completed:
            # Update blueprint status to ready for compilation
            response = supabase.table("agent_blueprints")\
                .update({
                    "todos_completion_status": "completed",
                    "todos_completed_at": datetime.now(timezone.utc).isoformat(),
                    "compilation_status": "ready_for_compilation"
                })\
                .eq("id", blueprint_id)\
                .execute()
            
            return response.data and len(response.data) > 0
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking todos completion: {str(e)}")
        return False

def update_blueprint_conversation_id(blueprint_id: str, conversation_id: str) -> bool:
    """
    Update a blueprint's conversation_id
    
    Args:
        blueprint_id: Blueprint UUID
        conversation_id: Conversation ID to link
    
    Returns:
        True if successful, False otherwise
    """
    try:
        response = supabase.table("agent_blueprints").update({
            "conversation_id": conversation_id
        }).eq("id", blueprint_id).execute()
        
        return response.data and len(response.data) > 0
        
    except Exception as e:
        logger.error(f"Error updating blueprint conversation_id: {str(e)}")
        return False

def update_blueprint(blueprint_id: str, updated_blueprint_data: dict) -> Optional[AgentBlueprint]:
    """
    Update a blueprint's agent_blueprint data
    
    Args:
        blueprint_id: Blueprint UUID
        updated_blueprint_data: New blueprint data
    
    Returns:
        Updated AgentBlueprint object if successful, None otherwise
    """
    try:
        response = supabase.table("agent_blueprints").update({
            "agent_blueprint": updated_blueprint_data
            # Note: updated_at column doesn't exist yet - run migration to add it
        }).eq("id", blueprint_id).execute()
        
        if response.data and len(response.data) > 0:
            blueprint_data = response.data[0]
            return AgentBlueprint(
                id=blueprint_data["id"],
                agent_id=blueprint_data["agent_id"],
                version=blueprint_data["version"],
                agent_blueprint=blueprint_data["agent_blueprint"],
                created_date=datetime.fromisoformat(blueprint_data["created_at"].replace("Z", "+00:00")),
                created_by=blueprint_data["created_by"],
                conversation_id=blueprint_data.get("conversation_id"),
                compiled_system_prompt=blueprint_data.get("compiled_system_prompt"),
                compiled_at=datetime.fromisoformat(blueprint_data["compiled_at"].replace("Z", "+00:00")) if blueprint_data.get("compiled_at") else None,
                compiled_by=blueprint_data.get("compiled_by"),
                compilation_status=blueprint_data.get("compilation_status", "draft"),
                implementation_todos=blueprint_data.get("implementation_todos", []),
                todos_completion_status=blueprint_data.get("todos_completion_status", "not_generated"),
                todos_generated_at=datetime.fromisoformat(blueprint_data["todos_generated_at"].replace("Z", "+00:00")) if blueprint_data.get("todos_generated_at") else None,
                todos_generated_by=blueprint_data.get("todos_generated_by"),
                todos_completed_at=datetime.fromisoformat(blueprint_data["todos_completed_at"].replace("Z", "+00:00")) if blueprint_data.get("todos_completed_at") else None,
                todos_completed_by=blueprint_data.get("todos_completed_by")
            )
        return None
        
    except Exception as e:
        logger.error(f"Error updating blueprint: {str(e)}")
        return None

def save_agent_collaboration_message(agent_id: str, blueprint_id: str, message_type: str, message_content: str, context_data: dict = None):
    """
    Save conversation message for agent collaboration history
    
    Args:
        agent_id: ID of the agent being collaborated on
        blueprint_id: ID of the blueprint being refined
        message_type: "user" or "ami"
        message_content: The actual message content
        context_data: Additional context data (optional)
    """
    try:
        data = {
            "agent_id": agent_id,
            "blueprint_id": blueprint_id,
            "message_type": message_type,
            "message_content": message_content,
            "context_data": context_data if context_data else {},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        response = supabase.table("agent_collaboration_history").insert(data).execute()
        
        if response.data:
            logger.info(f"Saved {message_type} message for agent {agent_id}, blueprint {blueprint_id}")
        else:
            logger.error(f"Failed to save collaboration message for agent {agent_id}")
            
    except Exception as e:
        logger.error(f"Error saving collaboration message for agent {agent_id}: {e}")

def get_agent_collaboration_history(agent_id: str, blueprint_id: str, limit: int = 20) -> list:
    """
    Retrieve conversation history for agent collaboration
    
    Args:
        agent_id: ID of the agent
        blueprint_id: ID of the blueprint
        limit: Maximum number of messages to retrieve
        
    Returns:
        List of conversation messages ordered by timestamp
    """
    try:
        response = supabase.table("agent_collaboration_history")\
            .select("message_type, message_content, context_data, created_at")\
            .eq("agent_id", agent_id)\
            .eq("blueprint_id", blueprint_id)\
            .order("created_at", desc=False)\
            .limit(limit)\
            .execute()
        
        if not response.data:
            logger.info(f"No conversation history found for agent {agent_id}")
            return []
        
        history = []
        for row in response.data:
            history.append({
                "message_type": row["message_type"],
                "message_content": row["message_content"], 
                "context_data": row.get("context_data", {}),
                "created_at": row.get("created_at", "")
            })
        
        logger.info(f"Retrieved {len(history)} conversation messages for agent {agent_id}")
        return history
        
    except Exception as e:
        logger.error(f"Error retrieving collaboration history for agent {agent_id}: {e}")
        return []