from supabase import create_client, Client
from typing import List, Dict, Optional
import os
from datetime import datetime, UTC
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
                 system_prompt: dict, tools_list: list, knowledge_list: list, status: str,
                 created_by: str, created_date: datetime, updated_date: datetime):
        self.id = id  # UUID
        self.agent_id = agent_id  # INT
        self.org_id = org_id  # UUID
        self.name = name
        self.description = description
        self.system_prompt = system_prompt  # JSONB field
        self.tools_list = tools_list  # JSONB field
        self.knowledge_list = knowledge_list  # JSONB field
        self.status = status  # 'active', 'deactive', 'delete'
        self.created_by = created_by  # User ID
        self.created_date = created_date
        self.updated_date = updated_date

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
        "created_date": datetime.now(UTC).isoformat()
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
        "created_date": datetime.now(UTC).isoformat()
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
            "joined_at": datetime.now(UTC).isoformat()
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

def create_agent(org_id: str, created_by: str, name: str, description: Optional[str] = None,
                 system_prompt: dict = None, tools_list: list = None, knowledge_list: list = None) -> Agent:
    """
    Create a new agent for an organization
    
    Args:
        org_id: Organization UUID
        created_by: User ID who created the agent
        name: Agent name
        description: Optional description
        system_prompt: JSON object for system prompt
        tools_list: List of tool names/IDs
        knowledge_list: List of knowledge base names/IDs
    
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
            "system_prompt": system_prompt or {},
            "tools_list": tools_list or [],
            "knowledge_list": knowledge_list or [],
            "status": "active",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat()
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
                system_prompt=agent_data.get("system_prompt", {}),
                tools_list=agent_data.get("tools_list", []),
                knowledge_list=agent_data.get("knowledge_list", []),
                status=agent_data["status"],
                created_by=agent_data["created_by"],
                created_date=datetime.fromisoformat(agent_data["created_at"].replace("Z", "+00:00")),
                updated_date=datetime.fromisoformat(agent_data["updated_at"].replace("Z", "+00:00"))
            )
        raise Exception("Failed to create agent")
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
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
                        system_prompt=agent_data.get("system_prompt", {}),
                        tools_list=agent_data.get("tools_list", []),
                        knowledge_list=agent_data.get("knowledge_list", []),
                        status=agent_data["status"],
                        created_by=agent_data["created_by"],
                        created_date=datetime.fromisoformat(agent_data["created_at"].replace("Z", "+00:00")),
                        updated_date=datetime.fromisoformat(agent_data["updated_at"].replace("Z", "+00:00"))
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
                system_prompt=agent_data.get("system_prompt", {}),
                tools_list=agent_data.get("tools_list", []),
                knowledge_list=agent_data.get("knowledge_list", []),
                status=agent_data["status"],
                created_by=agent_data["created_by"],
                created_date=datetime.fromisoformat(agent_data["created_at"].replace("Z", "+00:00")),
                updated_date=datetime.fromisoformat(agent_data["updated_at"].replace("Z", "+00:00"))
            )
        return None
        
    except Exception as e:
        logger.error(f"Error getting agent: {str(e)}")
        return None

def update_agent(agent_id: str, name: Optional[str] = None, description: Optional[str] = None,
                 system_prompt: Optional[dict] = None, tools_list: Optional[list] = None,
                 knowledge_list: Optional[list] = None, status: Optional[str] = None) -> Optional[Agent]:
    """
    Update an agent's information
    
    Args:
        agent_id: Agent UUID
        name: New name
        description: New description
        system_prompt: New system prompt
        tools_list: New tools list
        knowledge_list: New knowledge list
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
        if system_prompt is not None:
            update_data["system_prompt"] = system_prompt
        if tools_list is not None:
            update_data["tools_list"] = tools_list
        if knowledge_list is not None:
            update_data["knowledge_list"] = knowledge_list
        if status is not None:
            if status not in ["active", "deactive", "delete"]:
                raise ValueError("Status must be 'active', 'deactive', or 'delete'")
            update_data["status"] = status
        
        if not update_data:
            raise ValueError("No fields to update")
        
        update_data["updated_at"] = datetime.now(UTC).isoformat()
        
        response = supabase.table("org_agents").update(update_data).eq("id", agent_id).execute()
        
        if response.data and len(response.data) > 0:
            agent_data = response.data[0]
            return Agent(
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
            )
        return None
        
    except Exception as e:
        logger.error(f"Error updating agent: {str(e)}")
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
            "updated_at": datetime.now(UTC).isoformat()
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