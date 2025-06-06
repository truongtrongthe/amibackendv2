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