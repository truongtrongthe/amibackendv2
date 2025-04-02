from supabase import create_client, Client
from typing import List, Dict, Optional
import os
from datetime import datetime, UTC

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(
    spb_url,
    spb_key
)

class Organization:
    def __init__(self, id: str, org_id: int, name: str, description: Optional[str], created_date: datetime):
        self.id = id  # UUID
        self.org_id = org_id  # INT
        self.name = name
        self.description = description
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

def create_organization(name: str, description: Optional[str] = None) -> Organization:
    """
    Create a new organization with created_date
    """
    data = {
        "name": name,
        "description": description,
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
            created_date=datetime.fromisoformat(org_data["created_date"].replace("Z", "+00:00"))
        )
    raise Exception("Failed to create organization")

def update_organization(id: str, name: str, description: Optional[str] = None) -> Organization:
    """
    Update an organization's name and description
    """
    update_data = {
        "name": name,
        "description": description
    }
    
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
            created_date=datetime.fromisoformat(org_data["created_date"].replace("Z", "+00:00"))
        )
    return None

def create_brain(org_id: str, user_id: str, name: str) -> Brain:
    """
    Create a new brain record with default values including created_date
    """
    org_check = supabase.table("organization")\
        .select("id")\
        .eq("id", org_id)\
        .execute()
    
    if not org_check.data:
        raise ValueError(f"Organization with id {org_id} does not exist")

    temp_bank_name = f"temp_wisdom_bank_{org_id}"
    data = {
        "org_id": org_id,
        "name": name,
        "status": "training",
        "bank_name": temp_bank_name,
        "summary": f"Brain created by user {user_id}",
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
    response = supabase.table("brain")\
        .select("*")\
        .eq("org_id", org_id)\
        .execute()
    
    brains = []
    if response.data:
        for brain_data in response.data:
            brains.append(Brain(
                id=brain_data["id"],
                brain_id=brain_data["brain_id"],
                org_id=brain_data["org_id"],
                name=brain_data["name"],
                status=brain_data["status"],
                bank_name=brain_data["bank_name"],
                summary=brain_data["summary"],
                created_date=datetime.fromisoformat(brain_data["created_date"].replace("Z", "+00:00"))
            ))
    return brains

def get_brain_details(brain_id: int) -> Optional[Brain]:
    """
    Fetch details of a specific brain using the integer brain_id
    """
    response = supabase.table("brain")\
        .select("*")\
        .eq("brain_id", brain_id)\
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

# Example usage:
try:
    new_org = create_organization(
        name="Test Organization",
        description="A test organization"
    )
    print(f"Created organization: UUID={new_org.id}, OrgID={new_org.org_id}, Created={new_org.created_date}")

    updated_org = update_organization(
        id=new_org.id,
        name="Updated Test Org",
        description="Updated description"
    )
    print(f"Updated org: Name={updated_org.name}, Desc={updated_org.description}, Created={updated_org.created_date}")

    org = get_organization(new_org.id)
    if org:
        print(f"Org details: Name={org.name}, Created={org.created_date}")

    new_brain = create_brain(
        org_id=new_org.id,
        user_id="user-123",
        name="Test Brain"
    )
    print(f"Created brain: UUID={new_brain.id}, BrainID={new_brain.brain_id}, Created={new_brain.created_date}")

    updated_brain = update_brain(
        id=new_brain.id,
        new_name="Updated Test Brain",
        new_status="published"
    )
    print(f"Updated brain: Name={updated_brain.name}, Status={updated_brain.status}, Created={updated_brain.created_date}")

    brains = get_brains(new_org.id)
    for brain in brains:
        print(f"Brain: {brain.name}, BrainID: {brain.brain_id}, Created: {brain.created_date}")

    brain = get_brain_details(new_brain.brain_id)
    if brain:
        print(f"Brain details - Name: {brain.name}, Bank: {brain.bank_name}, Created: {brain.created_date}")

except Exception as e:
    print(f"Error: {str(e)}")