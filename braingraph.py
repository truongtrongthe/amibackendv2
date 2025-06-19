from supabase import create_client, Client
from typing import List, Dict, Optional
import os
from datetime import datetime, UTC
from utilities import logger

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(
    spb_url,
    spb_key
)

class BrainGraphVersion:
    def __init__(self, id: str, graph_id: str, version_number: int,
                 brain_ids: List[str], released_date: datetime, status: str):
        self.id = id  # UUID
        self.graph_id = graph_id  # UUID (references BrainGraph)
        self.version_number = version_number
        self.brain_ids = brain_ids  # List of brain UUIDs
        self.released_date = released_date
        self.status = status  # draft, published, archived

class BrainGraph:
    def __init__(self, id: str, org_id: str, name: str,
                 description: Optional[str], created_date: datetime):
        self.id = id  # UUID
        self.org_id = org_id  # UUID (references organization)
        self.name = name
        self.description = description
        self.created_date = created_date

def create_brain_graph(org_id: str, name: str, description: Optional[str] = None) -> BrainGraph:
    """
    Create a new brain graph for an organization
    
    Args:
        org_id: UUID of the organization
        name: Name of the brain graph
        description: Optional description
    
    Returns:
        BrainGraph object with UUID
    """
    data = {
        "org_id": org_id,
        "name": name,
        "description": description,
        "created_date": datetime.now(UTC).isoformat()
    }
    
    response = supabase.table("brain_graph").insert(data).execute()
    
    if response.data:
        graph_data = response.data[0]
        return BrainGraph(
            id=graph_data["id"],
            org_id=graph_data["org_id"],
            name=graph_data["name"],
            description=graph_data["description"],
            created_date=datetime.fromisoformat(graph_data["created_date"].replace("Z", "+00:00"))
        )
    raise Exception("Failed to create brain graph")

def create_brain_graph_version(graph_id: str, version_number: int, brain_ids: List[str]) -> BrainGraphVersion:
    """
    Create a new version for a brain graph
    
    Args:
        graph_id: UUID of the brain graph
        version_number: Version number for this release
        brain_ids: List of brain UUIDs included in this version
    
    Returns:
        BrainGraphVersion object with UUID
    """
    # Validate brain_ids
    if not validate_brain_ids(brain_ids):
        raise ValueError("One or more brain IDs do not exist")
        
    data = {
        "graph_id": graph_id,
        "version_number": version_number,
        "brain_ids": brain_ids,
        "released_date": datetime.now(UTC).isoformat(),
        "status": "training"  # New versions start as training
    }
    
    response = supabase.table("brain_graph_version").insert(data).execute()
    
    if response.data:
        version_data = response.data[0]
        return BrainGraphVersion(
            id=version_data["id"],
            graph_id=version_data["graph_id"],
            version_number=version_data["version_number"],
            brain_ids=version_data["brain_ids"],
            released_date=datetime.fromisoformat(version_data["released_date"].replace("Z", "+00:00")),
            status=version_data["status"]
        )
    raise Exception("Failed to create brain graph version")

def get_brain_graph(graph_id: str) -> Optional[BrainGraph]:
    """
    Fetch details of a specific brain graph using the UUID
    """
    response = supabase.table("brain_graph")\
        .select("*")\
        .eq("id", graph_id)\
        .execute()
    
    if response.data and len(response.data) > 0:
        graph_data = response.data[0]
        return BrainGraph(
            id=graph_data["id"],
            org_id=graph_data["org_id"],
            name=graph_data["name"],
            description=graph_data["description"],
            created_date=datetime.fromisoformat(graph_data["created_date"].replace("Z", "+00:00"))
        )
    return None

def get_brain_graph_versions(graph_id: str) -> List[BrainGraphVersion]:
    """
    Fetch all versions for a given brain graph
    """
    response = supabase.table("brain_graph_version")\
        .select("*")\
        .eq("graph_id", graph_id)\
        .order("version_number", desc=True)\
        .execute()
    
    versions = []
    if response.data:
        for version_data in response.data:
            versions.append(BrainGraphVersion(
                id=version_data["id"],
                graph_id=version_data["graph_id"],
                version_number=version_data["version_number"],
                brain_ids=version_data["brain_ids"],
                released_date=datetime.fromisoformat(version_data["released_date"].replace("Z", "+00:00")),
                status=version_data["status"]
            ))
    return versions

def update_brain_graph_version_status(version_id: str, new_status: str) -> BrainGraphVersion:
    """
    Update a brain graph version's status
    
    Args:
        version_id: UUID of the version
        new_status: New status (training, published)
    
    Returns:
        Updated BrainGraphVersion object
    """
    valid_statuses = {"training", "published"}
    if new_status not in valid_statuses:
        raise ValueError(f"Status must be one of {valid_statuses}")

    # If publishing, check if there's already a published version for this graph
    if new_status == "published":
        # Get the graph_id for this version
        version_response = supabase.table("brain_graph_version")\
            .select("graph_id")\
            .eq("id", version_id)\
            .execute()
        
        if version_response.data:
            graph_id = version_response.data[0]["graph_id"]
            
            # Check for existing published version
            published_response = supabase.table("brain_graph_version")\
                .select("id")\
                .eq("graph_id", graph_id)\
                .eq("status", "published")\
                .execute()
            
            if published_response.data:
                # Set existing published version back to training
                supabase.table("brain_graph_version")\
                    .update({"status": "training"})\
                    .eq("id", published_response.data[0]["id"])\
                    .execute()

    response = supabase.table("brain_graph_version")\
        .update({"status": new_status})\
        .eq("id", version_id)\
        .execute()
    
    if response.data and len(response.data) > 0:
        version_data = response.data[0]
        return BrainGraphVersion(
            id=version_data["id"],
            graph_id=version_data["graph_id"],
            version_number=version_data["version_number"],
            brain_ids=version_data["brain_ids"],
            released_date=datetime.fromisoformat(version_data["released_date"].replace("Z", "+00:00")),
            status=version_data["status"]
        )
    raise Exception("Failed to update brain graph version status or version not found")

def add_brains_to_version(version_id: str, brain_ids: List[str]) -> BrainGraphVersion:
    """
    Add brain UUIDs to a brain graph version
    
    Args:
        version_id: UUID of the version
        brain_ids: List of brain UUIDs to add
    
    Returns:
        Updated BrainGraphVersion object
    """
    # First get current brain_ids
    response = supabase.table("brain_graph_version")\
        .select("brain_ids", "status")\
        .eq("id", version_id)\
        .execute()
    
    if not response.data:
        raise Exception("Version not found")
    
    version_data = response.data[0]
    if version_data["status"] == "published":
        raise ValueError("Cannot modify a published version")
    
    current_brain_ids = version_data["brain_ids"]
    # Add new brain_ids (avoid duplicates)
    updated_brain_ids = list(set(current_brain_ids + brain_ids))
    
    # Update the version with new brain_ids
    update_response = supabase.table("brain_graph_version")\
        .update({"brain_ids": updated_brain_ids})\
        .eq("id", version_id)\
        .execute()
    
    if update_response.data and len(update_response.data) > 0:
        version_data = update_response.data[0]
        return BrainGraphVersion(
            id=version_data["id"],
            graph_id=version_data["graph_id"],
            version_number=version_data["version_number"],
            brain_ids=version_data["brain_ids"],
            released_date=datetime.fromisoformat(version_data["released_date"].replace("Z", "+00:00")),
            status=version_data["status"]
        )
    raise Exception("Failed to update brain graph version")

def remove_brains_from_version(version_id: str, brain_ids: List[str]) -> BrainGraphVersion:
    """
    Remove brain UUIDs from a brain graph version
    
    Args:
        version_id: UUID of the version
        brain_ids: List of brain UUIDs to remove
    
    Returns:
        Updated BrainGraphVersion object
    """
    # First get current brain_ids
    response = supabase.table("brain_graph_version")\
        .select("brain_ids", "status")\
        .eq("id", version_id)\
        .execute()
    
    if not response.data:
        raise Exception("Version not found")
    
    version_data = response.data[0]
    if version_data["status"] == "published":
        raise ValueError("Cannot modify a published version")
    
    current_brain_ids = version_data["brain_ids"]
    # Remove specified brain_ids
    updated_brain_ids = [bid for bid in current_brain_ids if bid not in brain_ids]
    
    # Update the version with new brain_ids
    update_response = supabase.table("brain_graph_version")\
        .update({"brain_ids": updated_brain_ids})\
        .eq("id", version_id)\
        .execute()
    
    if update_response.data and len(update_response.data) > 0:
        version_data = update_response.data[0]
        return BrainGraphVersion(
            id=version_data["id"],
            graph_id=version_data["graph_id"],
            version_number=version_data["version_number"],
            brain_ids=version_data["brain_ids"],
            released_date=datetime.fromisoformat(version_data["released_date"].replace("Z", "+00:00")),
            status=version_data["status"]
        )
    raise Exception("Failed to update brain graph version")

def validate_brain_ids(brain_ids: List[str]) -> bool:
    """
    Validate that all brain UUIDs exist in the brain table
    
    Args:
        brain_ids: List of brain UUIDs to validate
    
    Returns:
        True if all brain UUIDs exist, False otherwise
    """
    if not brain_ids:
        return True
        
    response = supabase.table("brain")\
        .select("id")\
        .in_("id", brain_ids)\
        .execute()
    
    if response.data:
        found_ids = {brain["id"] for brain in response.data}
        return len(found_ids) == len(brain_ids)
    return False

def update_brain_graph(graph_id: str, name: Optional[str] = None, description: Optional[str] = None) -> BrainGraph:
    """
    Update the name and/or description of a brain graph.
    Args:
        graph_id: UUID of the brain graph
        name: New name (optional)
        description: New description (optional)
    Returns:
        Updated BrainGraph object
    """
    update_data = {}
    if name is not None:
        update_data["name"] = name
    if description is not None:
        update_data["description"] = description
    if not update_data:
        raise ValueError("No update fields provided")
    response = supabase.table("brain_graph").update(update_data).eq("id", graph_id).execute()
    if response.data and len(response.data) > 0:
        graph_data = response.data[0]
        return BrainGraph(
            id=graph_data["id"],
            org_id=graph_data["org_id"],
            name=graph_data["name"],
            description=graph_data["description"],
            created_date=datetime.fromisoformat(graph_data["created_date"].replace("Z", "+00:00"))
        )
    raise Exception("Failed to update brain graph or brain graph not found") 