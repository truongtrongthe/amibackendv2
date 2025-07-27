from supabase import create_client, Client
from typing import List, Dict, Optional
import os
from datetime import datetime, UTC
from utilities import logger  # Assuming this is your logging utility

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(
    spb_url,
    spb_key
)


class AIA:
    def __init__(self, id: str, aia_id: int, org_id: str, task_type: str, name: Optional[str],
                 brain_ids: List[int], delivery_method_ids: List[int], created_date: datetime):
        self.id = id  # UUID
        self.aia_id = aia_id  # INT (assuming integer ID exists; adjust if not)
        self.org_id = org_id  # UUID of the organization
        self.task_type = task_type  # 'Chat' or 'Work'
        self.name = name
        self.brain_ids = brain_ids  # List of brain integer IDs
        self.delivery_method_ids = delivery_method_ids  # List of delivery method integer IDs
        self.created_date = created_date


def create_aia(org_id: str, task_type: str, name: Optional[str], brain_ids: List[int],
               delivery_method_ids: List[int]) -> AIA:
    """
    Create a new AIA with specified task type, linked brains, and delivery methods.

    Args:
        org_id: UUID of the organization
        task_type: Either 'Chat' or 'Work'
        name: Optional name for the AIA
        brain_ids: List of integer brain IDs to link
        delivery_method_ids: List of integer delivery method IDs to assign

    Returns:
        AIA object

    Raises:
        ValueError: If org_id or task_type is invalid, or if brain/delivery method IDs don't exist
        Exception: If creation fails
    """
    # Validate task_type
    valid_task_types = {"Chat", "Work"}
    if task_type not in valid_task_types:
        raise ValueError(f"task_type must be one of {valid_task_types}")

    # Check if organization exists
    org_check = supabase.table("organization").select("id").eq("id", org_id).execute()
    if not org_check.data:
        raise ValueError(f"Organization with id {org_id} does not exist")

    # Check if all brain_ids exist
    for brain_id in brain_ids:
        brain_check = supabase.table("brain").select("brain_id").eq("brain_id", brain_id).execute()
        if not brain_check.data:
            raise ValueError(f"Brain with id {brain_id} does not exist")

    # Check if all delivery_method_ids exist and match task_type
    for dm_id in delivery_method_ids:
        dm_check = supabase.table("delivery_method").select("id", "task_type").eq("id", dm_id).execute()
        if not dm_check.data:
            raise ValueError(f"Delivery method with id {dm_id} does not exist")
        if dm_check.data[0]["task_type"] != task_type:
            raise ValueError(f"Delivery method {dm_id} does not support task_type {task_type}")

    # Insert AIA
    data = {
        "org_id": org_id,
        "task_type": task_type,
        "name": name,
        "created_at": datetime.now(UTC).isoformat()
    }
    response = supabase.table("aia").insert(data).execute()

    if not response.data:
        raise Exception("Failed to create AIA")

    aia_data = response.data[0]
    aia_id = aia_data["id"]  # UUID

    # Link brains
    brain_links = [{"aia_id": aia_id, "brain_id": bid} for bid in brain_ids]
    if brain_links:
        supabase.table("aia_brain").insert(brain_links).execute()

    # Link delivery methods
    dm_links = [{"aia_id": aia_id, "delivery_method_id": dmid} for dmid in delivery_method_ids]
    if dm_links:
        supabase.table("aia_delivery_method").insert(dm_links).execute()

    # Note: Assuming 'id' is UUID and there's an integer 'aia_id'; adjust if only UUID exists
    return AIA(
        id=aia_data["id"],
        aia_id=aia_data.get("aia_id", 0),  # Adjust based on actual integer ID field name
        org_id=aia_data["org_id"],
        task_type=aia_data["task_type"],
        name=aia_data["name"],
        brain_ids=brain_ids,
        delivery_method_ids=delivery_method_ids,
        created_date=datetime.fromisoformat(aia_data["created_at"].replace("Z", "+00:00"))
    )


def update_aia(aia_id: str, task_type: Optional[str] = None, name: Optional[str] = None,
               brain_ids: Optional[List[int]] = None, delivery_method_ids: Optional[List[int]] = None) -> AIA:
    """
    Update an existing AIA's details.

    Args:
        aia_id: UUID of the AIA to update
        task_type: New task type ('Chat' or 'Work'), optional
        name: New name, optional
        brain_ids: New list of brain IDs, optional (replaces existing links)
        delivery_method_ids: New list of delivery method IDs, optional (replaces existing links)

    Returns:
        Updated AIA object

    Raises:
        ValueError: If task_type or IDs are invalid
        Exception: If update fails
    """
    # Check if AIA exists
    aia_check = supabase.table("aia").select("*").eq("id", aia_id).execute()
    if not aia_check.data:
        raise ValueError(f"AIA with id {aia_id} does not exist")
    current_aia = aia_check.data[0]

    # Prepare update data
    update_data = {}
    if task_type:
        valid_task_types = {"Chat", "Work"}
        if task_type not in valid_task_types:
            raise ValueError(f"task_type must be one of {valid_task_types}")
        update_data["task_type"] = task_type
    if name is not None:  # Allow empty string or None
        update_data["name"] = name

    # Update AIA table if there are changes
    if update_data:
        response = supabase.table("aia").update(update_data).eq("id", aia_id).execute()
        if not response.data:
            raise Exception("Failed to update AIA")
        updated_aia = response.data[0]
    else:
        updated_aia = current_aia

    # Handle brain links
    if brain_ids is not None:
        # Validate brain IDs
        for brain_id in brain_ids:
            brain_check = supabase.table("brain").select("brain_id").eq("brain_id", brain_id).execute()
            if not brain_check.data:
                raise ValueError(f"Brain with id {brain_id} does not exist")
        # Delete existing links
        supabase.table("aia_brain").delete().eq("aia_id", aia_id).execute()
        # Insert new links
        brain_links = [{"aia_id": aia_id, "brain_id": bid} for bid in brain_ids]
        if brain_links:
            supabase.table("aia_brain").insert(brain_links).execute()

    # Handle delivery method links
    if delivery_method_ids is not None:
        # Validate delivery method IDs and task_type compatibility
        task_type_to_check = task_type or current_aia["task_type"]
        for dm_id in delivery_method_ids:
            dm_check = supabase.table("delivery_method").select("id", "task_type").eq("id", dm_id).execute()
            if not dm_check.data:
                raise ValueError(f"Delivery method with id {dm_id} does not exist")
            if dm_check.data[0]["task_type"] != task_type_to_check:
                raise ValueError(f"Delivery method {dm_id} does not support task_type {task_type_to_check}")
        # Delete existing links
        supabase.table("aia_delivery_method").delete().eq("aia_id", aia_id).execute()
        # Insert new links
        dm_links = [{"aia_id": aia_id, "delivery_method_id": dmid} for dmid in delivery_method_ids]
        if dm_links:
            supabase.table("aia_delivery_method").insert(dm_links).execute()

    # Fetch final state
    final_response = supabase.table("aia").select("*").eq("id", aia_id).execute()
    if not final_response.data:
        raise Exception("Failed to retrieve updated AIA")
    aia_data = final_response.data[0]

    # Fetch linked brain and delivery method IDs
    brain_links = supabase.table("aia_brain").select("brain_id").eq("aia_id", aia_id).execute()
    brain_ids_final = [link["brain_id"] for link in brain_links.data] if brain_links.data else []
    dm_links = supabase.table("aia_delivery_method").select("delivery_method_id").eq("aia_id", aia_id).execute()
    dm_ids_final = [link["delivery_method_id"] for link in dm_links.data] if dm_links.data else []

    return AIA(
        id=aia_data["id"],
        aia_id=aia_data.get("aia_id", 0),  # Adjust based on actual integer ID field name
        org_id=aia_data["orga_id"],
        task_type=aia_data["task_type"],
        name=aia_data["name"],
        brain_ids=brain_ids_final,
        delivery_method_ids=dm_ids_final,
        created_date=datetime.fromisoformat(aia_data["created_at"].replace("Z", "+00:00"))
    )


def delete_aia(aia_id: str) -> None:
    """
    Delete an AIA by its UUID.

    Args:
        aia_id: UUID of the AIA to delete

    Raises:
        ValueError: If AIA does not exist
        Exception: If deletion fails
    """
    # Check if AIA exists
    aia_check = supabase.table("aia").select("id").eq("id", aia_id).execute()
    if not aia_check.data:
        raise ValueError(f"AIA with id {aia_id} does not exist")

    # Delete AIA (cascades to aia_brain and aia_delivery_method due to schema)
    response = supabase.table("aia").delete().eq("id", aia_id).execute()
    if not response.data:
        raise Exception("Failed to delete AIA")


def get_all_aias(org_id: str) -> List[AIA]:
    """
    Fetch all AIAs for a given organization using UUID org_id.

    Args:
        org_id: UUID of the organization

    Returns:
        List of AIA objects

    Raises:
        ValueError: If organization does not exist
    """
    # Check if organization exists
    org_check = supabase.table("organization").select("id").eq("id", org_id).execute()
    if not org_check.data:
        raise ValueError(f"Organization with id {org_id} does not exist")

    # Fetch all AIAs for the organization
    response = supabase.table("aia").select("*").eq("org_id", org_id).execute()

    aias = []
    if response.data:
        for aia_data in response.data:
            # Fetch linked brain IDs
            brain_links = supabase.table("aia_brain").select("brain_id").eq("aia_id", aia_data["id"]).execute()
            brain_ids = [link["brain_id"] for link in brain_links.data] if brain_links.data else []

            # Fetch linked delivery method IDs
            dm_links = supabase.table("aia_delivery_method").select("delivery_method_id").eq("aia_id", aia_data["id"]).execute()
            dm_ids = [link["delivery_method_id"] for link in dm_links.data] if dm_links.data else []

            aias.append(AIA(
                id=aia_data["id"],
                aia_id=aia_data.get("aia_id", 0),  # Adjust based on actual integer ID field name
                org_id=aia_data["org_id"],
                task_type=aia_data["task_type"],
                name=aia_data["name"],
                brain_ids=brain_ids,
                delivery_method_ids=dm_ids,
                created_date=datetime.fromisoformat(aia_data["created_at"].replace("Z", "+00:00"))
            ))
    return aias


def get_aia_detail(aia_id: str) -> Optional[AIA]:
    """
    Fetch details of a specific AIA using the UUID aia_id.

    Args:
        aia_id: UUID of the AIA

    Returns:
        AIA object if found, None otherwise
    """
    # Fetch AIA details
    response = supabase.table("aia").select("*").eq("id", aia_id).execute()

    if response.data and len(response.data) > 0:
        aia_data = response.data[0]

        # Fetch linked brain IDs
        brain_links = supabase.table("aia_brain").select("brain_id").eq("aia_id", aia_id).execute()
        brain_ids = [link["brain_id"] for link in brain_links.data] if brain_links.data else []

        # Fetch linked delivery method IDs
        dm_links = supabase.table("aia_delivery_method").select("delivery_method_id").eq("aia_id", aia_id).execute()
        dm_ids = [link["delivery_method_id"] for link in dm_links.data] if dm_links.data else []

        return AIA(
            id=aia_data["id"],
            aia_id=aia_data.get("aia_id", 0),  # Adjust based on actual integer ID field name
            org_id=aia_data["org_id"],
            task_type=aia_data["task_type"],
            name=aia_data["name"],
            brain_ids=brain_ids,
            delivery_method_ids=dm_ids,
            created_date=datetime.fromisoformat(aia_data["created_at"].replace("Z", "+00:00"))
        )
    return None

