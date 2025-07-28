from supabase import create_client, Client
from typing import List, Dict, Optional
import os
from datetime import datetime, UTC
from utilities import logger  # Assuming this is your logger utility

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(spb_url, spb_key)

class BrainLog:
    def __init__(self, entry_id: int, brain_id: str, entry_values: str, gap_analysis: Optional[str], created_date: datetime):
        self.entry_id = entry_id  # INT
        self.brain_id = brain_id  # UUID
        self.entry_values = entry_values  # TEXT
        self.gap_analysis = gap_analysis  # TEXT, optional
        self.created_date = created_date  # DATETIME (fixed from created_at)

def create_brain_log(brain_id: str, entry_values: str, gap_analysis: Optional[str] = None) -> BrainLog:
    """
    Create a new brain log entry with created_date
    """
    # Check if brain exists
    brain_check = supabase.table("brain")\
        .select("id")\
        .eq("id", brain_id)\
        .execute()
    
    if not brain_check.data:
        raise ValueError(f"Brain with UUID {brain_id} does not exist")

    data = {
        "brain_id": brain_id,
        "entry_values": entry_values,
        "gap_analysis": gap_analysis,
        "created_date": datetime.now(UTC).isoformat()
    }
    
    response = supabase.table("brain_logs").insert(data).execute()
    
    if response.data:
        log_data = response.data[0]
        return BrainLog(
            entry_id=log_data["entry_id"],
            brain_id=log_data["brain_id"],
            entry_values=log_data["entry_values"],
            gap_analysis=log_data["gap_analysis"],
            created_date=datetime.fromisoformat(log_data["created_date"].replace("Z", "+00:00"))
        )
    raise Exception("Failed to create brain log")

def update_brain_log(entry_id: int, entry_values: str, gap_analysis: Optional[str] = None) -> BrainLog:
    """
    Update a brain log's entry_values and gap_analysis using the entry_id
    """
    update_data = {
        "entry_values": entry_values,
        "gap_analysis": gap_analysis
    }
    
    response = supabase.table("brain_logs")\
        .update(update_data)\
        .eq("entry_id", entry_id)\
        .execute()
    
    if response.data and len(response.data) > 0:
        log_data = response.data[0]
        return BrainLog(
            entry_id=log_data["entry_id"],
            brain_id=log_data["brain_id"],
            entry_values=log_data["entry_values"],
            gap_analysis=log_data["gap_analysis"],
            created_date=datetime.fromisoformat(log_data["created_date"].replace("Z", "+00:00"))
        )
    raise Exception("Failed to update brain log or log not found")

def get_brain_logs(brain_id: str) -> List[BrainLog]:
    """
    Fetch all logs for a given brain using UUID brain_id
    """
    response = supabase.table("brain_logs")\
        .select("*")\
        .eq("brain_id", brain_id)\
        .execute()
    
    logs = []
    if response.data:
        for log_data in response.data:
            logs.append(BrainLog(
                entry_id=log_data["entry_id"],
                brain_id=log_data["brain_id"],
                entry_values=log_data["entry_values"],
                gap_analysis=log_data["gap_analysis"],
                created_date=datetime.fromisoformat(log_data["created_date"].replace("Z", "+00:00"))
            ))
    return logs

def get_brain_log_detail(entry_id: int) -> Optional[BrainLog]:
    """
    Fetch details of a specific brain log using the integer entry_id
    """
    response = supabase.table("brain_logs")\
        .select("*")\
        .eq("entry_id", entry_id)\
        .execute()
    
    if response.data and len(response.data) > 0:
        log_data = response.data[0]
        return BrainLog(
            entry_id=log_data["entry_id"],
            brain_id=log_data["brain_id"],
            entry_values=log_data["entry_values"],
            gap_analysis=log_data["gap_analysis"],
            created_date=datetime.fromisoformat(log_data["created_date"].replace("Z", "+00:00"))
        )
    return None

# Example usage
if __name__ == "__main__":
    try:
        # Create a log
        new_log = create_brain_log("b98d586e-108d-4c2f-8934-5baabb93d1d4", "Test accuracy: 95%", "Needs edge cases")
        print(f"Created log: {new_log.entry_id}")

        # Update a log
        updated_log = update_brain_log(new_log.entry_id, "Test accuracy: 96%", "Improved edge cases")
        print(f"Updated log: {updated_log.entry_values}")

        # Get all logs for a brain
        logs = get_brain_logs("b98d586e-108d-4c2f-8934-5baabb93d1d4")
        for log in logs:
            print(f"Log {log.entry_id}: {log.entry_values}")

        # Get a specific log
        log_detail = get_brain_log_detail(new_log.entry_id)
        print(f"Log detail: {log_detail.entry_values if log_detail else 'Not found'}")
    except Exception as e:
        print(f"Error: {e}")