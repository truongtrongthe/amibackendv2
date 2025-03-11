import os
from supabase import create_client, Client

# Load environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_knowledge_entry(raw_content: str, summary: str):
    """Insert a new knowledge entry into the knowledge_collection table."""
    data = {
        "raw_content": raw_content,
        "summary": summary
    }
   
    response = supabase.table("knowledge_collection").insert(data).execute()
    return response

def get_knowledge_entries(filters: dict = None):
    """Retrieve knowledge entries from the knowledge_collection table with optional filters."""
    query = supabase.table("knowledge_collection")
    if filters:
        query = query.select("*").filter(**filters)
    else:
        query = query.select("*")
    response = query.execute()
    return response

def update_knowledge_entry(entry_id: str, raw_content: str, summary: str):
    """Update a knowledge entry in the knowledge_collection table based on entry ID."""
    data = {
        "raw_content": raw_content,
        "summary": summary
    }
    response = supabase.table("knowledge_collection").update(data).eq("id", entry_id).execute()
    return response

def delete_knowledge_entry(entry_id: str):
    """Delete a knowledge entry from the knowledge_collection table based on entry ID."""
    response = supabase.table("knowledge_collection").delete().eq("id", entry_id).execute()
    return response
