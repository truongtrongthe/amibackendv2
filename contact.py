import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import logging

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

logger = logging.getLogger(__name__)

class ContactManager:
    def __init__(self):
        self.contacts_table = "contacts"
        self.profiles_table = "profiles"
        self.profile_versions_table = "profile_versions"

    # Create a new contact
    def create_contact(self, organization_id: str, type: str, first_name: str, last_name: str, email: str = None, phone: str = None, facebook_id: str = None, profile_picture_url: str = None) -> dict:
        if type not in ["partner", "customer"]:
            raise ValueError("Type must be 'partner' or 'customer'")
        
        contact_data = {
            "organization_id": organization_id,
            "type": type,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
            "facebook_id": facebook_id,
            "profile_picture_url": profile_picture_url,
            "created_at": datetime.utcnow().isoformat()
        }
        response = supabase.table(self.contacts_table).insert(contact_data).execute()
        return response.data[0] if response.data else None

    # Update an existing contact by id
    def update_contact(self, contact_id: int, organization_id: str = None, **kwargs) -> dict:
        allowed_fields = {"type", "first_name", "last_name", "email", "phone", "facebook_id", "profile_picture_url", "organization_id"}
        update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if "type" in update_data and update_data["type"] not in ["partner", "customer"]:
            raise ValueError("Type must be 'partner' or 'customer'")
        
        query = supabase.table(self.contacts_table).update(update_data).eq("id", contact_id)
        
        # If organization_id is provided, also filter by it
        if organization_id:
            query = query.eq("organization_id", organization_id)
            
        response = query.execute()
        return response.data[0] if response.data else None

    # Get all contacts
    def get_contacts(self, organization_id: str = None) -> list:
        """
        Fetch all contacts from the database, optionally filtered by organization_id
        """
        try:
            logger.info(f"Fetching contacts for organization_id: {organization_id or 'all'}")
            
            # Execute the query with error handling
            query = supabase.table(self.contacts_table).select("*")
            
            # Filter by organization_id if provided
            if organization_id:
                query = query.eq("organization_id", organization_id)
                
            response = query.execute()
            
            if not response.data:
                logger.info(f"No contacts found for organization_id: {organization_id or 'all'}")
                return []
                
            logger.info(f"Successfully fetched {len(response.data)} contacts")
            return response.data
            
        except Exception as e:
            logger.error(f"Error in get_contacts: {str(e)}")
            raise

    # Get contact details (including profile) by id
    def get_contact_details(self, contact_id: int, organization_id: str = None) -> dict:
        query = supabase.table(self.contacts_table).select("*, profiles(*)").eq("id", contact_id)
        
        # Filter by organization_id if provided
        if organization_id:
            query = query.eq("organization_id", organization_id)
            
        response = query.execute()
        return response.data[0] if response.data else None

    # Create a contact profile
    def create_contact_profile(self, contact_id: int, profile_summary: str = None, general_info: str = None,
                              personality: str = None, hidden_desires: str = None, linkedin_url: str = None,
                              social_media_urls: list = None, best_goals: list = None) -> dict:
        profile_data = {
            "contact_id": contact_id,
            "profile_summary": profile_summary,
            "general_info": general_info,
            "personality": personality,
            "hidden_desires": hidden_desires,
            "linkedin_url": linkedin_url,
            "social_media_urls": social_media_urls or [],
            "best_goals": best_goals or [],
            "updated_at": datetime.utcnow().isoformat()
        }
        response = supabase.table(self.profiles_table).insert(profile_data).execute()
        return response.data[0] if response.data else None

    # Update a contact profile by contact_id
    def update_contact_profile(self, contact_id: int, **kwargs) -> dict:
        allowed_fields = {
            "profile_summary", "general_info", "personality", "hidden_desires",
            "linkedin_url", "social_media_urls", "best_goals"
        }
        update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        response = supabase.table(self.profiles_table).update(update_data).eq("contact_id", contact_id).execute()
        return response.data[0] if response.data else None

    # Get profile summary by contact_id
    def get_profile_summary(self, contact_id: int) -> str:
        response = supabase.table(self.profiles_table).select("profile_summary").eq("contact_id", contact_id).execute()
        return response.data[0]["profile_summary"] if response.data else None

    # Get profile summary versions by contact_id
    def get_profile_summary_versions(self, contact_id: int) -> list:
        profile_response = supabase.table(self.profiles_table).select("id").eq("contact_id", contact_id).execute()
        if not profile_response.data:
            return []
        
        profile_id = profile_response.data[0]["id"]
        versions_response = (
            supabase.table(self.profile_versions_table)
            .select("summary, changed_at")
            .eq("profile_id", profile_id)
            .order("changed_at", desc=True)
            .execute()
        )
        return versions_response.data

    # Optional: Get contact by UUID
    def get_contact_by_uuid(self, contact_uuid: str, organization_id: str = None) -> dict:
        query = supabase.table(self.contacts_table).select("*").eq("uuid", contact_uuid)
        
        # Filter by organization_id if provided
        if organization_id:
            query = query.eq("organization_id", organization_id)
            
        response = query.execute()
        return response.data[0] if response.data else None

    # Get contact by Facebook ID
    def get_contact_by_facebook_id(self, facebook_id: str, organization_id: str = None) -> dict:
        query = supabase.table(self.contacts_table).select("*").eq("facebook_id", facebook_id)
        
        # Filter by organization_id if provided
        if organization_id:
            query = query.eq("organization_id", organization_id)
            
        response = query.execute()
        return response.data[0] if response.data else None

# Example usage
if __name__ == "__main__":
    cm = ContactManager()

    # Create a contact
    new_contact = cm.create_contact("org123", "customer", "Jane", "Smith", "jane@example.com", "555-5678")
    print("Created Contact:", new_contact)

    # Update contact
    updated_contact = cm.update_contact(new_contact["id"], email="jane.smith@example.com")
    print("Updated Contact:", updated_contact)

    # Get all contacts
    all_contacts = cm.get_contacts("org123")
    print("All Contacts:", all_contacts)

    # Create a profile
    profile = cm.create_contact_profile(
        new_contact["id"],
        profile_summary="Creative marketer seeking recognition",
        best_goals=[{"goal": "Close deal", "deadline": "2025-06-01"}]
    )
    print("Created Profile:", profile)

    # Update profile
    updated_profile = cm.update_contact_profile(
        new_contact["id"],
        profile_summary="Experienced marketer driving growth"
    )
    print("Updated Profile:", updated_profile)

    # Get contact details
    details = cm.get_contact_details(new_contact["id"], "org123")
    print("Contact Details:", details)

    # Get profile summary
    summary = cm.get_profile_summary(new_contact["id"])
    print("Profile Summary:", summary)

    # Get profile summary versions
    versions = cm.get_profile_summary_versions(new_contact["id"])
    print("Summary Versions:", versions)

    # Get contact by UUID
    contact_by_uuid = cm.get_contact_by_uuid(new_contact["uuid"], "org123")
    print("Contact by UUID:", contact_by_uuid)

    # Get contact by Facebook ID
    contact_by_facebook_id = cm.get_contact_by_facebook_id(new_contact["facebook_id"], "org123")
    print("Contact by Facebook ID:", contact_by_facebook_id)