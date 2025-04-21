import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import logging
import json

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
        self.analysis_table = "contact_analysis"

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
            query = supabase.table(self.contacts_table).select("*, profiles(*)")
            
            # Filter by organization_id if provided
            if organization_id:
                query = query.eq("organization_id", organization_id)
                
            response = query.execute()
            
            if not response.data:
                logger.info(f"No contacts found for organization_id: {organization_id or 'all'}")
                return []
            
            # Enrich contacts with tags based on sales data
            enriched_contacts = self._enrich_contacts_with_tags(response.data, organization_id)
                
            logger.info(f"Successfully fetched {len(enriched_contacts)} contacts")
            return enriched_contacts
            
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
        contact = response.data[0] if response.data else None
        
        if contact:
            # Enrich single contact with tags
            contacts_with_tags = self._enrich_contacts_with_tags([contact], organization_id)
            return contacts_with_tags[0] if contacts_with_tags else contact
        
        return None

    def _enrich_contacts_with_tags(self, contacts: list, organization_id: str = None) -> list:
        """
        Enrich contacts with tags based on sales priority and analysis data
        """
        if not contacts:
            return []
            
        try:
            # Get all contact IDs
            contact_ids = [contact["id"] for contact in contacts]
            
            # Fetch latest analysis for all contacts in one query
            analysis_query = supabase.table(self.analysis_table).select("*")
            
            if contact_ids:
                analysis_query = analysis_query.in_("contact_id", contact_ids)
            
            if organization_id:
                analysis_query = analysis_query.eq("organization_id", organization_id)
                
            analysis_response = analysis_query.order("analyzed_at", desc=True).execute()
            
            # Create a map of contact_id to their latest analysis
            analysis_map = {}
            for analysis in analysis_response.data:
                contact_id = analysis.get("contact_id")
                if contact_id not in analysis_map:
                    analysis_map[contact_id] = analysis
                    
            # Add tags to each contact based on analysis data
            for contact in contacts:
                contact_id = contact.get("id")
                contact_tags = []
                
                # Add basic type tag
                if contact.get("type"):
                    contact_tags.append(contact["type"])
                    
                # Add tags based on analysis if available
                if contact_id in analysis_map:
                    analysis = analysis_map[contact_id]
                    
                    # Add priority tag
                    priority = analysis.get("priority", "")
                    if "High" in priority:
                        contact_tags.append("hot_lead")
                    elif "Medium" in priority:
                        contact_tags.append("warm_lead")
                    elif "Low" in priority:
                        contact_tags.append("nurture_lead")
                    else:
                        contact_tags.append("unqualified_lead")
                    
                    # Add score range tag
                    score = analysis.get("sales_readiness_score", 0)
                    if score >= 80:
                        contact_tags.append("score_80_plus")
                    elif score >= 60:
                        contact_tags.append("score_60_plus")
                    elif score >= 40:
                        contact_tags.append("score_40_plus")
                    
                    # Parse score_breakdown
                    score_breakdown = {}
                    if analysis.get("score_breakdown"):
                        try:
                            if isinstance(analysis["score_breakdown"], str):
                                score_breakdown = json.loads(analysis["score_breakdown"])
                            else:
                                score_breakdown = analysis["score_breakdown"]
                        except (json.JSONDecodeError, TypeError):
                            score_breakdown = {}
                    
                    # Add tags for high scores in specific areas
                    if score_breakdown.get("urgency", 0) >= 15:
                        contact_tags.append("high_urgency")
                    if score_breakdown.get("explicit_interest", 0) >= 15:
                        contact_tags.append("high_interest")
                    if score_breakdown.get("decision_authority", 0) >= 10:
                        contact_tags.append("decision_maker")
                
                # Add tags based on profile data
                profile = contact.get("profiles", {})
                if profile:
                    # Check for goals with upcoming deadlines
                    if profile.get("best_goals") and isinstance(profile["best_goals"], list):
                        for goal in profile["best_goals"]:
                            if isinstance(goal, dict) and goal.get("deadline"):
                                try:
                                    deadline = datetime.strptime(goal["deadline"], '%Y-%m-%d')
                                    today = datetime.now()
                                    days_until_deadline = (deadline - today).days
                                    
                                    if days_until_deadline <= 7:
                                        contact_tags.append("deadline_this_week")
                                        break
                                    elif days_until_deadline <= 30:
                                        contact_tags.append("deadline_this_month")
                                        break
                                except (ValueError, TypeError):
                                    # Skip if date parsing fails
                                    pass
                
                # Add the tags to the contact
                contact["tags"] = list(set(contact_tags))  # Remove duplicates
                
            return contacts
                
        except Exception as e:
            logger.error(f"Error in _enrich_contacts_with_tags: {str(e)}")
            # Return original contacts if enrichment fails
            return contacts

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
        query = supabase.table(self.contacts_table).select("*, profiles(*)").eq("uuid", contact_uuid)
        
        # Filter by organization_id if provided
        if organization_id:
            query = query.eq("organization_id", organization_id)
            
        response = query.execute()
        contact = response.data[0] if response.data else None
        
        if contact:
            # Enrich single contact with tags
            contacts_with_tags = self._enrich_contacts_with_tags([contact], organization_id)
            return contacts_with_tags[0] if contacts_with_tags else contact
        
        return None

    # Get contact by Facebook ID
    def get_contact_by_facebook_id(self, facebook_id: str, organization_id: str = None) -> dict:
        query = supabase.table(self.contacts_table).select("*, profiles(*)").eq("facebook_id", facebook_id)
        
        # Filter by organization_id if provided
        if organization_id:
            query = query.eq("organization_id", organization_id)
            
        response = query.execute()
        contact = response.data[0] if response.data else None
        
        if contact:
            # Enrich single contact with tags
            contacts_with_tags = self._enrich_contacts_with_tags([contact], organization_id)
            return contacts_with_tags[0] if contacts_with_tags else contact
        
        return None

    # Get contacts filtered by tags
    def get_contacts_by_tags(self, tags: list, organization_id: str = None) -> list:
        """
        Fetch contacts and filter them by tags
        
        Args:
            tags: List of tags to filter by (contact must have at least one of these tags)
            organization_id: Optional organization ID to filter by
            
        Returns:
            List of contacts with the specified tags
        """
        try:
            logger.info(f"Fetching contacts with tags {tags} for organization_id: {organization_id or 'all'}")
            
            # Get all contacts first (they will be enriched with tags)
            all_contacts = self.get_contacts(organization_id)
            
            # No contacts found
            if not all_contacts:
                return []
            
            # Filter contacts by tags
            filtered_contacts = []
            for contact in all_contacts:
                contact_tags = contact.get("tags", [])
                # Add to filtered list if contact has any of the requested tags
                if any(tag in contact_tags for tag in tags):
                    filtered_contacts.append(contact)
            
            logger.info(f"Found {len(filtered_contacts)} contacts with tags {tags}")
            return filtered_contacts
            
        except Exception as e:
            logger.error(f"Error in get_contacts_by_tags: {str(e)}")
            return []

    # Get available tag categories
    def get_tag_categories(self) -> dict:
        """
        Get available tag categories and values for frontend filtering
        
        Returns:
            Dictionary of tag categories and their possible values
        """
        return {
            "priority": [
                "hot_lead",
                "warm_lead", 
                "nurture_lead",
                "unqualified_lead"
            ],
            "type": [
                "customer",
                "partner"
            ],
            "score": [
                "score_80_plus",
                "score_60_plus",
                "score_40_plus"
            ],
            "urgency": [
                "high_urgency",
                "deadline_this_week",
                "deadline_this_month"
            ],
            "qualifiers": [
                "high_interest",
                "decision_maker"
            ]
        }

    # Create or update a contact profile (2-in-1 method)
    def create_or_update_contact_profile(self, contact_id: int, **kwargs) -> dict:
        """
        Create a new profile if one doesn't exist for the contact, or update an existing profile.
        
        Args:
            contact_id: The ID of the contact
            **kwargs: Profile fields to set/update (profile_summary, general_info, personality, 
                     hidden_desires, linkedin_url, social_media_urls, best_goals)
                     
        Returns:
            The created or updated profile record
        """
        # Get contact details to check if profile exists
        contact = self.get_contact_details(contact_id)
        
        # Set allowed fields
        allowed_fields = {
            "profile_summary", "general_info", "personality", "hidden_desires",
            "linkedin_url", "social_media_urls", "best_goals"
        }
        
        # Filter kwargs to only include allowed fields
        profile_data = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        # Add updated_at timestamp
        profile_data["updated_at"] = datetime.utcnow().isoformat()
        
        # If profile exists, update it
        if contact and contact.get("profiles"):
            return self.update_contact_profile(contact_id, **profile_data)
        
        # Otherwise create a new profile
        # Add required contact_id
        profile_data["contact_id"] = contact_id
        
        # Ensure arrays have default empty values
        if "social_media_urls" not in profile_data:
            profile_data["social_media_urls"] = []
        if "best_goals" not in profile_data:
            profile_data["best_goals"] = []
            
        # Create profile
        response = supabase.table(self.profiles_table).insert(profile_data).execute()
        return response.data[0] if response.data else None

