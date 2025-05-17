from supabase import create_client, Client
from typing import List, Dict, Optional, Literal
import os
from datetime import datetime, UTC, timedelta
from utilities import logger  # Assuming this is your logging utility

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(
    spb_url,
    spb_key
)

class OrganizationUsage:
    def __init__(self, org_id: str):
        self.org_id = org_id
        
    def _get_date_range(self, period: Literal['day', 'week', 'month', 'year'] = 'day') -> tuple[str, str]:
        """Helper to get start and end date based on period"""
        current_date = datetime.now(UTC)
        
        if period == 'day':
            start_date = current_date.strftime("%Y-%m-%d")
            end_date = start_date
        elif period == 'week':
            # Start of week (Monday)
            start_date = (current_date - timedelta(days=current_date.weekday())).strftime("%Y-%m-%d")
            end_date = current_date.strftime("%Y-%m-%d")
        elif period == 'month':
            start_date = current_date.replace(day=1).strftime("%Y-%m-%d")
            end_date = current_date.strftime("%Y-%m-%d")
        elif period == 'year':
            start_date = current_date.replace(month=1, day=1).strftime("%Y-%m-%d")
            end_date = current_date.strftime("%Y-%m-%d")
        else:
            start_date = current_date.strftime("%Y-%m-%d")
            end_date = start_date
            
        return start_date, end_date

    def get_message_count(self, period: Literal['day', 'week', 'month', 'year'] = 'day') -> int:
        try:
            start_date, end_date = self._get_date_range(period)
            
            # Query the message_count table for the organization in the date range
            response = supabase.table("organization_usage") \
                .select("count") \
                .eq("org_id", self.org_id) \
                .gte("date", start_date) \
                .lte("date", end_date) \
                .eq("type", "message") \
                .execute()
                
            # Sum up all the counts
            total_count = sum(item["count"] for item in response.data) if response.data else 0
            return total_count
        except Exception as e:
            logger.error(f"Error getting message count for org {self.org_id}: {e}")
            return 0
            
    def get_reasoning_count(self, period: Literal['day', 'week', 'month', 'year'] = 'day') -> int:
        try:
            start_date, end_date = self._get_date_range(period)
            
            # Query the usage table for reasoning count
            response = supabase.table("organization_usage") \
                .select("count") \
                .eq("org_id", self.org_id) \
                .gte("date", start_date) \
                .lte("date", end_date) \
                .eq("type", "reasoning") \
                .execute()
                
            # Sum up all the counts
            total_count = sum(item["count"] for item in response.data) if response.data else 0
            return total_count
        except Exception as e:
            logger.error(f"Error getting reasoning count for org {self.org_id}: {e}")
            return 0
    
    def get_usage_summary(self, period: Literal['day', 'week', 'month', 'year'] = 'day') -> Dict:
        """Get a summary of all usage metrics for the organization"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            # Query all usage for the organization in the date range
            response = supabase.table("organization_usage") \
                .select("type, count") \
                .eq("org_id", self.org_id) \
                .gte("date", start_date) \
                .lte("date", end_date) \
                .execute()
                
            # Summarize by type
            summary = {}
            for item in response.data:
                usage_type = item["type"]
                count = item["count"]
                summary[usage_type] = summary.get(usage_type, 0) + count
                
            return summary
        except Exception as e:
            logger.error(f"Error getting usage summary for org {self.org_id}: {e}")
            return {}
    
    def add_message(self, count: int = 1) -> bool:
        """Add message usage count for the organization"""
        try:
            return self._add_usage("message", count)
        except Exception as e:
            logger.error(f"Error adding message count for org {self.org_id}: {e}")
            return False
            
    def add_reasoning(self, count: int = 1) -> bool:
        """Add reasoning usage count for the organization"""
        try:
            return self._add_usage("reasoning", count)
        except Exception as e:
            logger.error(f"Error adding reasoning count for org {self.org_id}: {e}")
            return False
    
    def _add_usage(self, usage_type: str, count: int) -> bool:
        """Generic method to add usage of any type"""
        try:
            # Get the current date in UTC
            current_date = datetime.now(UTC)
            date_str = current_date.strftime("%Y-%m-%d")

            # Query the usage table for the organization
            response = supabase.table("organization_usage") \
                .select("*") \
                .eq("org_id", self.org_id) \
                .eq("date", date_str) \
                .eq("type", usage_type) \
                .execute()

            # If the count already exists, update it
            if response.data:
                existing_count = response.data[0]["count"]  
                new_count = existing_count + count

                # Update the count
                supabase.table("organization_usage") \
                    .update({"count": new_count}) \
                    .eq("id", response.data[0]["id"]) \
                    .execute()
            else:
                # Create a new count entry
                supabase.table("organization_usage") \
                    .insert({
                        "org_id": self.org_id, 
                        "count": count, 
                        "date": date_str,
                        "type": usage_type
                    }) \
                    .execute()
                    
            # Add detailed usage record
            self._add_usage_detail(usage_type, count)
            return True
        except Exception as e:
            logger.error(f"Error adding {usage_type} count for org {self.org_id}: {e}")
            return False
    
    def _add_usage_detail(self, usage_type: str, count: int) -> None:
        """Add a detailed usage record for auditing purposes"""
        try:
            # Get the current timestamp in UTC
            timestamp = datetime.now(UTC).isoformat()
            
            # Insert the usage detail record
            supabase.table("usage_detail") \
                .insert({
                    "org_id": self.org_id,
                    "type": usage_type,
                    "count": count,
                    "timestamp": timestamp
                }) \
                .execute()
        except Exception as e:
            logger.error(f"Error adding usage detail for org {self.org_id}: {e}")

    def get_usage_details(self, 
                         usage_type: Optional[str] = None, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None,
                         limit: int = 100) -> List[Dict]:
        """Get detailed usage records with optional filtering"""
        try:
            query = supabase.table("usage_detail").select("*").eq("org_id", self.org_id)
            
            if usage_type:
                query = query.eq("type", usage_type)
                
            if start_date:
                query = query.gte("timestamp", f"{start_date}T00:00:00Z")
                
            if end_date:
                query = query.lte("timestamp", f"{end_date}T23:59:59Z")
                
            # Order by timestamp descending and limit results
            response = query.order("timestamp", desc=True).limit(limit).execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error getting usage details for org {self.org_id}: {e}")
            return []


