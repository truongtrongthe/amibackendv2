from supabase import create_client, Client
from typing import List, Dict, Optional, Literal, Any
import os
from datetime import datetime, UTC, timedelta
from utilities import logger  # Assuming this is your logging utility

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

# Log basic connection info
if spb_url and spb_key:
    logger.info(f"Initialized Supabase client: {spb_url}")
    supabase: Optional[Client] = create_client(spb_url, spb_key)
else:
    logger.error("Missing Supabase URL or key in environment variables")
    # Create a mock client for development/testing
    supabase: Optional[Client] = None

class OrganizationUsage:
    def __init__(self, org_id: str):
        self.org_id = org_id
        
    def is_configured(self) -> bool:
        """Check if usage tracking is properly configured"""
        return supabase is not None
        
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
            if not supabase:
                logger.warning("Supabase client not initialized, returning 0")
                return 0
                
            start_date, end_date = self._get_date_range(period)
            
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
            if not supabase:
                logger.warning("Supabase client not initialized, returning 0")
                return 0
                
            start_date, end_date = self._get_date_range(period)
            
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
            if not supabase:
                logger.warning("Supabase client not initialized, returning empty summary")
                return {}
                
            start_date, end_date = self._get_date_range(period)
            
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
            logger.info(f"Adding message count for org {self.org_id}: {count}")
            return self._add_usage("message", count)
        except Exception as e:
            logger.error(f"Error adding message count for org {self.org_id}: {e}")
            return False
            
    def add_reasoning(self, count: int = 1) -> bool:
        """Add reasoning usage count for the organization"""
        try:
            logger.info(f"Adding reasoning count for org {self.org_id}: {count}")
            return self._add_usage("reasoning", count)
        except Exception as e:
            logger.error(f"Error adding reasoning count for org {self.org_id}: {e}")
            return False
    
    def _add_usage(self, usage_type: str, count: int) -> bool:
        """Generic method to add usage of any type"""
        try:
            if not supabase:
                logger.warning("Supabase client not initialized, skipping usage tracking")
                return False
                
            if count <= 0:
                logger.warning(f"Skipping usage with non-positive count: {count}")
                return True
                
            # Get the current date in UTC
            current_date = datetime.now(UTC)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # First check if a record already exists for this org, type, and date
            existing_records = supabase.table("organization_usage") \
                .select("id, count") \
                .eq("org_id", self.org_id) \
                .eq("date", date_str) \
                .eq("type", usage_type) \
                .execute()
                
            if existing_records.data and len(existing_records.data) > 0:
                # Update existing record
                existing_record = existing_records.data[0]
                record_id = existing_record["id"]
                new_count = existing_record["count"] + count
                
                # Update the count
                supabase.table("organization_usage") \
                    .update({"count": new_count}) \
                    .eq("id", record_id) \
                    .execute()
                logger.info(f"Updated usage record: type={usage_type}, count={new_count}")
            else:
                # Insert new record
                data = {
                    "org_id": self.org_id,
                    "type": usage_type,
                    "count": count,
                    "date": date_str
                }
                supabase.table("organization_usage").insert(data).execute()
                logger.info(f"Created new usage record: type={usage_type}, count={count}")
                
            # Add a detailed usage record
            self._add_usage_detail(usage_type, count)
            return True
        except Exception as e:
            logger.error(f"Error adding {usage_type} usage: {e}")
            return False
    
    def _add_usage_detail(self, usage_type: str, count: int) -> None:
        """Add a detailed usage record for auditing purposes"""
        try:
            if not supabase:
                logger.warning("Supabase client not initialized, skipping usage detail tracking")
                return
                
            # Get the current timestamp in UTC
            timestamp = datetime.now(UTC).isoformat()
            
            # Add detailed record
            data = {
                "org_id": self.org_id,
                "type": usage_type,
                "count": count,
                "timestamp": timestamp
            }
            
            supabase.table("usage_detail").insert(data).execute()
            logger.info(f"Added usage detail record: type={usage_type}, count={count}")
        except Exception as e:
            logger.error(f"Error adding usage detail: {e}")

    def get_usage_details(self, 
                         usage_type: Optional[str] = None, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None,
                         limit: int = 100) -> List[Dict]:
        """Get detailed usage records with optional filtering"""
        try:
            if not supabase:
                logger.warning("Supabase client not initialized, returning empty details")
                return []
                
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

    def check_usage_limit(self, usage_type: str, limit: int, period: Literal['day', 'week', 'month', 'year'] = 'day') -> Dict[str, Any]:
        """Check if usage is within limits for a given type and period"""
        try:
            if not supabase:
                logger.warning("Supabase client not initialized, assuming unlimited usage")
                return {
                    "within_limit": True,
                    "current_usage": 0,
                    "limit": limit,
                    "remaining": limit,
                    "period": period
                }
                
            current_usage = 0
            if usage_type == "message":
                current_usage = self.get_message_count(period)
            elif usage_type == "reasoning":
                current_usage = self.get_reasoning_count(period)
            else:
                # For other types, get from summary
                summary = self.get_usage_summary(period)
                current_usage = summary.get(usage_type, 0)
                
            remaining = max(0, limit - current_usage)
            within_limit = current_usage < limit
            
            return {
                "within_limit": within_limit,
                "current_usage": current_usage,
                "limit": limit,
                "remaining": remaining,
                "period": period
            }
        except Exception as e:
            logger.error(f"Error checking usage limit for org {self.org_id}: {e}")
            return {
                "within_limit": True,  # Default to allowing usage on error
                "current_usage": 0,
                "limit": limit,
                "remaining": limit,
                "period": period
            }


