from supabase import create_client, Client
from typing import Optional
import os
from datetime import datetime, UTC
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from utilities import logger

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(spb_url, spb_key)

# Create router
router = APIRouter()

# Simple request model
class JoinAmiRequest(BaseModel):
    email: EmailStr
    name: str
    company: Optional[str] = None
    phone: Optional[str] = None
    message: Optional[str] = None

@router.post("/join-ami")
async def join_ami(request: JoinAmiRequest):
    """
    Join AMI waitlist - single endpoint to create waitlist entry
    """
    try:
        result = create_waitlist_entry(
            email=request.email,
            name=request.name,
            company=request.company,
            phone=request.phone,
            message=request.message
        )
        
        if result:
            return {"message": "Successfully joined AMI waitlist!", "status": "success"}
        
        raise HTTPException(status_code=500, detail="Failed to join waitlist")
    
    except ValueError as e:
        # Handle duplicate email case
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error joining AMI waitlist: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to join waitlist")

# Simple function to create waitlist entry (if needed elsewhere)
def create_waitlist_entry(email: str, name: str, company: str = None, phone: str = None, message: str = None):
    """
    Create a new waitlist entry
    """
    try:
        # Check if email already exists
        existing_check = supabase.table("waitlist")\
            .select("id")\
            .eq("email", email)\
            .execute()
        
        if existing_check.data:
            raise ValueError(f"Email {email} already exists in waitlist")

        data = {
            "email": email,
            "name": name,
            "company": company,
            "phone": phone,
            "message": message,
            "status": "pending",
            "created_date": datetime.now(UTC).isoformat()
        }
        
        response = supabase.table("waitlist").insert(data).execute()
        
        if response.data:
            return response.data[0]
        
        raise Exception("Failed to create waitlist entry")
    
    except Exception as e:
        logger.error(f"Error creating waitlist entry: {str(e)}")
        raise
