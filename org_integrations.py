from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime, timezone
from dataclasses import dataclass
from supabase import create_client, Client
import os
import json
import secrets
import uuid

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(spb_url, spb_key)

@dataclass
class OrganizationIntegration:
    id: UUID
    org_id: UUID
    integration_type: str
    name: str
    is_active: bool
    api_base_url: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_verify_token: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    config: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None

def create_integration(
    org_id: str,
    integration_type: str,
    name: str,
    api_base_url: str = None,
    webhook_url: str = None,
    webhook_verify_token: str = None,
    api_key: str = None,
    api_secret: str = None,
    access_token: str = None,
    refresh_token: str = None,
    token_expires_at: datetime = None,
    config: Dict[str, Any] = None,
    is_active: bool = False,
    base_domain: str = None  # Added for generating webhook URLs
) -> OrganizationIntegration:
    """
    Create a new integration for an organization
    
    Args:
        org_id: Organization UUID
        integration_type: Type of integration (odoo_crm, hubspot, salesforce, facebook, other)
        name: Display name for the integration
        api_base_url: Base URL for API calls
        webhook_url: URL for receiving webhooks
        webhook_verify_token: Token for verifying webhook requests (especially for Facebook)
        api_key: API key for authentication
        api_secret: API secret for authentication
        access_token: OAuth access token
        refresh_token: OAuth refresh token
        token_expires_at: Expiration time for access token
        config: Additional configuration as JSON
        is_active: Whether the integration is active
        base_domain: Base domain for generating webhook URLs (e.g., "https://yourdomain.com")
        
    Returns:
        OrganizationIntegration object with the created integration
    
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Validate required parameters
    if not org_id or not integration_type or not name:
        raise ValueError("org_id, integration_type, and name are required")
    
    # Validate integration_type
    valid_types = ["odoo_crm", "hubspot", "salesforce", "facebook", "google_drive", "other"]
    if integration_type not in valid_types:
        raise ValueError(f"integration_type must be one of {', '.join(valid_types)}")
    
    # Auto-generate webhook URL and verification token for Facebook integrations
    if integration_type == "facebook":
        # If no webhook_verify_token provided, generate a secure one
        if not webhook_verify_token:
            webhook_verify_token = secrets.token_urlsafe(32)  # Generate a secure 32-byte token
            print(f"Generated secure webhook verification token for Facebook integration")
        
        # Generate webhook URL if base_domain is provided
        if base_domain and not webhook_url:
            webhook_base = base_domain.rstrip('/')  # Remove trailing slash if present
            webhook_url = f"{webhook_base}/webhook?org_id={org_id}"
            print(f"Generated webhook URL: {webhook_url}")
        
        # Initialize config if needed
        if not config:
            config = {}
        elif isinstance(config, str):
            try:
                config = json.loads(config)
            except:
                config = {}
        
        # No warning about missing page_id - it's now optional
    
    # Prepare data for insertion
    integration_data = {
        "org_id": org_id,
        "integration_type": integration_type,
        "name": name,
        "is_active": is_active
    }
    
    # Add optional fields if provided
    if api_base_url:
        integration_data["api_base_url"] = api_base_url
    if webhook_url:
        integration_data["webhook_url"] = webhook_url
    if webhook_verify_token:
        integration_data["webhook_verify_token"] = webhook_verify_token
    if api_key:
        integration_data["api_key"] = api_key
    if api_secret:
        integration_data["api_secret"] = api_secret
    if access_token:
        integration_data["access_token"] = access_token
    if refresh_token:
        integration_data["refresh_token"] = refresh_token
    if token_expires_at:
        integration_data["token_expires_at"] = token_expires_at.isoformat()
    if config:
        integration_data["config"] = json.dumps(config)
    
    try:
        # Insert into database
        response = supabase.table("organization_integrations").insert(integration_data).execute()
        
        if not response.data:
            raise ValueError("Failed to create integration")
        
        # Convert response to OrganizationIntegration object
        integration_data = response.data[0]
        
        # Parse dates
        created_at = datetime.fromisoformat(integration_data["created_at"].replace("Z", "+00:00")) if integration_data.get("created_at") else None
        updated_at = datetime.fromisoformat(integration_data["updated_at"].replace("Z", "+00:00")) if integration_data.get("updated_at") else None
        token_expires = datetime.fromisoformat(integration_data["token_expires_at"].replace("Z", "+00:00")) if integration_data.get("token_expires_at") else None
        
        # Parse config
        config_data = integration_data.get("config", {})
        if isinstance(config_data, str):
            config_data = json.loads(config_data)
        
        return OrganizationIntegration(
            id=UUID(integration_data["id"]),
            org_id=UUID(integration_data["org_id"]),
            integration_type=integration_data["integration_type"],
            name=integration_data["name"],
            is_active=integration_data["is_active"],
            api_base_url=integration_data.get("api_base_url"),
            webhook_url=integration_data.get("webhook_url"),
            webhook_verify_token=integration_data.get("webhook_verify_token"),
            api_key=integration_data.get("api_key"),
            api_secret=integration_data.get("api_secret"),
            access_token=integration_data.get("access_token"),
            refresh_token=integration_data.get("refresh_token"),
            token_expires_at=token_expires,
            config=config_data,
            created_at=created_at,
            updated_at=updated_at
        )
    except Exception as e:
        raise ValueError(f"Failed to create integration: {str(e)}")

def get_org_integrations(org_id: str, active_only: bool = False, integration_type: str = None) -> List[OrganizationIntegration]:
    """
    Get all integrations for an organization
    
    Args:
        org_id: Organization UUID
        active_only: If True, return only active integrations
        integration_type: If provided, filter by integration type
        
    Returns:
        List of OrganizationIntegration objects
    """
    try:
        query = supabase.table("organization_integrations").select("*").eq("org_id", org_id)
        
        if active_only:
            query = query.eq("is_active", True)
            
        if integration_type:
            query = query.eq("integration_type", integration_type)
            
        response = query.execute()
        
        if not response.data:
            return []
        
        integrations = []
        for item in response.data:
            # Parse dates
            created_at = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")) if item.get("created_at") else None
            updated_at = datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")) if item.get("updated_at") else None
            token_expires = datetime.fromisoformat(item["token_expires_at"].replace("Z", "+00:00")) if item.get("token_expires_at") else None
            
            # Parse config
            config_data = item.get("config", {})
            if isinstance(config_data, str):
                config_data = json.loads(config_data)
            
            integration = OrganizationIntegration(
                id=UUID(item["id"]),
                org_id=UUID(item["org_id"]),
                integration_type=item["integration_type"],
                name=item["name"],
                is_active=item["is_active"],
                api_base_url=item.get("api_base_url"),
                webhook_url=item.get("webhook_url"),
                webhook_verify_token=item.get("webhook_verify_token"),
                api_key=item.get("api_key"),
                api_secret=item.get("api_secret"),
                access_token=item.get("access_token"),
                refresh_token=item.get("refresh_token"),
                token_expires_at=token_expires,
                config=config_data,
                created_at=created_at,
                updated_at=updated_at
            )
            integrations.append(integration)
            
        return integrations
    except Exception as e:
        print(f"Error getting organization integrations: {str(e)}")
        return []

def get_integration_by_id(integration_id: str) -> Optional[OrganizationIntegration]:
    """
    Get an integration by its ID
    
    Args:
        integration_id: Integration UUID
        
    Returns:
        OrganizationIntegration object or None if not found
    """
    try:
        response = supabase.table("organization_integrations").select("*").eq("id", integration_id).execute()
        
        if not response.data:
            return None
        
        item = response.data[0]
        
        # Parse dates
        created_at = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")) if item.get("created_at") else None
        updated_at = datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")) if item.get("updated_at") else None
        token_expires = datetime.fromisoformat(item["token_expires_at"].replace("Z", "+00:00")) if item.get("token_expires_at") else None
        
        # Parse config
        config_data = item.get("config", {})
        if isinstance(config_data, str):
            config_data = json.loads(config_data)
        
        return OrganizationIntegration(
            id=UUID(item["id"]),
            org_id=UUID(item["org_id"]),
            integration_type=item["integration_type"],
            name=item["name"],
            is_active=item["is_active"],
            api_base_url=item.get("api_base_url"),
            webhook_url=item.get("webhook_url"),
            webhook_verify_token=item.get("webhook_verify_token"),
            api_key=item.get("api_key"),
            api_secret=item.get("api_secret"),
            access_token=item.get("access_token"),
            refresh_token=item.get("refresh_token"),
            token_expires_at=token_expires,
            config=config_data,
            created_at=created_at,
            updated_at=updated_at
        )
    except Exception as e:
        print(f"Error getting integration by ID: {str(e)}")
        return None

def update_integration(
    integration_id: str,
    name: str = None,
    api_base_url: str = None,
    webhook_url: str = None,
    webhook_verify_token: str = None,
    api_key: str = None,
    api_secret: str = None,
    access_token: str = None,
    refresh_token: str = None,
    token_expires_at: datetime = None,
    config: Dict[str, Any] = None,
    is_active: bool = None
) -> Optional[OrganizationIntegration]:
    """
    Update an existing integration
    
    Args:
        integration_id: UUID of the integration to update
        name: New display name
        api_base_url: New base URL
        webhook_url: New webhook URL
        webhook_verify_token: New webhook verification token
        api_key: New API key
        api_secret: New API secret
        access_token: New access token
        refresh_token: New refresh token
        token_expires_at: New token expiry time
        config: New configuration
        is_active: New active status
        
    Returns:
        Updated OrganizationIntegration object or None if update failed
    """
    if not integration_id:
        raise ValueError("integration_id is required")
    
    update_data = {}
    
    # Add fields to update if provided
    if name is not None:
        update_data["name"] = name
    if api_base_url is not None:
        update_data["api_base_url"] = api_base_url
    if webhook_url is not None:
        update_data["webhook_url"] = webhook_url
    if webhook_verify_token is not None:
        update_data["webhook_verify_token"] = webhook_verify_token
    if api_key is not None:
        update_data["api_key"] = api_key
    if api_secret is not None:
        update_data["api_secret"] = api_secret
    if access_token is not None:
        update_data["access_token"] = access_token
    if refresh_token is not None:
        update_data["refresh_token"] = refresh_token
    if token_expires_at is not None:
        update_data["token_expires_at"] = token_expires_at.isoformat()
    if config is not None:
        update_data["config"] = json.dumps(config)
    if is_active is not None:
        update_data["is_active"] = is_active
    
    # If nothing to update, return current integration
    if not update_data:
        return get_integration_by_id(integration_id)
    
    try:
        response = supabase.table("organization_integrations").update(update_data).eq("id", integration_id).execute()
        
        if not response.data:
            return None
        
        item = response.data[0]
        
        # Parse dates
        created_at = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")) if item.get("created_at") else None
        updated_at = datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")) if item.get("updated_at") else None
        token_expires = datetime.fromisoformat(item["token_expires_at"].replace("Z", "+00:00")) if item.get("token_expires_at") else None
        
        # Parse config
        config_data = item.get("config", {})
        if isinstance(config_data, str):
            config_data = json.loads(config_data)
        
        return OrganizationIntegration(
            id=UUID(item["id"]),
            org_id=UUID(item["org_id"]),
            integration_type=item["integration_type"],
            name=item["name"],
            is_active=item["is_active"],
            api_base_url=item.get("api_base_url"),
            webhook_url=item.get("webhook_url"),
            webhook_verify_token=item.get("webhook_verify_token"),
            api_key=item.get("api_key"),
            api_secret=item.get("api_secret"),
            access_token=item.get("access_token"),
            refresh_token=item.get("refresh_token"),
            token_expires_at=token_expires,
            config=config_data,
            created_at=created_at,
            updated_at=updated_at
        )
    except Exception as e:
        print(f"Error updating integration: {str(e)}")
        return None

def delete_integration(integration_id: str) -> bool:
    """
    Delete an integration
    
    Args:
        integration_id: UUID of the integration to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    if not integration_id:
        return False
    
    try:
        response = supabase.table("organization_integrations").delete().eq("id", integration_id).execute()
        return len(response.data) > 0
    except Exception as e:
        print(f"Error deleting integration: {str(e)}")
        return False

def toggle_integration(integration_id: str, active: bool) -> Optional[OrganizationIntegration]:
    """
    Toggle the active status of an integration
    
    Args:
        integration_id: UUID of the integration
        active: New active status
        
    Returns:
        Updated OrganizationIntegration object or None if update failed
    """
    return update_integration(integration_id, is_active=active) 