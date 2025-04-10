import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

class ConversationManager:
    def __init__(self):
        self.conversations_table = "conversations"
        self.contacts_table = "contacts"

    # Create a new conversation
    def create_conversation(self, contact_id: int, title: str = None, 
                           conversation_data: Dict = None) -> dict:
        """
        Create a new conversation for a contact.
        
        Args:
            contact_id: The ID of the contact.
            title: Optional title for the conversation.
            conversation_data: Optional initial data for the conversation.
            
        Returns:
            The created conversation record.
        """
        conversation_data = conversation_data or {}
        
        conversation = {
            "contact_id": contact_id,
            "title": title,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "conversation_data": conversation_data
        }
        
        response = supabase.table(self.conversations_table).insert(conversation).execute()
        return response.data[0] if response.data else None

    # Get all conversations for a contact
    def get_conversations_by_contact(self, contact_id: int) -> List[Dict]:
        """
        Get all conversations for a specific contact.
        
        Args:
            contact_id: The ID of the contact.
            
        Returns:
            A list of conversation records.
        """
        response = supabase.table(self.conversations_table)\
            .select("*")\
            .eq("contact_id", contact_id)\
            .order("created_at", desc=True)\
            .execute()
        
        return response.data

    # Get a specific conversation by ID
    def get_conversation(self, conversation_id: int) -> Dict:
        """
        Get a specific conversation by its ID.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            The conversation record.
        """
        response = supabase.table(self.conversations_table)\
            .select("*")\
            .eq("id", conversation_id)\
            .execute()
        
        return response.data[0] if response.data else None

    # Get a specific conversation by UUID
    def get_conversation_by_uuid(self, uuid: str) -> Dict:
        """
        Get a specific conversation by its UUID.
        
        Args:
            uuid: The UUID of the conversation.
            
        Returns:
            The conversation record.
        """
        response = supabase.table(self.conversations_table)\
            .select("*")\
            .eq("uuid", uuid)\
            .execute()
        
        return response.data[0] if response.data else None

    # Update conversation data
    def update_conversation_data(self, conversation_id: int, 
                              conversation_data: Dict) -> Dict:
        """
        Update the conversation data for a specific conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            conversation_data: The new conversation data.
            
        Returns:
            The updated conversation record.
        """
        update_data = {
            "conversation_data": conversation_data,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = supabase.table(self.conversations_table)\
            .update(update_data)\
            .eq("id", conversation_id)\
            .execute()
        
        return response.data[0] if response.data else None

    # Update conversation title and status
    def update_conversation(self, conversation_id: int, title: str = None, 
                         status: str = None) -> Dict:
        """
        Update the title and/or status of a conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            title: Optional new title for the conversation.
            status: Optional new status for the conversation.
            
        Returns:
            The updated conversation record.
        """
        update_data = {"updated_at": datetime.utcnow().isoformat()}
        
        if title is not None:
            update_data["title"] = title
        
        if status is not None:
            if status not in ["active", "archived", "completed"]:
                raise ValueError("Status must be 'active', 'archived', or 'completed'")
            update_data["status"] = status
        
        response = supabase.table(self.conversations_table)\
            .update(update_data)\
            .eq("id", conversation_id)\
            .execute()
        
        return response.data[0] if response.data else None

    # Add message to conversation
    def add_message(self, conversation_id: int, 
                  message: Dict[str, Any]) -> Dict:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            message: The message to add (should include sender, content, timestamp).
            
        Returns:
            The updated conversation record.
        """
        # Get current conversation data
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"No conversation found with ID {conversation_id}")
        
        conversation_data = conversation.get("conversation_data", {})
        
        # Ensure messages array exists
        if "messages" not in conversation_data:
            conversation_data["messages"] = []
        
        # Add timestamp if not provided
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        # Add the new message
        conversation_data["messages"].append(message)
        
        # Update the conversation
        return self.update_conversation_data(conversation_id, conversation_data)

    # Get recent conversations for a contact with pagination
    def get_recent_conversations(self, contact_id: int, limit: int = 10, 
                              offset: int = 0) -> List[Dict]:
        """
        Get recent conversations for a contact with pagination.
        
        Args:
            contact_id: The ID of the contact.
            limit: Maximum number of conversations to return.
            offset: Offset for pagination.
            
        Returns:
            A list of conversation records.
        """
        response = supabase.table(self.conversations_table)\
            .select("*")\
            .eq("contact_id", contact_id)\
            .order("updated_at", desc=True)\
            .range(offset, offset + limit - 1)\
            .execute()
        
        return response.data

    # Search conversations by title or content
    def search_conversations(self, contact_id: int, search_term: str) -> List[Dict]:
        """
        Search conversations by title or content.
        
        Args:
            contact_id: The ID of the contact.
            search_term: The term to search for.
            
        Returns:
            A list of matching conversation records.
        """
        # For PostgreSQL JSON search
        response = supabase.table(self.conversations_table)\
            .select("*")\
            .eq("contact_id", contact_id)\
            .filter("title", "ilike", f"%{search_term}%")\
            .execute()
        
        return response.data

    # Add Facebook messenger message to conversation
    def add_fb_message(self, contact_id: int, platform_conversation_id: str, 
                      fb_message: Dict[str, Any], create_if_missing: bool = True) -> Dict:
        """
        Add a Facebook Messenger message to a conversation.
        
        Args:
            contact_id: The ID of the contact.
            platform_conversation_id: Facebook's conversation/thread ID.
            fb_message: The Facebook message data.
            create_if_missing: Create conversation if it doesn't exist.
            
        Returns:
            The updated conversation record.
        """
        # Try to find existing conversation by platform_conversation_id
        response = supabase.table(self.conversations_table)\
            .select("*")\
            .eq("contact_id", contact_id)\
            .eq("platform", "facebook")\
            .eq("platform_conversation_id", platform_conversation_id)\
            .execute()
        
        conversation = response.data[0] if response.data else None
        
        # Create new conversation if not found and create_if_missing is True
        if not conversation and create_if_missing:
            # Create title from the first message content (truncated)
            title = fb_message.get("content", "")[:50] + "..." if len(fb_message.get("content", "")) > 50 else fb_message.get("content", "")
            
            conversation_data = {"messages": []}
            new_conversation = {
                "contact_id": contact_id,
                "title": title,
                "platform": "facebook",
                "platform_conversation_id": platform_conversation_id,
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "conversation_data": conversation_data
            }
            
            response = supabase.table(self.conversations_table).insert(new_conversation).execute()
            conversation = response.data[0] if response.data else None
            
        if not conversation:
            raise ValueError(f"No conversation found and unable to create one")
        
        # Get current conversation data
        conversation_id = conversation["id"]
        conversation_data = conversation.get("conversation_data", {})
        
        # Ensure messages array exists
        if "messages" not in conversation_data:
            conversation_data["messages"] = []
        
        # Generate a unique message ID if not provided
        if "id" not in fb_message:
            fb_message["id"] = f"msg_{int(datetime.utcnow().timestamp())}_{len(conversation_data['messages'])}"
        
        # Set timestamp if not provided
        if "timestamp" not in fb_message:
            fb_message["timestamp"] = datetime.utcnow().isoformat()
        
        # Add platform type if not specified
        if "platform" not in fb_message:
            fb_message["platform"] = "facebook"
        
        # Add the new message
        conversation_data["messages"].append(fb_message)
        
        # Update the conversation with last message details for quick access
        update_data = {
            "conversation_data": conversation_data,
            "last_message_text": fb_message.get("content", ""),
            "last_message_timestamp": fb_message.get("timestamp"),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # If message is from contact, increment unread count
        if fb_message.get("sender_type") == "contact":
            update_data["unread_count"] = conversation.get("unread_count", 0) + 1
        
        response = supabase.table(self.conversations_table)\
            .update(update_data)\
            .eq("id", conversation_id)\
            .execute()
        
        return response.data[0] if response.data else None
    
    # Get or create conversation by platform ID
    def get_or_create_conversation_by_platform_id(self, contact_id: int, 
                                               platform: str, 
                                               platform_conversation_id: str) -> Dict:
        """
        Get a conversation by platform ID or create one if it doesn't exist.
        
        Args:
            contact_id: The ID of the contact.
            platform: The messaging platform (e.g., 'facebook').
            platform_conversation_id: The platform's conversation ID.
            
        Returns:
            The conversation record.
        """
        # Try to find existing conversation
        response = supabase.table(self.conversations_table)\
            .select("*")\
            .eq("contact_id", contact_id)\
            .eq("platform", platform)\
            .eq("platform_conversation_id", platform_conversation_id)\
            .execute()
        
        if response.data:
            return response.data[0]
        
        # Create new conversation
        new_conversation = {
            "contact_id": contact_id,
            "title": f"New {platform} conversation",
            "platform": platform,
            "platform_conversation_id": platform_conversation_id,
            "status": "active",
            "conversation_data": {"messages": []},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = supabase.table(self.conversations_table).insert(new_conversation).execute()
        return response.data[0] if response.data else None
    
    # Mark conversation as read
    def mark_conversation_as_read(self, conversation_id: int) -> Dict:
        """
        Mark all messages in a conversation as read.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            The updated conversation record.
        """
        update_data = {
            "unread_count": 0,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = supabase.table(self.conversations_table)\
            .update(update_data)\
            .eq("id", conversation_id)\
            .execute()
        
        return response.data[0] if response.data else None

# Example usage
if __name__ == "__main__":
    cm = ConversationManager()

    # Create a conversation
    conversation = cm.create_conversation(
        contact_id=1,
        title="Initial discussion",
        conversation_data={"messages": []}
    )
    print("Created Conversation:", conversation)

    # Add a message
    updated = cm.add_message(
        conversation_id=conversation["id"],
        message={
            "sender": "system",
            "content": "Hello, how can I help you today?",
            "type": "text"
        }
    )
    print("Updated Conversation:", updated)

    # Get conversations for a contact
    conversations = cm.get_conversations_by_contact(1)
    print("Conversations for Contact:", conversations) 