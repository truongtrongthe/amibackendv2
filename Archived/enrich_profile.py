import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import asyncio
from langchain_openai import ChatOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

from contact import ContactManager
from contactconvo import ConversationManager
from utilities import logger

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize OpenAI model for analysis
LLM = ChatOpenAI(model="gpt-4o")

class ProfileEnricher:
    def __init__(self):
        self.contact_manager = ContactManager()
        self.conversation_manager = ConversationManager()
    
    async def generate_profile_for_contact(self, contact_id: int, organization_id: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive profile for a contact based on their conversation history.
        
        Args:
            contact_id: The ID of the contact
            organization_id: Optional organization ID for filtering
            
        Returns:
            Dict: The updated profile data
        """
        logger.info(f"Generating profile for contact ID: {contact_id}")
        
        # Get contact details
        contact = self.contact_manager.get_contact_details(contact_id, organization_id)
        if not contact:
            logger.error(f"Contact with ID {contact_id} not found")
            raise ValueError(f"Contact with ID {contact_id} not found")
        
        # Get conversation history
        conversations = self.conversation_manager.get_conversations_by_contact(contact_id)
        if not conversations:
            logger.info(f"No conversations found for contact ID {contact_id}, creating empty profile")
            return self._create_empty_profile(contact_id)
        
        # Format conversations for analysis
        conversation_context = self._format_conversations_for_analysis(conversations)
        
        # Analyze conversations to extract profile data
        profile_data = await self._analyze_conversations(contact, conversation_context)
        
        # Update or create profile
        updated_profile = self.contact_manager.create_or_update_contact_profile(contact_id, **profile_data)
        
        return updated_profile
    
    def _create_empty_profile(self, contact_id: int) -> Dict:
        """
        Create an empty profile for a contact with no conversation history.
        
        Args:
            contact_id: The ID of the contact
            
        Returns:
            Dict: The empty profile data
        """
        empty_profile = {
            "profile_summary": "Insufficient conversation data to generate profile.",
            "general_info": "",
            "personality": "",
            "hidden_desires": "",
            "social_media_urls": [],
            "best_goals": []
        }
        
        return self.contact_manager.create_or_update_contact_profile(contact_id, **empty_profile)
    
    def _format_conversations_for_analysis(self, conversations: List[Dict]) -> str:
        """
        Format conversation data into a text format suitable for LLM analysis.
        
        Args:
            conversations: List of conversation records
            
        Returns:
            str: Formatted conversation text
        """
        conversation_text = ""
        
        # Sort conversations by date (oldest first)
        sorted_conversations = sorted(conversations, key=lambda x: x.get("created_at", ""))
        
        for idx, conversation in enumerate(sorted_conversations):
            conversation_text += f"CONVERSATION {idx+1} (Date: {conversation.get('created_at', 'Unknown')})\n"
            conversation_text += f"Title: {conversation.get('title', 'Untitled')}\n"
            
            # Extract messages from conversation data
            conversation_data = conversation.get("conversation_data", {})
            messages = conversation_data.get("messages", [])
            
            if not messages:
                conversation_text += "No messages in this conversation.\n\n"
                continue
            
            # Sort messages by timestamp
            sorted_messages = sorted(messages, key=lambda x: x.get("timestamp", ""))
            
            for message in sorted_messages:
                sender = message.get("sender", "Unknown")
                if sender == "system":
                    sender_label = "Agent"
                elif message.get("sender_type") == "contact":
                    sender_label = "Contact"
                else:
                    sender_label = sender.capitalize()
                
                content = message.get("content", "")
                timestamp = message.get("timestamp", "")
                
                conversation_text += f"{sender_label} ({timestamp}): {content}\n"
            
            conversation_text += "\n---\n\n"
        
        return conversation_text
    
    async def _analyze_conversations(self, contact: Dict, conversation_context: str) -> Dict:
        """
        Analyze conversation history to extract profile information.
        
        Args:
            contact: The contact data
            conversation_context: Formatted conversation history
            
        Returns:
            Dict: Extracted profile data
        """
        # Build analysis prompt
        prompt = self._build_profile_analysis_prompt(contact, conversation_context)
        
        # Call LLM for analysis
        try:
            result = LLM.invoke(prompt)
            analysis_result = result.content
            
            # Process analysis result
            profile_data = self._process_profile_analysis(analysis_result)
            
            return profile_data
        except Exception as e:
            logger.error(f"Error during conversation analysis: {str(e)}")
            # Return minimal profile data in case of error
            return {
                "profile_summary": f"Error analyzing conversations: {str(e)}",
                "general_info": "",
                "personality": "",
                "hidden_desires": "",
                "best_goals": []
            }
    
    def _build_profile_analysis_prompt(self, contact: Dict, conversation_context: str) -> str:
        """
        Build the prompt for profile analysis.
        
        Args:
            contact: The contact data
            conversation_context: Formatted conversation history
            
        Returns:
            str: The analysis prompt
        """
        contact_name = f"{contact.get('first_name', '')} {contact.get('last_name', '')}".strip()
        contact_type = contact.get('type', 'customer')
        
        return (
            f"TASK: Create a comprehensive profile for {contact_name} (Contact Type: {contact_type}) based on their conversation history.\n\n"
            f"CONVERSATION HISTORY:\n{conversation_context}\n\n"
            
            f"INSTRUCTIONS:\n"
            f"Based on the conversation history, extract and organize information about the contact into the following profile sections:\n\n"
            
            f"1. PROFILE SUMMARY: Create a 3-5 sentence executive summary of who this contact is, their main goals/needs, and key characteristics.\n\n"
            
            f"2. GENERAL INFO: Extract factual information about the contact such as:\n"
            f"   - Demographics (age, location, etc.)\n"
            f"   - Professional information (job, company, position)\n"
            f"   - Background information\n"
            f"   - Any products/services they've mentioned using\n\n"
            
            f"3. PERSONALITY: Describe the person's communication style, preferences, and personality traits you can infer from conversations:\n"
            f"   - How do they communicate?\n"
            f"   - What tone do they use?\n"
            f"   - Are they formal/informal, direct/indirect?\n"
            f"   - Any notable personality traits?\n\n"
            
            f"4. HIDDEN DESIRES: Based on the conversation, analyze what might be their underlying needs, pain points, or motivations:\n"
            f"   - What problems are they trying to solve?\n"
            f"   - What motivates their questions/requests?\n"
            f"   - What concerns or hesitations do they express?\n\n"
            
            f"5. BEST GOALS: List 1-3 specific goals this person appears to have, with each goal containing:\n"
            f"   - goal_name: A short name for the goal\n"
            f"   - description: A detailed description of what they're trying to achieve\n"
            f"   - importance: Low, Medium, or High priority based on context\n"
            f"   - status: Not Started, In Progress, or Completed\n"
            f"   - deadline: If mentioned (in YYYY-MM-DD format) or null\n\n"
            
            f"IMPORTANT GUIDELINES:\n"
            f"- Only include information that is evident from the conversation history\n"
            f"- For anything unclear or not mentioned, use minimalist language without speculation\n"
            f"- For sparse conversations, it's okay to note 'Insufficient information' in appropriate sections\n"
            f"- Format your output so it can be directly parsed into the contact profile\n"
            f"- For the BEST GOALS section, format as a structured array of 1-3 goals\n\n"
            
            f"FORMAT YOUR RESPONSE INTO THESE EXACT SECTIONS:\n"
            f"PROFILE_SUMMARY:\n"
            f"GENERAL_INFO:\n"
            f"PERSONALITY:\n"
            f"HIDDEN_DESIRES:\n"
            f"BEST_GOALS:\n"
        )
    
    def _process_profile_analysis(self, analysis_result: str) -> Dict:
        """
        Process the LLM analysis result into structured profile data.
        
        Args:
            analysis_result: The raw text result from LLM analysis
            
        Returns:
            Dict: Structured profile data
        """
        # Initialize default profile data
        profile_data = {
            "profile_summary": "",
            "general_info": "",
            "personality": "",
            "hidden_desires": "",
            "best_goals": []
        }
        
        # Extract sections using regex patterns
        import re
        
        # Extract profile summary
        summary_match = re.search(r'PROFILE_SUMMARY:(.*?)(?=GENERAL_INFO:|$)', analysis_result, re.DOTALL)
        if summary_match:
            profile_data["profile_summary"] = summary_match.group(1).strip()
        
        # Extract general info
        general_info_match = re.search(r'GENERAL_INFO:(.*?)(?=PERSONALITY:|$)', analysis_result, re.DOTALL)
        if general_info_match:
            profile_data["general_info"] = general_info_match.group(1).strip()
        
        # Extract personality
        personality_match = re.search(r'PERSONALITY:(.*?)(?=HIDDEN_DESIRES:|$)', analysis_result, re.DOTALL)
        if personality_match:
            profile_data["personality"] = personality_match.group(1).strip()
        
        # Extract hidden desires
        desires_match = re.search(r'HIDDEN_DESIRES:(.*?)(?=BEST_GOALS:|$)', analysis_result, re.DOTALL)
        if desires_match:
            profile_data["hidden_desires"] = desires_match.group(1).strip()
        
        # Extract best goals
        goals_match = re.search(r'BEST_GOALS:(.*?)(?=$)', analysis_result, re.DOTALL)
        if goals_match:
            goals_text = goals_match.group(1).strip()
            
            # Try to parse structured goals from the text
            try:
                goals = self._parse_goals_from_text(goals_text)
                profile_data["best_goals"] = goals
            except Exception as e:
                logger.error(f"Error parsing goals: {str(e)}")
                profile_data["best_goals"] = []
        
        return profile_data
    
    def _parse_goals_from_text(self, goals_text: str) -> List[Dict]:
        """
        Parse goals data from the text format into structured data.
        
        Args:
            goals_text: Text containing goals information
            
        Returns:
            List[Dict]: Structured goals data
        """
        goals = []
        
        # First try to parse as JSON
        try:
            # Check if the text is already JSON formatted
            if goals_text.strip().startswith('[') and goals_text.strip().endswith(']'):
                parsed_goals = json.loads(goals_text)
                if isinstance(parsed_goals, list):
                    return parsed_goals
        except:
            pass
        
        # If not JSON, try to parse as text
        import re
        
        # Look for numbered or bullet-point goals
        goal_blocks = re.split(r'\n\s*(?:\d+\.|\-|\*)\s*', goals_text)
        goal_blocks = [block.strip() for block in goal_blocks if block.strip()]
        
        for block in goal_blocks:
            goal = {
                "goal_name": "",
                "description": "",
                "importance": "Medium",
                "status": "Not Started",
                "deadline": None
            }
            
            # Extract goal name
            name_match = re.search(r'(?:Goal|Name):\s*(.*?)(?:\n|$)', block, re.IGNORECASE)
            if name_match:
                goal["goal_name"] = name_match.group(1).strip()
            else:
                # Use first sentence as name if not explicitly specified
                first_sentence = re.match(r'^(.*?[.!?])(?:\s|$)', block)
                if first_sentence:
                    goal["goal_name"] = first_sentence.group(1).strip()
                else:
                    goal["goal_name"] = block[:50].strip()
            
            # Extract description
            desc_match = re.search(r'(?:Description|Details):\s*(.*?)(?:\n|$)', block, re.IGNORECASE)
            if desc_match:
                goal["description"] = desc_match.group(1).strip()
            else:
                # Use remaining text as description if not specified
                goal["description"] = block
            
            # Extract importance
            importance_match = re.search(r'(?:Importance|Priority):\s*(Low|Medium|High)', block, re.IGNORECASE)
            if importance_match:
                goal["importance"] = importance_match.group(1).capitalize()
            
            # Extract status
            status_match = re.search(r'Status:\s*(Not Started|In Progress|Completed)', block, re.IGNORECASE)
            if status_match:
                goal["status"] = status_match.group(1)
            
            # Extract deadline
            deadline_match = re.search(r'Deadline:\s*(\d{4}-\d{2}-\d{2}|null|none)', block, re.IGNORECASE)
            if deadline_match:
                deadline = deadline_match.group(1).lower()
                if deadline not in ('null', 'none'):
                    goal["deadline"] = deadline
            
            goals.append(goal)
        
        # Use empty goals array if no goals found
        return goals

    async def batch_update_profiles(self, organization_id: Optional[str] = None, max_contacts: int = 100) -> List[Dict]:
        """
        Update profiles for multiple contacts in a batch process.
        
        Args:
            organization_id: Optional organization ID to filter contacts
            max_contacts: Maximum number of contacts to process in this batch
            
        Returns:
            List[Dict]: List of updated profile records
        """
        # Get contacts without profiles or with outdated profiles
        contacts_to_update = self._get_contacts_needing_profile_update(organization_id, max_contacts)
        
        if not contacts_to_update:
            logger.info("No contacts need profile updates")
            return []
        
        updated_profiles = []
        for contact in contacts_to_update:
            try:
                contact_id = contact["id"]
                logger.info(f"Updating profile for contact ID {contact_id}")
                
                profile = await self.generate_profile_for_contact(contact_id, organization_id)
                updated_profiles.append(profile)
                
                # Add a small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error updating profile for contact ID {contact.get('id')}: {str(e)}")
        
        return updated_profiles
    
    def _get_contacts_needing_profile_update(self, organization_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get a list of contacts that need profile updates.
        
        Args:
            organization_id: Optional organization ID to filter contacts
            limit: Maximum number of contacts to return
            
        Returns:
            List[Dict]: Contacts needing profile updates
        """
        # Get all contacts, optionally filtered by organization_id
        contacts = self.contact_manager.get_contacts(organization_id)
        
        # Filter contacts that need profile updates
        contacts_to_update = []
        
        for contact in contacts:
            # 1. No profile exists
            if not contact.get("profiles"):
                contacts_to_update.append(contact)
                continue
            
            # 2. Profile is older than 30 days
            profile = contact.get("profiles", {})
            if profile:
                updated_at = profile.get("updated_at")
                if updated_at:
                    try:
                        last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        now = datetime.utcnow()
                        days_since_update = (now - last_update).days
                        
                        if days_since_update > 30:
                            contacts_to_update.append(contact)
                            continue
                    except (ValueError, TypeError):
                        # If date parsing fails, assume update is needed
                        contacts_to_update.append(contact)
                        continue
            
            # 3. Profile is empty or has empty required fields
            if (profile and (
                not profile.get("profile_summary") or 
                not profile.get("general_info") or
                not profile.get("personality")
            )):
                contacts_to_update.append(contact)
                continue
        
        # Limit the number of contacts to process
        return contacts_to_update[:limit]

# Command line interface for profile enrichment
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich contact profiles based on conversation history")
    parser.add_argument("--contact_id", type=int, help="ID of the contact to update")
    parser.add_argument("--org_id", type=str, help="Organization ID to filter contacts")
    parser.add_argument("--batch", action="store_true", help="Run a batch update for all contacts needing updates")
    parser.add_argument("--max", type=int, default=100, help="Maximum number of contacts to process in batch mode")
    
    args = parser.parse_args()
    
    async def main():
        enricher = ProfileEnricher()
        
        if args.batch:
            logger.info(f"Starting batch profile update for up to {args.max} contacts")
            updated = await enricher.batch_update_profiles(args.org_id, args.max)
            logger.info(f"Updated {len(updated)} contact profiles")
        elif args.contact_id:
            logger.info(f"Updating profile for contact ID {args.contact_id}")
            profile = await enricher.generate_profile_for_contact(args.contact_id, args.org_id)
            logger.info(f"Profile updated successfully for contact ID {args.contact_id}")
        else:
            logger.error("Please specify either --contact_id or --batch")
    
    asyncio.run(main())
