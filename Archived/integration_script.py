import xmlrpc.client
import ssl
import os
from dotenv import load_dotenv
from contact import ContactManager

# Load environment variables
load_dotenv()

# Create an SSL context that doesn't verify certificates
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

# Odoo Connection details
odoo_url = os.getenv("ODOO_URL", "https://dev.opms.tech")
odoo_db = os.getenv("ODOO_DB", "erk")
odoo_username = os.getenv("ODOO_USERNAME", "dung.tt@bigholding.vn")
odoo_password = os.getenv("ODOO_PASSWORD", "Obd@2025")

# Default organization ID for contacts
DEFAULT_ORG_ID = os.getenv("DEFAULT_ORG_ID", "your-organization-id")

# Initialize ContactManager
contact_manager = ContactManager()

def connect_to_odoo():
    """Establish connection to Odoo and authenticate"""
    print("Connecting to Odoo...")
    
    # Connect to the common endpoint (for authentication)
    common = xmlrpc.client.ServerProxy(f"{odoo_url}/xmlrpc/2/common", context=context)
    
    # Authenticate and get the user ID
    uid = common.authenticate(odoo_db, odoo_username, odoo_password, {})
    
    if uid:
        print(f"Authentication successful! User ID: {uid}")
        # Connect to the object endpoint (for model operations)
        models = xmlrpc.client.ServerProxy(f"{odoo_url}/xmlrpc/2/object", context=context)
        return uid, models
    else:
        print("Authentication failed. Check credentials or database.")
        return None, None

def sync_contacts_to_crm(batch_size=50, offset=0, limit=None):
    """
    Sync contacts from Odoo to the CRM system via ContactManager
    
    Args:
        batch_size: Number of contacts to process in each batch
        offset: Starting offset in the Odoo partners list
        limit: Maximum number of contacts to sync (None for all)
    """
    uid, models = connect_to_odoo()
    if not uid or not models:
        return
    
    print(f"\nStarting contact sync from Odoo (offset: {offset})")
    
    try:
        # Count total customer contacts in Odoo
        customer_count = models.execute_kw(
            odoo_db, uid, odoo_password,
            'res.partner', 'search_count',
            [[('customer_rank', '>', 0)]]  # Filter for customers only
        )
        
        print(f"Found {customer_count} total customer contacts in Odoo")
        
        # Calculate total contacts to process
        total_to_process = min(customer_count, limit) if limit else customer_count
        print(f"Will process {total_to_process} contacts")
        
        # Process in batches
        processed_count = 0
        current_offset = offset
        
        while processed_count < total_to_process:
            batch_limit = min(batch_size, total_to_process - processed_count)
            
            print(f"\nFetching batch: offset {current_offset}, limit {batch_limit}")
            
            # Search for customer contacts
            partner_ids = models.execute_kw(
                odoo_db, uid, odoo_password,
                'res.partner', 'search',
                [[('customer_rank', '>', 0)]],  # Filter for customers only
                {'limit': batch_limit, 'offset': current_offset}
            )
            
            if not partner_ids:
                print("No more contacts found.")
                break
                
            # Read contact details
            partners = models.execute_kw(
                odoo_db, uid, odoo_password,
                'res.partner', 'read',
                [partner_ids],
                {'fields': [
                    'name', 'email', 'phone', 'mobile',
                    'street', 'city', 'country_id', 'create_date',
                    'function', 'title', 'website', 'comment',
                    'sale_order_ids'
                ]}
            )
            
            print(f"Processing {len(partners)} contacts")
            
            # Process each contact
            for partner in partners:
                try:
                    # Parse name into first/last
                    name_parts = partner['name'].split(' ', 1)
                    first_name = name_parts[0]
                    last_name = name_parts[1] if len(name_parts) > 1 else ""
                    
                    # Check if contact already exists in CRM by email
                    existing_contact = None
                    if partner['email']:
                        # You'll need to implement this method in ContactManager
                        # existing_contact = contact_manager.get_contact_by_email(partner['email'], DEFAULT_ORG_ID)
                        pass
                    
                    # If contact doesn't exist, create it
                    if not existing_contact:
                        print(f"Creating new contact: {partner['name']}")
                        contact = contact_manager.create_contact(
                            organization_id=DEFAULT_ORG_ID,
                            type="customer",
                            first_name=first_name,
                            last_name=last_name,
                            email=partner['email'],
                            phone=partner['phone'] or partner['mobile']
                        )
                        
                        if contact:
                            contact_id = contact['id']
                            
                            # Create profile with additional info
                            profile_data = {
                                "general_info": f"""
                                    Function: {partner.get('function', 'N/A')}
                                    Address: {partner.get('street', 'N/A')}, {partner.get('city', 'N/A')}
                                    Website: {partner.get('website', 'N/A')}
                                    Customer since: {partner.get('create_date', 'Unknown')}
                                    Notes: {partner.get('comment', '')}
                                """.strip(),
                                "profile_summary": f"Contact imported from Odoo: {partner['name']}"
                            }
                            
                            contact_manager.create_or_update_contact_profile(
                                contact_id=contact_id,
                                **profile_data
                            )
                            
                            # If we have order history, add it to the profile
                            if partner.get('sale_order_ids') and len(partner['sale_order_ids']) > 0:
                                # Get order details
                                orders = models.execute_kw(
                                    odoo_db, uid, odoo_password,
                                    'sale.order', 'read',
                                    [partner['sale_order_ids'][:5]],  # Get details for up to 5 orders
                                    {'fields': ['name', 'date_order', 'amount_total', 'state']}
                                )
                                
                                if orders:
                                    order_info = "Purchase History:\n"
                                    for order in orders:
                                        order_info += f"- Order {order['name']} ({order['date_order']}): ${order['amount_total']} ({order['state']})\n"
                                    
                                    # Update profile with order info
                                    profile_data["general_info"] += f"\n\n{order_info}"
                                    contact_manager.update_contact_profile(
                                        contact_id=contact_id,
                                        general_info=profile_data["general_info"]
                                    )
                    else:
                        # Update existing contact
                        print(f"Updating existing contact: {partner['name']}")
                        # This would need to be implemented
                        
                except Exception as e:
                    print(f"Error processing contact {partner['name']}: {str(e)}")
            
            # Update counters
            processed_count += len(partners)
            current_offset += batch_limit
            print(f"Processed {processed_count}/{total_to_process} contacts")
            
        print(f"\nContact sync completed. Processed {processed_count} contacts.")
        
    except Exception as e:
        print(f"Error during contact sync: {str(e)}")

if __name__ == "__main__":
    # Example usage
    print("Contact Sync Utility")
    print("====================")
    print("1. Sync all contacts")
    print("2. Sync specific range")
    
    choice = input("Enter option (1-2): ")
    
    if choice == '1':
        sync_contacts_to_crm()
    elif choice == '2':
        try:
            offset = int(input("Enter starting offset: "))
            limit = int(input("Enter maximum number of contacts to sync: "))
            sync_contacts_to_crm(offset=offset, limit=limit)
        except ValueError:
            print("Invalid input. Please enter numbers for offset and limit.")
    else:
        print("Invalid option selected.") 