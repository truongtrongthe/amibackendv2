import xmlrpc.client
import ssl

# Create an SSL context that doesn't verify certificates
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

# Connection details
url = "https://dev.opms.tech"
db = "erk"             # Replace with your database name
username = "dung.tt@bigholding.vn" # Replace with your Odoo username
password = "Obd@2025"    # Replace with your Odoo password

# Step 1: Connect to the common endpoint (for authentication)
common = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common", context=context)

# Step 2: Authenticate and get the user ID
uid = common.authenticate(db, username, password, {})

if uid:
    print(f"Authentication successful! User ID: {uid}")
else:
    print("Authentication failed. Check credentials or database.")
    exit()

# Step 3: Connect to the object endpoint (for model operations)
models = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/object", context=context)

# Function to fetch contacts with their order history
def fetch_contacts_with_orders(limit=25, offset=5000):
    try:
        print(f"\nFetching contacts with offset {offset} and limit {limit}...")
        
        # Search for customers (partners with customer_rank > 0)
        partner_ids = models.execute_kw(
            db, uid, password,
            'res.partner', 'search',
            [[('customer_rank', '>', 0)]],  # Filter for customers only
            {'limit': limit, 'offset': offset}  # Use offset and limit
        )
        
        total_contacts = len(partner_ids)
        print(f"Found {total_contacts} customer contacts (offset {offset}-{offset+limit})")
        
        # Read contact details including sale_order_ids
        partners = models.execute_kw(
            db, uid, password,
            'res.partner', 'read',
            [partner_ids],
            {'fields': [
                'name', 'email', 'phone', 'mobile', 
                'create_date', 'city', 'country_id',
                'sale_order_ids'  # Field linking to sales orders
            ]}
        )
        
        # Process each contact and their orders
        for i, partner in enumerate(partners):
            print(f"\n{'-'*80}")
            print(f"CONTACT {i+1}/{total_contacts}: {partner['name']}")
            print(f"Email: {partner['email'] or 'N/A'}")
            print(f"Phone: {partner['phone'] or 'N/A'} / Mobile: {partner['mobile'] or 'N/A'}")
            created = partner['create_date'] if partner['create_date'] else 'Unknown'
            print(f"Created: {created}")
            
            city = partner['city'] if partner['city'] else 'N/A'
            country = partner['country_id'][1] if partner['country_id'] else 'N/A'
            print(f"Location: {city}, {country}")
            
            # Get orders for this contact
            sale_order_ids = partner.get('sale_order_ids', [])
            
            if not sale_order_ids:
                print("\nNo purchase history found for this contact.")
                continue
            
            print(f"\nPURCHASE HISTORY: {len(sale_order_ids)} orders found")
            
            # Get order details
            orders = models.execute_kw(
                db, uid, password,
                'sale.order', 'read',
                [sale_order_ids],
                {'fields': [
                    'name', 'date_order', 'amount_total', 
                    'state', 'invoice_status', 'team_id',
                    'user_id', 'partner_invoice_id', 'partner_shipping_id',
                    'order_line'  # Include order lines
                ]}
            )
            
            # Sort orders by date (newest first)
            orders.sort(key=lambda x: x.get('date_order', ''), reverse=True)
            
            # Display order information
            for order_idx, order in enumerate(orders):
                print(f"\nOrder {order_idx+1}: {order['name']}")
                print(f"Date: {order['date_order']}")
                print(f"Total: {order['amount_total']}")
                print(f"Status: {order['state']}")
                print(f"Invoice Status: {order['invoice_status']}")
                
                sales_team = order['team_id'][1] if order['team_id'] else 'N/A'
                salesperson = order['user_id'][1] if order['user_id'] else 'N/A'
                print(f"Sales Team: {sales_team} / Salesperson: {salesperson}")
                
                # Get order line details
                if order['order_line']:
                    print("\nOrder Lines:")
                    order_lines = models.execute_kw(
                        db, uid, password,
                        'sale.order.line', 'read',
                        [order['order_line']],
                        {'fields': [
                            'product_id', 'product_uom_qty', 
                            'price_unit', 'price_subtotal'
                        ]}
                    )
                    
                    for line_idx, line in enumerate(order_lines):
                        product = line['product_id'][1] if line['product_id'] else 'N/A'
                        quantity = line['product_uom_qty']
                        unit_price = line['price_unit']
                        subtotal = line['price_subtotal']
                        print(f"  {line_idx+1}. {product} - Qty: {quantity} - Unit Price: {unit_price} - Subtotal: {subtotal}")
        
        return True
        
    except Exception as e:
        print(f"Error fetching contacts with orders: {e}")
        return False

# Run the contacts with orders function automatically
if __name__ == "__main__":
    fetch_contacts_with_orders(25, 5000)  # Fetch contacts with offset 5000 and limit 25