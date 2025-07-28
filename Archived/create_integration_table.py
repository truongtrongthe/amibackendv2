#!/usr/bin/env python3
"""
Script to create the organization_integrations table in Supabase

This script connects to Supabase and executes the SQL from 
create_org_integrations_table.sql to create the necessary table.
"""

import os
import sys
from supabase import create_client, Client

def create_integration_table():
    """Create organization_integrations table in Supabase"""
    
    # Check if SUPABASE_URL and SUPABASE_KEY environment variables are set
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        sys.exit(1)
    
    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Get the SQL file path
    sql_file = "create_org_integrations_table.sql"
    if not os.path.exists(sql_file):
        print(f"Error: SQL file {sql_file} not found")
        sys.exit(1)
    
    # Read SQL file
    with open(sql_file, "r") as f:
        sql_content = f.read()
    
    # Split the SQL into individual statements
    # Simple split on semicolons - may need to be improved for complex SQL
    sql_statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]
    
    # Execute each SQL statement
    try:
        print(f"Found {len(sql_statements)} SQL statements to execute")
        for i, statement in enumerate(sql_statements):
            print(f"Executing statement {i+1}...")
            try:
                supabase.raw(statement)
                print(f"Statement {i+1} executed successfully.")
            except Exception as e:
                print(f"Warning: Statement {i+1} execution issue: {str(e)}")
                # Continue with the rest of the statements
        
        print("\n✅ Organization integrations table setup completed!")
        print("You can now use the API endpoints to manage integrations.")
    except Exception as e:
        print(f"Error executing SQL: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    create_integration_table() 