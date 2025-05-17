# Usage Tracking Troubleshooting Guide

## Issues Identified

The main issue with the usage tracking system was that records were being logged in the code but not appearing in the database. 

Key problems:
1. **Unnecessary RLS Policies**: The original SQL script created restrictive Row Level Security policies.
2. **Overly Complex Implementation**: The code had excessive error handling and logging that made it hard to diagnose issues.

## Solution: Simplified Approach

We've simplified the implementation to match the approach used in other successful parts of the codebase like `brainlog.py` and `aia.py`:

1. **Simplified Tables**: Created tables without RLS policies using `create_simple_usage_tables.sql`
2. **Consistent Code Style**: Updated `usage.py` to follow the same code patterns as other files
3. **Direct Database Access**: Removed attempts to manipulate JWT tokens and service roles

## How To Use

1. Run the `create_simple_usage_tables.sql` script in your Supabase SQL editor

2. If you already have tables with restrictive RLS, run this SQL instead:
   ```sql
   -- Just disable RLS on the tables
   ALTER TABLE public.organization_usage DISABLE ROW LEVEL SECURITY;
   ALTER TABLE public.usage_detail DISABLE ROW LEVEL SECURITY;
   ```

3. Use the `OrganizationUsage` class normally:
   ```python
   from usage import OrganizationUsage
   
   # Track usage
   org_usage = OrganizationUsage("your-org-id")
   org_usage.add_message(5)  # Track 5 messages
   org_usage.add_reasoning(1)  # Track 1 reasoning operation
   ```

## Further Diagnostics

If you're still experiencing issues:
1. Check that your Supabase URL and API key in environment variables are correct
2. Verify the tables exist in your Supabase database
3. Run a simple query directly in the Supabase SQL editor to test permissions
4. Check the application logs for specific error messages 