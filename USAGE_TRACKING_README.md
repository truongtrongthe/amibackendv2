# Organization Usage Tracking System

This system provides comprehensive usage tracking for organizations in Supabase, allowing you to monitor and analyze message counts and reasoning usage across different time periods.

## Features

- Track message and reasoning usage for organizations
- Aggregate usage data by day, week, month, and year
- Store detailed usage records for audit and analysis
- Row-level security to ensure data isolation between organizations
- Optimized database schema with appropriate indexes
- Helper functions for common operations

## Setup Instructions

1. **Create the database tables**

Run the SQL script `create_organization_usage_tables.sql` in your Supabase SQL editor. This script will:
- Create the `organization_usage` and `usage_detail` tables
- Add proper indexes for performance
- Set up row-level security
- Create helper functions for common operations

2. **Import the Python code**

Use the `usage.py` file in your application to interact with the usage tracking system.

3. **Optional: Test with example data**

Run the `usage_examples.sql` script to insert test data and see example queries.

## Usage

### Python Usage

```python
from usage import OrganizationUsage

# Initialize for a specific organization
org_usage = OrganizationUsage("your-org-id-here")

# Add message usage
org_usage.add_message(1)  # Add 1 message by default
org_usage.add_message(5)  # Add 5 messages

# Add reasoning usage
org_usage.add_reasoning(2)  # Add 2 reasoning units

# Get usage counts for different periods
daily_messages = org_usage.get_message_count('day')
weekly_messages = org_usage.get_message_count('week')
monthly_messages = org_usage.get_message_count('month')
yearly_messages = org_usage.get_message_count('year')

# Get reasoning counts for different periods
daily_reasoning = org_usage.get_reasoning_count('day')
weekly_reasoning = org_usage.get_reasoning_count('week')

# Get usage summary for all types
monthly_summary = org_usage.get_usage_summary('month')
# Returns: {'message': 450, 'reasoning': 320, ...}

# Get detailed usage records
details = org_usage.get_usage_details(
    usage_type='message',
    start_date='2023-01-01',
    end_date='2023-01-31',
    limit=100
)
```

### SQL Queries

Several example SQL queries are provided in the `usage_examples.sql` file:

1. Daily usage by organization and type
2. Weekly usage aggregation
3. Monthly usage comparison
4. Usage comparison across organizations
5. Hourly usage patterns from detailed logs
6. Message to reasoning ratio analysis

## Database Schema

### organization_usage Table
- `id`: UUID primary key
- `org_id`: Organization UUID
- `type`: Usage type (e.g., 'message', 'reasoning')
- `count`: Usage count
- `date`: Date of usage
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### usage_detail Table
- `id`: UUID primary key
- `org_id`: Organization UUID
- `type`: Usage type (e.g., 'message', 'reasoning')
- `count`: Usage count
- `timestamp`: Exact timestamp of usage
- `metadata`: JSONB field for additional data
- `created_at`: Creation timestamp

## Helper Functions

- `increment_organization_usage(org_id, type, count, date)`: Adds usage count, handling conflicts
- `get_organization_usage_summary(org_id, start_date, end_date)`: Returns usage summary for a period

## Security

Row-level security is enabled for both tables to ensure organizations can only access their own data.

## Additional Types

You can track additional usage types beyond messages and reasoning by adding new types to the `type` field and extending the OrganizationUsage class.

## Performance Considerations

- The schema is optimized with appropriate indexes
- For high-volume systems, consider implementing batch inserts
- For analytics, you may want to create materialized views for common queries 