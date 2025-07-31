# Database Migrations - Org Agent Blueprint System

## Overview

This directory contains database migrations to transform the org_agent system from basic configuration to a comprehensive blueprint-based architecture with compilation support.

## Migration Order

**IMPORTANT:** Run these migrations in the exact order specified:

### 1. Drop Old Schema
```bash
psql -d your_database -f migration_drop_old_org_agents.sql
```
- Safely removes existing org_agents table and related objects
- Creates backup-friendly structure

### 2. Create New Schema  
```bash
psql -d your_database -f migration_create_new_org_agents_schema.sql
```
- Creates new `org_agents` table (lightweight management)
- Creates new `agent_blueprints` table (versioned definitions)
- Sets up indexes, views, and helper functions
- Establishes proper foreign key relationships

### 3. Add Compilation Support
```bash
psql -d your_database -f migration_add_compiled_prompt.sql
```
- Adds compilation fields to `agent_blueprints` table
- Creates compilation management functions
- Sets up compilation status tracking
- Adds performance indexes for compilation queries

## Architecture Changes

### OLD SYSTEM:
```sql
org_agents:
  - Basic info + embedded JSON configs
  - Single table approach
  - No versioning
  - No compilation concept
```

### NEW SYSTEM:
```sql
org_agents (lightweight):
  - Basic agent management
  - References current blueprint
  
agent_blueprints (versioned):
  - Complete blueprint definitions
  - Version tracking
  - Compilation support
  - Conversation linking
```

## Key Features Added

### 1. **Blueprint Versioning**
- Multiple blueprint versions per agent
- Easy rollback capabilities
- Complete change history

### 2. **Human-Controlled Compilation** 
- Draft → Compiled workflow
- User-triggered compilation
- Permission-based access
- Audit trail

### 3. **Conversation Linking**
- Blueprint creation linked to human-AMI conversations
- Traceability of requirements
- Context preservation

### 4. **Advanced Querying**
- Views for common queries
- Performance-optimized indexes
- Statistical functions

## Post-Migration Steps

1. **Update Application Code**
   - `org_agent.py` has been updated with new blueprint endpoints
   - Import new functions from `orgdb.py`

2. **Test New Endpoints**
   ```bash
   # Create agent with blueprint
   POST /org-agents/with-blueprint
   
   # Compile blueprint
   POST /org-agents/{id}/blueprints/{blueprint_id}/compile
   
   # Get compiled versions
   GET /org-agents/{id}/compiled-blueprints
   ```

3. **Frontend Updates**
   - Add blueprint creation UI
   - Add compilation triggers
   - Update agent management interface

## Rollback Strategy

If rollback is needed:
1. Export any important data from new tables
2. Run `migration_drop_old_org_agents.sql` to clean up
3. Restore from backup
4. Revert `org_agent.py` to backup version

## Support

- **Migration Issues:** Check logs and foreign key constraints
- **Performance:** Ensure indexes are created properly  
- **Data Integrity:** Verify foreign key relationships
- **Compilation:** Test prompt generation with sample blueprints

---

**Note:** These migrations are designed for production safety with proper backups, foreign key constraints, and rollback procedures.