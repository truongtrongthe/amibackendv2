# SQL Scripts Directory

This directory contains all SQL scripts used for database schema creation, modifications, and data management.

## Database Schema Scripts

### **Core Tables**
- `users_table.sql` - User management table schema
- `email_verification_table.sql` - Email verification system table
- `chat.sql` - Chat system database schema

### **Feature Additions**
- `add_language_column.sql` - Language column addition script

## Script Categories

### **🔧 Table Creation Scripts**
Scripts that create new database tables and initial schema:
- `users_table.sql` - Creates users table with authentication fields
- `email_verification_table.sql` - Creates email verification tracking table
- `chat.sql` - Creates chat-related tables and indexes

### **📝 Schema Modifications**
Scripts that modify existing database structure:
- `add_language_column.sql` - Adds language support to existing tables

## Usage Guidelines

### **Running Scripts**
```bash
# For PostgreSQL
psql -d your_database -f script_name.sql

# For MySQL
mysql -u username -p database_name < script_name.sql
```

### **Order of Execution**
For fresh database setup, run in this order:
1. `users_table.sql` - Core user management
2. `email_verification_table.sql` - User verification features
3. `chat.sql` - Chat functionality
4. `add_language_column.sql` - Additional features

### **Migration Safety**
- ⚠️ **Always backup your database** before running modification scripts
- 🔍 **Review scripts** before execution in production
- 🧪 **Test in development** environment first
- 📋 **Document changes** in your migration log

## Related Migration Scripts

For org_agent system migrations, see:
- `migrations/` directory - Contains org_agent blueprint system migrations
- Migration scripts are separate from these core database scripts

## Organization Benefits

✅ **Centralized SQL Management** - All database scripts in one location
✅ **Easy Maintenance** - Clear organization and documentation
✅ **Version Control** - Better tracking of database changes
✅ **Professional Structure** - Clean separation from application code

## Support

For database-related issues:
1. Check script comments for specific requirements
2. Verify database compatibility (PostgreSQL, MySQL, etc.)
3. Ensure proper permissions for database modifications
4. Review foreign key dependencies before running scripts