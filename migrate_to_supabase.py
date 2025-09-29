#!/usr/bin/env python3
"""
Migration script for new Supabase instance
Run this after setting up your .env file with the new Supabase credentials
"""

import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_connection():
    """Test database connection"""
    try:
        from app.db.database import engine
        from sqlalchemy import text
        
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Database connection successful!")
            print(f"PostgreSQL version: {version}")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def run_migrations():
    """Run Alembic migrations"""
    try:
        import subprocess
        
        print("Running database migrations...")
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=backend_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Migrations completed successfully!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Migration failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return False

def verify_tables():
    """Verify that all tables were created"""
    try:
        from app.db.database import engine
        from sqlalchemy import text
        
        expected_tables = ['users', 'documents', 'share_access']
        
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))
            
            existing_tables = [row[0] for row in result.fetchall()]
            print(f"Existing tables: {existing_tables}")
            
            missing_tables = set(expected_tables) - set(existing_tables)
            if missing_tables:
                print(f"❌ Missing tables: {missing_tables}")
                return False
            else:
                print("✅ All required tables exist!")
                return True
                
    except Exception as e:
        print(f"❌ Error verifying tables: {e}")
        return False

def main():
    """Main migration process"""
    print("🚀 Starting Supabase migration...")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = backend_dir / ".env"
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file in the backend directory with your Supabase credentials.")
        print("See SUPABASE_MIGRATION_GUIDE.md for the template.")
        return False
    
    # Test connection
    print("\n1. Testing database connection...")
    if not test_connection():
        return False
    
    # Run migrations
    print("\n2. Running database migrations...")
    if not run_migrations():
        return False
    
    # Verify tables
    print("\n3. Verifying table creation...")
    if not verify_tables():
        return False
    
    print("\n🎉 Migration completed successfully!")
    print("Your new Supabase database is ready to use.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
