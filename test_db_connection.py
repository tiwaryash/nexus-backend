#!/usr/bin/env python3
"""
Database connection test script for Supabase PostgreSQL
"""
import os
import sys
import psycopg2
from urllib.parse import urlparse

def test_database_connection():
    # Get the database URL from environment variable
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ ERROR: DATABASE_URL environment variable is not set")
        print("\n📝 To fix this, run:")
        print('export DATABASE_URL="postgresql://postgres:Nexus24%2F7%40%40@db.vokkayimlhtrlwfonvtf.supabase.co:5432/postgres"')
        return False
    
    print(f"🔍 Testing connection to: {database_url}")
    
    # Parse the URL to get components
    try:
        parsed = urlparse(database_url)
        host = parsed.hostname
        port = parsed.port
        database = parsed.path[1:]  # Remove leading slash
        username = parsed.username
        
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Database: {database}")
        print(f"   Username: {username}")
        
    except Exception as e:
        print(f"❌ ERROR: Invalid DATABASE_URL format: {e}")
        return False
    
    # Test basic connectivity
    print("\n🌐 Testing network connectivity...")
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result != 0:
            print(f"❌ ERROR: Cannot reach {host}:{port}")
            print("\n🔧 Possible solutions:")
            print("1. Check if your Supabase project is active (not paused)")
            print("2. Verify the database URL is correct")
            print("3. Check your internet connection")
            print("4. Contact Supabase support if the issue persists")
            return False
        else:
            print(f"✅ Network connectivity to {host}:{port} is working")
    except Exception as e:
        print(f"❌ ERROR: Network test failed: {e}")
        return False
    
    # Test database connection
    print("\n🔐 Testing database authentication...")
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Database connection successful!")
        print(f"   PostgreSQL version: {version[0]}")
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ ERROR: Database connection failed: {e}")
        print("\n🔧 Possible solutions:")
        print("1. Verify your Supabase project is not paused")
        print("2. Check if the password is correct")
        print("3. Ensure the database exists")
        print("4. Get a fresh connection string from Supabase dashboard")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Supabase Database Connection Test")
    print("=" * 40)
    
    success = test_database_connection()
    
    if success:
        print("\n🎉 All tests passed! Your database connection is working.")
    else:
        print("\n💡 Next steps:")
        print("1. Log into your Supabase dashboard")
        print("2. Check if your project is paused (unpause if needed)")
        print("3. Get a fresh database connection string")
        print("4. Update your DATABASE_URL environment variable")
        sys.exit(1)
