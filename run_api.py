#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start ArtVLM API Server

Simple script to start the ArtVLM API server.
"""

import subprocess
import sys
import os

def main():
    """Start the ArtVLM API server."""
    print("🚀 Starting ArtVLM API Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the API server
        subprocess.run([sys.executable, "artvlm_api.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
