#!/usr/bin/env python3
"""
Streamlit App Runner for ArtVLM API Tester

This script provides an easy way to run the Streamlit application for testing
the ArtVLM API with structured feedback collection.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit application."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Path to the Streamlit app
    app_path = script_dir / "streamlit_app.py"
    
    # Check if the app file exists
    if not app_path.exists():
        print(f"❌ Error: Streamlit app not found at {app_path}")
        print("Please ensure streamlit_app.py is in the same directory as this script.")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version {streamlit.__version__} found")
    except ImportError:
        print("❌ Streamlit not found. Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "streamlit_requirements.txt"])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print("pip install -r streamlit_requirements.txt")
            sys.exit(1)
    
    # Run the Streamlit app
    print("🚀 Starting ArtVLM API Tester...")
    print("📱 The app will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Change to the script directory
        os.chdir(script_dir)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 ArtVLM API Tester stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
