#!/usr/bin/env python3
"""
Launcher script for the Auto-Spend Guard Streamlit app
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app"""
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} is installed")
    except ImportError:
        print("❌ Streamlit is not installed. Installing now...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0"], check=True)
        print("✅ Streamlit installed successfully")
    
    # Check if plotly is installed
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__} is installed")
    except ImportError:
        print("❌ Plotly is not installed. Installing now...")
        subprocess.run([sys.executable, "-m", "pip", "install", "plotly>=5.15.0"], check=True)
        print("✅ Plotly installed successfully")
    
    # Launch the Streamlit app
    print("🚀 Launching Auto-Spend Guard Streamlit app...")
    print("📱 The app will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
