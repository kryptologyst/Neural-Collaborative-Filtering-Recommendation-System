#!/usr/bin/env python3
"""Simple script to run the Neural Collaborative Filtering demo."""

import subprocess
import sys
from pathlib import Path


def main():
    """Main function to run the demo."""
    print("🎯 Neural Collaborative Filtering Demo")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src/demo.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Generate sample data if it doesn't exist
    data_dir = Path("data/raw")
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        print("📊 Generating sample data...")
        subprocess.run([sys.executable, "scripts/utils.py", "generate_data"])
    
    # Run the demo
    print("🚀 Starting Streamlit demo...")
    print("📱 Open your browser to http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/demo.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
