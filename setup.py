#!/usr/bin/env python3
"""Setup script for Neural Collaborative Filtering project."""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    
    requirements = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "loguru>=0.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "streamlit>=1.25.0"
    ]
    
    for req in requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], 
                         check=True, capture_output=True)
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            return False
    
    return True


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "assets",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}")
    
    return True


def generate_sample_data():
    """Generate sample data for testing."""
    print("📊 Generating sample data...")
    
    try:
        subprocess.run([sys.executable, "scripts/utils.py", "generate_data"], 
                      check=True, capture_output=True)
        print("✅ Sample data generated")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate sample data: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Neural Collaborative Filtering Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed during directory creation")
        sys.exit(1)
    
    # Generate sample data
    if not generate_sample_data():
        print("⚠️  Sample data generation failed, but setup can continue")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run tests: python3 test_system.py")
    print("2. Start demo: python3 run_demo.py")
    print("3. Train model: python3 src/train.py")


if __name__ == "__main__":
    main()
