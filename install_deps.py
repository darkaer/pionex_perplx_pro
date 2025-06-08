#!/usr/bin/env python3
"""
Simple installation script for AI Trading Bot dependencies
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")

    try:
        # Install the required packages
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "aiohttp>=3.8.0",
            "websockets>=10.0", 
            "pyyaml>=6.0",
            "python-dotenv>=1.0.0",
            "requests>=2.28.0",
            "cryptography>=3.4.8",
            "pandas>=1.5.0",
            "numpy>=1.24.0"
        ])

        print("✅ Dependencies installed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = install_dependencies()
    if success:
        print("\n✅ Ready to run the bot!")
        print("💡 Next steps:")
        print("   python debug.py  # Test the system")
        print("   python start.py run  # Start in testing mode")
    else:
        print("\n❌ Installation failed. Please check the errors above.")
        sys.exit(1)
