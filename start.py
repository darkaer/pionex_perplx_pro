#!/usr/bin/env python3
"""
Startup script for AI Trading Bot
Provides easy setup and execution
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Setup virtual environment and install dependencies"""
    print("🔧 Setting up AI Trading Bot environment...")

    # Create virtual environment
    if not Path("venv").exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)

    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = "venv\Scripts\activate"
        pip_path = "venv\Scripts\pip"
    else:  # Unix/Linux/macOS
        activate_script = "venv/bin/activate"
        pip_path = "venv/bin/pip"

    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)

    print("✅ Environment setup completed!")
    print(f"💡 To manually activate: source {activate_script}")

def check_configuration():
    """Check if configuration files exist"""
    required_files = [
        "config/base_config.yaml",
        "config/testing_config.yaml", 
        "config/production_config.yaml",
        ".env.template"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing configuration files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False

    print("✅ All configuration files present")
    return True

def check_environment_variables():
    """Check if .env file exists and remind about API keys"""
    env_file = Path(".env")

    if not env_file.exists():
        print("⚠️  No .env file found")
        print("💡 Copy .env.template to .env and add your API keys for production mode")
        return False

    print("✅ .env file found")

    # Check for critical environment variables
    with open(".env", "r") as f:
        content = f.read()

    if "your_perplexity_api_key_here" in content:
        print("⚠️  Please update your Perplexity API key in .env file")

    if "your_pionex_api_key_here" in content:
        print("⚠️  Please update your Pionex API credentials in .env file")

    return True

def validate_installation():
    """Validate that all components can be imported"""
    print("🔍 Validating installation...")

    try:
        # Test imports
        sys.path.insert(0, '.')

        from config_manager import ConfigManager
        from perplexity_api import PerplexityAPI
        from websocket_handler import PionexWebSocketHandler
        from risk_manager import RiskManager

        print("✅ All modules import successfully")

        # Test configuration loading
        config_manager = ConfigManager()
        config = config_manager.load_config("testing")
        print("✅ Configuration loads successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def run_bot(mode="testing", validate_only=False):
    """Run the trading bot"""
    print(f"🚀 Starting AI Trading Bot in {mode} mode...")

    if validate_only:
        print("🔍 Validation mode - configuration only")

    # Import and run main
    try:
        import asyncio
        from main import main

        # Set command line arguments
        sys.argv = ["main.py", "--mode", mode]
        if validate_only:
            sys.argv.append("--validate-only")

        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
        return False

    return True

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description='AI Trading Bot Startup Script')
    parser.add_argument('action', nargs='?', default='run',
                       choices=['setup', 'validate', 'run', 'test'],
                       help='Action to perform')
    parser.add_argument('--mode', choices=['testing', 'production'],
                       default='testing', help='Trading mode')

    args = parser.parse_args()

    print("🤖 AI Trading Bot Startup Script")
    print("=" * 50)

    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("config/backups").mkdir(parents=True, exist_ok=True)

    if args.action == 'setup':
        setup_environment()
        check_configuration()
        check_environment_variables()

    elif args.action == 'validate':
        if not check_configuration():
            sys.exit(1)
        check_environment_variables()
        if not validate_installation():
            sys.exit(1)
        print("✅ All validations passed!")

    elif args.action == 'test':
        print("🧪 Running in test/validation mode...")
        if not validate_installation():
            sys.exit(1)
        run_bot(mode="testing", validate_only=True)

    elif args.action == 'run':
        print(f"🚀 Running bot in {args.mode} mode...")
        if not validate_installation():
            sys.exit(1)

        if args.mode == "production":
            confirm = input("⚠️  Running in PRODUCTION mode with real money. Continue? (yes/no): ")
            if confirm.lower() != 'yes':
                print("👍 Cancelled for safety")
                sys.exit(0)

        run_bot(mode=args.mode)

    print("\n👋 Startup script completed")

if __name__ == "__main__":
    main()
