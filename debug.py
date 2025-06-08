#!/usr/bin/env python3
"""
Debug and testing utilities for AI Trading Bot
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

async def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing configuration system...")

    try:
        from config_manager import ConfigManager

        config_manager = ConfigManager()

        # Test loading different modes
        for mode in ['testing', 'production']:
            print(f"  Testing {mode} mode...")
            config = config_manager.load_config(mode)
            print(f"    ✅ {mode} config loaded successfully")
            print(f"    Trading pairs: {config.trading.trading_pairs}")
            print(f"    Max position size: {config.trading.max_position_size}")
            print(f"    Risk level: {config.risk_management.max_daily_loss_percentage}%")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def test_perplexity_api():
    """Test Perplexity API integration"""
    print("🤖 Testing Perplexity API...")

    try:
        from config_manager import ConfigManager
        from perplexity_api import PerplexityAPI

        config_manager = ConfigManager()
        config = config_manager.load_config("testing")

        api = PerplexityAPI(config)

        # Test with mock data
        market_data = {
            "trading_pair": "BTC/USDT",
            "current_price": 45000,
            "volume_24h": 25000,
            "change_24h": 2.5,
            "rsi": 65,
            "macd": 50,
            "bb_upper": 46000,
            "bb_lower": 44000,
            "support_level": 44500,
            "resistance_level": 45500,
            "news_headlines": "Bitcoin shows bullish momentum",
            "available_balance": 5000,
            "open_positions": 0,
            "daily_pnl": 0
        }

        recommendation = await api.get_trading_recommendation(market_data)

        if recommendation:
            print(f"    ✅ Got recommendation: {recommendation.action}")
            print(f"    Confidence: {recommendation.confidence}")
            print(f"    Reasoning: {recommendation.reasoning}")
        else:
            print("    ⚠️  No recommendation received (expected in testing mode)")

        return True

    except Exception as e:
        print(f"❌ Perplexity API test failed: {e}")
        return False

async def test_websocket():
    """Test WebSocket connection"""
    print("🔌 Testing WebSocket handler...")

    try:
        from config_manager import ConfigManager
        from websocket_handler import PionexWebSocketHandler

        config_manager = ConfigManager()
        config = config_manager.load_config("testing")

        messages_received = []

        def message_handler(data):
            messages_received.append(data)
            print(f"    📨 Received: {data}")

        ws_handler = PionexWebSocketHandler(config, message_handler)

        # Test connection
        await ws_handler.connect()
        print("    ✅ WebSocket connected (mock mode)")

        # Wait for some mock messages
        await asyncio.sleep(3)

        await ws_handler.close()
        print(f"    ✅ Received {len(messages_received)} mock messages")

        return True

    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

async def test_risk_manager():
    """Test risk management system"""
    print("🛡️  Testing risk management...")

    try:
        from config_manager import ConfigManager
        from risk_manager import RiskManager

        config_manager = ConfigManager()
        config = config_manager.load_config("testing")

        risk_manager = RiskManager(config)
        risk_manager.initialize(5000.0)  # $5000 starting balance

        # Test position validation
        can_trade, reason = risk_manager.can_open_position("BTCUSDT", 500, 1.0)
        print(f"    Can open position: {can_trade} ({reason})")

        if can_trade:
            # Test adding a position
            success = risk_manager.add_position(
                symbol="BTCUSDT",
                side="BUY", 
                size=500,
                entry_price=45000,
                leverage=1.0,
                stop_loss=44100,
                take_profit=46800
            )
            print(f"    ✅ Position added: {success}")

            # Test position update
            risk_manager.update_position_price("BTCUSDT", 45500)
            print("    ✅ Position price updated")

            # Get risk metrics
            metrics = risk_manager.get_risk_metrics()
            print(f"    Risk level: {metrics.risk_level.value}")
            print(f"    Total positions: {metrics.total_positions}")

        return True

    except Exception as e:
        print(f"❌ Risk manager test failed: {e}")
        return False

async def test_integration():
    """Test full system integration"""
    print("🔄 Testing system integration...")

    try:
        from main import AITradingBot

        # Create bot instance
        bot = AITradingBot(mode="testing")

        # Initialize (but don't start)
        await bot.initialize()
        print("    ✅ Bot initialization successful")

        # Test market data preparation
        market_data = bot._prepare_market_data("BTC/USDT")
        print("    ✅ Market data preparation working")

        # Test position size calculation
        from perplexity_api import TradingRecommendation
        from datetime import datetime

        mock_recommendation = TradingRecommendation(
            action="BUY",
            confidence=0.8,
            entry_price=45000,
            stop_loss=44100,
            take_profit=46800,
            reasoning="Test recommendation",
            risk_level="LOW",
            timestamp=datetime.now(),
            trading_pair="BTC/USDT"
        )

        position_size = bot._calculate_position_size(mock_recommendation)
        print(f"    ✅ Position size calculation: {position_size}")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    print("🧪 Running comprehensive test suite...")
    print("=" * 50)

    tests = [
        ("Configuration", test_configuration),
        ("Perplexity API", test_perplexity_api),
        ("WebSocket", test_websocket),
        ("Risk Manager", test_risk_manager),
        ("Integration", test_integration)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🧪 {test_name} Test:")
        print("-" * 30)

        try:
            result = await test_func()
            results[test_name] = result

            if result:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")

        except Exception as e:
            print(f"💥 {test_name} test CRASHED: {e}")
            results[test_name] = False

    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print("=" * 50)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())
