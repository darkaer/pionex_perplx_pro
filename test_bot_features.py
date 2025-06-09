import asyncio
from main import AITradingBot

async def test_bot_features():
    print("\n=== AI Trading Bot Feature Test (Dry Run) ===\n")
    bot = AITradingBot(mode="testing")
    await bot.initialize()

    # Test WebSocket connection
    print("[1] Testing WebSocket connection...")
    try:
        await bot.websocket_handler.connect()
        print("    WebSocket connection: SUCCESS")
    except Exception as e:
        print(f"    WebSocket connection: FAILED ({e})")

    # Test balance fetch (simulated)
    print("[2] Testing balance fetch...")
    try:
        balance = await bot._fetch_pionex_balance()
        print(f"    Simulated balance: {balance}")
    except Exception as e:
        print(f"    Balance fetch: FAILED ({e})")

    # Test order placement (simulated)
    print("[3] Testing simulated order placement...")
    try:
        # Use a dummy symbol and small size for dry run
        symbol = "BTC_USDT"
        order = await bot._place_pionex_order(symbol, "BUY", "LIMIT", size=0.001, price=100.0)
        print(f"    Simulated order placement: {order}")
    except Exception as e:
        print(f"    Order placement: FAILED ({e})")

    # Test open orders fetch (simulated)
    print("[4] Testing open orders fetch...")
    try:
        open_orders = await bot._get_pionex_open_orders("BTC_USDT")
        print(f"    Simulated open orders: {open_orders}")
    except Exception as e:
        print(f"    Open orders fetch: FAILED ({e})")

    # Test order cancellation (simulated)
    print("[5] Testing order cancellation...")
    try:
        # Use a dummy order_id for dry run
        result = await bot._cancel_pionex_order("BTC_USDT", "dummy_order_id")
        print(f"    Simulated order cancellation: {result}")
    except Exception as e:
        print(f"    Order cancellation: FAILED ({e})")

    print("\n=== Feature Test Complete ===\n")

if __name__ == "__main__":
    asyncio.run(test_bot_features()) 