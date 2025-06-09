import asyncio
from main import AITradingBot
import aiohttp

async def fetch_perp_symbols():
    url = "https://api.pionex.com/api/v1/common/symbols?type=PERP"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            if data.get("result"):
                symbols = [s["symbol"] for s in data["data"]["symbols"]]
                print("\nAvailable PERP symbols:")
                for s in symbols:
                    print(f"  {s}")
                return set(symbols)
            else:
                print("Failed to fetch PERP symbols.")
                return set()

def validate_perp_pairs(config_pairs, available_perp_symbols):
    print("\nValidating trading pairs in config...")
    all_valid = True
    for pair in config_pairs:
        if pair not in available_perp_symbols:
            print(f"  [INVALID] {pair} is not a valid PERP symbol!")
            all_valid = False
        else:
            print(f"  [OK] {pair}")
    return all_valid

async def test_bot_features():
    print("\n=== AI Trading Bot Feature Test (PERP Market) ===\n")
    # Fetch available PERP symbols
    available_perp_symbols = await fetch_perp_symbols()

    bot = AITradingBot(mode="testing")
    await bot.initialize()
    config_pairs = bot.config.trading.trading_pairs
    # Validate trading pairs
    if not validate_perp_pairs(config_pairs, available_perp_symbols):
        print("\nERROR: One or more trading pairs are not valid PERP symbols. Please update your config.")
        return

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
        # Use a valid PERP symbol and small size for dry run
        symbol = next(iter(available_perp_symbols)) if available_perp_symbols else "BTC_USDT_PERP"
        order = await bot._place_pionex_order(symbol, "BUY", "LIMIT", size=0.001, price=100.0)
        print(f"    Simulated order placement: {order}")
    except Exception as e:
        print(f"    Order placement: FAILED ({e})")

    # Test open orders fetch (simulated)
    print("[4] Testing open orders fetch...")
    try:
        open_orders = await bot._get_pionex_open_orders(symbol)
        print(f"    Simulated open orders: {open_orders}")
    except Exception as e:
        print(f"    Open orders fetch: FAILED ({e})")

    # Test order cancellation (simulated)
    print("[5] Testing order cancellation...")
    try:
        # Use a dummy order_id for dry run
        result = await bot._cancel_pionex_order(symbol, "dummy_order_id")
        print(f"    Simulated order cancellation: {result}")
    except Exception as e:
        print(f"    Order cancellation: FAILED ({e})")

    print("\n=== Feature Test Complete ===\n")

if __name__ == "__main__":
    asyncio.run(test_bot_features()) 