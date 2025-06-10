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
                symbol_info_map = {s["symbol"]: s for s in data["data"]["symbols"]}
                print("\nAvailable PERP symbols:")
                for s in symbols:
                    print(f"  {s}")
                return set(symbols), symbol_info_map
            else:
                print("Failed to fetch PERP symbols.")
                return set(), {}

def validate_perp_pairs(config_pairs, available_perp_symbols):
    print("\nValidating trading pairs in config...")
    all_valid = True
    for pair in config_pairs:
        # Accept both BTC_USDT and BTC/USDT as valid if present in available symbols
        normalized_pair = pair.replace("/", "_")
        if normalized_pair not in available_perp_symbols:
            print(f"  [INVALID] {pair} is not a valid PERP symbol!")
            all_valid = False
        else:
            print(f"  [OK] {pair}")
    return all_valid

async def test_bot_features():
    print("\n=== AI Trading Bot Feature Test (PERP Market) ===\n")
    # Fetch available PERP symbols and their info
    available_perp_symbols, symbol_info_map = await fetch_perp_symbols()

    # Set mode to 'production' for real API testing. Change to 'testing' for simulation.
    mode = "production"
    bot = AITradingBot(mode=mode)
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
        # Use a valid PERP symbol and correct min size for dry run
        symbol = next(iter(available_perp_symbols)) if available_perp_symbols else "BTC_USDT"
        api_symbol = symbol.replace('_PERP', '')
        min_size = float(symbol_info_map[symbol]["minOrderSize"]) if symbol in symbol_info_map else 1.0
        order = await bot._place_pionex_order(api_symbol, "BUY", "LIMIT", size=min_size, price=100.0)
        print(f"    Simulated order placement: {order}")
    except Exception as e:
        print(f"    Order placement: FAILED ({e})")

    # Test open orders fetch (simulated)
    print("[4] Testing open orders fetch...")
    try:
        open_orders = await bot._get_pionex_open_orders(api_symbol)
        print(f"    Simulated open orders: {open_orders}")
    except Exception as e:
        print(f"    Open orders fetch: FAILED ({e})")

    # Test order cancellation (simulated)
    print("[5] Testing order cancellation...")
    try:
        # Use a dummy order_id for dry run
        result = await bot._cancel_pionex_order(api_symbol, "dummy_order_id")
        print(f"    Simulated order cancellation: {result}")
    except Exception as e:
        print(f"    Order cancellation: FAILED ({e})")

    print("\n=== Feature Test Complete ===\n")

async def test_print_valid_perp_symbols():
    print("\n=== Test: fetch_valid_perp_symbols ===")
    bot = AITradingBot(mode="production")
    await bot.initialize()
    symbols = await bot.fetch_valid_perp_symbols()
    print(f"Valid PERP symbols ({len(symbols)}): {sorted(symbols)}")

async def test_validate_leverage_and_margin():
    print("\n=== Test: _validate_leverage_and_margin (try both symbol formats) ===")
    bot = AITradingBot(mode="production")
    await bot.initialize()
    symbols = ["BTC_USDT_PERP", "BTC_USDT"]
    size = 0.001  # Safer small size for BTC
    price = 70000  # Example price for BTC, adjust as needed
    leverage = 20
    for symbol in symbols:
        print(f"\nTesting symbol: {symbol}")
        is_valid = await bot.is_valid_perp_symbol(symbol) if symbol.endswith('_PERP') else True
        if not is_valid:
            print(f"Symbol {symbol} is not a valid PERP symbol. Skipping.")
            continue
        valid, reason = await bot._validate_leverage_and_margin(symbol, size, price, leverage)
        print(f"Validation result for {symbol}: {valid}, Reason: {reason}")
        if valid:
            break
    else:
        print("Neither symbol format worked for leverage/margin validation.")

async def test_monitor_liquidation_and_auto_reduce():
    print("\n=== Test: _monitor_liquidation_and_auto_reduce (try both symbol formats) ===")
    bot = AITradingBot(mode="production")
    await bot.initialize()
    symbols = ["BTC_USDT_PERP", "BTC_USDT"]
    for symbol in symbols:
        print(f"\nTesting symbol: {symbol}")
        is_valid = await bot.is_valid_perp_symbol(symbol) if symbol.endswith('_PERP') else True
        if not is_valid:
            print(f"Symbol {symbol} is not a valid PERP symbol. Skipping.")
            continue
        task = asyncio.create_task(bot._monitor_liquidation_and_auto_reduce(symbol, poll_interval=10))
        print("Monitor started. Waiting 30 seconds...")
        await asyncio.sleep(30)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print("Monitor task cancelled.")
        break
    else:
        print("Neither symbol format worked for liquidation monitor.")

async def test_position_size_by_balance():
    print("\n=== Test: calculate_position_size_by_balance ===")
    bot = AITradingBot(mode="production")  # or "testing"
    await bot.initialize()
    symbol = "BTC_USDT"
    price = 70000  # Example price, adjust as needed
    percentage = 0.05  # 5% of balance
    size = await bot.calculate_position_size_by_balance(symbol, price, percentage)
    print(f"Calculated position size for {symbol} at {price} USDT using {percentage*100}% of balance: {size}")

if __name__ == "__main__":
    import sys
    import asyncio
    tests = {
        "perp_symbols": test_print_valid_perp_symbols,
        "leverage": test_validate_leverage_and_margin,
        "liquidation": test_monitor_liquidation_and_auto_reduce,
        "position_size": test_position_size_by_balance,
        "all": None
    }
    if len(sys.argv) > 1 and sys.argv[1] in tests:
        if sys.argv[1] == "all":
            async def run_all():
                await test_print_valid_perp_symbols()
                await test_validate_leverage_and_margin()
                await test_monitor_liquidation_and_auto_reduce()
                await test_position_size_by_balance()
            asyncio.run(run_all())
        else:
            asyncio.run(tests[sys.argv[1]]())
    else:
        print("Usage: python test_bot_features.py [perp_symbols|leverage|liquidation|position_size|all]") 