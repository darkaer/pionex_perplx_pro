"""
Main AI Trading Bot Application
Integrates Perplexity Labs AI with Pionex trading execution
"""

import asyncio
import logging
import signal
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import os
import pandas as pd
import ta
import aiohttp
import hmac
import hashlib
import time
import uuid
import math
from pionex_python.restful.Orders import Orders
try:
    from dotenv import load_dotenv
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Import our modules
from config_manager import ConfigManager, BotConfig
from perplexity_api import PerplexityAPI, TradingRecommendation
from websocket_handler import PionexWebSocketHandler
from risk_manager import RiskManager, RiskMetrics

class AITradingBot:
    """Main AI Trading Bot orchestrating all components"""

    def __init__(self, config_path: str = "config", mode: str = "testing"):
        self.config_path = config_path
        self.mode = mode
        self.config: Optional[BotConfig] = None
        self.logger = self._setup_logging()

        # Core components
        self.config_manager = ConfigManager(config_path)
        self.perplexity_api: Optional[PerplexityAPI] = None
        self.websocket_handler: Optional[PionexWebSocketHandler] = None
        self.risk_manager: Optional[RiskManager] = None

        # Runtime state
        self.running = False
        self.last_analysis_time = datetime.now()
        self.market_data: Dict[str, Any] = {}

        # Performance tracking
        self.start_time = datetime.now()
        self.trade_count = 0
        self.total_pnl = 0.0

        self.tasks = []  # Store references to main async tasks

        # Initialize Pionex Orders client
        api_key = os.getenv('TRADING_BOT_PIONEX_API_KEY')
        api_secret = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')
        self.orders_client = None
        if api_key and api_secret:
            self.orders_client = Orders(api_key, api_secret)
        else:
            self.logger.warning("Pionex API credentials not set. Real order placement will fail.")

        self.ohlcv_update_minutes = int(os.getenv("TRADING_BOT_OHLCV_UPDATE_MINUTES", 30))
        self._ohlcv_update_task = None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/trading_bot.log', mode='a')
            ]
        )

        # Create logs directory
        Path('logs').mkdir(exist_ok=True)

        return logging.getLogger(__name__)

    async def initialize(self):
        """Initialize all bot components"""
        try:
            self.logger.info(f"Initializing AI Trading Bot in {self.mode} mode...")

            # Load configuration
            self.config = self.config_manager.load_config(self.mode)

            # Set analysis interval from config
            self.analysis_interval = getattr(self.config.ai_prompts, 'analysis_interval', 300)
            self.logger.info(f"Analysis interval set to {self.analysis_interval} seconds.")

            # Initialize components
            self.perplexity_api = PerplexityAPI(self.config)
            self.websocket_handler = PionexWebSocketHandler(mode=self.mode)
            self.risk_manager = RiskManager(self.config)

            # Initialize risk manager with starting balance
            env_balance = os.getenv("TRADING_BOT_INITIAL_BALANCE")
            if env_balance is not None:
                try:
                    starting_balance = float(env_balance)
                    self.logger.info(f"Using starting balance from environment: ${starting_balance}")
                except ValueError:
                    self.logger.warning(f"Invalid TRADING_BOT_INITIAL_BALANCE value: {env_balance}, using default.")
                    starting_balance = 5000.0 if self.mode == "testing" else 10000.0
            elif hasattr(self.config, "testing") and hasattr(self.config.testing, "initial_balance"):
                starting_balance = self.config.testing.initial_balance
                self.logger.info(f"Using starting balance from config: ${starting_balance}")
            else:
                starting_balance = 5000.0 if self.mode == "testing" else 10000.0
                self.logger.info(f"Using default starting balance: ${starting_balance}")
            self.risk_manager.initialize(starting_balance)

            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Fetch historical OHLCV for all trading pairs
            for trading_pair in self.config.trading.trading_pairs:
                symbol = self._get_symbol(trading_pair)
                await self._update_ohlcv_for_symbol(symbol, interval="1M", hours=48)

            # After initial OHLCV fetch:
            if self._ohlcv_update_task is None:
                self._ohlcv_update_task = asyncio.create_task(self._ohlcv_update_loop())

            self.logger.info("Bot initialization completed successfully")

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def _signal_handler(self, signum, frame):
        if not self.running:
            self.logger.info("Shutdown already in progress, please wait...")
            return
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    async def start(self):
        """Start the trading bot"""
        try:
            self.running = True
            self.logger.info("Starting AI Trading Bot...")

            # Connect to WebSocket
            await self.websocket_handler.connect()

            # Subscribe to trade data for all trading pairs
            for trading_pair in self.config.trading.trading_pairs:
                symbol = self._get_symbol(trading_pair)
                await self.websocket_handler.subscribe("TRADE", symbol)

            # Start main trading loop and balance sync as tasks
            loop = asyncio.get_running_loop()
            self.tasks = [
                loop.create_task(self._trading_loop(), name="trading_loop"),
                loop.create_task(self._websocket_loop(), name="websocket_loop"),
                loop.create_task(self._monitoring_loop(), name="monitoring_loop"),
                loop.create_task(self._sync_balance_loop(), name="sync_balance_loop")
            ]
            self.logger.info("All main tasks started.")
            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
        finally:
            await self.shutdown()

    async def _trading_loop(self):
        """Main trading logic loop"""
        self.logger.info("Starting trading loop...")

        while self.running:
            try:
                # Check if enough time has passed for next analysis
                if self._should_run_analysis():
                    await self._run_trading_analysis()

                # Check for position management
                await self._manage_positions()

                # Wait before next iteration, checking self.running every 5 seconds
                trade_loop_time = float(os.getenv("TRADE_LOOP_TIME", 30))
                elapsed = 0
                interval = 5
                while elapsed < trade_loop_time and self.running:
                    sleep_time = min(interval, trade_loop_time - elapsed)
                    await asyncio.sleep(sleep_time)
                    elapsed += sleep_time

            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _websocket_loop(self):
        """WebSocket message handling loop"""
        try:
            while self.running:
                await asyncio.sleep(1)  # Keep the coroutine alive
        except Exception as e:
            self.logger.error(f"WebSocket loop error: {str(e)}")

    async def _monitoring_loop(self):
        """Performance monitoring loop"""
        while self.running:
            try:
                await self._log_performance_metrics()
                await asyncio.sleep(300)  # Log every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(300)

    def _should_run_analysis(self) -> bool:
        """Check if enough time has passed for next analysis"""
        time_elapsed = datetime.now() - self.last_analysis_time
        return time_elapsed.total_seconds() >= self.analysis_interval

    async def _run_trading_analysis(self):
        """Run AI analysis and potentially execute trades"""
        try:
            self.logger.info("Running AI trading analysis...")

            for trading_pair in self.config.trading.trading_pairs:
                await self._analyze_pair(trading_pair)

            self.last_analysis_time = datetime.now()

        except Exception as e:
            self.logger.error(f"Error in trading analysis: {str(e)}")

    async def _analyze_pair(self, trading_pair: str):
        """Analyze a specific trading pair"""
        try:
            # Prepare market data for analysis
            market_data = self._prepare_market_data(trading_pair)

            # Get AI recommendation
            recommendation = await self.perplexity_api.get_trading_recommendation(market_data)

            if not recommendation:
                self.logger.warning(f"No recommendation received for {trading_pair}")
                return

            # Check if recommendation meets confidence threshold
            if recommendation.confidence < self.config.ai_prompts.confidence_threshold:
                self.logger.info(f"Recommendation for {trading_pair} below confidence threshold: {recommendation.confidence}")
                return

            # Validate with risk management
            await self._process_recommendation(recommendation)

        except Exception as e:
            self.logger.error(f"Error analyzing {trading_pair}: {str(e)}")

    def _prepare_market_data(self, trading_pair: str) -> Dict[str, Any]:
        """Prepare market data for AI analysis"""
        # Get current market data (this would come from WebSocket in real implementation)
        current_data = self.market_data.get(trading_pair, {})

        # Mock data for testing
        if self.mode == "testing":
            import random
            current_data = {
                "current_price": 45000 + random.randint(-1000, 1000),
                "volume_24h": random.randint(10000, 50000),
                "change_24h": random.uniform(-5, 5),
                "rsi": random.uniform(30, 70),
                "macd": random.uniform(-100, 100),
                "bb_upper": 46000,
                "bb_lower": 44000,
                "support_level": 44500,
                "resistance_level": 45500,
                "ema_20": 45200 + random.randint(-200, 200),
                "ma_50": 45000 + random.randint(-200, 200)
            }
        else:
            # Try to calculate indicators if OHLCV data is available
            ohlcv = current_data.get("ohlcv")  # Expecting a DataFrame or dict with columns: open, high, low, close, volume
            if ohlcv is not None:
                if isinstance(ohlcv, dict):
                    df = pd.DataFrame(ohlcv)
                else:
                    df = ohlcv
                # Ensure we have enough data
                if len(df) >= 52:
                    # EMA (9)
                    ema_9 = ta.trend.ema_indicator(df['close'], window=9).iloc[-1]
                    # EMA (20)
                    ema_20 = ta.trend.ema_indicator(df['close'], window=20).iloc[-1]
                    # MA (50)
                    ma_50 = ta.trend.sma_indicator(df['close'], window=50).iloc[-1]
                    # VWAP
                    vwap = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume']).iloc[-1]
                else:
                    ema_9 = ema_20 = ma_50 = vwap = 'N/A'
            else:
                ema_9 = ema_20 = ma_50 = vwap = 'N/A'
            current_data["ema_9"] = ema_9
            current_data["ema_20"] = ema_20
            current_data["ma_50"] = ma_50
            current_data["vwap"] = vwap

        # Get risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()

        return {
            "trading_pair": trading_pair,
            "current_price": current_data.get("current_price", 0),
            "volume_24h": current_data.get("volume_24h", 0),
            "change_24h": current_data.get("change_24h", 0),
            "rsi": current_data.get("rsi", "N/A"),
            "macd": current_data.get("macd", "N/A"),
            "bb_upper": current_data.get("bb_upper", "N/A"),
            "bb_lower": current_data.get("bb_lower", "N/A"),
            "ema_9": current_data.get("ema_9", "N/A"),
            "ema_20": current_data.get("ema_20", "N/A"),
            "ma_50": current_data.get("ma_50", "N/A"),
            "support_level": current_data.get("support_level", "N/A"),
            "resistance_level": current_data.get("resistance_level", "N/A"),
            "news_headlines": "Recent market analysis shows mixed signals",
            "available_balance": risk_metrics.available_balance,
            "open_positions": risk_metrics.total_positions,
            "daily_pnl": risk_metrics.daily_pnl,
            "vwap": current_data.get("vwap", "N/A"),
        }

    async def calculate_position_size_by_balance(self, symbol, price, percentage=None):
        """
        Calculate position size as a set percentage of available balance.
        :param symbol: Trading symbol (e.g., 'BTC_USDT')
        :param price: Current price of the asset
        :param percentage: Fraction of balance to use (e.g., 0.05 for 5%). If None, use config value.
        :return: Position size in base asset (e.g., BTC)
        """
        if percentage is None:
            percentage = self.config.trading.max_position_size
        balance = await self._fetch_pionex_balance()
        if balance is None:
            self.logger.error("Could not fetch balance for position sizing.")
            return 0
        notional = balance * percentage
        size = notional / price
        self.logger.info(f"Calculated position size: {size} {symbol.split('_')[0]} (using {percentage*100:.2f}% of balance: {balance} USDT at price {price})")
        return size

    async def _process_recommendation(self, recommendation: TradingRecommendation):
        """Process AI recommendation through risk management"""
        try:
            symbol = self._get_symbol(recommendation.trading_pair)

            # Skip if HOLD recommendation
            if recommendation.action == "HOLD":
                self.logger.info(f"AI recommends HOLD for {recommendation.trading_pair}")
                return

            # Calculate position size as a set percentage of available balance
            position_size = await self.calculate_position_size_by_balance(
                symbol, recommendation.entry_price, self.config.trading.max_position_size
            )

            # Check with risk manager
            can_trade, reason = self.risk_manager.can_open_position(
                symbol, 
                position_size, 
                self.config.trading.default_leverage
            )

            if not can_trade:
                self.logger.warning(f"Trade rejected by risk manager: {reason}")
                return

            # Validate stop loss from AI response
            sl_from_ai = recommendation.stop_loss
            if sl_from_ai is not None:
                valid_sl, sl_reason = self.risk_manager.validate_stop_loss(
                    recommendation.entry_price,
                    sl_from_ai,
                    recommendation.action
                )
                if valid_sl:
                    self.logger.info(f"Using AI-provided stop loss: {sl_from_ai}")
                    recommendation.stop_loss = sl_from_ai
                else:
                    self.logger.warning(f"AI stop loss invalid: {sl_reason}")
                    # Auto-correct only if invalid
                    sl_pct = self.config.trading.stop_loss_percentage / 100
                    if recommendation.action == "BUY":
                        recommendation.stop_loss = recommendation.entry_price * (1 - sl_pct)
                    elif recommendation.action == "SELL":
                        recommendation.stop_loss = recommendation.entry_price * (1 + sl_pct)
                    self.logger.warning(f"Auto-corrected stop loss to {recommendation.stop_loss:.2f}")
            else:
                # Auto-correct only if missing
                sl_pct = self.config.trading.stop_loss_percentage / 100
                if recommendation.action == "BUY":
                    recommendation.stop_loss = recommendation.entry_price * (1 - sl_pct)
                elif recommendation.action == "SELL":
                    recommendation.stop_loss = recommendation.entry_price * (1 + sl_pct)
                self.logger.warning(f"No AI stop loss provided. Auto-set to {recommendation.stop_loss:.2f}")

            # Validate take profit from AI response
            tp_from_ai = recommendation.take_profit
            if tp_from_ai is not None and tp_from_ai != recommendation.entry_price:
                self.logger.info(f"Using AI-provided take profit: {tp_from_ai}")
                recommendation.take_profit = tp_from_ai
            else:
                # Auto-correct only if missing or invalid
                tp_pct = self.config.trading.take_profit_percentage / 100
                if recommendation.action == "BUY":
                    recommendation.take_profit = recommendation.entry_price * (1 + tp_pct)
                elif recommendation.action == "SELL":
                    recommendation.take_profit = recommendation.entry_price * (1 - tp_pct)
                self.logger.warning(f"No valid AI take profit provided. Auto-set to {recommendation.take_profit:.2f}")

            # Execute trade
            await self._execute_trade(recommendation, position_size)

        except Exception as e:
            self.logger.error(f"Error processing recommendation: {str(e)}")

    async def _execute_trade(self, recommendation: TradingRecommendation, position_size: float):
        """Execute trade based on recommendation"""
        try:
            if self.mode == "testing":
                # Simulate trade execution
                success = await self._simulate_trade(recommendation, position_size)
            else:
                # Execute real trade (would integrate with actual Pionex API)
                success = await self._execute_real_trade(recommendation, position_size)

            if success:
                # Add position to risk manager
                symbol = self._get_symbol(recommendation.trading_pair)
                self.risk_manager.add_position(
                    symbol=symbol,
                    side=recommendation.action,
                    size=position_size,
                    entry_price=recommendation.entry_price,
                    leverage=self.config.trading.default_leverage,
                    stop_loss=recommendation.stop_loss,
                    take_profit=recommendation.take_profit
                )

                self.trade_count += 1
                self.logger.info(f"Trade executed: {recommendation.action} {position_size} {recommendation.trading_pair}")

        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")

    async def _simulate_trade(self, recommendation: TradingRecommendation, position_size: float) -> bool:
        """Simulate trade execution for testing"""
        self.logger.info(f"SIMULATED TRADE: {recommendation.action} {position_size} {recommendation.trading_pair} @ ${recommendation.entry_price}")

        # Simulate some latency
        await asyncio.sleep(0.5)

        # Mock success rate (95% for testing)
        import random
        return random.random() < 0.95

    async def _execute_real_trade(self, recommendation, position_size: float) -> bool:
        self.logger.info(f"REAL TRADE: {recommendation.action} {position_size} {recommendation.trading_pair}")
        try:
            symbol = self._get_symbol(recommendation.trading_pair)
            side = recommendation.action
            leverage_int = int(round(self.config.trading.default_leverage))
            market_type = self.config.trading.market_type.lower()
            api_symbol = self._get_api_symbol(symbol)
            order_type = recommendation.order_type if hasattr(recommendation, 'order_type') and recommendation.order_type else "LIMIT"
            order_payload = {
                "symbol": api_symbol,
                "side": side,
                "type": order_type
            }

            # Spot market
            if market_type == "spot":
                if side == "BUY":
                    # position_size is in base asset (e.g., BTC), convert to USDT
                    usdt_to_spend = position_size * recommendation.entry_price
                    order_payload["amount"] = str(usdt_to_spend)
                else:  # SELL
                    order_payload["size"] = str(position_size)    # Asset amount
                if order_type.upper() == "LIMIT":
                    order_payload["price"] = str(recommendation.entry_price)
            # Futures market
            elif market_type == "perpetual":
                order_payload["size"] = str(position_size)  # contracts
                order_payload["leverage"] = leverage_int
                if order_type.upper() == "LIMIT":
                    order_payload["price"] = str(recommendation.entry_price)
            else:
                self.logger.error(f"Unknown market type: {market_type}")
                return False

            self.logger.info(f"Order payload: {order_payload}")
            await self._place_pionex_order_payload(order_payload)
            return True
        except Exception as e:
            self.logger.error(f"Exception in _execute_real_trade: {e}")
            return False

    async def _place_pionex_order_payload(self, order_payload):
        """
        Place an order using pionex_python.restful.Orders instead of manual REST logic.
        """
        if not self.orders_client:
            self.logger.error("Pionex Orders client not initialized. Cannot place order.")
            return None
        try:
            # Orders client is synchronous, so run in executor
            import concurrent.futures
            loop = asyncio.get_running_loop()
            def place_order():
                return self.orders_client.new_order(**order_payload)
            result = await loop.run_in_executor(None, place_order)
            if result and result.get('code') == 0:
                self.logger.info(f"Order placed successfully: {result.get('data')}")
                return result.get('data')
            else:
                self.logger.error(f"Order placement failed: {result}")
                return None
        except Exception as e:
            self.logger.error(f"Exception placing Pionex order via Orders client: {e}")
            return None

    async def _manage_positions(self):
        """Monitor and manage open positions"""
        try:
            for symbol, position in list(self.risk_manager.positions.items()):
                # Update position with current price (mock for testing)
                if self.mode == "testing":
                    import random
                    # Simulate price movement
                    price_change = random.uniform(-0.02, 0.02)  # ±2%
                    new_price = position.current_price * (1 + price_change)
                    self.risk_manager.update_position_price(symbol, new_price)

        except Exception as e:
            self.logger.error(f"Error managing positions: {str(e)}")

    def _handle_market_data(self, data: Dict[str, Any]):
        """Handle incoming market data from WebSocket"""
        try:
            # Only handle trade ticks for OHLCV aggregation
            if 'channel' in data and 'ticker' in data['channel']:
                symbol = data.get('data', {}).get('symbol', '')
                if symbol:
                    self.market_data[symbol] = data['data']

            # --- Real-time OHLCV aggregation ---
            if data.get('topic') == 'TRADE':
                symbol = data.get('symbol')
                trade = data.get('data')
                if not symbol or not trade:
                    return
                # trade: {"price": "...", "size": "...", "side": "BUY"/"SELL", "time": 1680000000000}
                price = float(trade.get('price'))
                volume = float(trade.get('size'))
                ts = int(trade.get('time'))
                minute_ts = ts - (ts % 60000)  # Start of the minute
                # Get or create OHLCV DataFrame
                ohlcv = self.market_data.setdefault(symbol, {}).get('ohlcv')
                if ohlcv is None or ohlcv.empty:
                    # Create new DataFrame
                    df = pd.DataFrame([{
                        'time': minute_ts,
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': volume
                    }])
                    self.market_data[symbol]['ohlcv'] = df
                    return
                df = ohlcv
                # If last bar is current minute, update it
                if df.iloc[-1]['time'] == minute_ts:
                    df.at[df.index[-1], 'high'] = max(df.iloc[-1]['high'], price)
                    df.at[df.index[-1], 'low'] = min(df.iloc[-1]['low'], price)
                    df.at[df.index[-1], 'close'] = price
                    df.at[df.index[-1], 'volume'] += volume
                else:
                    # New minute, append new row
                    new_row = {
                        'time': minute_ts,
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': volume
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    # Keep only last 48H (2880 rows)
                    if len(df) > 2880:
                        df = df.iloc[-2880:]
                    self.market_data[symbol]['ohlcv'] = df
        except Exception as e:
            self.logger.error(f"Error handling market data: {str(e)}")

    async def _log_performance_metrics(self):
        """Log current performance metrics"""
        try:
            risk_metrics = self.risk_manager.get_risk_metrics()

            uptime = datetime.now() - self.start_time

            self.logger.info(f"PERFORMANCE METRICS:")
            self.logger.info(f"  Uptime: {uptime}")
            self.logger.info(f"  Total Trades: {self.trade_count}")
            self.logger.info(f"  Total P&L: ${risk_metrics.total_pnl:.2f}")
            self.logger.info(f"  Daily P&L: ${risk_metrics.daily_pnl:.2f}")
            self.logger.info(f"  Win Rate: {risk_metrics.win_rate:.1f}%")
            self.logger.info(f"  Max Drawdown: {risk_metrics.max_drawdown:.2f}%")
            self.logger.info(f"  Risk Level: {risk_metrics.risk_level.value}")
            self.logger.info(f"  Open Positions: {risk_metrics.total_positions}")

        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {str(e)}")

    async def _sync_balance_loop(self):
        """Periodically fetch and update the real account balance from Pionex"""
        if self.mode != "production":
            return
        while self.running:
            try:
                balance = await self._fetch_pionex_balance()
                if balance is not None:
                    self.risk_manager.update_balance(balance)
            except Exception as e:
                self.logger.error(f"Error syncing balance: {e}")
            await asyncio.sleep(60)  # Sync every 60 seconds

    async def _fetch_pionex_balance(self) -> float:
        """Fetch account balance from Pionex REST API (USDT only)"""
        api_key = os.getenv('TRADING_BOT_PIONEX_API_KEY')
        api_secret = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')
        if not api_key or not api_secret:
            self.logger.error("Pionex API credentials not set for balance sync.")
            return None
        try:
            url = "https://api.pionex.com/api/v1/account/balances"
            path_url = "/api/v1/account/balances"
            method = "GET"
            timestamp = str(int(time.time() * 1000))
            query = f"timestamp={timestamp}"
            full_url = f"{url}?{query}"
            string_to_sign = f"{method}{path_url}?{query}"
            # self.logger.info(f"[ALT2 SIGN] String to sign: {string_to_sign}")
            signature = hmac.new(api_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
            headers = {
                "PIONEX-KEY": api_key,
                "PIONEX-SIGNATURE": signature,
                "PIONEX-TIMESTAMP": timestamp
            }
            # self.logger.info(f"Requesting balance: {full_url}")
            # self.logger.info(f"Headers: {headers}")
            async with aiohttp.ClientSession() as session:
                async with session.get(full_url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Find USDT balance
                        for asset in data.get("data", {}).get("balances", []):
                            if asset.get("coin") == "USDT":
                                return float(asset.get("free", 0))
                        self.logger.warning("USDT balance not found in response.")
                        self.logger.warning(f"Full balance response: {data}")
                        return None
                    else:
                        self.logger.error(f"Failed to fetch balance: {resp.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Exception fetching Pionex balance: {e}")
            return None

    async def _set_futures_leverage(self, symbol: str, leverage) -> bool:
        api_symbol = self._get_api_symbol(symbol)
        api_key = os.getenv('TRADING_BOT_PIONEX_API_KEY')
        api_secret = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')
        if not api_key or not api_secret:
            self.logger.error("Pionex API credentials not set for leverage adjustment.")
            return False
        try:
            url = "https://api.pionex.com/api/v1/futures/leverage"
            path_url = "/api/v1/futures/leverage"
            method = "POST"
            timestamp = str(int(time.time() * 1000))
            leverage_int = int(round(leverage))
            body = {
                "symbol": api_symbol,
                "leverage": leverage_int,
                "timestamp": int(timestamp)
            }
            import json as pyjson
            body_json = pyjson.dumps(body, separators=(',', ':'))
            string_to_sign = f"{method}{path_url}{timestamp}{body_json}"
            signature = hmac.new(api_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
            headers = {
                "PIONEX-KEY": api_key,
                "PIONEX-SIGNATURE": signature,
                "PIONEX-TIMESTAMP": timestamp,
                "Content-Type": "application/json"
            }
            self.logger.info(f"Setting leverage {leverage_int}x for {api_symbol}...")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=body_json) as resp:
                    try:
                        resp_data = await resp.json()
                    except Exception:
                        resp_data = await resp.text()
                        self.logger.error(f"Non-JSON response: {resp_data}")
                        return False
                    if resp.status == 200 and resp_data.get("result"):
                        self.logger.info(f"Leverage set successfully: {resp_data['data']}")
                        return True
                    else:
                        self.logger.error(f"Leverage set failed: {resp.status} {resp_data}")
                        return False
        except Exception as e:
            self.logger.error(f"Exception during leverage adjustment: {e}")
            return False

    async def shutdown(self):
        """Graceful shutdown of the bot"""
        try:
            self.logger.info("Shutting down AI Trading Bot...")

            # Cancel all running tasks except the current one
            current_task = asyncio.current_task()
            for task in self.tasks:
                if task is not current_task and not task.done():
                    self.logger.info(f"Cancelling task: {task.get_name()}")
                    task.cancel()
            # Wait for all tasks to finish with timeout
            try:
                await asyncio.wait([t for t in self.tasks if t is not current_task], timeout=10)
                self.logger.info("All main tasks cancelled.")
            except Exception as e:
                self.logger.error(f"Error waiting for tasks to cancel: {e}")

            # Close WebSocket connection
            if self.websocket_handler:
                self.logger.info("Closing WebSocket handler...")
                await self.websocket_handler.close()

            # Log final performance
            self.logger.info("Logging final performance metrics...")
            await self._log_performance_metrics()

            # Save configuration backup
            if self.config_manager and self.config:
                self.logger.info("Saving configuration backup...")
                self.config_manager.save_config_backup()

            # Cancel OHLCV update task if running
            if self._ohlcv_update_task:
                self._ohlcv_update_task.cancel()
                try:
                    await self._ohlcv_update_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("Shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    # Helper function to get correct symbol format
    def _get_symbol(self, trading_pair: str) -> str:
        # Always use underscore format for Pionex API (no _PERP)
        return trading_pair.replace("/", "_")

    def _get_api_symbol(self, symbol: str) -> str:
        # Remove _PERP suffix for Pionex API calls
        return symbol.replace('_PERP', '')

    async def _place_pionex_order(self, symbol, side, order_type, size, price=None):
        """Simulate or place an order for test compatibility and production."""
        if self.mode == "testing":
            # Simulate minOrderSize check
            if size is None or size <= 0:
                raise ValueError("Order size must be positive")
            # Simulate order placement
            return {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "size": size,
                "price": price,
                "status": "simulated"
            }
        else:
            # Production: Place real order via Pionex REST API
            api_key = os.getenv('TRADING_BOT_PIONEX_API_KEY')
            api_secret = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')
            if not api_key or not api_secret:
                self.logger.error("Pionex API credentials not set for order placement.")
                return None
            try:
                import json as pyjson
                market_type = getattr(self.config.trading, 'market_type', 'spot').lower()
                # --- Spot notional validation ---
                if market_type == "spot":
                    # Fetch min_notional from /api/v1/common/symbols
                    min_notional = 11  # Default fallback
                    try:
                        url_info = f"https://api.pionex.com/api/v1/common/symbols?symbol={symbol}"
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url_info) as resp:
                                data = await resp.json()
                                if data.get("result") and data["data"]["symbols"]:
                                    min_notional_str = data["data"]["symbols"][0].get("minAmount")
                                    if min_notional_str:
                                        min_notional = float(min_notional_str)
                    except Exception as e:
                        self.logger.warning(f"Could not fetch min_notional for {symbol}, using default: {e}")
                    notional = float(price) * float(size)
                    if notional < min_notional:
                        msg = f"Order notional ({notional}) is below min_notional ({min_notional}) for {symbol}. Increase size or price."
                        self.logger.error(msg)
                        raise ValueError(msg)
                url = "https://api.pionex.com/api/v1/trade/order"
                path_url = "/api/v1/trade/order"
                order_payload = {
                    "clientOrderId": str(uuid.uuid4()),
                    "symbol": symbol,
                    "side": side,
                    "type": order_type,
                    "size": str(size)
                }
                if price is not None and order_type.upper() == "LIMIT":
                    order_payload["price"] = str(price)
                method = "POST"
                timestamp = str(int(time.time() * 1000))
                query = f"timestamp={timestamp}"
                full_url = f"{url}?{query}"
                body_json = pyjson.dumps(order_payload, separators=(',', ':'))
                string_to_sign = f"{method}{path_url}?{query}{body_json}"
                signature = hmac.new(api_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
                headers = {
                    "PIONEX-KEY": api_key,
                    "PIONEX-SIGNATURE": signature,
                    "PIONEX-TIMESTAMP": timestamp,
                    "Content-Type": "application/json"
                }
                self.logger.info(f"Placing order: {body_json}")
                async with aiohttp.ClientSession() as session:
                    async with session.post(full_url, headers=headers, data=body_json) as resp:
                        try:
                            resp_data = await resp.json()
                        except Exception:
                            resp_data = await resp.text()
                            self.logger.error(f"Non-JSON response: {resp_data}")
                            return None
                        if resp.status == 200 and resp_data.get("result"):
                            self.logger.info(f"Order placed successfully: {resp_data['data']}")
                            return resp_data["data"]
                        else:
                            self.logger.error(f"Order placement failed: {resp.status} {resp_data}")
                            return resp_data
            except Exception as e:
                self.logger.error(f"Exception placing Pionex order: {e}")
                return None

    async def _get_pionex_open_orders(self, symbol):
        """Simulate or fetch open orders for a symbol."""
        if self.mode == "testing":
            # Return a simulated open order list
            return [
                {
                    "order_id": "simulated_order_id",
                    "symbol": symbol,
                    "side": "BUY",
                    "size": 1.0,
                    "price": 100.0,
                    "status": "open"
                }
            ]
        else:
            # Production: Fetch open orders from Pionex REST API
            api_key = os.getenv('TRADING_BOT_PIONEX_API_KEY')
            api_secret = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')
            if not api_key or not api_secret:
                self.logger.error("Pionex API credentials not set for open orders fetch.")
                return None
            try:
                url = "https://api.pionex.com/api/v1/trade/openOrders"
                path_url = "/api/v1/trade/openOrders"
                method = "GET"
                timestamp = str(int(time.time() * 1000))
                query = f"symbol={symbol}&timestamp={timestamp}"
                full_url = f"{url}?{query}"
                string_to_sign = f"{method}{path_url}?{query}"
                signature = hmac.new(api_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
                headers = {
                    "PIONEX-KEY": api_key,
                    "PIONEX-SIGNATURE": signature,
                    "PIONEX-TIMESTAMP": timestamp
                }
                self.logger.info(f"Fetching open orders for {symbol}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(full_url, headers=headers) as resp:
                        try:
                            resp_data = await resp.json()
                        except Exception:
                            resp_data = await resp.text()
                            self.logger.error(f"Non-JSON response: {resp_data}")
                            return None
                        if resp.status == 200 and resp_data.get("result"):
                            self.logger.info(f"Open orders fetched: {resp_data['data']}")
                            return resp_data["data"]
                        else:
                            self.logger.error(f"Open orders fetch failed: {resp.status} {resp_data}")
                            return resp_data
            except Exception as e:
                self.logger.error(f"Exception fetching open orders: {e}")
                return None

    async def _cancel_pionex_order(self, symbol, order_id):
        """Simulate or cancel an order by order ID."""
        if self.mode == "testing":
            # Simulate successful cancellation
            return {"order_id": order_id, "symbol": symbol, "status": "cancelled (simulated)"}
        else:
            # Production: Cancel order via Pionex REST API
            api_key = os.getenv('TRADING_BOT_PIONEX_API_KEY')
            api_secret = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')
            if not api_key or not api_secret:
                self.logger.error("Pionex API credentials not set for order cancellation.")
                return None
            try:
                url = "https://api.pionex.com/api/v1/trade/order"
                path_url = "/api/v1/trade/order"
                method = "DELETE"
                timestamp = str(int(time.time() * 1000))
                import json as pyjson
                body = {
                    "symbol": symbol,
                    "orderId": order_id
                }
                body_json = pyjson.dumps(body, separators=(',', ':'))
                query = f"timestamp={timestamp}"
                full_url = f"{url}?{query}"
                string_to_sign = f"{method}{path_url}?{query}{body_json}"
                signature = hmac.new(api_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
                headers = {
                    "PIONEX-KEY": api_key,
                    "PIONEX-SIGNATURE": signature,
                    "PIONEX-TIMESTAMP": timestamp,
                    "Content-Type": "application/json"
                }
                self.logger.info(f"Cancelling order {order_id} for {symbol}")
                async with aiohttp.ClientSession() as session:
                    async with session.delete(full_url, headers=headers, data=body_json) as resp:
                        try:
                            resp_data = await resp.json()
                        except Exception:
                            resp_data = await resp.text()
                            self.logger.error(f"Non-JSON response: {resp_data}")
                            return None
                        if resp.status == 200 and resp_data.get("result"):
                            self.logger.info(f"Order cancelled: {resp_data['data']}")
                            return resp_data["data"]
                        else:
                            self.logger.error(f"Order cancellation failed: {resp.status} {resp_data}")
                            return resp_data
            except Exception as e:
                self.logger.error(f"Exception cancelling order: {e}")
                return None

    async def fetch_valid_perp_symbols(self):
        """Fetch the list of valid PERP (futures) symbols from Pionex API."""
        url = "https://api.pionex.com/api/v1/common/symbols?type=PERP"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    data = await resp.json()
                    if data.get("result"):
                        symbols = [s["symbol"] for s in data["data"]["symbols"]]
                        self.logger.info(f"Fetched {len(symbols)} valid PERP symbols from Pionex.")
                        return set(symbols)
                    else:
                        self.logger.error("Failed to fetch PERP symbols from Pionex.")
                        return set()
        except Exception as e:
            self.logger.error(f"Exception fetching PERP symbols: {e}")
            return set()

    async def is_valid_perp_symbol(self, symbol):
        """Check if a symbol is a valid PERP symbol on Pionex."""
        valid_symbols = await self.fetch_valid_perp_symbols()
        if symbol in valid_symbols:
            return True
        self.logger.warning(f"Symbol {symbol} is not a valid PERP symbol on Pionex.")
        return False

    async def _validate_leverage_and_margin(self, symbol, size, price, leverage):
        """
        Check margin and set leverage if needed before placing a leveraged order.
        Leverage is set from config/environment variable. No leverage bracket API validation is performed.
        NOTE: Manual transfer of funds from spot to futures is required. The API does not support this for most users.
        """
        # 1. Calculate required margin
        notional = size * price
        required_margin = notional / leverage
        # 2. Fetch futures account balance
        balance = await self._fetch_pionex_balance()
        if balance is None or balance < required_margin:
            self.logger.error(f"Insufficient margin: required {required_margin}, available {balance}. Please transfer funds manually from spot to futures in the Pionex app/web.")
            return False, "Insufficient margin"
        # 3. Set leverage
        set_result = await self._set_futures_leverage(symbol, leverage)
        if not set_result:
            self.logger.error("Failed to set leverage before order placement.")
            return False, "Failed to set leverage"
        return True, "OK"

    async def _monitor_liquidation_and_auto_reduce(self, symbol, reduce_pct=0.1, threshold=0.85, poll_interval=30):
        """Monitor position risk and auto-reduce if marginRatio > threshold."""
        while self.running:
            try:
                api_key = os.getenv('TRADING_BOT_PIONEX_API_KEY')
                api_secret = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')
                if not api_key or not api_secret:
                    self.logger.error("Pionex API credentials not set for liquidation monitoring.")
                    await asyncio.sleep(poll_interval)
                    continue
                url = f"https://api.pionex.com/api/v1/futures/positionRisk?symbol={symbol}"
                path_url = "/api/v1/futures/positionRisk"
                method = "GET"
                query = f"symbol={symbol}"
                timestamp = str(int(time.time() * 1000))
                string_to_sign = f"{method}{path_url}?{query}{timestamp}"
                signature = hmac.new(api_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
                headers = {
                    "PIONEX-KEY": api_key,
                    "PIONEX-SIGNATURE": signature,
                    "PIONEX-TIMESTAMP": timestamp
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as resp:
                        try:
                            resp_data = await resp.json()
                        except Exception:
                            resp_data = await resp.text()
                            self.logger.error(f"Non-JSON response: {resp_data}")
                            await asyncio.sleep(poll_interval)
                            continue
                        if resp.status == 200 and resp_data.get("result"):
                            risk = resp_data["data"]
                            margin_ratio = float(risk.get("marginRatio", 0))
                            current_leverage = float(risk.get("leverage", 1))
                            if margin_ratio > threshold:
                                self.logger.warning(f"Margin ratio {margin_ratio} exceeds threshold {threshold}. Reducing position.")
                                # Reduce position by reduce_pct (close part of the position)
                                # This is a placeholder: you must implement position reduction logic
                                # e.g., place an opposite order of size*reduce_pct
                                # self.logger.info(f"Would reduce position by {reduce_pct*100}% here.")
                                # Example:
                                # await self._place_pionex_order(symbol, 'SELL', 'MARKET', size_to_reduce)
                                pass
                            else:
                                self.logger.info(f"Margin ratio {margin_ratio} is safe.")
                        else:
                            self.logger.error(f"Failed to fetch position risk: {resp.status} {resp_data}")
                await asyncio.sleep(poll_interval)
            except Exception as e:
                self.logger.error(f"Exception in liquidation monitor: {e}")
                await asyncio.sleep(poll_interval)

    async def _fetch_historical_ohlcv(self, symbol: str, interval: str = "1M", hours: int = 48, after: int = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles from Pionex REST API for the given symbol and interval.
        If 'after' is provided, only fetch candles after that timestamp (ms).
        Returns a DataFrame with columns: ['time', 'open', 'high', 'low', 'close', 'volume']
        """
        base_url = "https://api.pionex.com/api/v1/market/klines"
        now = int(time.time() * 1000)
        ms_per_candle = 60_000 if interval == "1M" else 300_000  # Only 1M supported for now
        total_candles = hours * 60 if interval == "1M" else hours * 12
        max_limit = 500
        candles = []
        end_time = now
        session = aiohttp.ClientSession()
        try:
            # If after is set, fetch only new candles
            if after is not None:
                # Always fetch up to now, but only as many as needed
                for _ in range(math.ceil(total_candles / max_limit)):
                    limit = max_limit
                    params = {
                        "symbol": symbol,
                        "interval": interval,
                        "limit": limit,
                        "startTime": after + ms_per_candle  # fetch strictly after 'after'
                    }
                    async with session.get(base_url, params=params) as resp:
                        if resp.status != 200:
                            self.logger.error(f"Failed to fetch klines for {symbol}: {resp.status}")
                            break
                        data = await resp.json()
                        if not data.get("result"):
                            self.logger.error(f"Klines fetch error for {symbol}: {data}")
                            break
                        klines = data["data"]["klines"]
                        if not klines:
                            break
                        candles += klines
                        if len(klines) < limit:
                            break  # No more data
                        # Advance after to last candle
                        after = klines[-1]["time"]
                # Convert to DataFrame
                new_df = pd.DataFrame([
                    {
                        "time": k["time"],
                        "open": float(k["open"]),
                        "high": float(k["high"]),
                        "low": float(k["low"]),
                        "close": float(k["close"]),
                        "volume": float(k["volume"])
                    } for k in candles
                ])
                return new_df
            # Else, fetch full window (old behavior)
            for _ in range(math.ceil(total_candles / max_limit)):
                limit = min(max_limit, total_candles - len(candles))
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                    "endTime": end_time
                }
                async with session.get(base_url, params=params) as resp:
                    if resp.status != 200:
                        self.logger.error(f"Failed to fetch klines for {symbol}: {resp.status}")
                        break
                    data = await resp.json()
                    if not data.get("result"):
                        self.logger.error(f"Klines fetch error for {symbol}: {data}")
                        break
                    klines = data["data"]["klines"]
                    if not klines:
                        break
                    candles = klines + candles  # prepend to keep order
                    if len(klines) < limit:
                        break  # No more data
                    end_time = klines[0]["time"] - ms_per_candle
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    "time": k["time"],
                    "open": float(k["open"]),
                    "high": float(k["high"]),
                    "low": float(k["low"]),
                    "close": float(k["close"]),
                    "volume": float(k["volume"])
                } for k in candles
            ])
            return df
        except Exception as e:
            self.logger.error(f"Exception fetching historical klines for {symbol}: {e}")
            return pd.DataFrame()
        finally:
            await session.close()

    async def _update_ohlcv_for_symbol(self, symbol: str, interval: str = "1M", hours: int = 48):
        """
        Update OHLCV DataFrame for a symbol by fetching only new candles and appending them.
        """
        ohlcv = self.market_data.get(symbol, {}).get("ohlcv")
        if ohlcv is not None and not ohlcv.empty:
            last_ts = int(ohlcv["time"].max())
            new_df = await self._fetch_historical_ohlcv(symbol, interval=interval, hours=hours, after=last_ts)
            if not new_df.empty:
                combined = pd.concat([ohlcv, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["time"]).sort_values("time")
                # Keep only last 2880 rows (48H)
                if len(combined) > 2880:
                    combined = combined.iloc[-2880:]
                self.market_data[symbol]["ohlcv"] = combined.reset_index(drop=True)
                self.logger.info(f"Appended {len(new_df)} new candles to {symbol} OHLCV.")
            else:
                self.logger.info(f"No new candles for {symbol}.")
        else:
            # No data yet, fetch full window
            df = await self._fetch_historical_ohlcv(symbol, interval=interval, hours=hours)
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
            self.market_data[symbol]["ohlcv"] = df
            self.logger.info(f"Fetched initial OHLCV for {symbol}.")

    async def _update_all_ohlcv(self, interval: str = "1M", hours: int = 48):
        """
        Update OHLCV for all trading pairs.
        """
        for trading_pair in self.config.trading.trading_pairs:
            symbol = self._get_symbol(trading_pair)
            await self._update_ohlcv_for_symbol(symbol, interval=interval, hours=hours)

    async def _ohlcv_update_loop(self):
        """Background task to periodically update OHLCV for all symbols."""
        while True:
            try:
                self.logger.info(f"Starting periodic OHLCV update for all symbols...")
                await self._update_all_ohlcv(interval="1M", hours=48)
                self.logger.info(f"Periodic OHLCV update complete. Next in {self.ohlcv_update_minutes} minutes.")
            except Exception as e:
                self.logger.error(f"Error in OHLCV update loop: {e}")
            await asyncio.sleep(self.ohlcv_update_minutes * 60)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI Trading Bot')
    parser.add_argument('--mode', choices=['testing', 'production'], 
                       default='testing', help='Trading mode')
    parser.add_argument('--config', default='config', 
                       help='Configuration directory path')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration without running')

    args = parser.parse_args()

    bot = AITradingBot(config_path=args.config, mode=args.mode)

    try:
        await bot.initialize()

        if args.validate_only:
            print(f"✅ Configuration validation successful for {args.mode} mode")
            return

        await bot.start()

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"❌ Bot failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
