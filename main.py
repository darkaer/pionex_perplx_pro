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
        self.analysis_interval = 300  # 5 minutes

        # Performance tracking
        self.start_time = datetime.now()
        self.trade_count = 0
        self.total_pnl = 0.0

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

            self.logger.info("Bot initialization completed successfully")

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
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
                symbol = trading_pair.replace("/", "_")
                await self.websocket_handler.subscribe("TRADE", symbol)

            # Start main trading loop and balance sync
            await asyncio.gather(
                self._trading_loop(),
                self._websocket_loop(),
                self._monitoring_loop(),
                self._sync_balance_loop()
            )

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

                # Wait before next iteration
                trade_loop_time = float(os.getenv("TRADE_LOOP_TIME", 30))
                await asyncio.sleep(trade_loop_time)  # Configurable trade loop interval

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
                    # EMA (20)
                    ema_20 = ta.trend.ema_indicator(df['close'], window=20).iloc[-1]
                    # MA (50)
                    ma_50 = ta.trend.sma_indicator(df['close'], window=50).iloc[-1]
                else:
                    ema_20 = ma_50 = 'N/A'
            else:
                ema_20 = ma_50 = 'N/A'
            current_data["ema_20"] = ema_20
            current_data["ma_50"] = ma_50

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
            "ema_20": current_data.get("ema_20", "N/A"),
            "ma_50": current_data.get("ma_50", "N/A"),
            "support_level": current_data.get("support_level", "N/A"),
            "resistance_level": current_data.get("resistance_level", "N/A"),
            "news_headlines": "Recent market analysis shows mixed signals",
            "available_balance": risk_metrics.available_balance,
            "open_positions": risk_metrics.total_positions,
            "daily_pnl": risk_metrics.daily_pnl
        }

    async def _process_recommendation(self, recommendation: TradingRecommendation):
        """Process AI recommendation through risk management"""
        try:
            symbol = recommendation.trading_pair.replace("/", "")

            # Skip if HOLD recommendation
            if recommendation.action == "HOLD":
                self.logger.info(f"AI recommends HOLD for {recommendation.trading_pair}")
                return

            # Calculate position size based on risk
            position_size = self._calculate_position_size(recommendation)

            # Check with risk manager
            can_trade, reason = self.risk_manager.can_open_position(
                symbol, 
                position_size, 
                self.config.trading.default_leverage
            )

            if not can_trade:
                self.logger.warning(f"Trade rejected by risk manager: {reason}")
                return

            # Validate stop loss
            if recommendation.stop_loss:
                valid_sl, sl_reason = self.risk_manager.validate_stop_loss(
                    recommendation.entry_price,
                    recommendation.stop_loss,
                    recommendation.action
                )

                if not valid_sl:
                    self.logger.warning(f"Invalid stop loss: {sl_reason}")
                    return

            # Execute trade
            await self._execute_trade(recommendation, position_size)

        except Exception as e:
            self.logger.error(f"Error processing recommendation: {str(e)}")

    def _calculate_position_size(self, recommendation: TradingRecommendation) -> float:
        """Calculate appropriate position size based on confidence and risk"""
        base_size = self.config.trading.max_position_size

        # Adjust based on confidence
        confidence_multiplier = recommendation.confidence

        # Adjust based on risk level
        risk_multipliers = {
            "LOW": 1.0,
            "MEDIUM": 0.8,
            "HIGH": 0.5
        }

        risk_multiplier = risk_multipliers.get(recommendation.risk_level, 0.5)

        # Calculate final position size
        position_size = base_size * confidence_multiplier * risk_multiplier

        return min(position_size, self.config.trading.max_position_size)

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
                symbol = recommendation.trading_pair.replace("/", "")
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

    async def _execute_real_trade(self, recommendation: TradingRecommendation, position_size: float) -> bool:
        """Execute real trade through Pionex API"""
        self.logger.info(f"REAL TRADE: {recommendation.action} {position_size} {recommendation.trading_pair}")
        try:
            symbol = recommendation.trading_pair.replace("/", "_")
            side = recommendation.action
            # Determine order type and parameters
            if hasattr(recommendation, 'order_type') and recommendation.order_type == "MARKET":
                # MARKET order
                # For market BUY, use amount (USDT to spend); for market SELL, use size (quantity)
                if side == "BUY":
                    order_result = await self._place_pionex_order(symbol, side, "MARKET", amount=position_size)
                else:
                    order_result = await self._place_pionex_order(symbol, side, "MARKET", size=position_size)
            else:
                # LIMIT order (default)
                order_result = await self._place_pionex_order(
                    symbol, side, "LIMIT", size=position_size, price=recommendation.entry_price, ioc=True)
            if order_result:
                self.logger.info(f"Order placed: {order_result}")
                return True
            else:
                self.logger.error("Order placement failed.")
                return False
        except Exception as e:
            self.logger.error(f"Exception in _execute_real_trade: {e}")
            return False

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
            if 'channel' in data and 'ticker' in data['channel']:
                symbol = data.get('data', {}).get('symbol', '')
                if symbol:
                    self.market_data[symbol] = data['data']

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

    async def _place_pionex_order(self, symbol, side, order_type, size=None, price=None, amount=None, ioc=False):
        api_key = os.getenv('TRADING_BOT_PIONEX_API_KEY')
        api_secret = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')
        if not api_key or not api_secret:
            self.logger.error("Pionex API credentials not set for order placement.")
            return None
        try:
            url = "https://api.pionex.com/api/v1/trade/order"
            path_url = "/api/v1/trade/order"
            method = "POST"
            timestamp = str(int(time.time() * 1000))
            client_order_id = str(uuid.uuid4())
            body = {
                "clientOrderId": client_order_id,
                "symbol": symbol,
                "side": side,
                "type": order_type
            }
            if size is not None:
                body["size"] = str(size)
            if price is not None:
                body["price"] = str(price)
            if amount is not None:
                body["amount"] = str(amount)
            if ioc:
                body["IOC"] = True

            import json as pyjson
            body_json = pyjson.dumps(body, separators=(',', ':'))  # No spaces, compact

            # Add timestamp as query param
            query = f"timestamp={timestamp}"
            full_url = f"{url}?{query}"
            # Signature: method + path_url + "?" + query + body_json
            string_to_sign = f"{method}{path_url}?{query}{body_json}"
            self.logger.info(f"[ORDER SIGN] String to sign: {string_to_sign}")
            signature = hmac.new(api_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
            headers = {
                "PIONEX-KEY": api_key,
                "PIONEX-SIGNATURE": signature,
                "PIONEX-TIMESTAMP": timestamp,
                "Content-Type": "application/json"
            }
            self.logger.info(f"Placing order: {body_json}")
            self.logger.info(f"Requesting order: {full_url}")
            self.logger.info(f"Headers: {headers}")
            async with aiohttp.ClientSession() as session:
                async with session.post(full_url, headers=headers, data=body_json) as resp:
                    resp_data = await resp.json()
                    if resp.status == 200 and resp_data.get("result"):
                        self.logger.info(f"Order placed successfully: {resp_data['data']}")
                        return resp_data["data"]
                    else:
                        self.logger.error(f"Order placement failed: {resp.status} {resp_data}")
                        return None
        except Exception as e:
            self.logger.error(f"Exception placing Pionex order: {e}")
            return None

    async def _place_pionex_mass_order(self, symbol, orders):
        """
        Place multiple LIMIT orders at once.
        orders: list of dicts, each with keys: side, price, size, (optional) clientOrderId
        """
        api_key = os.getenv('TRADING_BOT_PIONEX_API_KEY')
        api_secret = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')
        if not api_key or not api_secret:
            self.logger.error("Pionex API credentials not set for mass order placement.")
            return None
        try:
            url = "https://api.pionex.com/api/v1/trade/massOrder"
            path_url = "/api/v1/trade/massOrder"
            method = "POST"
            timestamp = str(int(time.time() * 1000))
            import uuid
            # Ensure all orders have clientOrderId and type LIMIT
            for order in orders:
                if "clientOrderId" not in order:
                    order["clientOrderId"] = str(uuid.uuid4())
                order["type"] = "LIMIT"  # Only LIMIT supported
            body = {
                "symbol": symbol,
                "orders": orders
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
            self.logger.info(f"Placing mass order: {body_json}")
            self.logger.info(f"Headers: {headers}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=body_json) as resp:
                    resp_data = await resp.json()
                    if resp.status == 200 and resp_data.get("result"):
                        self.logger.info(f"Mass order placed successfully: {resp_data['data']}")
                        return resp_data["data"]
                    else:
                        self.logger.error(f"Mass order placement failed: {resp.status} {resp_data}")
                        return None
        except Exception as e:
            self.logger.error(f"Exception placing Pionex mass order: {e}")
            return None

    async def _cancel_pionex_order(self, symbol, order_id):
        """
        Cancel a single order by symbol and orderId.
        """
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
            body = {
                "symbol": symbol,
                "orderId": order_id
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
            self.logger.info(f"Cancelling order: {body_json}")
            self.logger.info(f"Headers: {headers}")
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers, data=body_json) as resp:
                    resp_data = await resp.json()
                    if resp.status == 200 and resp_data.get("result"):
                        self.logger.info(f"Order cancelled successfully: {order_id}")
                        return True
                    else:
                        self.logger.error(f"Order cancellation failed: {resp.status} {resp_data}")
                        return False
        except Exception as e:
            self.logger.error(f"Exception cancelling Pionex order: {e}")
            return False

    async def _get_pionex_open_orders(self, symbol):
        """
        Get all open orders for a symbol.
        """
        api_key = os.getenv('TRADING_BOT_PIONEX_API_KEY')
        api_secret = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')
        if not api_key or not api_secret:
            self.logger.error("Pionex API credentials not set for open orders query.")
            return None
        try:
            url = f"https://api.pionex.com/api/v1/trade/openOrders?symbol={symbol}"
            path_url = "/api/v1/trade/openOrders"
            method = "GET"
            query = f"symbol={symbol}"
            timestamp = str(int(time.time() * 1000))
            # Signature: method + path_url + "?" + query + timestamp
            string_to_sign = f"{method}{path_url}?{query}{timestamp}"
            signature = hmac.new(api_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
            headers = {
                "PIONEX-KEY": api_key,
                "PIONEX-SIGNATURE": signature,
                "PIONEX-TIMESTAMP": timestamp
            }
            self.logger.info(f"Getting open orders: {url}")
            self.logger.info(f"Headers: {headers}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    resp_data = await resp.json()
                    if resp.status == 200 and resp_data.get("result"):
                        self.logger.info(f"Open orders fetched: {resp_data['data']['orders']}")
                        return resp_data["data"]["orders"]
                    else:
                        self.logger.error(f"Open orders fetch failed: {resp.status} {resp_data}")
                        return None
        except Exception as e:
            self.logger.error(f"Exception fetching open orders: {e}")
            return None

    async def shutdown(self):
        """Graceful shutdown of the bot"""
        try:
            self.logger.info("Shutting down AI Trading Bot...")

            # Close WebSocket connection
            if self.websocket_handler:
                await self.websocket_handler.close()

            # Log final performance
            await self._log_performance_metrics()

            # Save configuration backup
            if self.config_manager and self.config:
                self.config_manager.save_config_backup()

            self.logger.info("Shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")


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
