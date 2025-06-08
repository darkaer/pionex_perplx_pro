"""
WebSocket Handler for Pionex API Integration
Handles both testing (mock) and production WebSocket connections
"""

import asyncio
import json
import logging
import hmac
import hashlib
import base64
import time
from typing import Dict, Any, Optional, Callable
from urllib.parse import urlencode
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI

class PionexWebSocketHandler:
    """Handles WebSocket connections to Pionex API with authentication"""

    def __init__(self, config, message_handler: Callable[[Dict], None] = None):
        self.config = config
        self.message_handler = message_handler
        self.websocket = None
        self.is_connected = False
        self.is_authenticated = False
        self.reconnect_count = 0
        self.logger = logging.getLogger(__name__)

        # Message handlers for different message types
        self.handlers = {
            'ticker': self._handle_ticker,
            'trade': self._handle_trade,
            'balance': self._handle_balance,
            'order': self._handle_order,
            'position': self._handle_position,
            'error': self._handle_error
        }

    async def connect(self, endpoint_type: str = "private"):
        """Connect to appropriate WebSocket endpoint"""
        try:
            url = self._get_websocket_url(endpoint_type)

            if self.config.trading.mode == "testing":
                await self._connect_mock(url)
            else:
                await self._connect_production(url, endpoint_type)

        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {str(e)}")
            await self._handle_reconnection()

    def _get_websocket_url(self, endpoint_type: str) -> str:
        """Get appropriate WebSocket URL based on mode and endpoint type"""
        if self.config.trading.mode == "testing":
            return self.config.websocket.mock_url

        if endpoint_type == "public":
            return self.config.websocket.public_url
        return self.config.websocket.production_url

    async def _connect_mock(self, url: str):
        """Connect to mock WebSocket for testing"""
        self.logger.info("Connecting to mock WebSocket endpoint...")

        try:
            # For testing, create a mock WebSocket that simulates responses
            self.websocket = MockWebSocket()
            self.is_connected = True
            self.is_authenticated = True
            self.logger.info("Connected to mock WebSocket successfully")

            # Start mock data generation
            asyncio.create_task(self._generate_mock_data())

        except Exception as e:
            self.logger.error(f"Mock WebSocket connection failed: {str(e)}")
            raise

    async def _connect_production(self, url: str, endpoint_type: str):
        """Connect to production Pionex WebSocket"""
        self.logger.info("Connecting to Pionex WebSocket...")

        try:
            # Add authentication parameters for private endpoints
            if endpoint_type == "private":
                url = self._add_auth_params(url)

            # Connect to WebSocket
            self.websocket = await websockets.connect(
                url,
                ping_interval=self.config.websocket.ping_interval,
                ping_timeout=10,
                close_timeout=10
            )

            self.is_connected = True
            self.logger.info("Connected to Pionex WebSocket successfully")

            # Authenticate if required
            if endpoint_type == "private":
                await self._authenticate()

        except Exception as e:
            self.logger.error(f"Production WebSocket connection failed: {str(e)}")
            raise

    def _add_auth_params(self, url: str) -> str:
        """Add authentication parameters to WebSocket URL"""
        timestamp = str(int(time.time() * 1000))

        # Create signature for authentication
        signature = self._generate_signature(
            method="GET",
            path="/ws",
            query_params={},
            timestamp=timestamp
        )

        # Add authentication parameters
        auth_params = {
            'apiKey': self.config.api.pionex_api_key,
            'timestamp': timestamp,
            'signature': signature
        }

        # Append to URL
        separator = '&' if '?' in url else '?'
        return f"{url}{separator}{urlencode(auth_params)}"

    def _generate_signature(self, method: str, path: str, query_params: Dict, timestamp: str) -> str:
        """Generate HMAC SHA256 signature for Pionex API"""
        # Prepare the string to sign
        query_string = urlencode(sorted(query_params.items()))
        string_to_sign = f"{method}{path}{query_string}{timestamp}"

        # Generate signature
        signature = hmac.new(
            self.config.api.pionex_secret_key.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    async def _authenticate(self):
        """Send authentication message for private WebSocket"""
        auth_message = {
            "method": "LOGIN",
            "params": {
                "apiKey": self.config.api.pionex_api_key,
                "timestamp": str(int(time.time() * 1000))
            },
            "id": 1
        }

        await self.send_message(auth_message)

        # Wait for authentication response
        auth_response = await self.websocket.recv()
        response_data = json.loads(auth_response)

        if response_data.get("result", {}).get("status") == "success":
            self.is_authenticated = True
            self.logger.info("WebSocket authentication successful")
        else:
            raise Exception(f"Authentication failed: {response_data}")

    async def send_message(self, message: Dict):
        """Send message through WebSocket"""
        if not self.is_connected or not self.websocket:
            raise Exception("WebSocket not connected")

        try:
            message_str = json.dumps(message)
            await self.websocket.send(message_str)
            self.logger.debug(f"Sent message: {message_str}")

        except Exception as e:
            self.logger.error(f"Failed to send message: {str(e)}")
            raise

    async def listen(self):
        """Listen for incoming messages"""
        while self.is_connected:
            try:
                if self.config.trading.mode == "testing":
                    # Mock listening - messages are generated automatically
                    await asyncio.sleep(1)
                    continue

                # Real WebSocket listening
                message = await self.websocket.recv()
                await self._process_message(message)

            except ConnectionClosed:
                self.logger.warning("WebSocket connection closed")
                self.is_connected = False
                await self._handle_reconnection()
                break

            except Exception as e:
                self.logger.error(f"Error receiving message: {str(e)}")
                await asyncio.sleep(1)

    async def _process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            self.logger.debug(f"Received message: {message}")

            # Determine message type and handle accordingly
            message_type = self._determine_message_type(data)

            if message_type in self.handlers:
                await self.handlers[message_type](data)

            # Call external message handler if provided
            if self.message_handler:
                self.message_handler(data)

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")

    def _determine_message_type(self, data: Dict) -> str:
        """Determine the type of incoming message"""
        if 'channel' in data:
            return data['channel'].split('.')[0]  # e.g., 'ticker.BTC-USDT' -> 'ticker'
        elif 'method' in data:
            return data['method'].lower()
        elif 'error' in data:
            return 'error'
        else:
            return 'unknown'

    async def _handle_ticker(self, data: Dict):
        """Handle ticker update messages"""
        self.logger.debug(f"Ticker update: {data}")

    async def _handle_trade(self, data: Dict):
        """Handle trade execution messages"""
        self.logger.info(f"Trade executed: {data}")

    async def _handle_balance(self, data: Dict):
        """Handle balance update messages"""
        self.logger.info(f"Balance update: {data}")

    async def _handle_order(self, data: Dict):
        """Handle order status messages"""
        self.logger.info(f"Order update: {data}")

    async def _handle_position(self, data: Dict):
        """Handle position update messages"""
        self.logger.info(f"Position update: {data}")

    async def _handle_error(self, data: Dict):
        """Handle error messages"""
        self.logger.error(f"WebSocket error: {data}")

    async def _handle_reconnection(self):
        """Handle WebSocket reconnection logic"""
        if self.reconnect_count >= self.config.websocket.reconnect_attempts:
            self.logger.error("Maximum reconnection attempts reached")
            return

        self.reconnect_count += 1
        wait_time = self.config.websocket.reconnect_delay * self.reconnect_count

        self.logger.info(f"Reconnecting in {wait_time} seconds (attempt {self.reconnect_count})")
        await asyncio.sleep(wait_time)

        try:
            await self.connect()
            self.reconnect_count = 0  # Reset on successful connection
        except Exception as e:
            self.logger.error(f"Reconnection failed: {str(e)}")

    async def _generate_mock_data(self):
        """Generate mock data for testing"""
        while self.is_connected and self.config.trading.mode == "testing":
            # Generate mock ticker data
            mock_ticker = {
                "channel": "ticker.BTC-USDT",
                "data": {
                    "symbol": "BTC-USDT",
                    "price": 45000 + (time.time() % 1000 - 500),  # Fluctuating price
                    "volume": 1234.56,
                    "change": 2.5
                }
            }

            if self.message_handler:
                self.message_handler(mock_ticker)

            await asyncio.sleep(5)  # Send mock data every 5 seconds

    async def close(self):
        """Close WebSocket connection"""
        self.is_connected = False

        if self.websocket and self.config.trading.mode != "testing":
            await self.websocket.close()

        self.logger.info("WebSocket connection closed")


class MockWebSocket:
    """Mock WebSocket for testing purposes"""

    def __init__(self):
        self.closed = False

    async def send(self, message: str):
        """Mock send method"""
        pass

    async def recv(self) -> str:
        """Mock receive method"""
        await asyncio.sleep(1)
        return json.dumps({"status": "mock_response"})

    async def close(self):
        """Mock close method"""
        self.closed = True
