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
import os

logger = logging.getLogger(__name__)

class PionexWebSocketHandler:
    """
    Proper Pionex WebSocket handler with authentication
    Fixes HTTP 200 error by implementing correct authentication
    """

    def __init__(self, api_key: str = None, api_secret: str = None, mode: str = "production"):
        # Allow API key/secret from env if not provided
        self.api_key = api_key or os.getenv('PIONEX_API_KEY')
        self.api_secret = api_secret or os.getenv('PIONEX_API_SECRET')
        self.mode = mode
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.subscriptions = set()

    def generate_authenticated_url(self) -> str:
        """
        Generate properly authenticated Pionex WebSocket URL
        This fixes the HTTP 200 error by including required authentication parameters
        """
        if self.mode == "testing":
            # For testing mode, use public stream (no auth required)
            return "wss://ws.pionex.com/wsPub"

        # Production mode requires authentication
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials required for production mode")

        # Step 1: Get current timestamp in milliseconds
        timestamp = str(int(time.time() * 1000))

        # Step 2: Create query parameters
        params = {
            'key': self.api_key,
            'timestamp': timestamp
        }

        # Step 3: Sort parameters by key in ascending ASCII order
        sorted_params = sorted(params.items())
        query_string = urlencode(sorted_params)

        # Step 4: Create PATH_URL
        path = "/ws"
        path_url = f"{path}?{query_string}"

        # Step 5: Concatenate "websocket_auth"
        message_to_sign = f"{path_url}websocket_auth"

        # Step 6: Generate HMAC SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'), 
            message_to_sign.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()

        # Step 7: Create final authenticated URL
        authenticated_url = f"wss://ws.pionex.com/ws?key={self.api_key}&timestamp={timestamp}&signature={signature}"

        logger.info(f"Generated authenticated WebSocket URL")
        logger.debug(f"Message signed: {message_to_sign}")
        logger.debug(f"Signature: {signature}")

        return authenticated_url

    async def connect(self) -> bool:
        """
        Connect to Pionex WebSocket with proper authentication
        Returns True if connection successful, False otherwise
        """
        try:
            # Generate authenticated URL
            ws_url = self.generate_authenticated_url()
            # logger.info(f"Connecting to WebSocket URL: {ws_url}")

            # Connect with proper headers
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=10,   # Wait 10 seconds for pong
                close_timeout=10
            )

            self.is_connected = True
            logger.info("✅ WebSocket connection established successfully!")

            # Start message handler
            asyncio.create_task(self._message_handler())

            return True

        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 200:
                logger.error("❌ HTTP 200 Error - Authentication failed!")
                logger.error("Check your API credentials and signature generation")
            else:
                logger.error(f"❌ WebSocket connection failed with status {e.status_code}: {e}")
            return False

        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            return False

    async def disconnect(self):
        """Gracefully disconnect from WebSocket"""
        if self.websocket and self.is_connected:
            try:
                await self.websocket.close()
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
            finally:
                self.is_connected = False
                self.websocket = None

    async def subscribe(self, topic: str, symbol: str):
        """
        Subscribe to a topic for a specific symbol

        Args:
            topic: TRADE, DEPTH, ORDER, or FILL
            symbol: Trading pair (e.g., "BTC_USDT")
        """
        if not self.is_connected:
            logger.error("Not connected to WebSocket")
            return False

        subscription_msg = {
            "op": "SUBSCRIBE",
            "topic": topic,
            "symbol": symbol
        }

        try:
            await self.websocket.send(json.dumps(subscription_msg))
            self.subscriptions.add(f"{topic}:{symbol}")
            logger.info(f"Subscribed to {topic} for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to {topic}:{symbol} - {e}")
            return False

    async def unsubscribe(self, topic: str, symbol: str):
        """Unsubscribe from a topic"""
        if not self.is_connected:
            logger.error("Not connected to WebSocket")
            return False

        unsubscribe_msg = {
            "op": "UNSUBSCRIBE", 
            "topic": topic,
            "symbol": symbol
        }

        try:
            await self.websocket.send(json.dumps(unsubscribe_msg))
            self.subscriptions.discard(f"{topic}:{symbol}")
            logger.info(f"Unsubscribed from {topic} for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {topic}:{symbol} - {e}")
            return False

    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                await self._process_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            self.is_connected = False

    async def _process_message(self, message: str):
        """Process individual messages"""
        try:
            data = json.loads(message)

            # Handle different message types
            if data.get("type") == "PING":
                # Respond to server ping
                pong_response = {"type": "PONG", "timestamp": int(time.time() * 1000)}
                await self.websocket.send(json.dumps(pong_response))
                logger.debug("Responded to PING with PONG")

            elif data.get("op") in ["SUBSCRIBE", "UNSUBSCRIBE"]:
                # Subscription confirmation
                logger.info(f"Subscription response: {data}")

            elif data.get("topic"):
                # Market data or account data
                await self._handle_data_message(data)

            else:
                logger.debug(f"Unknown message type: {data}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _handle_data_message(self, data: Dict[str, Any]):
        """Handle market data and account data messages"""
        topic = data.get("topic")
        symbol = data.get("symbol")

        if topic == "TRADE":
            logger.info(f"Trade data for {symbol}: {data.get('data')}")
        elif topic == "DEPTH":
            logger.info(f"Depth data for {symbol}: {data.get('data')}")
        elif topic == "ORDER":
            logger.info(f"Order update: {data.get('data')}")
        elif topic == "FILL":
            logger.info(f"Fill update: {data.get('data')}")
        else:
            logger.info(f"Data message: {data}")

# Example usage and testing function
async def test_websocket_connection():
    """Test the WebSocket connection with proper authentication"""
    # Get API credentials from environment variables
    api_key = os.getenv('PIONEX_API_KEY')
    api_secret = os.getenv('PIONEX_API_SECRET')

    if not api_key or not api_secret:
        print("❌ API credentials not found in environment variables")
        print("Set PIONEX_API_KEY and PIONEX_API_SECRET environment variables")
        return

    # Create WebSocket handler
    ws_handler = PionexWebSocketHandler(api_key, api_secret, mode="production")

    try:
        # Test connection
        print("Testing WebSocket connection...")
        success = await ws_handler.connect()

        if success:
            print("✅ Connection successful! Testing subscriptions...")

            # Test subscription to market data
            await ws_handler.subscribe("TRADE", "BTC_USDT")

            # Keep connection alive for 10 seconds to receive messages
            await asyncio.sleep(10)

            # Clean disconnect
            await ws_handler.disconnect()
            print("✅ Test completed successfully")
        else:
            print("❌ Connection failed - check logs above for details")

    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run test
    asyncio.run(test_websocket_connection())
