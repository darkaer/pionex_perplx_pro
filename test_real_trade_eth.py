from dotenv import load_dotenv
load_dotenv()

import os
import logging
from pionex_python.restful.Orders import Orders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv('TRADING_BOT_PIONEX_API_KEY')
API_SECRET = os.getenv('TRADING_BOT_PIONEX_SECRET_KEY')

assert API_KEY and API_SECRET, "API credentials not set in environment!"

orders_client = Orders(API_KEY, API_SECRET)

order_payload = {
    "symbol": "ETH_USDT",
    "side": "BUY",
    "type": "MARKET",
    "amount": "12"  # Spend 12 USDT to buy ETH
}

logger.info(f"Placing test spot BUY order: {order_payload}")
try:
    result = orders_client.new_order(**order_payload)
    logger.info(f"Order response: {result}")
    print("Order response:", result)
except Exception as e:
    logger.error(f"Exception placing test order: {e}")
    print("Exception placing test order:", e) 