"""
Perplexity Labs API Integration for AI Trading Recommendations
Generates trading signals using advanced language models with real-time market data
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradingRecommendation:
    """Trading recommendation from AI analysis"""
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    risk_level: str  # LOW, MEDIUM, HIGH
    timestamp: datetime
    trading_pair: str

class PerplexityAPI:
    """Interface to Perplexity Labs API for trading analysis"""

    def __init__(self, config):
        self.config = config
        self.api_key = config.api.perplexity_api_key
        self.model = config.api.perplexity_model
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.logger = logging.getLogger(__name__)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests

    async def get_trading_recommendation(self, market_data: Dict[str, Any]) -> Optional[TradingRecommendation]:
        """Get AI trading recommendation based on market data"""
        try:
            # Rate limiting
            await self._handle_rate_limiting()

            if self.config.trading.mode == "testing" and not self.api_key:
                return self._generate_mock_recommendation(market_data)

            # Prepare the prompt with market data
            prompt = self._prepare_market_analysis_prompt(market_data)

            # Make API request
            response = await self._make_api_request(prompt)

            # Parse and validate response
            recommendation = self._parse_recommendation(response, market_data["trading_pair"])

            self.logger.info(f"Generated recommendation for {market_data['trading_pair']}: {recommendation.action} (confidence: {recommendation.confidence})")

            return recommendation

        except Exception as e:
            self.logger.error(f"Error getting trading recommendation: {str(e)}")
            return None

    async def assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current portfolio risk using AI analysis"""
        try:
            await self._handle_rate_limiting()

            if self.config.trading.mode == "testing" and not self.api_key:
                return self._generate_mock_risk_assessment(portfolio_data)

            # Prepare risk assessment prompt
            prompt = self._prepare_risk_assessment_prompt(portfolio_data)

            # Make API request
            response = await self._make_api_request(prompt)

            # Parse risk assessment
            risk_assessment = self._parse_risk_assessment(response)

            return risk_assessment

        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {str(e)}")
            return {"risk_level": "UNKNOWN", "recommendations": []}

    def _prepare_market_analysis_prompt(self, market_data: Dict[str, Any]) -> str:
        """Prepare market analysis prompt with current data"""
        prompt_template = self.config.ai_prompts.market_analysis_prompt

        # Fill in the template with actual market data
        filled_prompt = prompt_template.format(
            trading_pair=market_data.get("trading_pair", "BTC/USDT"),
            current_price=market_data.get("current_price", 0),
            volume_24h=market_data.get("volume_24h", 0),
            change_24h=market_data.get("change_24h", 0),
            rsi=market_data.get("rsi", "N/A"),
            macd=market_data.get("macd", "N/A"),
            bb_upper=market_data.get("bb_upper", "N/A"),
            bb_lower=market_data.get("bb_lower", "N/A"),
            support_level=market_data.get("support_level", "N/A"),
            resistance_level=market_data.get("resistance_level", "N/A"),
            news_headlines=market_data.get("news_headlines", "No recent news"),
            available_balance=market_data.get("available_balance", 0),
            open_positions=market_data.get("open_positions", 0),
            daily_pnl=market_data.get("daily_pnl", 0)
        )

        return filled_prompt

    def _prepare_risk_assessment_prompt(self, portfolio_data: Dict[str, Any]) -> str:
        """Prepare risk assessment prompt with portfolio data"""
        prompt_template = self.config.ai_prompts.risk_assessment_prompt

        filled_prompt = prompt_template.format(
            action=portfolio_data.get("proposed_action", "HOLD"),
            trading_pair=portfolio_data.get("trading_pair", "BTC/USDT"),
            position_size=portfolio_data.get("position_size", 0),
            leverage=portfolio_data.get("leverage", 1.0),
            volatility=portfolio_data.get("volatility", 0),
            daily_losses=portfolio_data.get("daily_losses", 0),
            consecutive_losses=portfolio_data.get("consecutive_losses", 0),
            max_drawdown=portfolio_data.get("max_drawdown", 0),
            open_positions_count=portfolio_data.get("open_positions_count", 0)
        )

        return filled_prompt

    async def _make_api_request(self, prompt: str) -> Dict[str, Any]:
        """Make request to Perplexity API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.config.ai_prompts.system_prompt
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": self.config.ai_prompts.max_response_tokens,
            "temperature": self.config.ai_prompts.temperature,
            "stream": False
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")

    def _parse_recommendation(self, api_response: Dict[str, Any], trading_pair: str) -> TradingRecommendation:
        """Parse API response into TradingRecommendation object"""
        try:
            # Extract the AI response text
            content = api_response["choices"][0]["message"]["content"]

            # Parse the structured response
            recommendation = self._extract_recommendation_fields(content)

            return TradingRecommendation(
                action=recommendation.get("action", "HOLD"),
                confidence=float(recommendation.get("confidence", 0.5)),
                entry_price=self._safe_float(recommendation.get("entry_price")),
                stop_loss=self._safe_float(recommendation.get("stop_loss")),
                take_profit=self._safe_float(recommendation.get("take_profit")),
                reasoning=recommendation.get("reasoning", "No reasoning provided"),
                risk_level=recommendation.get("risk_level", "MEDIUM"),
                timestamp=datetime.now(),
                trading_pair=trading_pair
            )

        except Exception as e:
            self.logger.error(f"Error parsing recommendation: {str(e)}")
            # Return default safe recommendation
            return TradingRecommendation(
                action="HOLD",
                confidence=0.0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                reasoning=f"Error parsing AI response: {str(e)}",
                risk_level="HIGH",
                timestamp=datetime.now(),
                trading_pair=trading_pair
            )

    def _extract_recommendation_fields(self, content: str) -> Dict[str, Any]:
        """Extract structured fields from AI response text"""
        fields = {}
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()

                if key in ['action', 'confidence', 'entry_price', 'stop_loss', 'take_profit', 'reasoning', 'risk_level']:
                    fields[key] = value

        # Handle common variations
        if 'buy' in content.lower():
            fields['action'] = 'BUY'
        elif 'sell' in content.lower():
            fields['action'] = 'SELL'
        else:
            fields['action'] = 'HOLD'

        return fields

    def _parse_risk_assessment(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse risk assessment response"""
        try:
            content = api_response["choices"][0]["message"]["content"]

            # Extract risk level and recommendations
            risk_assessment = {
                "risk_level": "MEDIUM",
                "recommendations": [],
                "analysis": content
            }

            # Simple parsing for risk level
            content_lower = content.lower()
            if 'high risk' in content_lower or 'dangerous' in content_lower:
                risk_assessment["risk_level"] = "HIGH"
            elif 'low risk' in content_lower or 'safe' in content_lower:
                risk_assessment["risk_level"] = "LOW"

            return risk_assessment

        except Exception as e:
            self.logger.error(f"Error parsing risk assessment: {str(e)}")
            return {
                "risk_level": "HIGH",
                "recommendations": ["Error in risk assessment"],
                "analysis": f"Error: {str(e)}"
            }

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        try:
            # Remove any non-numeric characters except decimal point
            if isinstance(value, str):
                value = ''.join(c for c in value if c.isdigit() or c == '.')
            return float(value)
        except (ValueError, TypeError):
            return None

    async def _handle_rate_limiting(self):
        """Handle API rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            await asyncio.sleep(wait_time)

        self.last_request_time = asyncio.get_event_loop().time()

    def _generate_mock_recommendation(self, market_data: Dict[str, Any]) -> TradingRecommendation:
        """Generate mock recommendation for testing"""
        import random

        actions = ["BUY", "SELL", "HOLD"]
        action = random.choice(actions)

        # Generate realistic mock data
        current_price = market_data.get("current_price", 45000)

        mock_rec = TradingRecommendation(
            action=action,
            confidence=round(random.uniform(0.6, 0.9), 2),
            entry_price=current_price if action != "HOLD" else None,
            stop_loss=current_price * 0.98 if action == "BUY" else current_price * 1.02 if action == "SELL" else None,
            take_profit=current_price * 1.04 if action == "BUY" else current_price * 0.96 if action == "SELL" else None,
            reasoning=f"Mock {action} recommendation based on simulated analysis",
            risk_level=random.choice(["LOW", "MEDIUM"]),
            timestamp=datetime.now(),
            trading_pair=market_data.get("trading_pair", "BTC/USDT")
        )

        self.logger.info(f"Generated mock recommendation: {action} (confidence: {mock_rec.confidence})")
        return mock_rec

    def _generate_mock_risk_assessment(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock risk assessment for testing"""
        return {
            "risk_level": "LOW",
            "recommendations": ["Mock risk assessment - safe to proceed"],
            "analysis": "Mock analysis indicates acceptable risk levels for testing mode"
        }
