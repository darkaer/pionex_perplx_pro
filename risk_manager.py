"""
Risk Management System for AI Trading Bot
Implements comprehensive risk controls and position management
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    side: str  # BUY or SELL
    size: float
    entry_price: float
    current_price: float
    leverage: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def pnl(self) -> float:
        """Calculate current P&L"""
        if self.side == "BUY":
            return (self.current_price - self.entry_price) * self.size * self.leverage
        else:
            return (self.entry_price - self.current_price) * self.size * self.leverage

    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L as percentage"""
        if self.entry_price == 0:
            return 0
        return (self.pnl / (self.entry_price * self.size)) * 100

@dataclass
class RiskMetrics:
    """Risk metrics for the trading account"""
    total_balance: float
    available_balance: float
    total_pnl: float
    daily_pnl: float
    max_drawdown: float
    win_rate: float
    consecutive_losses: int
    total_positions: int
    risk_level: RiskLevel

class RiskManager:
    """Comprehensive risk management system"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Risk tracking
        self.positions: Dict[str, Position] = {}
        self.daily_trades: List[Dict] = []
        self.daily_start_balance = 0
        self.emergency_stop_triggered = False
        self.consecutive_losses = 0
        self.max_drawdown = 0
        self.starting_balance = 0

        # Performance tracking
        self.trade_history: List[Dict] = []
        self.daily_pnl_history: List[float] = []

    def initialize(self, starting_balance: float):
        """Initialize risk manager with starting balance"""
        self.starting_balance = starting_balance
        self.daily_start_balance = starting_balance
        self.logger.info(f"Risk manager initialized with balance: ${starting_balance:,.2f}")

    def can_open_position(self, symbol: str, position_size: float, leverage: float = 1.0) -> tuple[bool, str]:
        """Check if a new position can be opened based on risk rules"""

        # Check emergency stop
        if self.emergency_stop_triggered:
            return False, "Emergency stop is active"

        # Check consecutive losses
        if self.consecutive_losses >= self.config.risk_management.max_consecutive_losses:
            return False, f"Maximum consecutive losses reached ({self.consecutive_losses})"

        # Check daily loss limit
        daily_pnl = self.calculate_daily_pnl()
        daily_loss_limit = self.daily_start_balance * (self.config.risk_management.max_daily_loss_percentage / 100)

        if daily_pnl <= -daily_loss_limit:
            return False, f"Daily loss limit exceeded: {daily_pnl:.2f}"

        # Check position size limits
        max_position_value = self.starting_balance * self.config.risk_management.max_drawdown_percentage / 100
        position_value = position_size * leverage

        if position_value > max_position_value:
            return False, f"Position size too large: ${position_value:.2f} > ${max_position_value:.2f}"

        # Check maximum drawdown
        current_drawdown = self.calculate_drawdown()
        if current_drawdown >= self.config.risk_management.max_drawdown_percentage:
            return False, f"Maximum drawdown reached: {current_drawdown:.2f}%"

        # Check if position already exists
        if symbol in self.positions:
            return False, f"Position already exists for {symbol}"

        return True, "Position approved"

    def validate_stop_loss(self, entry_price: float, stop_loss: float, side: str) -> tuple[bool, str]:
        """Validate stop loss level"""
        if side == "BUY":
            if stop_loss >= entry_price:
                return False, "Stop loss must be below entry price for BUY orders"
            loss_percentage = ((entry_price - stop_loss) / entry_price) * 100
        else:
            if stop_loss <= entry_price:
                return False, "Stop loss must be above entry price for SELL orders"
            loss_percentage = ((stop_loss - entry_price) / entry_price) * 100

        max_stop_loss = self.config.trading.stop_loss_percentage * 2  # Allow up to 2x configured stop loss
        if loss_percentage > max_stop_loss:
            return False, f"Stop loss too wide: {loss_percentage:.2f}% > {max_stop_loss:.2f}%"

        return True, "Stop loss valid"

    def add_position(self, symbol: str, side: str, size: float, entry_price: float, 
                    leverage: float = 1.0, stop_loss: float = None, take_profit: float = None) -> bool:
        """Add a new position"""
        try:
            position = Position(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            self.positions[symbol] = position
            self.logger.info(f"Added position: {side} {size} {symbol} @ ${entry_price}")

            return True

        except Exception as e:
            self.logger.error(f"Error adding position: {str(e)}")
            return False

    def update_position_price(self, symbol: str, current_price: float):
        """Update position with current market price"""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price

            # Check for stop loss or take profit triggers
            self._check_position_exits(symbol)

    def _check_position_exits(self, symbol: str):
        """Check if position should be closed due to stop loss or take profit"""
        position = self.positions[symbol]

        should_close = False
        reason = ""

        # Check stop loss
        if position.stop_loss:
            if position.side == "BUY" and position.current_price <= position.stop_loss:
                should_close = True
                reason = "Stop loss triggered"
            elif position.side == "SELL" and position.current_price >= position.stop_loss:
                should_close = True
                reason = "Stop loss triggered"

        # Check take profit
        if position.take_profit and not should_close:
            if position.side == "BUY" and position.current_price >= position.take_profit:
                should_close = True
                reason = "Take profit triggered"
            elif position.side == "SELL" and position.current_price <= position.take_profit:
                should_close = True
                reason = "Take profit triggered"

        # Check position timeout
        position_age = datetime.now() - position.timestamp
        max_age = timedelta(hours=self.config.risk_management.position_timeout_hours)

        if position_age > max_age and not should_close:
            should_close = True
            reason = "Position timeout"

        if should_close:
            self.logger.warning(f"Position {symbol} should be closed: {reason}")
            # Automatically close the position
            self.close_position(symbol, position.current_price, reason)

    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual close") -> Optional[Dict]:
        """Close a position and record the trade"""
        if symbol not in self.positions:
            self.logger.warning(f"Cannot close position {symbol}: position not found")
            return None

        position = self.positions[symbol]
        position.current_price = exit_price

        # Calculate final P&L
        final_pnl = position.pnl

        # Record trade
        trade_record = {
            "symbol": symbol,
            "side": position.side,
            "size": position.size,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "leverage": position.leverage,
            "pnl": final_pnl,
            "pnl_percentage": position.pnl_percentage,
            "duration": datetime.now() - position.timestamp,
            "reason": reason,
            "timestamp": datetime.now()
        }

        self.trade_history.append(trade_record)
        self.daily_trades.append(trade_record)

        # Update consecutive losses
        if final_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Remove position
        del self.positions[symbol]

        self.logger.info(f"Closed position {symbol}: P&L ${final_pnl:.2f} ({reason})")

        # Check for emergency stop conditions
        self._check_emergency_conditions()

        return trade_record

    def calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        daily_pnl = sum(trade["pnl"] for trade in self.daily_trades)

        # Add unrealized P&L from open positions
        for position in self.positions.values():
            daily_pnl += position.pnl

        return daily_pnl

    def calculate_drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        current_balance = self.starting_balance + sum(trade["pnl"] for trade in self.trade_history)

        if self.starting_balance == 0:
            return 0

        drawdown = ((self.starting_balance - current_balance) / self.starting_balance) * 100

        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        return drawdown

    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        total_pnl = sum(trade["pnl"] for trade in self.trade_history)
        daily_pnl = self.calculate_daily_pnl()

        # Calculate win rate
        profitable_trades = len([t for t in self.trade_history if t["pnl"] > 0])
        total_trades = len(self.trade_history)
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

        # Determine risk level
        risk_level = self._assess_risk_level()

        return RiskMetrics(
            total_balance=self.starting_balance + total_pnl,
            available_balance=self.starting_balance + total_pnl - sum(p.size * p.entry_price for p in self.positions.values()),
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            max_drawdown=self.max_drawdown,
            win_rate=win_rate,
            consecutive_losses=self.consecutive_losses,
            total_positions=len(self.positions),
            risk_level=risk_level
        )

    def _assess_risk_level(self) -> RiskLevel:
        """Assess current risk level"""
        if self.emergency_stop_triggered:
            return RiskLevel.CRITICAL

        drawdown = self.calculate_drawdown()
        daily_loss_pct = abs(self.calculate_daily_pnl() / self.daily_start_balance * 100)

        if (drawdown >= self.config.risk_management.max_drawdown_percentage * 0.8 or 
            daily_loss_pct >= self.config.risk_management.max_daily_loss_percentage * 0.8 or
            self.consecutive_losses >= self.config.risk_management.max_consecutive_losses - 1):
            return RiskLevel.HIGH

        if (drawdown >= self.config.risk_management.max_drawdown_percentage * 0.5 or
            daily_loss_pct >= self.config.risk_management.max_daily_loss_percentage * 0.5 or
            self.consecutive_losses >= 2):
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _check_emergency_conditions(self):
        """Check if emergency stop should be triggered"""
        if not self.config.risk_management.emergency_stop_enabled:
            return

        # Check daily loss limit
        daily_pnl = self.calculate_daily_pnl()
        daily_loss_limit = self.daily_start_balance * (self.config.risk_management.max_daily_loss_percentage / 100)

        if daily_pnl <= -daily_loss_limit:
            self._trigger_emergency_stop("Daily loss limit exceeded")
            return

        # Check max drawdown
        if self.calculate_drawdown() >= self.config.risk_management.max_drawdown_percentage:
            self._trigger_emergency_stop("Maximum drawdown reached")
            return

        # Check consecutive losses
        if self.consecutive_losses >= self.config.risk_management.max_consecutive_losses:
            self._trigger_emergency_stop("Maximum consecutive losses reached")
            return

    def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        if not self.emergency_stop_triggered:
            self.emergency_stop_triggered = True
            self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")

            # In a full implementation, this would:
            # 1. Close all open positions
            # 2. Cancel all pending orders
            # 3. Send notifications
            # 4. Log the event

    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of new trading day)"""
        self.daily_trades.clear()
        self.daily_start_balance = self.starting_balance + sum(trade["pnl"] for trade in self.trade_history)
        self.logger.info("Daily metrics reset")

    def can_resume_trading(self) -> tuple[bool, str]:
        """Check if trading can be resumed after emergency stop"""
        if not self.emergency_stop_triggered:
            return True, "No emergency stop active"

        # Implement conditions for resuming trading
        # For now, require manual reset
        return False, "Emergency stop active - manual reset required"

    def reset_emergency_stop(self):
        """Manually reset emergency stop"""
        self.emergency_stop_triggered = False
        self.consecutive_losses = 0
        self.logger.warning("Emergency stop manually reset")

    def update_balance(self, new_balance: float):
        """Update the starting and daily balance for real-time syncing"""
        self.starting_balance = new_balance
        self.daily_start_balance = new_balance
        self.logger.info(f"Risk manager balance updated to: ${new_balance:,.2f}")
