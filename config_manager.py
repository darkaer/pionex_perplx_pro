"""
Configuration Manager for AI Trading Bot
Handles YAML configuration files, environment variables, and mode switching
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import copy

@dataclass
class AIPromptConfig:
    """Configuration for AI prompts and behavior"""
    system_prompt: str = ""
    market_analysis_prompt: str = ""
    risk_assessment_prompt: str = ""
    confidence_threshold: float = 0.7
    max_response_tokens: int = 1000
    temperature: float = 0.3
    analysis_interval: int = 300  # Default analysis interval in seconds

@dataclass
class TradingConfig:
    """Trading parameters and settings"""
    mode: str = "testing"  # testing or production
    trading_pairs: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    max_position_size: float = 0.1  # 10% of balance
    default_leverage: float = 1.0
    stop_loss_percentage: float = 4.0
    take_profit_percentage: float = 4.0
    order_timeout_minutes: int = 30

@dataclass
class RiskManagementConfig:
    """Risk management parameters"""
    max_daily_loss_percentage: float = 5.0
    max_consecutive_losses: int = 3
    max_drawdown_percentage: float = 10.0
    position_timeout_hours: int = 24
    emergency_stop_enabled: bool = True
    volatility_threshold: float = 5.0

@dataclass
class APIConfig:
    """API configuration for external services"""
    perplexity_api_key: str = ""
    perplexity_model: str = "sonar-pro"
    pionex_api_key: str = ""
    pionex_secret_key: str = ""
    pionex_base_url: str = "https://api.pionex.com"
    pionex_ws_url: str = "wss://ws.pionex.com/ws"

@dataclass
class WebSocketConfig:
    """WebSocket connection configuration"""
    mock_url: str = "ws://localhost:8080/mock"
    production_url: str = "wss://ws.pionex.com/ws"
    public_url: str = "wss://ws.pionex.com/ws/Pub"
    ping_interval: int = 30
    reconnect_attempts: int = 5
    reconnect_delay: int = 5

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file_enabled: bool = True
    file_path: str = "logs/trading_bot.log"
    console_enabled: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5

@dataclass
class NotificationConfig:
    """Notification settings"""
    discord_webhook_url: str = ""
    email_enabled: bool = False
    email_smtp_server: str = ""
    email_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)

@dataclass
class BotConfig:
    """Complete bot configuration"""
    ai_prompts: AIPromptConfig = field(default_factory=AIPromptConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    api: APIConfig = field(default_factory=APIConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

class ConfigManager:
    """Manages configuration loading, validation, and environment overrides"""

    def __init__(self, config_path: str = "config", env_prefix: str = "TRADING_BOT"):
        self.config_path = Path(config_path)
        self.env_prefix = env_prefix
        self.config: Optional[BotConfig] = None
        self.logger = logging.getLogger(__name__)

    def load_config(self, mode: str = "testing") -> BotConfig:
        """Load configuration based on mode (testing/production)"""
        try:
            # Load base configuration
            base_config_file = self.config_path / "base_config.yaml"
            base_config = self._load_yaml_file(base_config_file)

            # Load mode-specific configuration
            mode_config_file = self.config_path / f"{mode}_config.yaml"
            mode_config = self._load_yaml_file(mode_config_file)

            # Merge configurations
            merged_config = self._merge_configs(base_config, mode_config)

            # Apply environment variable overrides
            merged_config = self._apply_env_overrides(merged_config)

            # Convert to dataclass
            self.config = self._dict_to_dataclass(merged_config)

            # Validate configuration
            self._validate_config()

            self.logger.info(f"Configuration loaded successfully for mode: {mode}")
            return self.config

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not file_path.exists():
            self.logger.warning(f"Configuration file not found: {file_path}")
            return {}

        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            self.logger.error(f"Error loading YAML file {file_path}: {str(e)}")
            raise

    def _merge_configs(self, base_config: Dict, mode_config: Dict) -> Dict:
        """Merge base and mode-specific configurations"""
        merged = copy.deepcopy(base_config)

        def deep_merge(target: Dict, source: Dict):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value

        deep_merge(merged, mode_config)
        return merged

    def _apply_env_overrides(self, config: Dict) -> Dict:
        """Apply environment variable overrides"""
        env_mappings = {
            # API credentials
            f"{self.env_prefix}_PERPLEXITY_API_KEY": ["api", "perplexity_api_key"],
            f"{self.env_prefix}_PIONEX_API_KEY": ["api", "pionex_api_key"],
            f"{self.env_prefix}_PIONEX_SECRET_KEY": ["api", "pionex_secret_key"],
            # Trading config overrides
            f"{self.env_prefix}_TRADING_MODE": ["trading", "mode"],
            f"{self.env_prefix}_MAX_DAILY_LOSS": ["risk_management", "max_daily_loss_percentage"],
            f"{self.env_prefix}_MAX_POSITION_SIZE": ["trading", "max_position_size"],
            f"{self.env_prefix}_DEFAULT_LEVERAGE": ["trading", "default_leverage"],
            f"{self.env_prefix}_STOP_LOSS": ["trading", "stop_loss_percentage"],
            f"{self.env_prefix}_TAKE_PROFIT": ["trading", "take_profit_percentage"],
            # Risk management overrides
            f"{self.env_prefix}_MAX_CONSECUTIVE_LOSSES": ["risk_management", "max_consecutive_losses"],
            f"{self.env_prefix}_MAX_DRAWDOWN": ["risk_management", "max_drawdown_percentage"],
            f"{self.env_prefix}_POSITION_TIMEOUT": ["risk_management", "position_timeout_hours"],
            f"{self.env_prefix}_EMERGENCY_STOP": ["risk_management", "emergency_stop_enabled"],
            # Notification settings
            f"{self.env_prefix}_DISCORD_WEBHOOK": ["notifications", "discord_webhook_url"],
            # Advanced settings
            f"{self.env_prefix}_PERPLEXITY_MODEL": ["api", "perplexity_model"],
            f"{self.env_prefix}_LOG_LEVEL": ["logging", "level"],
            f"{self.env_prefix}_WS_PING_INTERVAL": ["websocket", "ping_interval"],
            f"{self.env_prefix}_WS_RECONNECT_ATTEMPTS": ["websocket", "reconnect_attempts"],
            f"{self.env_prefix}_ANALYSIS_INTERVAL": ["ai_prompts", "analysis_interval"],
        }

        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(config, path, self._convert_env_value(value))

        return config

    def _set_nested_value(self, config: Dict, path: List[str], value: Any):
        """Set nested dictionary value"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Try to convert to boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'

        # Return as string
        return value

    def _dict_to_dataclass(self, config_dict: Dict) -> BotConfig:
        """Convert dictionary to BotConfig dataclass"""
        try:
            return BotConfig(
                ai_prompts=AIPromptConfig(**config_dict.get('ai_prompts', {})),
                trading=TradingConfig(**config_dict.get('trading', {})),
                risk_management=RiskManagementConfig(**config_dict.get('risk_management', {})),
                api=APIConfig(**config_dict.get('api', {})),
                websocket=WebSocketConfig(**config_dict.get('websocket', {})),
                logging=LoggingConfig(**config_dict.get('logging', {})),
                notifications=NotificationConfig(**config_dict.get('notifications', {}))
            )
        except Exception as e:
            self.logger.error(f"Error converting config to dataclass: {str(e)}")
            raise

    def _validate_config(self):
        """Validate configuration parameters"""
        if not self.config:
            raise ValueError("Configuration not loaded")

        # Validate API keys for production mode (warn only during validation)
        if self.config.trading.mode == "production":
            if not self.config.api.perplexity_api_key:
                self.logger.warning("Perplexity API key not set - required for production trading")
            if not self.config.api.pionex_api_key or not self.config.api.pionex_secret_key:
                self.logger.warning("Pionex API credentials not set - required for production trading")

        # Validate risk parameters
        if self.config.risk_management.max_daily_loss_percentage <= 0 or self.config.risk_management.max_daily_loss_percentage > 50:
            raise ValueError("Max daily loss percentage must be between 0 and 50")

        if self.config.trading.max_position_size <= 0 or self.config.trading.max_position_size > 1:
            raise ValueError("Max position size must be between 0 and 1")

        # Validate trading pairs
        if not self.config.trading.trading_pairs:
            raise ValueError("At least one trading pair must be configured")

        self.logger.info("Configuration validation passed")

    def get_websocket_url(self, endpoint_type: str = "private") -> str:
        """Get appropriate WebSocket URL based on mode and endpoint type"""
        if self.config.trading.mode == "testing":
            return self.config.websocket.mock_url

        if endpoint_type == "public":
            return self.config.websocket.public_url
        return self.config.websocket.production_url

    def save_config_backup(self, backup_name: str = None):
        """Save current configuration as backup"""
        if not self.config:
            raise ValueError("No configuration loaded to backup")

        if not backup_name:
            from datetime import datetime
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_path = self.config_path / "backups" / f"{backup_name}.yaml"
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass back to dict
        config_dict = self._dataclass_to_dict(self.config)

        with open(backup_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration backup saved: {backup_path}")

    def _dataclass_to_dict(self, obj) -> Dict:
        """Convert dataclass to dictionary"""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, field_def in obj.__dataclass_fields__.items():
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    result[field_name] = self._dataclass_to_dict(value)
                else:
                    result[field_name] = value
            return result
        return obj
