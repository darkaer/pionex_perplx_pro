# AI Trading Bot: Perplexity Labs + Pionex Integration

A sophisticated cryptocurrency trading bot that leverages Perplexity Labs' advanced AI capabilities with Pionex's automated trading infrastructure for intelligent, risk-managed trading operations.

## 🚀 Key Features

- **AI-Powered Analysis**: Integration with Perplexity Labs for real-time market analysis and trading recommendations
- **Risk Management**: Multi-layer risk controls including stop-loss, position sizing, and emergency stops
- **Dual Mode Operation**: Safe testing mode with paper trading and production mode for live trading
- **WebSocket Integration**: Real-time market data processing and trade execution
- **Configurable AI Prompts**: Customizable AI behavior and analysis parameters
- **Comprehensive Monitoring**: Real-time performance tracking and logging

## 📋 System Requirements

- Python 3.8 or higher
- Internet connection for API access
- Perplexity Labs API key (for AI analysis)
- Pionex account and API credentials (for live trading)

## 🔧 Installation

### Quick Start (Recommended)

```bash
# Clone or download the bot files
# Navigate to the bot directory

# Run the setup script
python start.py setup

# Test the installation
python start.py validate

# Run in testing mode
python start.py run
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env

# Edit .env with your API keys (for production)
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.template` to `.env` and configure:

```bash
# Required for production mode
TRADING_BOT_PERPLEXITY_API_KEY=your_perplexity_api_key
TRADING_BOT_PIONEX_API_KEY=your_pionex_api_key  
TRADING_BOT_PIONEX_SECRET_KEY=your_pionex_secret_key

# Optional overrides
TRADING_BOT_TRADING_MODE=testing
TRADING_BOT_MAX_DAILY_LOSS=5.0
TRADING_BOT_MAX_POSITION_SIZE=0.1
```

### Configuration Files

The bot uses YAML configuration files in the `config/` directory:

- `base_config.yaml`: Base settings shared across all modes
- `testing_config.yaml`: Safe settings for paper trading
- `production_config.yaml`: Optimized settings for live trading

### AI Prompt Customization

Customize the AI behavior by editing the prompts in `base_config.yaml`:

```yaml
ai_prompts:
  system_prompt: |
    You are a professional cryptocurrency trading AI...

  market_analysis_prompt: |
    Analyze the current market conditions for {trading_pair}...

  confidence_threshold: 0.7
  temperature: 0.3
```

## 🎮 Usage

### Testing Mode (Safe - No Real Money)

```bash
# Quick start in testing mode
python start.py run

# Or directly with main.py
python main.py --mode testing

# Validate configuration only
python main.py --mode testing --validate-only
```

### Production Mode (Live Trading)

```bash
# Production mode with confirmation
python start.py run --mode production

# Direct execution
python main.py --mode production
```

⚠️ **Warning**: Production mode trades with real money. Always test thoroughly first.

### Debug and Testing

```bash
# Run comprehensive tests
python debug.py

# Test specific components
python start.py test
```

## 📊 Risk Management Features

### Position-Level Controls
- Configurable stop-loss percentages
- Automatic take-profit execution  
- Dynamic position sizing based on confidence
- Maximum leverage limits

### Account-Level Protection
- Daily loss limits with automatic halt
- Maximum drawdown protection
- Consecutive loss detection
- Position timeout mechanisms

### System-Level Safety
- Emergency stop functionality
- API rate limiting compliance
- Graceful error handling and recovery
- Comprehensive audit logging

## 🔌 API Integration

### Perplexity Labs
- Real-time market analysis using advanced language models
- Customizable prompts for trading strategies
- Confidence scoring and risk assessment
- Support for multiple AI models

### Pionex WebSocket
- Real-time market data feeds
- Authenticated private data access
- Automatic reconnection handling
- Support for both testing and production endpoints

## 📁 File Structure

```
ai-trading-bot/
├── main.py                    # Main application entry point
├── config_manager.py          # Configuration management system
├── perplexity_api.py          # Perplexity Labs API integration
├── websocket_handler.py       # Pionex WebSocket handler
├── risk_manager.py            # Risk management system
├── start.py                   # Startup and setup script
├── debug.py                   # Testing and debugging utilities
├── requirements.txt           # Python dependencies
├── .env.template              # Environment variable template
├── config/
│   ├── base_config.yaml       # Base configuration
│   ├── testing_config.yaml    # Testing mode settings
│   ├── production_config.yaml # Production mode settings
│   └── backups/               # Configuration backups
└── logs/                      # Application logs
```

## 🧪 Testing

The bot includes comprehensive testing capabilities:

```bash
# Run all tests
python debug.py

# Individual test categories
python start.py validate     # Configuration validation
python start.py test        # Quick functionality test
```

### Test Coverage
- Configuration loading and validation
- API integration testing
- WebSocket connection testing
- Risk management validation
- Full system integration testing

## 📈 Performance Monitoring

The bot provides real-time monitoring including:

- Total P&L tracking
- Daily performance metrics
- Win rate calculations
- Maximum drawdown analysis
- Risk level assessment
- Position management status

### Logging
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- File-based logging with rotation
- Console output for real-time monitoring
- Detailed trade execution logs

## ⚡ Trading Modes Comparison

| Feature | Testing Mode | Production Mode |
|---------|-------------|----------------|
| Real Money | ❌ Paper Trading | ✅ Live Trading |
| API Keys Required | ❌ Optional | ✅ Required |
| Position Size | 5% max | 15% max |
| Leverage | 1x only | Up to 2x |
| Risk Controls | Very Conservative | Balanced |
| Confidence Threshold | 80% | 65% |
| Logging Level | DEBUG | INFO |

## 🛡️ Security Best Practices

- Store API keys in environment variables, never in code
- Use IP whitelisting on exchange APIs when available
- Enable 2FA on all exchange accounts
- Regularly rotate API keys
- Monitor logs for suspicious activity
- Start with small position sizes in production

## 🆘 Troubleshooting

### Common Issues

**Configuration Errors**
```bash
python start.py validate
python debug.py
```

**API Connection Issues**
- Verify API keys in `.env` file
- Check API key permissions on exchanges
- Ensure IP whitelisting is configured

**WebSocket Connection Failures**
- Check internet connectivity
- Verify exchange API status
- Review firewall settings

### Emergency Procedures

**Stop All Trading**
- Press Ctrl+C to stop the bot gracefully
- Emergency stop triggers automatically on risk limits
- Manual reset available through risk manager

**Reset Configuration**
```bash
# Restore from backup
cp config/backups/backup_YYYYMMDD_HHMMSS.yaml config/base_config.yaml
```

## 📞 Support and Contributing

### Getting Help
1. Check the logs in `logs/trading_bot.log`
2. Run diagnostic tests with `python debug.py`
3. Review configuration with `python start.py validate`

### Contributing
- Follow existing code structure and documentation
- Add tests for new features
- Update configuration examples
- Maintain backward compatibility

## ⚖️ Legal and Risk Disclaimer

**High Risk Warning**: Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. The value of cryptocurrencies can be extremely volatile and unpredictable.

**Software Disclaimer**: This software is provided "as is" without warranty of any kind. The developers are not responsible for any financial losses incurred through the use of this software.

**Regulatory Compliance**: Users are responsible for ensuring compliance with all applicable laws and regulations in their jurisdiction.

**No Investment Advice**: This software does not provide investment advice. All trading decisions are made by the user or their configured algorithms.

## 📜 License

This project is provided for educational and research purposes. Users are responsible for compliance with all applicable laws and regulations.

---

**Remember**: Always test thoroughly in testing mode before using production mode with real money. Start with small amounts and gradually increase as you gain confidence in the system.
