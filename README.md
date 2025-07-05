# ETF Swing Trading System

A Python-based ETF swing trading system designed to generate long-term capital growth with controlled drawdowns. The system focuses exclusively on ETFs (sector-specific, thematic, leveraged) and implements regime-aware filtering with intelligent data caching.

## 🎯 Key Objectives

- **Risk Management**: Target 20% max drawdown with decades-long time horizon
- **Time Efficiency**: Daily monitoring of just 5-10 minutes
- **Performance**: Must outperform benchmarks (SPY, sector ETFs)
- **Future-Ready**: Architected to handle both ETFs and individual stocks

## ✨ Features

- **🔄 Intelligent Data Caching**: 95%+ reduction in API calls with healing strategy
- **📊 Market Regime Detection**: Volatility, trend, sector rotation, and risk sentiment analysis
- **🎯 Complete Trade Setup Suite**: 8 comprehensive setups covering all major trading patterns
- **💻 Production CLI**: Daily screening with export capabilities
- **📈 Technical Analysis**: RSI, ATR, Bollinger Bands, moving averages with caching

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment recommended

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ETFtrade
```

2. **Set up virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize database**
```bash
python init_database.py
```

### Daily Workflow

**Morning Screening (5 minutes)**
```bash
# Activate environment
source .venv/bin/activate

# Screen for trade opportunities
python screener.py --regime-filter --export-csv

# Check specific setup
python screener.py --setup trend_pullback --min-confidence 0.7
```

**Data Management**
```bash
# Update market data cache
python screener.py --update-data

# View cache statistics
python screener.py --cache-stats

# Force full refresh if needed
python screener.py --force-refresh
```

## 📁 Project Structure

```
ETFtrade/
├── screener.py          # Main CLI for daily screening
├── data_cache.py        # Intelligent data caching engine
├── regime_detection.py  # Market regime analysis
├── trade_setups.py      # Core trading strategies
├── init_database.py     # Database initialization
├── etf_list.csv        # Curated ETF universe (~50 symbols)
├── journal.db          # SQLite database (created after init)
└── requirements.txt    # Python dependencies
```

## 🔧 Core Components

### Market Regime Detection
- **Volatility Regime**: VIX-based (low <20, medium 20-30, high >30)
- **Trend Regime**: SPY distance from 200-day SMA
- **Sector Rotation**: Growth vs Value ratios (QQQ/IWM, XLK/XLF)
- **Risk Sentiment**: Defensive vs aggressive ETF performance

### Trade Setups (8 Total)
**Core Momentum & Mean Reversion:**
- **Trend Pullback**: 3-8% pullbacks in trending markets
- **Breakout Continuation**: Volume-confirmed breakouts above 20-day highs
- **Oversold Mean Reversion**: RSI <30 + below lower Bollinger Band
- **Regime Rotation**: Sector positioning based on regime changes

**Advanced Pattern Recognition:**
- **Gap Fill Reversal**: Trade ETFs gapping down ≥2% with reversal signals
- **Relative Strength Momentum**: Buy ETFs outperforming SPY during weakness
- **Volatility Contraction**: ATR compression setups before expansion
- **Dividend/Distribution Play**: Technical setups in dividend sectors

### Data Caching System
- **Smart Refresh**: Always refreshes last 5 trading days
- **Healing Strategy**: Ensures 200+ day buffer for SMA200 calculations
- **95% API Reduction**: Dramatically reduces yfinance API calls
- **Technical Indicators**: Pre-calculated and cached (SMA, RSI, ATR, Bollinger Bands)

## 📊 Usage Examples

### Environment Setup
```bash
# IMPORTANT: Always activate virtual environment when starting work
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Initialize database with market data (first time only)
python init_database.py --bootstrap core      # Essential ETFs for regime detection (recommended)
python init_database.py --bootstrap priority  # Core + high-priority trading ETFs
python init_database.py --bootstrap all       # All ETFs (slower, comprehensive)
```

### Flask Web Application
```bash
# Start Flask development server
cd flask_app && python app.py                            # http://localhost:5000
cd flask_app && FLASK_ENV=production python app.py       # Production mode
cd flask_app && python -c "from app import create_app; create_app().run(debug=True, port=5001)"  # Custom port

# Web Interface Features:
# - Dashboard: System status monitoring and overview
# - Screener: Full web interface for ETF/stock screening with regime detection
# - Regime: Real-time market regime analysis with historical charts
# - Data: Cache management, data updates, and universe maintenance
# - Backtest: Web-based backtesting with walk-forward analysis
```

### Daily Trading Workflow (CLI)
```bash
# Basic screening
python screener.py --regime-filter --export-csv          # Find trade candidates (uses cache)
python screener.py --update-data --regime-filter         # Update data + find candidates
python screener.py --cache-stats                         # Check data cache status

# Specific setup screening
python screener.py --setup breakout_continuation --min-confidence 0.6
python screener.py --setup gap_fill_reversal
python screener.py --setup relative_strength_momentum
python screener.py --setup volatility_contraction
python screener.py --setup dividend_distribution_play
python screener.py --setup elder_triple_screen
python screener.py --setup institutional_volume_climax
python screener.py --setup failed_breakdown_reversal
python screener.py --setup earnings_expectation_reset
python screener.py --setup elder_force_impulse

# Export results
python screener.py --regime-filter --export-json
```

### Data Management
```bash
# Smart data refresh (recommended)
python screener.py --update-data                         # Smart refresh market data
python screener.py --cache-stats                         # View cache statistics
python screener.py --force-refresh                       # Force full data refresh

# Cache functionality
python data_cache.py                                     # Test cache functionality
python regime_detection.py                               # Update regime detection
```

### Trading & Analysis (Future)
```bash
# Trade management (when journal.py is complete)
python journal.py --add-trade SYMBOL --setup trend_pullback --risk 0.02
python journal.py --update-trade ID --exit-price 45.50 --notes "target hit"
python journal.py --open-trades --correlations           # Check current positions

# Weekly workflow
python report.py --weekly --vs-benchmarks                # Full performance review
python journal.py --weekly-review --regime-analysis      # Trade analysis by regime

# Backtesting & optimization
python backtest.py --setup trend_pullback --walk-forward # Test single setup
python backtest.py --all-setups --regime-aware           # Test all strategies
python backtest.py --optimize --start-date 2020-01-01    # Parameter optimization

# Analysis & reports
python report.py --regime-performance --since 2023-01-01 # Performance by regime
python report.py --setup-analysis --export-charts        # Setup effectiveness
python analysis/treasury_analysis.py                     # Treasury analysis tools
jupyter notebook reports/weekly_review.ipynb             # Interactive analysis
```

## 🛠️ Development

### Code Quality & Testing
```bash
# Linting and formatting
ruff check . && ruff format .

# Type checking
mypy . --strict

# Run tests
pytest tests/ -v                                         # Run all tests
pytest tests/test_screener.py::test_regime_detection     # Test specific function
```

### Debugging & Diagnostics
```bash
# Enable debug mode
ETF_DEBUG=1 python screener.py --cache-stats            # Debug cache behavior
ETF_DEBUG=1 python screener.py --update-data            # Debug data refresh logic

# Database operations
sqlite3 journal.db ".backup backup_$(date +%Y%m%d).db"  # Manual DB backup
sqlite3 journal.db ".schema"                            # View database schema
```

### Database Schema
The system uses SQLite with these key tables:
- `instruments`: ETF universe with sector/theme tagging
- `price_data`: OHLCV data cache
- `indicators`: Technical indicators cache
- `market_regimes`: Daily regime detection results

## 📈 Performance

- **Cache Efficiency**: 6,400+ price records, 37,000+ indicator values cached
- **API Optimization**: 95%+ reduction in yfinance API calls
- **Speed**: Screening 50+ ETFs in seconds vs minutes
- **Data Quality**: Healing strategy ensures indicator accuracy
- **Complete Coverage**: 8 trade setups covering all major patterns and market conditions

## 🎯 Success Metrics

- Beat SPY by 2%+ annually after costs
- Stay within 20% maximum drawdown
- Positive performance across market regimes
- <30 minutes daily maintenance time

## 📋 Development Status

- ✅ **Phase 1**: Strategy Foundation (Complete - 8 Trade Setups)
- 🔄 **Phase 2**: Screener + Backtest Engine (Screener Complete)
- 📋 **Phase 3**: Trade Journal
- 📋 **Phase 4**: Reporting Tools
- 📋 **Phase 5**: Optimization & Expansion

## 🤝 Contributing

This is a personal trading system. For questions or suggestions, please open an issue.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.

## 📄 License

[Add your license here]