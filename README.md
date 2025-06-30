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

### Basic Screening
```bash
# Screen all setups with regime filtering
python screener.py --regime-filter --export-csv

# Focus on specific setups
python screener.py --setup breakout_continuation --min-confidence 0.6
python screener.py --setup gap_fill_reversal
python screener.py --setup relative_strength_momentum
python screener.py --setup volatility_contraction

# Export results to JSON
python screener.py --regime-filter --export-json
```

### Data Management
```bash
# Check cache status
python screener.py --cache-stats

# Update data for specific symbols
python data_cache.py  # Runs cache test

# View regime detection
python regime_detection.py
```

## 🛠️ Development

### Code Quality
```bash
# Linting and formatting
ruff check . && ruff format .

# Type checking
mypy . --strict

# Run tests
pytest tests/ -v
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