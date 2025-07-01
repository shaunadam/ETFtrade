# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based ETF swing trading system designed to generate long-term capital growth with controlled drawdowns. The system focuses exclusively on ETFs (sector-specific, thematic, leveraged) and is designed for limited time availability with daily monitoring of just 5-10 minutes.

**Key Objectives:**
- Target 20% max drawdown with decades-long time horizon
- Trade frequency: 1-2 hours weeknights + Sunday sessions
- Must outperform benchmarks (SPY, sector ETFs)
- Future-proofed to handle both ETFs and individual stocks

## Project Structure & Components

The system is organized into 5 main phases:

### Core Components
- **screener.py**: Production CLI for ETF screening with regime-aware filtering and export capabilities
- **data_cache.py**: Intelligent data caching engine with 95%+ API call reduction
- **regime_detection.py**: Market regime detection across volatility, trend, sector rotation, and risk sentiment
- **trade_setups.py**: Core trade setup implementations with regime validation
- **backtest.py**: Walk-forward backtesting engine with regime analysis  
- **journal.py**: SQLite-based trade journal with correlation tracking
- **report.py**: Performance reporting vs benchmarks
- **etf_list.csv**: Curated list of ~50 high-quality ETFs with tagging
- **journal.db**: SQLite database for trades, regimes, price data, and technical indicators

### Database Schema (Future-Proofed)
```sql
instruments: id, symbol, name, type (ETF/stock), sector, tags
price_data: symbol, date, open, high, low, close, volume, updated_at
indicators: symbol, date, indicator_name, value, updated_at  
trades: id, instrument_id, setup, entry_date, exit_date, size, entry_price, exit_price, r_planned, r_actual, notes, regime_at_entry
setups: id, name, description, parameters
snapshots: id, trade_id, date, price, notes, chart_path
market_regimes: date, volatility_regime, trend_regime, sector_rotation, risk_on_off, vix_level, spy_vs_sma200, growth_value_ratio, risk_on_off_ratio
```

## Technical Stack & Dependencies

**Core Libraries:**
- `yfinance`: Market data (free tier)
- `pandas`, `numpy`: Data manipulation
- `matplotlib`/`plotly`: Visualization
- `sqlite3`: Database (built-in)
- `jupyter`: Analysis notebooks

**Development Tools:**
- `pytest`: Testing framework
- `ruff`: Linting and formatting
- `mypy`: Type checking

## Common Development Commands

```bash
# Environment setup (first time)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Initialize database with market data (REQUIRED for first-time setup)
python init_database.py --bootstrap core      # Essential ETFs for regime detection (recommended)
python init_database.py --bootstrap priority  # Core + high-priority trading ETFs
python init_database.py --bootstrap all       # All ETFs (slower, comprehensive)
python init_database.py --skip-data           # Schema only (manual data loading required)

# IMPORTANT: Always activate virtual environment when starting work
# Run this once per session before any Python commands:
source .venv/bin/activate

# Daily trading workflow
python screener.py --regime-filter --export-csv          # Find trade candidates
python screener.py --cache-stats                         # Check data cache status
python screener.py --setup breakout_continuation --min-confidence 0.6  # Specific setup scan
python screener.py --setup gap_fill_reversal             # Gap reversal opportunities
python screener.py --setup relative_strength_momentum    # Relative strength plays
python screener.py --setup volatility_contraction        # Low volatility setups
python screener.py --setup dividend_distribution_play    # Dividend timing plays
python journal.py --open-trades --correlations           # Check current positions
python report.py --daily --vs-spy                        # Quick performance check

# Weekly workflow
python report.py --weekly --vs-benchmarks                # Full performance review
python journal.py --weekly-review --regime-analysis      # Trade analysis by regime

# Backtesting & optimization
python backtest.py --setup trend_pullback --walk-forward # Test single setup
python backtest.py --all-setups --regime-aware           # Test all strategies
python backtest.py --optimize --start-date 2020-01-01    # Parameter optimization

# Trade management
python journal.py --add-trade SYMBOL --setup trend_pullback --risk 0.02
python journal.py --update-trade ID --exit-price 45.50 --notes "target hit"
python journal.py --calculate-r --all-open               # Update R multiples

# Data management
python screener.py --update-data                         # Smart refresh market data
python screener.py --force-refresh                       # Force full data refresh
python screener.py --cache-stats                         # View cache statistics
python data_cache.py                                     # Test cache functionality
python regime_detection.py                               # Update regime detection
python journal.py --update-correlations                  # Refresh correlation matrix
python journal.py --backup-db --compress                 # Backup trade data

# Analysis & reports
python report.py --regime-performance --since 2023-01-01 # Performance by regime
python report.py --setup-analysis --export-charts        # Setup effectiveness
jupyter notebook reports/weekly_review.ipynb             # Interactive analysis

# Development & testing
pytest tests/ -v                                         # Run all tests
pytest tests/test_screener.py::test_regime_detection     # Test specific function
ruff check . && ruff format .                           # Code quality
mypy . --strict                                         # Type checking

# Database operations
sqlite3 journal.db ".backup backup_$(date +%Y%m%d).db"  # Manual DB backup
sqlite3 journal.db ".schema"                            # View database schema
```

## Trading Strategy Framework

### Market Regime Detection
- **Volatility Regime**: VIX levels (low <20, medium 20-30, high >30)
- **Trend Regime**: SPY distance from 200-day SMA  
- **Sector Rotation**: Growth vs Value (QQT/IWM, XLK/XLF ratios)
- **Risk-On/Risk-Off**: Defensive vs aggressive ETF performance

### Core Trade Setups
- **Trend Pullback**: Works best in trending regimes
- **Breakout Continuation**: Avoid in high volatility regimes
- **Oversold Mean Reversion**: Effective in ranging markets
- **Regime Rotation**: Sector rotation based on regime changes
- **Gap Fill Reversal**: Trade ETFs gapping down with reversal signals
- **Relative Strength Momentum**: Buy ETFs outperforming SPY during weakness
- **Volatility Contraction**: Trade after ATR compression before expansion
- **Dividend/Distribution Play**: Technical setups in dividend sectors during stable regimes

### Risk Management Rules
- Max 2% capital risk per trade
- Max 3-4 concurrent positions
- Max 30% in correlated sectors
- R-based exits (2R target, -1R stop)
- ATR-based trailing stops

## ETF Universe Categories

Stored in etf_list.csv for now.

## Data Caching System

The system implements intelligent data caching to minimize API calls and improve performance:

### Key Features
- **Smart Refresh Strategy**: Always refreshes last 5 trading days
- **Healing Logic**: Ensures 200+ day buffer for SMA200 calculations
- **95% API Reduction**: Dramatically reduces yfinance API calls
- **Technical Indicators**: Pre-calculated and cached (SMA20/50/200, RSI, ATR, Bollinger Bands)
- **Graceful Fallback**: Seamless fallback to yfinance when cache misses

### Cache Management
```bash
# Check cache status
python screener.py --cache-stats

# Smart refresh (recommended)
python screener.py --update-data

# Force full refresh (if needed)
python screener.py --force-refresh

# Test cache functionality
python data_cache.py
```

### Performance Metrics
- **Cache Size**: 6,400+ price records, 37,000+ indicator values
- **Symbols Cached**: 50+ ETFs with full history
- **Speed Improvement**: Screening in seconds vs minutes
- **Data Quality**: Healing strategy ensures indicator accuracy

## Development Phases

1. **Phase 1**: Strategy Foundation ✅ (ETF universe, regime detection, trade setups, data caching)
2. **Phase 2**: Screener + Backtest Engine (screener complete, backtest pending)
3. **Phase 3**: Trade Journal (SQLite database, correlation tracking)
4. **Phase 4**: Reporting Tools (performance vs benchmarks)
5. **Phase 5**: Optimization & Expansion (strategy refinement, dynamic parameter optimization)

**Progress Tracking**: See [PROGRESS.md](PROGRESS.md) for detailed development status and completed tasks.

## Success Metrics
- Beat SPY by 2%+ annually after costs
- Stay within 20% maximum drawdown
- Positive performance across market regimes
- <30 minutes daily maintenance
- Stable walk-forward parameters

## Future Enhancements (Post Phase 5)

### Dynamic Production Parameter Optimization
- **Adaptive Screening**: Monthly/quarterly parameter optimization for live screening
- **Risk Controls**: Statistical significance requirements and overfitting prevention
- **Regime-Aware Adaptation**: Parameter updates based on current market regime
- **Live Walk-Forward**: Continuous optimization using rolling historical performance
- **Portfolio-Level Optimization**: Holistic parameter tuning across all setups and positions

## Development Preferences
- **Commit Messages**: Keep simple, 1 sentence, no "Generated with Claude Code" footers
- **Code Style**: Follow existing patterns, minimal comments unless needed