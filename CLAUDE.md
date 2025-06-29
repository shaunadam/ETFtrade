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
- **screener.py**: ETF screening with regime-aware filtering
- **backtest.py**: Walk-forward backtesting engine with regime analysis  
- **journal.py**: SQLite-based trade journal with correlation tracking
- **report.py**: Performance reporting vs benchmarks
- **etf_universe.csv**: Curated list of ~50 high-quality ETFs with tagging
- **journal.db**: SQLite database for trades, regimes, and correlations

### Database Schema (Future-Proofed)
```sql
instruments: id, symbol, name, type (ETF/stock), sector, tags
trades: id, instrument_id, setup, entry_date, exit_date, size, entry_price, exit_price, r_planned, r_actual, notes, regime_at_entry
setups: id, name, description, parameters
snapshots: id, trade_id, date, price, notes, chart_path
market_regimes: date, volatility_regime, trend_regime, sector_rotation, notes
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
# Environment setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Daily trading workflow
python screener.py --regime-filter --export-csv          # Find trade candidates
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
python screener.py --update-universe --check-liquidity   # Refresh ETF universe
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

### Risk Management Rules
- Max 2% capital risk per trade
- Max 3-4 concurrent positions
- Max 30% in correlated sectors
- R-based exits (2R target, -1R stop)
- ATR-based trailing stops

## ETF Universe Categories

Stored in etf_list.csv for now.

## Development Phases

1. **Phase 1**: Strategy Foundation (ETF universe, regime detection, trade setups)
2. **Phase 2**: Screener + Backtest Engine (walk-forward analysis)
3. **Phase 3**: Trade Journal (SQLite database, correlation tracking)
4. **Phase 4**: Reporting Tools (performance vs benchmarks)
5. **Phase 5**: Optimization & Expansion (strategy refinement)

**Progress Tracking**: See [PROGRESS.md](PROGRESS.md) for detailed development status and completed tasks.

## Success Metrics
- Beat SPY by 2%+ annually after costs
- Stay within 20% maximum drawdown
- Positive performance across market regimes
- <30 minutes daily maintenance
- Stable walk-forward parameters

## Development Preferences
- **Commit Messages**: Keep simple, 1 sentence, no "Generated with Claude Code" footers
- **Code Style**: Follow existing patterns, minimal comments unless needed