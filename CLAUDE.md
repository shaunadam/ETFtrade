# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This is a Python-based ETF and stock swing trading system designed to generate long-term capital growth with controlled drawdowns. The system trades both ETFs and individual stocks with equal priority, featuring both CLI tools and a Flask web application for comprehensive trading workflow management.

**Key Objectives:**
- Target 20% max drawdown with decades-long time horizon
- Trade frequency: 1-2 hours weeknights + Sunday sessions
- Must outperform benchmarks (SPY, sector ETFs)
- Future-proofed to handle both ETFs and individual stocks

## Architecture & Components

### Core Components
- **screener.py**: Production CLI for ETF/stock screening with regime-aware filtering
- **data_cache.py**: Intelligent data caching engine with 95%+ API call reduction
- **regime_detection.py**: Market regime detection across 4 dimensions
- **trade_setups.py**: Core trade setup implementations with regime validation
- **backtest.py**: Walk-forward backtesting engine with regime analysis
- **flask_app/**: Complete modular Flask web application with dark Bootstrap 5 theme

### Database Schema
```sql
instruments: id, symbol, name, type (ETF/stock), sector, tags
price_data: symbol, date, open, high, low, close, volume, updated_at
indicators: symbol, date, indicator_name, value, updated_at
trades: id, instrument_id, setup, entry_date, exit_date, size, entry_price, exit_price, r_planned, r_actual, notes, regime_at_entry
market_regimes: date, volatility_regime, trend_regime, sector_rotation, risk_on_off, vix_level, spy_vs_sma200, growth_value_ratio, risk_on_off_ratio
```

## Technical Stack

**Core Libraries:**
- `yfinance`: Market data (free tier)
- `pandas`, `numpy`: Data manipulation
- `matplotlib`/`plotly`: Visualization
- `sqlite3`: Database (built-in)

**Web Framework:**
- `flask`: Web application framework
- `flask-sqlalchemy`: Database ORM integration
- `python-dotenv`: Environment configuration

**Development Tools:**
- `pytest`: Testing framework
- `ruff`: Linting and formatting
- `mypy`: Type checking

## Trading Strategy Framework

### Market Regime Detection
- **Volatility Regime**: VIX levels (low <20, medium 20-30, high >30)
- **Trend Regime**: SPY distance from 200-day SMA
- **Sector Rotation**: Growth vs Value (QQQ/IWM, XLK/XLF ratios)
- **Risk-On/Risk-Off**: Defensive vs aggressive ETF performance

### Trade Setups (13 Total)
Core setups covering momentum, mean reversion, gaps, volatility, dividends, multi-timeframe analysis, institutional behavior, event-driven opportunities, and Elder's advanced indicators.

### Risk Management Rules
- Max 2% capital risk per trade
- Max 3-4 concurrent positions
- Max 30% in correlated sectors
- R-based exits (2R target, -1R stop)
- ATR-based trailing stops

## Data Caching System

### Key Features
- **Smart Refresh Strategy**: Always refreshes last 5 trading days
- **Healing Logic**: Ensures 200+ day buffer for SMA200 calculations
- **95% API Reduction**: Dramatically reduces yfinance API calls
- **Technical Indicators**: Pre-calculated and cached (SMA20/50/200, RSI, ATR, Bollinger Bands, EMA13, Force Index, MACD)

## Development Phases

1. **Phase 1**: Strategy Foundation ✅
2. **Phase 2**: Screener + Backtest Engine ✅
3. **Phase 3**: Flask Web Application ✅
4. **Phase 4**: Trade Journal Integration (In Progress)
5. **Phase 5**: Reporting Tools
6. **Phase 6**: Optimization & Expansion

## Success Metrics
- Beat SPY by 2%+ annually after costs
- Stay within 20% maximum drawdown
- Positive performance across market regimes
- <30 minutes daily maintenance

## Development Preferences
- **Commit Messages**: Keep simple, 1 sentence, no "Generated with Claude Code" footers
- **Code Style**: Follow existing patterns, minimal comments unless needed
- **File Creation**: NEVER create files unless absolutely necessary - always prefer editing existing files
- **Documentation**: NEVER proactively create documentation files unless explicitly requested
- This is always a feature branch - no backwards compatibility needed
- When in doubt, we choose clarity over cleverness
- Avoid complex abstractions or "clever" code. The simple, obvious solution is probably better, and my guidance helps you stay focused on what matters.

## Problem-Solving Together
When you're stuck or confused:

- Stop - Don't spiral into complex solutions
- Delegate - Consider spawning agents for parallel investigation
- Ultrathink - For complex problems, say "I need to ultrathink through this challenge" to engage deeper reasoning
- Step back - Re-read the requirements
- Simplify - The simple solution is usually correct
- Ask - "I see two approaches: [A] vs [B]. Which do you prefer?"
- **My insights on better approaches are valued - please ask for them!**