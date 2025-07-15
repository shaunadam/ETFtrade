# Claude.md: Project Briefing Document

## 1. Project Architecture Overview
This project is a Python-based ETF and stock swing trading system designed for long-term capital growth with a focus on controlled drawdowns, aiming to outperform benchmarks like SPY.

*   **Core Purpose**: To generate long-term capital growth with controlled drawdowns, specifically targeting an outperformance of benchmarks like SPY while maintaining a maximum drawdown within 20%. The system is designed for daily monitoring requiring less than 30 minutes of maintenance time.
*   **Dual Interface Approach**:
    *   **Command-Line Interface (CLI)**: Provides core operations such as screening, data caching, regime detection, trade setup execution, and backtesting. Key CLI tools include `screener.py` (main CLI for daily screening), `data_cache.py` (intelligent data caching engine), `regime_detection.py` (market regime analysis), `trade_setups.py` (core trading strategies), and `backtest.py` (backtesting engine).
    *   **Flask Web Application**: A modular web application offering a professional user interface with interactive dashboards, charting, and comprehensive trading workflow management. It integrates all CLI functionalities and provides features like real-time progress updates and configuration UIs.
*   **Technical Stack**: The system relies on a robust technical stack:
    *   **Python**: The primary programming language for all core logic and scripting.
    *   **Flask**: A micro web framework used for building the web application, providing routing, templating, and request handling.
    *   **Pandas**: Utilized extensively for data manipulation, analysis, and handling time-series data (e.g., OHLCV data, technical indicators).
    *   **NumPy**: Provides fundamental numerical operations, often used in conjunction with Pandas for high-performance array computations.
    *   **Matplotlib/Plotly**: Used for generating interactive charts and visualizations within the Flask web application and for analysis.
    *   **SQLite**: Serves as the lightweight, file-based database for persistent data storage, including cached market data, indicators, and trade journal entries.
*   **Key Objectives**:
    *   **Risk Management**: Target 20% maximum drawdown over a decades-long time horizon.
    *   **Time Efficiency**: Enable daily monitoring and operation within 5-10 minutes.
    *   **Performance**: Consistently outperform benchmarks (e.g., SPY, sector ETFs).
    *   **Future-Ready**: Architected to seamlessly handle both ETFs and individual stocks.

## 2. Core Components Deep Dive

*   **2.1 Intelligent Data Caching (`data_cache.py`)**:
    *   **Purpose**: Minimizes API calls by efficiently caching historical price data and calculated technical indicators.
    *   **Mechanisms**: Implements a "smart refresh" strategy that primarily updates the most recent 5 trading days and a "healing strategy" to ensure a sufficient data buffer (200+ days) for accurate SMA200 calculations.
    *   **Cached Data**: Stores OHLCV data, and pre-calculated technical indicators such as Simple Moving Averages (SMA), Relative Strength Index (RSI), Average True Range (ATR), and Bollinger Bands (plus others).

*   **2.2 Market Regime Detection (`regime_detection.py`, `regime_validator.py`)**:
    *   **Purpose**: Analyzes various market conditions to inform trading strategies and filter signals, adapting to different market environments.
    *   **Detected Regimes**:
        *   **Volatility Regime**: Determined by VIX levels (low <20, medium 20-30, high >30).
        *   **Trend Regime**: Assessed by SPY's distance from its 200-day Simple Moving Average.
        *   **Sector Rotation**: Analyzes performance ratios between Growth and Value ETFs (e.g., QQQ/IWM, XLK/XLF).
        *   **Risk Sentiment**: Evaluates the relative performance of Defensive vs. Aggressive ETFs.
    *   **Influence on Trading**: `regime_validator.py` uses these detected regimes to filter trade signals, ensuring that only signals aligned with the current market environment are considered.

*   **2.3 Comprehensive Trade Setup Suite (`trade_setups.py`)**:
    *   **Purpose**: Implements a variety of core trading strategies and advanced pattern recognition techniques to identify trade opportunities.
    *   **Implemented Setups (8 Total)**:
        *   **Core Momentum & Mean Reversion**:
            *   **Trend Pullback**: Identifies 3-8% pullbacks within established trending markets.
            *   **Breakout Continuation**: Scans for volume-confirmed breakouts above 20-day highs.
            *   **Oversold Mean Reversion**: Targets ETFs with RSI <30 and prices below their lower Bollinger Band.
            *   **Regime Rotation**: Positions based on identified sector shifts driven by market regime changes.
        *   **Advanced Pattern Recognition**:
            *   **Gap Fill Reversal**: Trades ETFs gapping down ≥2% with subsequent reversal signals.
            *   **Relative Strength Momentum**: Buys ETFs demonstrating outperformance relative to SPY during broader market weakness.
            *   **Volatility Contraction**: Identifies setups where Average True Range (ATR) has compressed, anticipating future price expansion.
            *   **Dividend/Distribution Play**: Focuses on technical setups within dividend-paying sectors.
            *   **Elder Triple Screen**: Applies a multi-timeframe approach using trend, momentum, and volume indicators.
            *   **Institutional Volume Climax**: Detects potential reversals based on extreme volume spikes.
            *   **Failed Breakdown Reversal**: Identifies instances where a breakdown below support fails, leading to a reversal.
            *   **Earnings Expectation Reset**: Analyzes post-earnings price action for potential reversals or continuations.
            *   **Elder Force Impulse**: Utilizes Elder's Force Index and Impulse System for trend and momentum confirmation.

*   **2.4 Risk Management & Trade Execution (`portfolio_risk.py`, `trade_manager.py`)**:
    *   **Purpose**: Ensures controlled drawdowns and adherence to predefined risk limits throughout the trading process.
    *   **Key Aspects**:
        *   **Position Sizing**: Calculates appropriate position sizes based on risk-per-trade and account size.
        *   **Capital Risk**: Enforces strict limits on capital risked per individual trade.
        *   **Sector Concentration**: Monitors and limits exposure to specific sectors to prevent over-concentration.
        *   **Trade Lifecycle**: `trade_manager.py` handles the entry, ongoing management, and closing of trades, integrating risk checks at each stage.

## 3. Connection of Markdown Files

The markdown files in the root of this repository serve as critical documentation and planning artifacts for the project:

*   **`README.md`**: Acts as the primary entry point for human users. It provides a high-level overview of the project's objectives, features, quick start guide, project structure, core components, usage examples for both CLI and Flask app, performance metrics, and development status. It outlines the initial focus on ETFs and the future-proofing for individual stocks.
*   **`CLAUDE.md` (This Document)**: Specifically designed as a detailed briefing document for AI assistants. It elaborates on the project's architecture, key components, technical stack, trading strategy framework, data caching system, market regime detection, development phases, and success metrics, providing a comprehensive yet concise overview.
*   **`PLAN.md`**: Serves as the short-term plan, outlining current in-progress tasks and confirmed progress, as well as future work items and the development roadmap.

Collectively, these files provide a comprehensive view of the project's goals, architecture, current status, and future development roadmap, tailored for different audiences.

## 4. Development Roadmap & Status

The project is structured into distinct development phases, progressing from foundational strategy development to advanced features like trade journaling, reporting, and optimization.

*   **Completed Phases**:
    *   **Phase 1: Strategy Foundation**: All 8 core Trade Setups have been implemented.
    *   **Phase 2: Screener + Backtest Engine**: The `screener.py` CLI tool is complete, enabling daily screening for trade opportunities. Backtest engine has been built, but does not work correctly. It is in backlog for now.
*   **Future Work Items (from `PLAN.md`)**:
    *   **Trade Journal Integration**: Implement comprehensive trade journal functionality, including correlation tracking and a web interface for trade management.
    *   **Risk Management Calculator**: Develop tools for position sizing (risk-based, account size, entry/stop/target inputs), portfolio risk assessment (exposure, correlation-adjusted risk, drawdown projections), and pre-trade validation (size recommendations, guideline compliance).
    *   **Reporting Tools**: Create robust performance reporting against benchmarks, daily/weekly reporting, and regime-based performance analysis.
    *   **Optimization & Expansion**: Focus on strategy refinement, building parameter optimization tools, and implementing dynamic production parameter optimization.

