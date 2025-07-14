# PLAN.md: Consolidated Project Progress and Plan

This document consolidates the current in-progress elements and confirmed progress across various project planning and development files.

## 1. Overall Project Status

The project is in a strong position, with foundational elements and core CLI tools (Phase 1 & 2) largely complete. The basic Flask Web Application (Phase 3) is also production-ready. Current efforts are focused on enhancing the Flask application (Phase 4) and planning for future integrations like trade journaling, reporting, and optimization. A significant ongoing effort is the integration of individual stock support.

## 2. Key In-Progress & Next Development Areas

### 2.1 Flask Web Application Enhancements (Phase 4)

*   **Current Focus**:
    *   Implementing charting on the stock screener.
    *   Reviewing and tidying up existing functionality (e.g., completing placeholder implementations).
    *   Standardizing progress bars for clear user feedback across all operations.
    *   Enabling users to select stocks from screener results for backtesting.
    *   Identifying useful tools for available data.
*   **Next Major Task**:
    *   **Risk Management Calculator (Phase 4.3)**:
        *   Position Sizing Calculator (risk-based, account size, entry/stop/target inputs).
        *   Risk/Reward Analysis (R-multiple, breakeven, expected value).
        *   Portfolio Risk Assessment (exposure, correlation-adjusted risk, drawdown projections).
        *   Trade Validation (pre-trade assessment, size recommendations, guideline compliance).
*   **Code Quality & Architecture Improvements (Phase 4.5)**:
    *   Complete remaining placeholder implementations in the service layer.
    *   Update inline documentation.

### 2.2 Individual Stock Support Integration (Ongoing)

*   **High Priority Tasks**:
    *   Update `init_database.py` for stock data loading and bootstrapping (`stocks_core`, `stocks_priority`, `stocks_all`).
    *   Create `backfill_stocks.py` for historical stock data backfilling and caching.
    *   Adjust `trade_setups.py` parameters for stocks (volatility, volume, position sizing, confidence).
    *   Update `screener.py` to include instrument type filtering (`--type stock`, `--type all`).
*   **Medium Priority Tasks**:
    *   Enhance `backtest.py` with separate performance tracking for stocks vs. ETFs, regime analysis for stocks, sector performance breakdown, and correlation analysis.
    *   Implement portfolio-level risk controls (concentration limits, sector limits, correlation monitoring).
    *   Develop stock-specific analysis tools (e.g., `stock_analyzer.py`).
*   **Low Priority Tasks**:
    *   Earnings calendar integration.
    *   Advanced correlation analysis.
    *   Dynamic parameter optimization for stocks.

## 3. Confirmed Progress & Completed Tasks

*   **Phase 1: Strategy Foundation**: Complete (Market regime detection, 13 trade setups, data caching, database schema).
*   **Phase 2: Screener + Backtest Engine**: Mostly Complete (Production CLI screener, export capabilities, walk-forward backtesting engine).
*   **Phase 3: Basic Flask Web Application**: Complete (Modular Flask app, dark Bootstrap 5 theme, service layer integration, interactive dashboards, API endpoints).
*   **Phase 4.1 & 4.2 (Flask Enhancements)**: Complete (Interactive screener charts, standardized progress bars, enhanced export, selective backtesting from screener).
*   **Backtesting Overhaul (from `BACKTEST.md`)**: Phase 1 & 2 largely complete.

## 4. Future Development Phases (Pending)

*   **Phase 5: Trade Journal Integration**: Implement trade journal functionality, correlation tracking, and web interface for trade management.
*   **Phase 6: Reporting Tools**: Develop performance reporting vs. benchmarks, daily/weekly reporting, and regime-based performance analysis.
*   **Phase 7: Optimization & Expansion**: Focus on strategy refinement, parameter optimization tools, and dynamic production parameter optimization.

## 5. Key Technical Considerations & Next Steps

*   **Risk Management**: A critical focus area, with the Risk Management Calculator being the immediate next development task. This will involve significant work on portfolio-level controls and stock-specific sizing adjustments.
*   **Data Backfilling**: The `backfill_stocks.py` script is essential for populating historical data for individual stocks.
*   **Parameter Tuning**: Adjusting parameters in `trade_setups.py` and other modules will be crucial for optimizing stock performance.
*   **Testing**: Incremental testing is planned, starting with core stock data loading and progressing to full backtesting and risk system validation.
