# PLAN.md: Consolidated Project Progress and Plan

This document consolidates the current in-progress elements and confirmed progress across various project planning and development files.

## 1. Overall Project Status

The project is in a strong position, with foundational elements and core CLI tools (Phase 1 & 2) largely complete. The basic Flask Web Application (Phase 3) is also production-ready. Current efforts are focused on enhancing the Flask application and planning for future integrations like trade journaling, reporting, and optimization. A significant ongoing effort is the integration of individual stock support.


*   **Current Focus**:
    *   Bug Fix: Screener doesn't screen individual stocks if you select All. It only returns ETFs
    
*   **Next Major Task**:
    *   **Risk Management Calculator (Phase 4.3)**:
        *   Position Sizing Calculator (risk-based, account size, entry/stop/target inputs).
        *   Risk/Reward Analysis (R-multiple, breakeven, expected value).
        *   Portfolio Risk Assessment (exposure, correlation-adjusted risk, drawdown projections).
        *   Trade Validation (pre-trade assessment, size recommendations, guideline compliance).
*   **Code Quality & Architecture Improvements (Phase 4.5)**:
    *   Complete remaining placeholder implementations in the service layer.
    *   Update inline documentation.

## Future Work Items:

### Individual Stock Support Integration (Ongoing)

*   **High Priority Tasks**:
    *   Update `init_database.py` for stock data loading and bootstrapping (`stocks_core`, `stocks_priority`, `stocks_all`).
    *   Create `backfill_stocks.py` for historical stock data backfilling and caching.
    *   Adjust `trade_setups.py` parameters for stocks (volatility, volume, position sizing, confidence).
    *   Update `screener.py` to include instrument type filtering (`--type stock`, `--type all`).
*   **Medium Priority Tasks**:
    *   Enhance `backtest.py` with separate performance tracking for stocks vs. ETFs, regime analysis for stocks, sector performance breakdown, and correlation analysis.
    *   Implement portfolio-level risk controls (concentration limits, sector limits, correlation monitoring).
    *   Develop stock-specific analysis tools (e.g., `stock_analyzer.py`).


*   **Trade Journal Integration**: Implement trade journal functionality, correlation tracking, and web interface for trade management.
*   **Reporting Tools**: Develop performance reporting vs. benchmarks, daily/weekly reporting, and regime-based performance analysis.
*   **Optimization & Expansion**: Focus on strategy refinement, parameter optimization tools, and dynamic production parameter optimization.
