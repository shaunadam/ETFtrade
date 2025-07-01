 Comprehensive Plan: Adding Individual Stock Support                                                                                                             │ │
│ │                                                                                                                                                                 │ │
│ │ Overview                                                                                                                                                        │ │
│ │                                                                                                                                                                 │ │
│ │ The current ETF trading system is well-architected and already future-proofed for individual stocks. The database schema includes a type field in the           │ │
│ │ instruments table that accepts 'ETF', 'Stock', and 'ETN'. The plan involves extending the existing infrastructure rather than creating separate systems.        │ │
│ │                                                                                                                                                                 │ │
│ │ 1. Database & Data Management Changes                                                                                                                           │ │
│ │                                                                                                                                                                 │ │
│ │ 1.1 Update init_database.py                                                                                                                                     │ │
│ │                                                                                                                                                                 │ │
│ │ - Add load_stock_data() function similar to load_etf_data()                                                                                                     │ │
│ │ - Modify bootstrap_market_data() to handle stocks with new bootstrap levels:                                                                                    │ │
│ │   - stocks_core: Essential stocks for testing (10-15 symbols)                                                                                                   │ │
│ │   - stocks_priority: High-volume, liquid stocks (30-40 symbols)                                                                                                 │ │
│ │   - stocks_all: All 56 stocks from stocks_list.csv                                                                                                              │ │
│ │ - Update command-line arguments to support stock bootstrapping:                                                                                                 │ │
│ │ python init_database.py --bootstrap stocks_core                                                                                                                 │ │
│ │ python init_database.py --bootstrap stocks_priority                                                                                                             │ │
│ │ python init_database.py --bootstrap stocks_all                                                                                                                  │ │
│ │                                                                                                                                                                 │ │
│ │ 1.2 Enhance data_cache.py                                                                                                                                       │ │
│ │                                                                                                                                                                 │ │
│ │ - No major changes needed - already handles any symbol generically                                                                                              │ │
│ │ - Add stock-specific caching considerations:                                                                                                                    │ │
│ │   - Higher volatility stocks may need more frequent updates                                                                                                     │ │
│ │   - Add debug logging for stock vs ETF caching behavior                                                                                                         │ │
│ │   - Consider different healing strategies for individual stocks                                                                                                 │ │
│ │                                                                                                                                                                 │ │
│ │ 1.3 Create Backfill Task                                                                                                                                        │ │
│ │                                                                                                                                                                 │ │
│ │ - New script: backfill_stocks.py                                                                                                                                │ │
│ │   - Load all 56 stocks from stocks_list.csv                                                                                                                     │ │
│ │   - Fetch 2+ years of historical data                                                                                                                           │ │
│ │   - Calculate and cache all technical indicators                                                                                                                │ │
│ │   - Handle potential API rate limiting                                                                                                                          │ │
│ │   - Progress reporting and error handling                                                                                                                       │ │
│ │                                                                                                                                                                 │ │
│ │ 2. Trading Strategy & Setup Changes                                                                                                                             │ │
│ │                                                                                                                                                                 │ │
│ │ 2.1 Update trade_setups.py                                                                                                                                      │ │
│ │                                                                                                                                                                 │ │
│ │ - Parameter adjustments for stocks:                                                                                                                             │ │
│ │   - Volatility multipliers: Stocks typically need wider stops (2.5-3.0x ATR vs 2.0x for ETFs)                                                                   │ │
│ │   - Volume thresholds: Higher requirements (3.0x vs 2.0x for breakouts)                                                                                         │ │
│ │   - Position sizing: More conservative for individual stocks (1.5% vs 2% risk)                                                                                  │ │
│ │   - Confidence thresholds: Higher minimums (0.6 vs 0.5) due to higher stock volatility                                                                          │ │
│ │ - Stock-specific considerations:                                                                                                                                │ │
│ │   - Sector correlation checks: Prevent over-concentration in single sectors                                                                                     │ │
│ │   - Market cap filtering: Different parameters for large-cap vs mid-cap stocks                                                                                  │ │
│ │   - Beta adjustments: Factor in stock beta for position sizing                                                                                                  │ │
│ │   - Earnings seasonality: Enhanced earnings expectation reset setup                                                                                             │ │
│ │ - New setup validations:                                                                                                                                        │ │
│ │   - Check if instrument is stock vs ETF in validate_signal()                                                                                                    │ │
│ │   - Apply different confidence adjustments                                                                                                                      │ │
│ │   - Sector diversification rules                                                                                                                                │ │
│ │                                                                                                                                                                 │ │
│ │ 2.2 Regime Detection Considerations                                                                                                                             │ │
│ │                                                                                                                                                                 │ │
│ │ - regime_detection.py requires minimal changes                                                                                                                  │ │
│ │ - Regime detection stays ETF-based (SPY, QQQ, etc.) as intended                                                                                                 │ │
│ │ - Individual stocks respond to market regimes but don't define them                                                                                             │ │
│ │                                                                                                                                                                 │ │
│ │ 3. Screening & Analysis Updates                                                                                                                                 │ │
│ │                                                                                                                                                                 │ │
│ │ 3.1 Update screener.py                                                                                                                                          │ │
│ │                                                                                                                                                                 │ │
│ │ - Add instrument type filtering:                                                                                                                                │ │
│ │   - --type etf or --type stock or --type all                                                                                                                    │ │
│ │   - Default remains ETF for backward compatibility                                                                                                              │ │
│ │ - Enhance symbol retrieval in SetupManager.get_all_symbols()                                                                                                    │ │
│ │ - Add stock-specific output formatting                                                                                                                          │ │
│ │ - Sector-based screening options                                                                                                                                │ │
│ │                                                                                                                                                                 │ │
│ │ 3.2 Update backtest.py                                                                                                                                          │ │
│ │                                                                                                                                                                 │ │
│ │ - Separate performance tracking for stocks vs ETFs                                                                                                              │ │
│ │ - Enhanced regime analysis showing how stocks performed vs ETFs in different regimes                                                                            │ │
│ │ - Sector performance breakdown                                                                                                                                  │ │
│ │ - Individual stock correlation analysis                                                                                                                         │ │
│ │ - Benchmark comparisons (SPY for market, sector ETFs for sector comparison)                                                                                     │ │
│ │                                                                                                                                                                 │ │
│ │ 4. Risk Management Enhancements                                                                                                                                 │ │
│ │                                                                                                                                                                 │ │
│ │ 4.1 Portfolio-Level Risk Controls                                                                                                                               │ │
│ │                                                                                                                                                                 │ │
│ │ - Maximum position concentration: 5% max in any single stock                                                                                                    │ │
│ │ - Sector concentration limits: 20% max in any sector across stocks                                                                                              │ │
│ │ - Stock vs ETF allocation: Track total allocation to each asset class                                                                                           │ │
│ │ - Correlation monitoring: Enhanced correlation tracking for individual stocks                                                                                   │ │
│ │                                                                                                                                                                 │ │
│ │ 4.2 Position Sizing Adjustments                                                                                                                                 │ │
│ │                                                                                                                                                                 │ │
│ │ - Beta-adjusted sizing: Incorporate stock beta into position calculations                                                                                       │ │
│ │ - Liquidity requirements: Higher volume thresholds for individual stocks                                                                                        │ │
│ │ - Market cap considerations: Different sizing for large-cap vs mid-cap                                                                                          │ │
│ │                                                                                                                                                                 │ │
│ │ 5. New Components Needed                                                                                                                                        │ │
│ │                                                                                                                                                                 │ │
│ │ 5.1 stock_analyzer.py (New)                                                                                                                                     │ │
│ │                                                                                                                                                                 │ │
│ │ - Stock-specific analysis functions                                                                                                                             │ │
│ │ - Earnings calendar integration (future enhancement)                                                                                                            │ │
│ │ - Sector rotation analysis                                                                                                                                      │ │
│ │ - Individual stock correlation tracking                                                                                                                         │ │
│ │                                                                                                                                                                 │ │
│ │ 5.2 Enhanced Journal System                                                                                                                                     │ │
│ │                                                                                                                                                                 │ │
│ │ - Separate performance tracking for stocks vs ETFs                                                                                                              │ │
│ │ - Sector allocation monitoring                                                                                                                                  │ │
│ │ - Stock-specific trade notes and analysis                                                                                                                       │ │
│ │ - Enhanced correlation tracking                                                                                                                                 │ │
│ │                                                                                                                                                                 │ │
│ │ 6. Testing Strategy                                                                                                                                             │ │
│ │                                                                                                                                                                 │ │
│ │ 6.1 Separate Test Suites                                                                                                                                        │ │
│ │                                                                                                                                                                 │ │
│ │ - Unit tests: Test stock-specific modifications to existing setups                                                                                              │ │
│ │ - Integration tests: Test stock data loading and caching                                                                                                        │ │
│ │ - Backtest validation: Compare stock vs ETF performance in different regimes                                                                                    │ │
│ │ - Risk system tests: Verify portfolio-level risk controls                                                                                                       │ │
│ │                                                                                                                                                                 │ │
│ │ 6.2 Gradual Rollout                                                                                                                                             │ │
│ │                                                                                                                                                                 │ │
│ │ 1. Phase 1: Load stock data and test caching (10 core stocks)                                                                                                   │ │
│ │ 2. Phase 2: Test trade setups with adjusted parameters (20 stocks)                                                                                              │ │
│ │ 3. Phase 3: Full screening and backtesting (all 56 stocks)                                                                                                      │ │
│ │ 4. Phase 4: Live paper trading integration                                                                                                                      │ │
│ │                                                                                                                                                                 │ │
│ │ 7. Configuration & Commands                                                                                                                                     │ │
│ │                                                                                                                                                                 │ │
│ │ 7.1 New Command Patterns                                                                                                                                        │ │
│ │                                                                                                                                                                 │ │
│ │ # Data management                                                                                                                                               │ │
│ │ python init_database.py --bootstrap stocks_core                                                                                                                 │ │
│ │ python backfill_stocks.py --symbols-from stocks_list.csv                                                                                                        │ │
│ │                                                                                                                                                                 │ │
│ │ # Screening                                                                                                                                                     │ │
│ │ python screener.py --type stock --sector Technology --min-confidence 0.6                                                                                        │ │
│ │ python screener.py --type all --regime-filter --export-csv                                                                                                      │ │
│ │                                                                                                                                                                 │ │
│ │ # Backtesting                                                                                                                                                   │ │
│ │ python backtest.py --type stock --setup trend_pullback --walk-forward                                                                                           │ │
│ │ python backtest.py --type all --regime-aware --compare-types                                                                                                    │ │
│ │                                                                                                                                                                 │ │
│ │ # Analysis                                                                                                                                                      │ │
│ │ python stock_analyzer.py --sector Technology --correlation-matrix                                                                                               │ │
│ │ python report.py --type stock --vs-sector-etf --monthly                                                                                                         │ │
│ │                                                                                                                                                                 │ │
│ │ 8. File Changes Summary                                                                                                                                         │ │
│ │                                                                                                                                                                 │ │
│ │ Modified Files:                                                                                                                                                 │ │
│ │                                                                                                                                                                 │ │
│ │ 1. init_database.py: Add stock loading and bootstrap functions                                                                                                  │ │
│ │ 2. trade_setups.py: Parameter adjustments and stock-specific validation                                                                                         │ │
│ │ 3. screener.py: Add type filtering and stock-specific formatting                                                                                                │ │
│ │ 4. backtest.py: Enhanced performance tracking and comparisons                                                                                                   │ │
│ │ 5. data_cache.py: Minor stock-specific considerations                                                                                                           │ │
│ │                                                                                                                                                                 │ │
│ │ New Files:                                                                                                                                                      │ │
│ │                                                                                                                                                                 │ │
│ │ 1. backfill_stocks.py: Stock data backfill utility                                                                                                              │ │
│ │ 2. stock_analyzer.py: Stock-specific analysis tools                                                                                                             │ │
│ │ 3. stocks_list.csv: Already provided                                                                                                                            │ │
│ │                                                                                                                                                                 │ │
│ │ Updated Configuration:                                                                                                                                          │ │
│ │                                                                                                                                                                 │ │
│ │ 1. Command-line interfaces: Enhanced with stock support                                                                                                         │ │
│ │ 2. Database queries: Type-aware symbol retrieval                                                                                                                │ │
│ │ 3. Risk management: Portfolio-level controls                                                                                                                    │ │
│ │                                                                                                                                                                 │ │
│ │ 9. Implementation Priority                                                                                                                                      │ │
│ │                                                                                                                                                                 │ │
│ │ High Priority (Core Functionality):                                                                                                                             │ │
│ │                                                                                                                                                                 │ │
│ │ 1. Update init_database.py stock loading                                                                                                                        │ │
│ │ 2. Create backfill_stocks.py                                                                                                                                    │ │
│ │ 3. Adjust trade_setups.py parameters                                                                                                                            │ │
│ │ 4. Update screener.py type filtering                                                                                                                            │ │
│ │                                                                                                                                                                 │ │
│ │ Medium Priority (Enhanced Features):                                                                                                                            │ │
│ │                                                                                                                                                                 │ │
│ │ 1. Enhanced backtest.py comparisons                                                                                                                             │ │
│ │ 2. Risk management controls                                                                                                                                     │ │
│ │ 3. Stock-specific analysis tools                                                                                                                                │ │
│ │                                                                                                                                                                 │ │
│ │ Low Priority (Future Enhancements):                                                                                                                             │ │
│ │                                                                                                                                                                 │ │
│ │ 1. Earnings calendar integration                                                                                                                                │ │
│ │ 2. Advanced correlation analysis                                                                                                                                │ │
│ │ 3. Dynamic parameter optimization for stocks                                                                                                                    │ │
│ │                                                                                                                                                                 │ │
│ │ This plan maintains the excellent existing architecture while thoughtfully extending it to support individual stocks with appropriate risk controls and         │ │
│ │ performance tracking.             