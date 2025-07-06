# Backtest Performance Optimization Plan

## Performance Issue Analysis

### Current Problem
The backtesting system is unacceptably slow when processing multiple symbols. Observed behavior:
- Several seconds between "Filtering to 2 selected..." log messages
- Slow day-by-day processing loop
- Poor user experience when running backtests through the Flask frontend

### Root Cause Investigation
The main bottleneck is in the `_run_standard_backtest()` method which:
1. Loops through each trading day sequentially
2. Calls `_get_signals_for_date()` for each day
3. Performs symbol filtering on every iteration
4. Makes individual database/cache lookups for price data
5. Processes positions one by one with repeated price lookups

## Optimization Strategy

### 1. Pre-compute Symbol Mapping (High Impact)
**Problem**: Symbol filtering logic runs on every trading day
**Solution**: 
- Move symbol filtering from daily loop to backtest initialization
- Create symbol mapping dictionary once at start
- Eliminate repeated string normalization operations
- Cache available symbols for the entire backtest period

### 2. Batch Data Pre-loading (High Impact)
**Problem**: Individual price lookups for each day/symbol combination
**Solution**:
- Pre-load all required price data for the entire backtest period
- Replace `_get_price_for_date()` calls with cached dictionary lookups
- Reduce database/cache hits from O(days × symbols) to O(symbols)
- Create price data matrix for vectorized operations

### 3. Optimize Position Management (Medium Impact)
**Problem**: Individual position updates with repeated price lookups
**Solution**:
- Batch position updates with pre-loaded price data
- Vectorize MAE/MFE calculations where possible
- Reduce individual price lookups in `_update_positions()`
- Cache ATR and other technical indicators

### 4. Reduce Logging Overhead (Low Impact)
**Problem**: Frequent logging operations during tight loops
**Solution**:
- Move frequent info logs to debug level
- Add progress indicators instead of per-day logging
- Reduce I/O operations during performance-critical loops
- Implement progress bars for user feedback

### 5. Setup Scanning Optimization (Medium Impact)
**Problem**: Repeated technical indicator calculations
**Solution**:
- Cache technical indicators between runs
- Use vectorized operations for signal detection
- Optimize regime filtering logic
- Pre-compute common indicators like SMA, RSI, ATR

## Implementation Order

### Phase 1: Core Performance Fixes (Immediate)
1. **Symbol Mapping Pre-computation**
   - Extract symbol filtering from daily loop
   - Create symbol lookup dictionary at initialization
   - Eliminate repeated string operations

2. **Batch Data Pre-loading**
   - Pre-load all price data for backtest period
   - Replace individual lookups with dictionary access
   - Create price data structures for efficient access

### Phase 2: Advanced Optimizations (Next)
3. **Position Management Optimization**
   - Batch position updates
   - Vectorize calculations where possible
   - Reduce price lookup redundancy

4. **Progress Tracking**
   - Add progress bars for user feedback
   - Replace verbose logging with progress indicators
   - Improve Flask frontend feedback

### Phase 3: Technical Optimizations (Later)
5. **Technical Indicator Caching**
   - Cache calculated indicators
   - Use vectorized operations
   - Optimize regime detection

## Testing Strategy

### Performance Testing
- Create small test dataset (2 symbols, 30 days)
- Measure performance before and after each optimization
- Use Python profiling tools to identify remaining bottlenecks
- Test with larger datasets (10+ symbols, 1+ years)

### Regression Testing
- Ensure optimization doesn't change backtest results
- Compare trade counts, returns, and metrics
- Test edge cases (no trades, all trades, etc.)

### User Experience Testing
- Test Flask frontend with progress indicators
- Verify cancellation functionality works
- Ensure error handling remains robust

## Success Metrics

### Performance Targets
- **Primary**: 2-symbol, 1-year backtest completes in <30 seconds
- **Secondary**: 10-symbol, 1-year backtest completes in <2 minutes
- **Stretch**: Progress updates every 5-10 seconds during execution

### Code Quality Targets
- No regression in backtest accuracy
- Maintain existing error handling
- Keep configuration system intact
- Preserve all existing functionality

## Implementation Notes

### Files to Modify
- `backtest.py`: Main performance optimizations
- `flask_app/services/backtest_service.py`: Progress tracking
- `flask_app/templates/backtest/index.html`: Progress indicators
- `backtest_config.py`: Add performance-related config options

### Preservation Requirements
- Keep all existing functionality
- Maintain backward compatibility
- Preserve configuration system
- Keep comprehensive logging (at debug level)

## Next Steps

1. **Immediate**: Implement Phase 1 optimizations
2. **Short-term**: Add progress tracking for user feedback
3. **Medium-term**: Complete Phase 2 optimizations
4. **Long-term**: Profile and optimize remaining bottlenecks

## Risk Mitigation

- **Incremental changes**: Test each optimization separately
- **Regression testing**: Verify no accuracy changes
- **Rollback plan**: Keep original code structure intact
- **Performance monitoring**: Track improvements at each step