# Backtesting System Improvement Plan

## Overview
Comprehensive improvements to resolve symbol leakage, performance issues, and incomplete Flask integration.

## Issues Identified

### 1. Symbol Leakage in Flask App
When testing 2 symbols in frontend, logs show other symbols being tested - this indicates the Flask integration isn't properly filtering/limiting symbols to the selected instruments.

### 2. General Slowness
The backtesting is running slow, likely due to inefficient data access patterns and unnecessary computation across all symbols rather than just selected ones.

### 3. Poor Integration
The Flask service layer has placeholder/incomplete implementations for selected instrument filtering with TODO comments in production code.

### 4. CLI vs Web Inconsistency
The CLI backtest.py works well but the web integration doesn't properly utilize its capabilities for selective instrument backtesting.

## Root Causes Analysis

1. **Incomplete Flask Integration**: The `_backtest_selected_instruments` methods in `backtest_service.py` are incomplete with TODO comments
2. **Inefficient Data Access**: The backtest engine scans all symbols when it should only process selected ones
3. **No Proper Instrument Filtering**: The current system doesn't properly filter at the data level
4. **Unnecessary Computation**: Running full market scans when only specific instruments are selected
5. **Missing Progress Tracking**: No granular progress feedback for long-running operations

## Phase 1: Core Engine Improvements

### 1. Add Instrument Filtering to BacktestEngine
- Modify `run_backtest()` to accept `instrument_filter` parameter
- Update `_get_signals_for_date()` to only process filtered symbols
- Add `backtest_selected_instruments()` method for direct instrument filtering

### 2. Optimize Data Access Patterns
- Pre-fetch data only for selected instruments
- Cache filtered symbol lists to avoid repeated database queries
- Add lazy loading for unnecessary data

### 3. Improve Progress Tracking
- Add progress callback support to BacktestEngine
- Implement granular progress reporting (symbol-by-symbol, date-by-date)
- Add cancellation support for long-running operations

## Phase 2: Flask Integration Fixes

### 1. Complete BacktestService Implementation
- Remove placeholder methods and implement proper instrument filtering
- Fix `_backtest_selected_instruments()` to actually filter symbols
- Add proper error handling and logging

### 2. Enhance API Endpoints
- Add progress tracking endpoint for real-time updates
- Implement proper validation for selected instruments
- Add cancellation endpoint for long-running backtests

### 3. Frontend Integration
- Update backtest UI to show proper progress for selected instruments
- Add validation to prevent running backtests on invalid symbol selections
- Implement proper loading states and cancellation

## Phase 3: Performance Optimizations

### 1. Caching Improvements
- Cache regime detection results to avoid repeated calculations
- Pre-calculate technical indicators for selected instruments only
- Implement smart caching for backtest results

### 2. Parallel Processing
- Add support for parallel symbol processing where possible
- Implement async operations for data fetching
- Add batch processing for multiple date ranges

## Phase 4: Testing & Validation

### 1. Add Integration Tests
- Test symbol filtering works correctly
- Verify no symbol leakage in Flask app
- Performance benchmarks for different instrument counts

### 2. Create Comprehensive Documentation
- Document all improvements and architectural changes
- Include performance benchmarks and optimization guidelines
- Add troubleshooting guide for common issues

## Key Deliverables
- Fixed symbol leakage in Flask app
- 50%+ performance improvement for selective backtesting
- Complete Flask integration without placeholder methods
- Comprehensive documentation updates
- Integration tests ensuring reliability

## Success Criteria
- Web app only processes selected instruments (no symbol leakage)
- Backtesting 2 symbols completes in <30 seconds
- Progress tracking shows accurate real-time updates
- All TODO comments removed from production code

## Implementation Priority
1. **High Priority**: Fix symbol leakage and complete Flask integration
2. **Medium Priority**: Performance optimizations and progress tracking
3. **Low Priority**: Parallel processing and advanced caching

## Files to Modify
- `backtest.py` - Core engine improvements
- `flask_app/services/backtest_service.py` - Complete integration
- `flask_app/blueprints/backtest/__init__.py` - API enhancements
- `flask_app/templates/backtest/index.html` - Frontend improvements
- `tests/test_backtest.py` - Integration tests

## Testing Strategy
- Unit tests for instrument filtering logic
- Integration tests for Flask API endpoints
- Performance benchmarks for different symbol counts
- End-to-end tests for web workflow