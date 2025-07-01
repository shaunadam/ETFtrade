#!/usr/bin/env python3
"""
Detailed analysis of treasury rate data for Sharpe ratio implementation.
Tests data quality, missing values, and provides practical examples.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_risk_free_data(symbol, name, test_period='2y'):
    """Analyze a risk-free rate ticker for Sharpe ratio suitability."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {symbol} - {name}")
    print(f"{'='*60}")
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=test_period)
        
        if hist.empty:
            print(f"❌ No data available for {symbol}")
            return None
        
        # Basic stats
        print(f"📊 DATA OVERVIEW:")
        print(f"   Total data points: {len(hist)}")
        print(f"   Date range: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Missing values: {hist['Close'].isna().sum()}")
        
        # For yield tickers, analyze the rate values
        if symbol.startswith('^'):
            print(f"   Current rate: {hist['Close'].iloc[-1]:.4f}%")
            print(f"   2-year average: {hist['Close'].mean():.4f}%")
            print(f"   2-year range: {hist['Close'].min():.4f}% - {hist['Close'].max():.4f}%")
            print(f"   Volatility (annualized): {hist['Close'].std() * np.sqrt(252):.4f}%")
            
            # Test for data quality
            zero_values = (hist['Close'] == 0).sum()
            negative_values = (hist['Close'] < 0).sum()
            print(f"   Zero values: {zero_values}")
            print(f"   Negative values: {negative_values}")
            
            # Recent data availability
            recent_data = hist.tail(10)
            print(f"\n📈 RECENT VALUES (last 10 trading days):")
            for date, row in recent_data.iterrows():
                print(f"   {date.strftime('%Y-%m-%d')}: {row['Close']:.4f}%")
                
        else:
            # For ETFs, analyze price and volume
            print(f"   Current price: ${hist['Close'].iloc[-1]:.2f}")
            print(f"   2-year return: {((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100:.2f}%")
            print(f"   Average volume: {hist['Volume'].mean():,.0f}")
            print(f"   Price volatility (annualized): {(hist['Close'].pct_change().std() * np.sqrt(252) * 100):.2f}%")
        
        # Data consistency checks
        business_days = pd.bdate_range(hist.index[0], hist.index[-1])
        expected_days = len(business_days)
        actual_days = len(hist)
        coverage = actual_days / expected_days * 100
        
        print(f"\n🔍 DATA QUALITY:")
        print(f"   Expected trading days: {expected_days}")
        print(f"   Actual data points: {actual_days}")
        print(f"   Data coverage: {coverage:.1f}%")
        
        # Check for gaps
        hist_dates = set(hist.index.date)
        business_dates = set(business_days.date)
        missing_dates = business_dates - hist_dates
        
        if missing_dates:
            print(f"   Missing dates: {len(missing_dates)} (showing first 5)")
            for date in sorted(missing_dates)[:5]:
                print(f"     {date}")
        else:
            print(f"   Missing dates: 0 ✅")
            
        return {
            'symbol': symbol,
            'data_points': len(hist),
            'coverage': coverage,
            'missing_values': hist['Close'].isna().sum(),
            'current_value': hist['Close'].iloc[-1],
            'data_quality_score': coverage - (hist['Close'].isna().sum() / len(hist) * 100)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        return None

def calculate_sharpe_examples(risk_free_symbol='^IRX'):
    """Show practical examples of Sharpe ratio calculation."""
    print(f"\n{'='*80}")
    print(f"SHARPE RATIO CALCULATION EXAMPLES")
    print(f"{'='*80}")
    
    try:
        # Get risk-free rate data
        rf_ticker = yf.Ticker(risk_free_symbol)
        rf_data = rf_ticker.history(period='1y')
        
        # Get SPY data for example
        spy = yf.Ticker('SPY')
        spy_data = spy.history(period='1y')
        
        if rf_data.empty or spy_data.empty:
            print("❌ Insufficient data for examples")
            return
        
        # Align dates
        common_dates = rf_data.index.intersection(spy_data.index)
        rf_rates = rf_data.loc[common_dates, 'Close'] / 100  # Convert percentage to decimal
        spy_prices = spy_data.loc[common_dates, 'Close']
        
        print(f"📊 EXAMPLE: SPY vs {risk_free_symbol} (past year)")
        print(f"   Data points: {len(common_dates)}")
        print(f"   Date range: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}")
        
        # Calculate daily returns
        spy_returns = spy_prices.pct_change().dropna()
        
        # Calculate annualized metrics
        annual_spy_return = ((spy_prices.iloc[-1] / spy_prices.iloc[0]) ** (252/len(spy_prices)) - 1)
        annual_rf_rate = rf_rates.mean()  # Already annualized
        annual_spy_vol = spy_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_spy_return - annual_rf_rate) / annual_spy_vol
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   SPY Annual Return: {annual_spy_return:.2%}")
        print(f"   Risk-Free Rate: {annual_rf_rate:.2%}")
        print(f"   SPY Volatility: {annual_spy_vol:.2%}")
        print(f"   Excess Return: {(annual_spy_return - annual_rf_rate):.2%}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
        
        # Show implementation code
        print(f"\n💻 IMPLEMENTATION CODE:")
        print(f"""
# Get risk-free rate data
rf_ticker = yf.Ticker('{risk_free_symbol}')
rf_data = rf_ticker.history(period='1y')
rf_annual_rate = rf_data['Close'].mean() / 100  # Convert % to decimal

# Calculate portfolio metrics
portfolio_annual_return = 0.12  # Example: 12% annual return
portfolio_annual_vol = 0.18     # Example: 18% annual volatility

# Calculate Sharpe ratio
sharpe_ratio = (portfolio_annual_return - rf_annual_rate) / portfolio_annual_vol
print(f"Sharpe Ratio: {{sharpe_ratio:.3f}}")

# For daily Sharpe calculations:
daily_rf_rate = (1 + rf_annual_rate) ** (1/252) - 1
daily_excess_returns = daily_portfolio_returns - daily_rf_rate
daily_sharpe = daily_excess_returns.mean() / daily_excess_returns.std() * np.sqrt(252)
""")
        
    except Exception as e:
        print(f"❌ Error in Sharpe calculation example: {e}")

def test_data_update_frequency():
    """Test how frequently different data sources update."""
    print(f"\n{'='*80}")
    print(f"DATA UPDATE FREQUENCY TEST")
    print(f"{'='*80}")
    
    symbols = ['^IRX', '^TNX', 'BIL', 'SHY']
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Get last 5 days of data
            recent = ticker.history(period='5d')
            
            if not recent.empty:
                last_update = recent.index[-1]
                days_ago = (datetime.now().date() - last_update.date()).days
                
                print(f"\n📅 {symbol}:")
                print(f"   Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')} ({days_ago} days ago)")
                print(f"   Recent data points: {len(recent)}")
                
                # Check if data is current
                if days_ago <= 1:
                    print(f"   Status: ✅ Current")
                elif days_ago <= 3:
                    print(f"   Status: ⚠️  Slightly delayed")
                else:
                    print(f"   Status: ❌ Outdated")
                    
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")

def main():
    print("TREASURY RATE DATA DETAILED ANALYSIS")
    print("For ETF Trading System Sharpe Ratio Implementation")
    
    # Analyze key candidates
    candidates = [
        ('^IRX', '3-Month Treasury Bill Yield'),
        ('^TNX', '10-Year Treasury Constant Maturity Rate'),
        ('^FVX', '5-Year Treasury Constant Maturity Rate'),
        ('BIL', 'SPDR Bloomberg 1-3 Month T-Bill ETF'),
        ('SHY', 'iShares 1-3 Year Treasury Bond ETF'),
    ]
    
    results = []
    for symbol, name in candidates:
        result = analyze_risk_free_data(symbol, name)
        if result:
            results.append(result)
    
    # Summary comparison
    if results:
        print(f"\n{'='*80}")
        print(f"SUMMARY COMPARISON")
        print(f"{'='*80}")
        print(f"{'Symbol':<8} {'Data Points':<12} {'Coverage %':<12} {'Missing':<8} {'Quality Score':<12}")
        print(f"{'-'*60}")
        
        for r in sorted(results, key=lambda x: x['data_quality_score'], reverse=True):
            print(f"{r['symbol']:<8} {r['data_points']:<12} {r['coverage']:<12.1f} {r['missing_values']:<8} {r['data_quality_score']:<12.1f}")
    
    # Practical examples
    calculate_sharpe_examples('^IRX')
    
    # Update frequency test
    test_data_update_frequency()
    
    # Final recommendations
    print(f"\n{'='*80}")
    print(f"FINAL RECOMMENDATIONS FOR ETF TRADING SYSTEM")
    print(f"{'='*80}")
    
    print(f"""
🏆 RECOMMENDED IMPLEMENTATION:

1. PRIMARY: ^IRX (3-Month Treasury Bill Yield)
   - Use for: Daily Sharpe ratio calculations
   - Pros: True risk-free rate, excellent data quality, daily updates
   - Cons: None significant
   - Implementation: rf_rate = yf.Ticker('^IRX').history(period='1d')['Close'].iloc[-1] / 100

2. SECONDARY: ^TNX (10-Year Treasury)
   - Use for: Longer-term performance analysis, benchmark comparisons
   - Pros: Stable, widely used, good for regime detection
   - Cons: Less appropriate for short-term trades
   - Implementation: Similar to ^IRX

3. BACKUP: BIL ETF
   - Use for: When yield tickers fail, correlation analysis
   - Pros: ETF liquidity, can track distributions
   - Cons: Need to calculate implied yield from price changes
   - Implementation: Calculate yield from price appreciation + distributions

🔧 INTEGRATION SUGGESTIONS:

1. Add to data_cache.py:
   - Cache ^IRX daily values
   - Implement fallback to ^TNX if ^IRX fails
   - Store in indicators table as 'risk_free_rate'

2. Add to trade_setups.py:
   - Include current risk-free rate in trade evaluation
   - Use for position sizing based on risk-adjusted returns

3. Add to report.py:
   - Calculate and display Sharpe ratios for all strategies
   - Compare against risk-free rate benchmarks
   - Track regime changes in risk-free environment

4. Error handling:
   - Implement fallback chain: ^IRX -> ^TNX -> BIL -> fixed rate (e.g., 2%)
   - Cache last known good value for continuity
   - Alert if data is >3 days old
""")

if __name__ == "__main__":
    main()