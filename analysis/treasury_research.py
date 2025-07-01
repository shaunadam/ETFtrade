#!/usr/bin/env python3
"""
Research script to investigate treasury rate and risk-free rate data available through yfinance.
This will help determine the best options for calculating Sharpe ratios in the trading system.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_ticker(symbol, description):
    """Test a ticker symbol and return basic info about available data."""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get recent data (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            return {
                'symbol': symbol,
                'description': description,
                'status': 'NO DATA',
                'latest_value': None,
                'data_points': 0,
                'date_range': None
            }
        
        # Get info if available
        info = {}
        try:
            info = ticker.info
        except:
            info = {}
        
        return {
            'symbol': symbol,
            'description': description,
            'status': 'SUCCESS',
            'latest_value': hist['Close'].iloc[-1] if len(hist) > 0 else None,
            'latest_date': hist.index[-1].strftime('%Y-%m-%d') if len(hist) > 0 else None,
            'data_points': len(hist),
            'date_range': f"{hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}" if len(hist) > 0 else None,
            'avg_volume': hist['Volume'].mean() if 'Volume' in hist.columns and not hist['Volume'].isna().all() else None,
            'info_available': len(info) > 0,
            'long_name': info.get('longName', 'N/A') if info else 'N/A'
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'description': description,
            'status': f'ERROR: {str(e)}',
            'latest_value': None,
            'data_points': 0,
            'date_range': None
        }

def main():
    print("=" * 80)
    print("TREASURY RATE AND RISK-FREE RATE DATA RESEARCH")
    print("=" * 80)
    print()
    
    # Define symbols to test
    test_symbols = [
        # Treasury Yield Tickers (Yahoo Finance format)
        ('^TNX', '10-Year Treasury Constant Maturity Rate'),
        ('^IRX', '3-Month Treasury Bill Yield'),
        ('^FVX', '5-Year Treasury Constant Maturity Rate'),
        ('^TYX', '30-Year Treasury Constant Maturity Rate'),
        ('^DJI', 'Dow Jones (test control)'),  # Control to verify yfinance is working
        
        # Treasury ETFs
        ('SHY', 'iShares 1-3 Year Treasury Bond ETF'),
        ('IEF', 'iShares 7-10 Year Treasury Bond ETF'),
        ('TLT', 'iShares 20+ Year Treasury Bond ETF'),
        ('BIL', 'SPDR Bloomberg 1-3 Month T-Bill ETF'),
        ('SCHO', 'Schwab Short-Term U.S. Treasury ETF'),
        ('SCHR', 'Schwab Intermediate-Term U.S. Treasury ETF'),
        ('SCHQ', 'Schwab Long-Term U.S. Treasury ETF'),
        
        # Money Market and Short-term Rates
        ('SPAXX', 'Fidelity Government Money Market Fund'),  # May not work
        ('VMFXX', 'Vanguard Federal Money Market Fund'),      # May not work
        
        # Additional Treasury-related symbols
        ('DGS10', 'FRED 10-Year Treasury Rate'),              # May not work directly
        ('DGS3MO', 'FRED 3-Month Treasury Rate'),             # May not work directly
        ('TB3MS', 'FRED 3-Month Treasury Bill Rate'),         # May not work directly
        
        # International alternatives
        ('GLD', 'SPDR Gold Trust ETF'),  # As alternative risk-free proxy
        ('CASH', 'JPMorgan Ultra-Short Income ETF'),
    ]
    
    results = []
    
    print("Testing ticker symbols...")
    print()
    
    for symbol, description in test_symbols:
        print(f"Testing {symbol}: {description}")
        result = test_ticker(symbol, description)
        results.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"  ✓ SUCCESS - Latest: {result['latest_value']:.4f}% on {result['latest_date']}")
            print(f"    Data points: {result['data_points']}, Range: {result['date_range']}")
        else:
            print(f"  ✗ {result['status']}")
        print()
    
    # Summary and recommendations
    print("=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    # Working treasury yield tickers
    working_yields = [r for r in results if r['status'] == 'SUCCESS' and r['symbol'].startswith('^')]
    if working_yields:
        print("✓ WORKING TREASURY YIELD TICKERS:")
        for r in working_yields:
            print(f"  {r['symbol']}: {r['description']}")
            print(f"    Latest: {r['latest_value']:.4f}% | Data points: {r['data_points']}")
    else:
        print("✗ No working treasury yield tickers found")
    
    print()
    
    # Working treasury ETFs
    working_etfs = [r for r in results if r['status'] == 'SUCCESS' and not r['symbol'].startswith('^') and r['symbol'] not in ['GLD']]
    if working_etfs:
        print("✓ WORKING TREASURY ETFs:")
        for r in working_etfs:
            volume_str = f" | Avg Volume: {r['avg_volume']:,.0f}" if r['avg_volume'] else ""
            print(f"  {r['symbol']}: {r['description']}")
            print(f"    Latest: ${r['latest_value']:.2f} | Data points: {r['data_points']}{volume_str}")
    
    print()
    
    # Recommendations for Sharpe ratio calculation
    print("RECOMMENDATIONS FOR SHARPE RATIO CALCULATION:")
    print("-" * 50)
    
    if any(r['symbol'] == '^IRX' and r['status'] == 'SUCCESS' for r in results):
        print("🏆 PRIMARY RECOMMENDATION: ^IRX (3-Month Treasury Bill)")
        print("   - Most commonly used risk-free rate for Sharpe ratios")
        print("   - Represents actual risk-free rate, not bond prices")
        print("   - Updated regularly, good data availability")
        print()
    
    if any(r['symbol'] == '^TNX' and r['status'] == 'SUCCESS' for r in results):
        print("🥈 SECONDARY RECOMMENDATION: ^TNX (10-Year Treasury)")
        print("   - Good for longer-term investment horizons")
        print("   - More stable than short-term rates")
        print("   - Widely used benchmark")
        print()
    
    if any(r['symbol'] == 'BIL' and r['status'] == 'SUCCESS' for r in results):
        print("🥉 ETF ALTERNATIVE: BIL (1-3 Month T-Bill ETF)")
        print("   - Can calculate implied yield from price changes")
        print("   - More liquid than direct rate tickers")
        print("   - Good for portfolio correlation analysis")
        print()
    
    print("IMPLEMENTATION NOTES:")
    print("- Treasury yield tickers (^IRX, ^TNX) provide yields as percentages")
    print("- Convert to decimal for Sharpe ratio: yield / 100")
    print("- For daily Sharpe ratios, convert annual rate: (1 + annual_rate)^(1/252) - 1")
    print("- ETF yields need to be calculated from distributions and price appreciation")
    
    # Test longer historical data for the best candidates
    print()
    print("=" * 80)
    print("HISTORICAL DATA AVAILABILITY TEST")
    print("=" * 80)
    
    best_candidates = ['^IRX', '^TNX', 'BIL', 'SHY']
    for symbol in best_candidates:
        if any(r['symbol'] == symbol and r['status'] == 'SUCCESS' for r in results):
            print(f"\nTesting historical data for {symbol}:")
            try:
                ticker = yf.Ticker(symbol)
                # Test 5 years of data
                hist_5y = ticker.history(period='5y')
                hist_max = ticker.history(period='max')
                
                if not hist_5y.empty:
                    print(f"  5-year data: {len(hist_5y)} points ({hist_5y.index[0].strftime('%Y-%m-%d')} to {hist_5y.index[-1].strftime('%Y-%m-%d')})")
                
                if not hist_max.empty:
                    print(f"  Max data: {len(hist_max)} points ({hist_max.index[0].strftime('%Y-%m-%d')} to {hist_max.index[-1].strftime('%Y-%m-%d')})")
                    
                    # Show sample values for yield tickers
                    if symbol.startswith('^'):
                        recent_avg = hist_max['Close'].tail(30).mean()
                        print(f"  Recent 30-day average: {recent_avg:.4f}%")
                        
            except Exception as e:
                print(f"  Error testing historical data: {e}")

if __name__ == "__main__":
    main()