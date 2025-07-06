#!/usr/bin/env python3
"""Test script for enhanced backtest engine with professional metrics."""

import sys
from datetime import datetime, timedelta
from backtest import BacktestEngine, PerformanceMetrics
from trade_setups import SetupType

def test_enhanced_backtest():
    """Test the enhanced backtest engine with professional metrics."""
    
    print("🧪 Testing Enhanced Backtest Engine")
    print("=" * 50)
    
    # Initialize engine
    engine = BacktestEngine()
    
    # Test with a short time period to ensure we have data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months back
    
    print(f"📅 Testing period: {start_date.date()} to {end_date.date()}")
    
    try:
        # Run a simple backtest with one setup
        print("\n🚀 Running backtest with trend pullback setup...")
        
        results = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            setup_types=[SetupType.TREND_PULLBACK],
            walk_forward=False,
            regime_aware=True
        )
        
        if 'performance' in results:
            metrics = results['performance']
            
            print("\n📊 Basic Metrics:")
            print(f"   Total Trades: {metrics.total_trades}")
            print(f"   Win Rate: {metrics.win_rate:.2%}")
            print(f"   Total Return: {metrics.total_return:.2%}")
            print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
            print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            
            print("\n🎯 Professional Metrics (Phase 1):")
            print(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
            print(f"   Information Ratio: {metrics.information_ratio:.2f}")
            print(f"   Annual Return: {metrics.annual_return:.2%}")
            print(f"   Downside Deviation: {metrics.downside_deviation:.2%}")
            print(f"   Benchmark Return: {metrics.benchmark_return:.2%}")
            print(f"   Tracking Error: {metrics.tracking_error:.2%}")
            
            print("\n📈 MAE/MFE Analysis:")
            print(f"   Max Adverse Excursion: {metrics.max_adverse_excursion:.2f}%")
            print(f"   Max Favorable Excursion: {metrics.max_favorable_excursion:.2f}%")
            print(f"   Average MAE: {metrics.avg_mae:.2f}%")
            print(f"   Average MFE: {metrics.avg_mfe:.2f}%")
            
            print("\n✅ Enhanced backtest completed successfully!")
            
        else:
            print("❌ No performance metrics found in results")
            
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = test_enhanced_backtest()
    sys.exit(0 if success else 1)