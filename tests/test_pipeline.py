#!/usr/bin/env python3
"""
Test pipeline for enhanced backtest engine.
"""

from datetime import datetime
from screener import ETFScreener
from backtest import BacktestEngine, OptimizationParameters

def test_pipeline():
    """Test the screener -> backtest pipeline."""
    
    print("=" * 60)
    print("TESTING SCREENER -> BACKTEST PIPELINE")
    print("=" * 60)
    
    # Step 1: Get current signals from screener
    print("\n🔍 Step 1: Running screener to find triggered signals...")
    screener = ETFScreener()
    
    # Get signals from all setups with regime filtering
    signals = screener.screen_etfs(
        setup_filter=None,  # All setups
        min_confidence=0.5,
        max_signals=10,
        regime_filter=True
    )
    
    print(f"📈 Found {len(signals)} triggered signals")
    if signals:
        print("\nTriggered signals:")
        for i, signal in enumerate(signals, 1):
            print(f"  {i}. {signal.symbol} - {signal.setup_type.value} (confidence: {signal.confidence:.1%})")
    else:
        print("❌ No signals found - cannot test backtest pipeline")
        return
    
    # Step 2: Backtest only those specific signals
    print(f"\n🎯 Step 2: Backtesting {len(signals)} triggered signals...")
    
    engine = BacktestEngine()
    results = engine.backtest_from_signals(
        signals=signals,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 6, 15)
    )
    
    # Step 3: Display results
    print("\n" + "=" * 60)
    print("PIPELINE BACKTEST RESULTS")
    print("=" * 60)
    
    perf = results['performance']
    print(f"Signals Fed to Backtest: {results['signals_tested']}")
    print(f"Unique Symbols: {results['unique_symbols']}")
    print(f"Setups Tested: {', '.join(results['setups_tested'])}")
    print(f"Period: 2025-01-01 to 2025-06-15")
    print("-" * 60)
    print(f"Total Trades: {perf.total_trades}")
    print(f"Win Rate: {perf.win_rate:.1%}")
    print(f"Average R-Multiple: {perf.avg_r_multiple:.2f}")
    print(f"Total Return: {perf.total_return:.1%}")
    print(f"Maximum Drawdown: {perf.max_drawdown:.1%}")
    print(f"Sharpe Ratio: {perf.sharpe_ratio:.2f}")
    print(f"Profit Factor: {perf.profit_factor:.2f}")
    print(f"Average Days Held: {perf.avg_days_held:.1f}")
    print("=" * 60)
    
    # Show individual trades if any
    if results['trades']:
        print("\nIndividual Trades:")
        for trade in results['trades']:
            status = trade['status']
            pnl = trade.get('pnl', 0) or 0
            r_mult = trade.get('r_multiple', 0) or 0
            print(f"  {trade['symbol']} ({trade['setup_type']}): "
                  f"${pnl:.2f} (R: {r_mult:.2f}) - {status}")
    
    print(f"\n✅ Pipeline test complete!")
    print(f"📊 Successfully tested {len(signals)} triggered signal(s)")

def test_enhanced_features():
    """Test enhanced features: parameter optimization and regime analysis."""
    
    print("\n" + "="*60)
    print("TESTING ENHANCED BACKTEST FEATURES")
    print("="*60)
    
    # Test 1: Parameter optimization classes
    print("\n1. Testing parameter optimization classes...")
    params = OptimizationParameters(
        stop_loss_pct=0.04,
        profit_target_r=2.5,
        confidence_threshold=0.7
    )
    print(f"   ✅ Created params: stop_loss={params.stop_loss_pct:.1%}, "
          f"target={params.profit_target_r:.1f}R, confidence={params.confidence_threshold:.2f}")
    
    # Test 2: CSV backtest with existing signals
    print("\n2. Testing CSV-based backtesting...")
    try:
        engine = BacktestEngine()
        csv_file = "etf_signals_20250630_080303.csv"
        start_date = datetime(2025, 6, 1)
        end_date = datetime(2025, 6, 30)
        
        results = engine.backtest_from_csv(csv_file, start_date, end_date)
        
        perf = results['performance']
        print(f"   ✅ CSV backtest completed:")
        print(f"      - Total trades: {perf.total_trades}")
        print(f"      - Win rate: {perf.win_rate:.1%}")
        print(f"      - Signals tested: {results.get('signals_tested', 0)}")
        
    except Exception as e:
        print(f"   ⚠️ CSV backtest error: {e}")
    
    # Test 3: Basic functionality check
    print("\n3. Testing enhanced engine initialization...")
    engine = BacktestEngine()
    print(f"   ✅ Engine with optimization params: {type(engine.current_params).__name__}")
    print(f"   ✅ Default stop loss: {engine.current_params.stop_loss_pct:.1%}")
    print(f"   ✅ Default target: {engine.current_params.profit_target_r:.1f}R")
    
    print("\n✅ Enhanced features test complete!")

if __name__ == "__main__":
    test_pipeline()
    test_enhanced_features()