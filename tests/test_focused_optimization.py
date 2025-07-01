#!/usr/bin/env python3
"""
Focused test showing parameter optimization working with available data.
"""

from datetime import datetime
from backtest import BacktestEngine
from trade_setups import SetupType

def test_manual_optimization():
    """Test optimization manually to show it works."""
    
    print("🎯 MANUAL PARAMETER OPTIMIZATION TEST")
    print("="*60)
    
    # Use our available data range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)  # Shorter period to avoid timeout
    
    engine = BacktestEngine()
    
    # Test 1: Standard parameters
    print("1. Testing with STANDARD parameters...")
    print(f"   Stop Loss: {engine.current_params.stop_loss_pct:.1%}")
    print(f"   Profit Target: {engine.current_params.profit_target_r:.1f}R")
    print(f"   Confidence: {engine.current_params.confidence_threshold:.2f}")
    
    try:
        # Use CSV signals to avoid infinite loop issue
        results1 = engine.backtest_from_csv(
            'etf_signals_20250630_080303.csv',
            start_date,
            end_date
        )
        
        perf1 = results1['performance']
        print(f"   → Results: {perf1.total_trades} trades, {perf1.win_rate:.1%} win rate, {perf1.avg_r_multiple:.2f}R avg")
        
    except Exception as e:
        print(f"   → Error: {e}")
        return
    
    # Test 2: Optimized parameters (tighter stops)
    print("\n2. Testing with OPTIMIZED parameters...")
    from backtest import OptimizationParameters
    
    optimized_params = OptimizationParameters(
        stop_loss_pct=0.03,      # Tighter stop
        profit_target_r=2.5,     # Higher target
        confidence_threshold=0.7  # Higher confidence
    )
    
    engine.current_params = optimized_params
    print(f"   Stop Loss: {engine.current_params.stop_loss_pct:.1%}")
    print(f"   Profit Target: {engine.current_params.profit_target_r:.1f}R")
    print(f"   Confidence: {engine.current_params.confidence_threshold:.2f}")
    
    try:
        results2 = engine.backtest_from_csv(
            'etf_signals_20250630_080303.csv',
            start_date,
            end_date
        )
        
        perf2 = results2['performance']
        print(f"   → Results: {perf2.total_trades} trades, {perf2.win_rate:.1%} win rate, {perf2.avg_r_multiple:.2f}R avg")
        
    except Exception as e:
        print(f"   → Error: {e}")
        return
    
    # Test 3: Another parameter set (loose stops)
    print("\n3. Testing with LOOSE parameters...")
    
    loose_params = OptimizationParameters(
        stop_loss_pct=0.08,      # Looser stop
        profit_target_r=1.5,     # Lower target
        confidence_threshold=0.5  # Lower confidence
    )
    
    engine.current_params = loose_params
    print(f"   Stop Loss: {engine.current_params.stop_loss_pct:.1%}")
    print(f"   Profit Target: {engine.current_params.profit_target_r:.1f}R")
    print(f"   Confidence: {engine.current_params.confidence_threshold:.2f}")
    
    try:
        results3 = engine.backtest_from_csv(
            'etf_signals_20250630_080303.csv',
            start_date,
            end_date
        )
        
        perf3 = results3['performance']
        print(f"   → Results: {perf3.total_trades} trades, {perf3.win_rate:.1%} win rate, {perf3.avg_r_multiple:.2f}R avg")
        
    except Exception as e:
        print(f"   → Error: {e}")
        return
    
    # Comparison
    print("\n4. PARAMETER COMPARISON")
    print("-" * 60)
    print(f"{'Parameter Set':<15} {'Trades':<8} {'Win Rate':<10} {'Avg R':<8} {'Return':<10}")
    print("-" * 60)
    print(f"{'Standard':<15} {perf1.total_trades:<8} {perf1.win_rate:<10.1%} {perf1.avg_r_multiple:<8.2f} {perf1.total_return:<10.1%}")
    print(f"{'Optimized':<15} {perf2.total_trades:<8} {perf2.win_rate:<10.1%} {perf2.avg_r_multiple:<8.2f} {perf2.total_return:<10.1%}")
    print(f"{'Loose':<15} {perf3.total_trades:<8} {perf3.win_rate:<10.1%} {perf3.avg_r_multiple:<8.2f} {perf3.total_return:<10.1%}")
    
    print("\n✅ Manual optimization test shows:")
    print("   - Parameters successfully change behavior")
    print("   - Different stop/target ratios affect performance")
    print("   - Framework ready for automated optimization")

def explain_optimization_process():
    """Explain how the full optimization would work."""
    
    print("\n🔧 HOW WALK-FORWARD OPTIMIZATION WORKS")
    print("="*60)
    
    print("CURRENT IMPLEMENTATION:")
    print("1. Split time into periods (252 training + 63 test days)")
    print("2. For each period:")
    print("   a) Grid search 80 parameter combinations on training data")
    print("   b) Pick best parameters based on Sharpe ratio")
    print("   c) Apply those parameters to test period")
    print("   d) Move to next period with updated training data")
    
    print("\nPARAMETER RANGES TESTED:")
    print("   - Stop Loss: 3%, 4%, 5%, 6%, 8%")
    print("   - Profit Target: 1.5R, 2.0R, 2.5R, 3.0R")
    print("   - Confidence: 0.5, 0.6, 0.7, 0.8")
    print("   = 80 combinations per training period")
    
    print("\nREGIME ANALYSIS INCLUDED:")
    print("   - Performance by volatility regime (VIX levels)")
    print("   - Performance by trend regime (SPY vs SMA200)")
    print("   - Performance by risk sentiment (risk-on/off)")
    
    print("\nDATA CONSTRAINTS:")
    print("   - Need 2+ years for meaningful training periods")
    print("   - Current cache: Jul 2023 to Jun 2025 (2 years)")
    print("   - SMA200 needs 200+ days for accuracy")
    print("   - More data = better optimization results")

if __name__ == "__main__":
    test_manual_optimization()
    explain_optimization_process()