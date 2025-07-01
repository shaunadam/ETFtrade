#!/usr/bin/env python3
"""
Test walk-forward optimization to clearly show before/after parameter changes.
"""

from datetime import datetime
from backtest import BacktestEngine
from trade_setups import SetupType

def test_walk_forward_optimization():
    """Test walk-forward optimization with clear before/after comparison."""
    
    print("🧪 TESTING WALK-FORWARD PARAMETER OPTIMIZATION")
    print("="*70)
    
    engine = BacktestEngine()
    
    # Test 1: Standard backtest (no optimization)
    print("\n1. STANDARD BACKTEST (Static Parameters)")
    print("-" * 50)
    print(f"Using fixed parameters:")
    print(f"  - Stop Loss: {engine.current_params.stop_loss_pct:.1%}")
    print(f"  - Profit Target: {engine.current_params.profit_target_r:.1f}R") 
    print(f"  - Confidence: {engine.current_params.confidence_threshold:.2f}")
    
    # Short period with cached data 
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    try:
        standard_results = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            setup_types=[SetupType.REGIME_ROTATION],
            walk_forward=False,
            regime_aware=True
        )
        
        std_perf = standard_results['performance']
        print(f"\nStandard Results:")
        print(f"  - Total Trades: {std_perf.total_trades}")
        print(f"  - Win Rate: {std_perf.win_rate:.1%}")
        print(f"  - Avg R-Multiple: {std_perf.avg_r_multiple:.2f}")
        print(f"  - Total Return: {std_perf.total_return:.1%}")
        print(f"  - Sharpe Ratio: {std_perf.sharpe_ratio:.2f}")
        
    except Exception as e:
        print(f"Standard backtest error: {e}")
        return
    
    # Test 2: Walk-forward optimization
    print(f"\n2. WALK-FORWARD OPTIMIZATION (Adaptive Parameters)")
    print("-" * 50)
    
    try:
        wf_results = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            setup_types=[SetupType.REGIME_ROTATION],
            walk_forward=True,
            regime_aware=True
        )
        
        wf_perf = wf_results['performance']
        print(f"\nWalk-Forward Results:")
        print(f"  - Total Trades: {wf_perf.total_trades}")
        print(f"  - Win Rate: {wf_perf.win_rate:.1%}")
        print(f"  - Avg R-Multiple: {wf_perf.avg_r_multiple:.2f}")
        print(f"  - Total Return: {wf_perf.total_return:.1%}")
        print(f"  - Sharpe Ratio: {wf_perf.sharpe_ratio:.2f}")
        
        # Show optimization history
        if 'optimization_history' in wf_results:
            opt_history = wf_results['optimization_history']
            print(f"\nParameter Evolution ({len(opt_history)} periods):")
            for i, period in enumerate(opt_history):
                params = period['optimal_params']
                print(f"  Period {i+1}: stop={params['stop_loss_pct']:.1%}, "
                      f"target={params['profit_target_r']:.1f}R, "
                      f"conf={params['confidence_threshold']:.2f}")
        
        # Show regime performance if available
        if 'regime_performance' in wf_results and wf_results['regime_performance']:
            regime_perf = wf_results['regime_performance']
            print(f"\nRegime Performance Analysis:")
            
            if regime_perf.volatility_low:
                vol_low = regime_perf.volatility_low
                print(f"  - Low Vol: {vol_low.total_trades} trades, {vol_low.win_rate:.1%} win rate")
            
            if regime_perf.trend_up:
                trend_up = regime_perf.trend_up
                print(f"  - Uptrend: {trend_up.total_trades} trades, {trend_up.win_rate:.1%} win rate")
        
    except Exception as e:
        print(f"Walk-forward error: {e}")
        return
    
    # Test 3: Comparison
    print(f"\n3. PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<20} {'Standard':<12} {'Walk-Forward':<12} {'Improvement'}")
    print("-" * 58)
    
    metrics = [
        ('Total Return', f"{std_perf.total_return:.1%}", f"{wf_perf.total_return:.1%}"),
        ('Win Rate', f"{std_perf.win_rate:.1%}", f"{wf_perf.win_rate:.1%}"),
        ('Avg R-Multiple', f"{std_perf.avg_r_multiple:.2f}", f"{wf_perf.avg_r_multiple:.2f}"),
        ('Sharpe Ratio', f"{std_perf.sharpe_ratio:.2f}", f"{wf_perf.sharpe_ratio:.2f}"),
        ('Total Trades', f"{std_perf.total_trades}", f"{wf_perf.total_trades}")
    ]
    
    for metric, std_val, wf_val in metrics:
        # Calculate improvement for numeric metrics
        try:
            std_num = float(std_val.rstrip('%'))
            wf_num = float(wf_val.rstrip('%'))
            if std_num != 0:
                improvement = f"{((wf_num - std_num) / abs(std_num)) * 100:+.1f}%"
            else:
                improvement = "N/A"
        except:
            improvement = "N/A"
        
        print(f"{metric:<20} {std_val:<12} {wf_val:<12} {improvement}")
    
    print("="*70)
    print("✅ Walk-forward optimization test complete!")

if __name__ == "__main__":
    test_walk_forward_optimization()