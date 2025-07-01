#!/usr/bin/env python3
"""
ETF screener for identifying trade opportunities.

This is the primary CLI interface for daily trading workflow,
implementing regime-aware filtering and export capabilities.

Usage:
    python screener.py --regime-filter --export-csv
    python screener.py --setup trend_pullback --min-confidence 0.7
    python screener.py --update-data --export-json
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from data_cache import DataCache
from regime_detection import RegimeDetector
from trade_setups import SetupManager, SetupType, TradeSignal


class ETFScreener:
    """ETF screening engine with regime-aware filtering."""
    
    def __init__(self, db_path: str = "journal.db"):
        self.db_path = db_path
        self.data_cache = DataCache(db_path)
        self.regime_detector = RegimeDetector(db_path)
        self.setup_manager = SetupManager(db_path)
    
    def screen_etfs(self, 
                   setup_filter: Optional[str] = None,
                   min_confidence: float = 0.5,
                   max_signals: int = 100,
                   regime_filter: bool = True,
                   update_data: bool = False) -> List[TradeSignal]:
        """
        Screen ETFs for trade opportunities.
        
        Args:
            setup_filter: Filter by specific setup type
            min_confidence: Minimum confidence threshold
            max_signals: Maximum number of signals to return
            regime_filter: Apply regime-aware filtering
            update_data: Whether to update market data before screening
            
        Returns:
            List of trade signals
        """
        print("🔍 Screening ETF universe for trade opportunities...")
        
        # Update market data only if requested
        if update_data:
            print("🔄 Updating market data...")
            self.data_cache.update_market_data()
        
        # Get current market regime
        current_regime = self.regime_detector.detect_current_regime()
        print(f"\n📊 Current Market Regime:")
        print(f"   Volatility: {current_regime.volatility_regime.value}")
        print(f"   Trend: {current_regime.trend_regime.value}")
        print(f"   Sector Rotation: {current_regime.sector_rotation.value}")
        print(f"   Risk Sentiment: {current_regime.risk_sentiment.value}")
        
        # Get signals from all setups or specific setup
        if setup_filter:
            try:
                setup_type = SetupType(setup_filter)
                setup = self.setup_manager.setups[setup_type]
                symbols = self.setup_manager.get_all_symbols()
                signals = setup.scan_for_signals(symbols)
                print(f"\n🎯 Scanning with {setup_filter} setup...")
            except (ValueError, KeyError):
                print(f"❌ Invalid setup type: {setup_filter}")
                print(f"Available setups: {[s.value for s in SetupType]}")
                return []
        else:
            print(f"\n🎯 Scanning with all setups...")
            all_signals = self.setup_manager.scan_all_setups(max_signals_per_setup=5)
            signals = []
            for setup_signals in all_signals.values():
                signals.extend(setup_signals)
        
        # Filter by confidence
        signals = [s for s in signals if s.confidence >= min_confidence]
        
        # Apply regime filtering if requested
        if regime_filter:
            filtered_signals = []
            for signal in signals:
                setup = self.setup_manager.setups[signal.setup_type]
                is_valid, confidence = setup.validate_signal(signal.symbol, current_regime)
                if is_valid:
                    filtered_signals.append(signal)
            signals = filtered_signals
            print(f"   Regime filtering applied")
        
        # Sort by confidence and limit results
        signals.sort(key=lambda x: x.confidence, reverse=True)
        signals = signals[:max_signals]
        
        print(f"\n📈 Found {len(signals)} qualifying signals")
        return signals
    
    def display_signals(self, signals: List[TradeSignal]) -> None:
        """Display trade signals in console format."""
        if not signals:
            print("🚫 No signals found matching criteria")
            return
        
        print(f"\n{'='*80}")
        print(f"{'TRADE SIGNALS':^80}")
        print(f"{'='*80}")
        
        for i, signal in enumerate(signals, 1):
            risk_reward = (signal.target_price - signal.entry_price) / signal.risk_per_share
            
            print(f"\n{i}. {signal.symbol} - {signal.setup_type.value.upper()}")
            print(f"   {'─'*60}")
            print(f"   Entry Price:    ${signal.entry_price:.2f}")
            print(f"   Stop Loss:      ${signal.stop_loss:.2f}")
            print(f"   Target Price:   ${signal.target_price:.2f}")
            print(f"   Risk/Share:     ${signal.risk_per_share:.2f}")
            print(f"   Position Size:  {signal.position_size:.0f} shares")
            print(f"   Risk/Reward:    {risk_reward:.1f}:1")
            print(f"   Confidence:     {signal.confidence:.1%}")
            print(f"   Signal:         {signal.signal_strength.value}")
            print(f"   Notes:          {signal.notes}")
    
    def export_csv(self, signals: List[TradeSignal], filename: Optional[str] = None) -> str:
        """Export signals to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"etf_signals_{timestamp}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'symbol', 'setup_type', 'signal_strength', 'confidence',
                'entry_price', 'stop_loss', 'target_price', 'risk_per_share',
                'position_size', 'risk_reward_ratio', 'notes'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for signal in signals:
                risk_reward = (signal.target_price - signal.entry_price) / signal.risk_per_share
                
                writer.writerow({
                    'symbol': signal.symbol,
                    'setup_type': signal.setup_type.value,
                    'signal_strength': signal.signal_strength.value,
                    'confidence': f"{signal.confidence:.3f}",
                    'entry_price': f"{signal.entry_price:.2f}",
                    'stop_loss': f"{signal.stop_loss:.2f}",
                    'target_price': f"{signal.target_price:.2f}",
                    'risk_per_share': f"{signal.risk_per_share:.2f}",
                    'position_size': f"{signal.position_size:.0f}",
                    'risk_reward_ratio': f"{risk_reward:.2f}",
                    'notes': signal.notes
                })
        
        print(f"📄 Signals exported to {filename}")
        return filename
    
    def export_json(self, signals: List[TradeSignal], filename: Optional[str] = None) -> str:
        """Export signals to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"etf_signals_{timestamp}.json"
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "signals_count": len(signals),
            "signals": []
        }
        
        for signal in signals:
            risk_reward = (signal.target_price - signal.entry_price) / signal.risk_per_share
            
            signal_data = {
                "symbol": signal.symbol,
                "setup_type": signal.setup_type.value,
                "signal_strength": signal.signal_strength.value,
                "confidence": signal.confidence,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "target_price": signal.target_price,
                "risk_per_share": signal.risk_per_share,
                "position_size": signal.position_size,
                "risk_reward_ratio": risk_reward,
                "notes": signal.notes,
                "regime_context": {
                    "volatility": signal.regime_context.volatility_regime.value,
                    "trend": signal.regime_context.trend_regime.value,
                    "sector_rotation": signal.regime_context.sector_rotation.value,
                    "risk_sentiment": signal.regime_context.risk_sentiment.value
                }
            }
            export_data["signals"].append(signal_data)
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2)
        
        print(f"📄 Signals exported to {filename}")
        return filename
    
    def show_cache_stats(self) -> None:
        """Display data cache statistics."""
        stats = self.data_cache.get_cache_stats()
        print(f"\n📊 Data Cache Statistics:")
        print(f"   Price records:      {stats['price_records']:,}")
        print(f"   Indicator records:  {stats['indicator_records']:,}")
        print(f"   Symbols cached:     {stats['symbols_cached']}")
        print(f"   Date range:         {stats['date_range']}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="ETF Screener - Find trade opportunities with regime-aware filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python screener.py --regime-filter --export-csv
  python screener.py --setup trend_pullback --min-confidence 0.7
  python screener.py --update-data --max-signals 5
  python screener.py --cache-stats
        """
    )
    
    # Screening options
    parser.add_argument('--setup', type=str, 
                       choices=[s.value for s in SetupType],
                       help='Filter by specific setup type')
    
    parser.add_argument('--min-confidence', type=float, default=0.5,
                       help='Minimum confidence threshold (default: 0.5)')
    
    parser.add_argument('--max-signals', type=int, default=10,
                       help='Maximum number of signals (default: 10)')
    
    parser.add_argument('--regime-filter', action='store_true',
                       help='Apply regime-aware filtering')
    
    # Data management
    parser.add_argument('--update-data', action='store_true',
                       help='Update market data before screening')
    
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force full data refresh')
    
    # Export options
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV file')
    
    parser.add_argument('--export-json', action='store_true',
                       help='Export results to JSON file')
    
    parser.add_argument('--output-file', type=str,
                       help='Custom output filename')
    
    # Utility options
    parser.add_argument('--cache-stats', action='store_true',
                       help='Show data cache statistics')
    
    args = parser.parse_args()
    
    # Initialize screener
    screener = ETFScreener()
    
    # Handle utility commands
    if args.cache_stats:
        screener.show_cache_stats()
        return
    
    # Screen for signals (data update handled within screen_etfs if needed)
    signals = screener.screen_etfs(
        setup_filter=args.setup,
        min_confidence=args.min_confidence,
        max_signals=args.max_signals,
        regime_filter=args.regime_filter,
        update_data=args.update_data or args.force_refresh
    )
    
    # Force refresh requires separate handling
    if args.force_refresh and not args.update_data:
        print("🔄 Force refreshing market data...")
        screener.data_cache.update_market_data(force_full_refresh=True)
        print("✅ Market data force refreshed")
    
    # Display results
    screener.display_signals(signals)
    
    # Export if requested
    if args.export_csv:
        screener.export_csv(signals, args.output_file)
    
    if args.export_json:
        screener.export_json(signals, args.output_file)
    
    if not signals:
        print("\n💡 Try adjusting parameters:")
        print("   • Lower --min-confidence threshold")
        print("   • Remove --regime-filter")
        print("   • Try different --setup types")
        print("   • Use --update-data to refresh market data")


if __name__ == "__main__":
    main()