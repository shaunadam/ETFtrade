#!/usr/bin/env python3
"""
Backfill historical stock data and technical indicators.

This script loads historical price data and calculates technical indicators
for all stocks in the stocks_list.csv file, storing them in the database cache.

Usage:
    python backfill_stocks.py --all                    # Load all stocks
    python backfill_stocks.py --core                   # Load core stocks only
    python backfill_stocks.py --priority               # Load priority stocks
    python backfill_stocks.py --symbols AAPL MSFT     # Load specific symbols
    python backfill_stocks.py --from-csv stocks_list.csv --limit 10  # Load first 10 from CSV
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Optional
import sqlite3

from data_cache import DataCache


class StockBackfiller:
    """Backfill historical stock data with progress tracking and error handling."""
    
    def __init__(self, db_path: str = "journal.db"):
        self.db_path = db_path
        self.data_cache = DataCache(db_path)
        
        # Define stock groups (same as in init_database.py)
        self.core_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "JNJ", "UNH"
        ]
        
        self.priority_stocks = [
            "AMD", "CRM", "NFLX", "SHOP", "ROKU", "ZM", "SNOW", "PLTR", "NET", "DDOG", 
            "ZS", "MRNA", "GILD", "BAC", "GS", "XOM", "CVX", "FCX", "NEM", "UBER"
        ]
    
    def get_stocks_from_csv(self, csv_path: str) -> List[str]:
        """Load stock symbols from CSV file."""
        if not Path(csv_path).exists():
            print(f"❌ CSV file not found: {csv_path}")
            return []
        
        symbols = []
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    symbols.append(row['symbol'])
            
            print(f"📊 Found {len(symbols)} stocks in {csv_path}")
            return symbols
            
        except Exception as e:
            print(f"❌ Error reading CSV file: {e}")
            return []
    
    def get_stocks_from_database(self) -> List[str]:
        """Get all stock symbols from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT symbol FROM instruments WHERE type = 'Stock' ORDER BY symbol")
                symbols = [row[0] for row in cursor.fetchall()]
            
            print(f"📊 Found {len(symbols)} stocks in database")
            return symbols
            
        except Exception as e:
            print(f"❌ Error reading from database: {e}")
            return []
    
    def backfill_symbols(self, symbols: List[str], period: str = "2y", 
                        delay_seconds: float = 0.1) -> dict:
        """
        Backfill historical data for given symbols.
        
        Args:
            symbols: List of stock symbols to backfill
            period: Period for historical data (2y, 5y, etc.)
            delay_seconds: Delay between API calls to avoid rate limiting
            
        Returns:
            Dict with success/failure counts and failed symbols
        """
        if not symbols:
            print("❌ No symbols to process")
            return {"success": 0, "failed": 0, "failed_symbols": []}
        
        print(f"🔄 Backfilling {len(symbols)} stocks with {period} of historical data...")
        print(f"   Rate limiting: {delay_seconds}s delay between requests")
        
        success_count = 0
        failed_count = 0
        failed_symbols = []
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"   [{i:2d}/{len(symbols)}] {symbol}...", end=" ", flush=True)
                
                # Use force refresh to ensure we get fresh data
                data = self.data_cache.get_cached_data(symbol, period, force_refresh=True)
                
                if not data.empty:
                    print(f"✅ {len(data)} days")
                    success_count += 1
                else:
                    print("❌ No data")
                    failed_count += 1
                    failed_symbols.append(symbol)
                
                # Rate limiting to avoid overwhelming yfinance
                if i < len(symbols):  # Don't delay after the last symbol
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                failed_count += 1
                failed_symbols.append(symbol)
                
                # Continue with next symbol
                time.sleep(delay_seconds)
        
        # Summary
        print(f"\n📈 Backfill completed:")
        print(f"   ✅ Success: {success_count}/{len(symbols)} stocks")
        print(f"   ❌ Failed: {failed_count}/{len(symbols)} stocks")
        
        if failed_symbols:
            print(f"   Failed symbols: {', '.join(failed_symbols[:10])}")
            if len(failed_symbols) > 10:
                print(f"   ... and {len(failed_symbols) - 10} more")
        
        return {
            "success": success_count,
            "failed": failed_count,
            "failed_symbols": failed_symbols
        }
    
    def verify_data_quality(self, symbols: List[str]) -> dict:
        """Verify that backfilled data has good quality."""
        print(f"\n🔍 Verifying data quality for {len(symbols)} stocks...")
        
        quality_issues = []
        short_history = []
        missing_indicators = []
        
        for symbol in symbols:
            try:
                data = self.data_cache.get_cached_data(symbol, "1y")
                
                if data.empty:
                    quality_issues.append(f"{symbol}: No data")
                    continue
                
                # Check for minimum history (at least 200 days for SMA200)
                if len(data) < 200:
                    short_history.append(f"{symbol}: {len(data)} days")
                
                # Check for key technical indicators
                required_indicators = ['SMA20', 'SMA50', 'SMA200', 'RSI', 'ATR']
                missing = [ind for ind in required_indicators if ind not in data.columns or data[ind].isna().all()]
                if missing:
                    missing_indicators.append(f"{symbol}: {', '.join(missing)}")
                    
            except Exception as e:
                quality_issues.append(f"{symbol}: {str(e)[:30]}...")
        
        # Report quality issues
        total_issues = len(quality_issues) + len(short_history) + len(missing_indicators)
        
        if total_issues == 0:
            print("✅ All stocks passed data quality checks")
        else:
            print(f"⚠️  Found {total_issues} data quality issues:")
            
            if quality_issues:
                print(f"   Data issues ({len(quality_issues)}): {', '.join(quality_issues[:3])}")
                if len(quality_issues) > 3:
                    print(f"   ... and {len(quality_issues) - 3} more")
            
            if short_history:
                print(f"   Short history ({len(short_history)}): {', '.join(short_history[:3])}")
                if len(short_history) > 3:
                    print(f"   ... and {len(short_history) - 3} more")
            
            if missing_indicators:
                print(f"   Missing indicators ({len(missing_indicators)}): {', '.join(missing_indicators[:2])}")
                if len(missing_indicators) > 2:
                    print(f"   ... and {len(missing_indicators) - 2} more")
        
        return {
            "quality_issues": quality_issues,
            "short_history": short_history,
            "missing_indicators": missing_indicators
        }


def main():
    """Main backfill function."""
    parser = argparse.ArgumentParser(description="Backfill historical stock data")
    
    # Symbol selection options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                      help="Backfill all stocks from database")
    group.add_argument("--core", action="store_true",
                      help="Backfill core stocks only")
    group.add_argument("--priority", action="store_true",
                      help="Backfill core + priority stocks")
    group.add_argument("--from-csv", type=str,
                      help="Load symbols from CSV file")
    group.add_argument("--symbols", nargs="+",
                      help="Specific symbols to backfill")
    
    # Configuration options
    parser.add_argument("--period", default="2y",
                       help="Historical data period (default: 2y)")
    parser.add_argument("--delay", type=float, default=0.1,
                       help="Delay between API calls in seconds (default: 0.1)")
    parser.add_argument("--limit", type=int,
                       help="Limit number of symbols to process")
    parser.add_argument("--verify", action="store_true",
                       help="Verify data quality after backfill")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually backfilling")
    
    args = parser.parse_args()
    
    # Initialize backfiller
    backfiller = StockBackfiller()
    
    # Determine symbols to process
    if args.all:
        symbols = backfiller.get_stocks_from_database()
    elif args.core:
        symbols = backfiller.core_stocks
        print(f"📊 Using core stocks: {len(symbols)} symbols")
    elif args.priority:
        symbols = backfiller.core_stocks + backfiller.priority_stocks
        print(f"📊 Using core + priority stocks: {len(symbols)} symbols")
    elif args.from_csv:
        symbols = backfiller.get_stocks_from_csv(args.from_csv)
    elif args.symbols:
        symbols = args.symbols
        print(f"📊 Using specified symbols: {len(symbols)} symbols")
    else:
        print("❌ No symbol selection option provided")
        sys.exit(1)
    
    if not symbols:
        print("❌ No symbols to process")
        sys.exit(1)
    
    # Apply limit if specified
    if args.limit:
        original_count = len(symbols)
        symbols = symbols[:args.limit]
        print(f"📊 Limited to first {len(symbols)} of {original_count} symbols")
    
    # Dry run mode
    if args.dry_run:
        print(f"\n🔍 DRY RUN - Would process {len(symbols)} symbols:")
        for i, symbol in enumerate(symbols, 1):
            print(f"   {i:2d}. {symbol}")
        print(f"\nRun without --dry-run to execute backfill")
        sys.exit(0)
    
    # Execute backfill
    print(f"\n🚀 Starting backfill process...")
    print(f"   Period: {args.period}")
    print(f"   Delay: {args.delay}s")
    print(f"   Symbols: {len(symbols)}")
    
    results = backfiller.backfill_symbols(symbols, args.period, args.delay)
    
    # Verify data quality if requested
    if args.verify and results["success"] > 0:
        successful_symbols = [s for s in symbols if s not in results["failed_symbols"]]
        backfiller.verify_data_quality(successful_symbols)
    
    # Exit with error code if too many failures
    failure_rate = results["failed"] / len(symbols)
    if failure_rate > 0.5:
        print(f"\n❌ High failure rate ({failure_rate:.1%}). Check network connection or API limits.")
        sys.exit(1)
    elif failure_rate > 0.2:
        print(f"\n⚠️  Moderate failure rate ({failure_rate:.1%}). Some symbols may have issues.")
    
    print(f"\n✅ Backfill completed successfully!")


if __name__ == "__main__":
    main()