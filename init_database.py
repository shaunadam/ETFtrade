#!/usr/bin/env python3
"""
Initialize the ETF trading system SQLite database with schema and ETF data.
Future-proofed to handle both ETFs and individual stocks.
"""

import sqlite3
import csv
import sys
from pathlib import Path

def create_database_schema(cursor):
    """Create all database tables with future-proof schema."""
    
    # Instruments table - handles both ETFs and stocks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS instruments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL CHECK (type IN ('ETF', 'Stock', 'ETN')),
            sector TEXT,
            theme TEXT,
            geography TEXT,
            leverage TEXT,
            volatility_profile TEXT,
            avg_volume_req TEXT,
            tags TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Trade setups table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS setups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            parameters TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Trades table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument_id INTEGER NOT NULL,
            setup_id INTEGER,
            entry_date DATE NOT NULL,
            exit_date DATE,
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            r_planned REAL,
            r_actual REAL,
            notes TEXT,
            regime_at_entry TEXT,
            status TEXT DEFAULT 'open' CHECK (status IN ('open', 'closed', 'cancelled')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (instrument_id) REFERENCES instruments (id),
            FOREIGN KEY (setup_id) REFERENCES setups (id)
        )
    """)
    
    # Trade snapshots table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            date DATE NOT NULL,
            price REAL NOT NULL,
            notes TEXT,
            chart_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES trades (id)
        )
    """)
    
    # Market regimes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_regimes (
            date DATE PRIMARY KEY,
            volatility_regime TEXT NOT NULL,
            trend_regime TEXT NOT NULL,
            sector_rotation TEXT NOT NULL,
            risk_on_off TEXT NOT NULL,
            vix_level REAL,
            spy_vs_sma200 REAL,
            growth_value_ratio REAL,
            risk_on_off_ratio REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Correlation tracking table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument1_id INTEGER NOT NULL,
            instrument2_id INTEGER NOT NULL,
            correlation_30d REAL,
            correlation_90d REAL,
            correlation_252d REAL,
            date_calculated DATE NOT NULL,
            FOREIGN KEY (instrument1_id) REFERENCES instruments (id),
            FOREIGN KEY (instrument2_id) REFERENCES instruments (id),
            UNIQUE(instrument1_id, instrument2_id, date_calculated)
        )
    """)
    
    # Price data caching table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(symbol, date)
        )
    """)
    
    # Technical indicators caching table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            indicator_name TEXT NOT NULL,
            value REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(symbol, date, indicator_name)
        )
    """)
    
    # Risk-free rate metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rate_metadata (
            symbol TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            source TEXT NOT NULL,
            data_type TEXT NOT NULL CHECK (data_type IN ('yield', 'etf_proxy')),
            priority INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Risk-free rates data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_free_rates (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            rate_type TEXT NOT NULL CHECK (rate_type IN ('yield', 'etf_return')),
            value REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(symbol, date),
            FOREIGN KEY (symbol) REFERENCES rate_metadata (symbol)
        )
    """)
    
    # Create performance indexes
    create_database_indexes(cursor)
    
    print("✅ Database schema created successfully")

def create_database_indexes(cursor):
    """Create performance indexes for all tables."""
    
    indexes = [
        # Single column indexes for frequent WHERE clauses
        "CREATE INDEX IF NOT EXISTS idx_price_data_symbol ON price_data(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_price_data_date ON price_data(date)",
        "CREATE INDEX IF NOT EXISTS idx_indicators_symbol ON indicators(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_indicators_date ON indicators(date)",
        "CREATE INDEX IF NOT EXISTS idx_risk_free_rates_symbol ON risk_free_rates(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_risk_free_rates_date ON risk_free_rates(date)",
        "CREATE INDEX IF NOT EXISTS idx_instruments_type ON instruments(type)",
        "CREATE INDEX IF NOT EXISTS idx_instruments_symbol ON instruments(symbol)",
        
        # Composite indexes for common query patterns
        "CREATE INDEX IF NOT EXISTS idx_price_data_symbol_date ON price_data(symbol, date)",
        "CREATE INDEX IF NOT EXISTS idx_indicators_symbol_date ON indicators(symbol, date)",
        "CREATE INDEX IF NOT EXISTS idx_indicators_symbol_name ON indicators(symbol, indicator_name)",
        "CREATE INDEX IF NOT EXISTS idx_risk_free_rates_symbol_date ON risk_free_rates(symbol, date)",
        
        # Trade-related indexes for future use
        "CREATE INDEX IF NOT EXISTS idx_trades_instrument_id ON trades(instrument_id)",
        "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
        "CREATE INDEX IF NOT EXISTS idx_trades_entry_date ON trades(entry_date)",
        "CREATE INDEX IF NOT EXISTS idx_snapshots_trade_id ON snapshots(trade_id)",
        "CREATE INDEX IF NOT EXISTS idx_correlations_date ON correlations(date_calculated)",
        "CREATE INDEX IF NOT EXISTS idx_market_regimes_date ON market_regimes(date)"
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)
    
    print("✅ Database indexes created successfully")

def load_etf_data(cursor, csv_file_path):
    """Load ETF data from CSV into instruments table."""
    
    if not Path(csv_file_path).exists():
        print(f"❌ ETF CSV file not found: {csv_file_path}")
        return False
    
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            etf_count = 0
            
            for row in reader:
                # Create tags from various fields for filtering
                tags = []
                if row.get('theme'): tags.append(row['theme'].lower().replace(' ', '_'))
                if row.get('geography'): tags.append(row['geography'].lower().replace(' ', '_'))
                if row.get('leverage'): tags.append(f"{row['leverage']}_leverage")
                if row.get('volatility_profile'): tags.append(f"{row['volatility_profile'].lower()}_vol")
                
                tags_str = ','.join(tags)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO instruments 
                    (symbol, name, type, sector, theme, geography, leverage, 
                     volatility_profile, avg_volume_req, tags, notes) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['symbol'],
                    row['name'],
                    row['type'],  # ETF, ETN from CSV
                    row.get('sector', ''),
                    row.get('theme', ''),
                    row.get('geography', ''),
                    row.get('leverage', ''),
                    row.get('volatility_profile', ''),
                    row.get('avg_volume_req', ''),
                    tags_str,
                    row.get('notes', '')
                ))
                etf_count += 1
            
            print(f"✅ Loaded {etf_count} ETFs into instruments table")
            return True
            
    except Exception as e:
        print(f"❌ Error loading ETF data: {e}")
        return False

def load_stock_data(cursor, csv_file_path):
    """Load stock data from CSV into instruments table."""
    
    if not Path(csv_file_path).exists():
        print(f"❌ Stock CSV file not found: {csv_file_path}")
        return False
    
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            stock_count = 0
            
            for row in reader:
                # Create tags from various fields for filtering
                tags = []
                if row.get('market_cap_category'): tags.append(row['market_cap_category'].lower().replace(' ', '_'))
                if row.get('beta_category'): tags.append(row['beta_category'].lower().replace(' ', '_'))
                if row.get('liquidity_category'): tags.append(row['liquidity_category'].lower().replace(' ', '_'))
                if row.get('tags'): tags.extend(row['tags'].split())
                
                tags_str = ','.join(tags)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO instruments 
                    (symbol, name, type, sector, theme, geography, leverage, 
                     volatility_profile, avg_volume_req, tags, notes) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['symbol'],
                    row['name'],
                    'Stock',  # Individual stock type
                    row.get('sector', ''),
                    row.get('market_cap_category', ''),  # Use market cap as theme for stocks
                    'US',  # All stocks in list are US-based
                    '1x',  # Individual stocks are unlevered
                    row.get('beta_category', ''),  # Use beta category as volatility profile
                    row.get('liquidity_category', ''),
                    tags_str,
                    f"Beta: {row.get('beta_category', 'Unknown')}, Market Cap: {row.get('market_cap_category', 'Unknown')}"
                ))
                stock_count += 1
            
            print(f"✅ Loaded {stock_count} stocks into instruments table")
            return True
            
    except Exception as e:
        print(f"❌ Error loading stock data: {e}")
        return False

def insert_default_setups(cursor):
    """Insert default trade setups (all 13 from trade_setups.py)."""
    
    setups = [
        ("trend_pullback", "Pullback in trending market", "{'pullback_pct': 0.05, 'trend_sma': 200}"),
        ("breakout_continuation", "Breakout with volume confirmation", "{'volume_multiplier': 1.5, 'breakout_period': 20}"),
        ("oversold_mean_reversion", "Mean reversion from oversold levels", "{'rsi_threshold': 30, 'bb_position': 'lower'}"),
        ("regime_rotation", "Sector rotation based on regime change", "{'regime_change_threshold': 0.1}"),
        ("gap_fill_reversal", "Gap fill reversal after overnight gaps", "{'gap_threshold': 0.02, 'reversal_confirmation': True}"),
        ("relative_strength_momentum", "Momentum based on relative strength vs SPY", "{'lookback_period': 20, 'min_outperformance': 0.05}"),
        ("volatility_contraction", "Trade after volatility compression", "{'atr_compression_ratio': 0.5, 'breakout_threshold': 1.5}"),
        ("dividend_distribution_play", "Technical setups around dividend dates", "{'days_before_ex': 5, 'defensive_sectors': True}"),
        ("elder_triple_screen", "Multi-timeframe trend following with precise entry timing", "{'weekly_ema_span': 65, 'rsi_threshold': 30, 'volume_multiplier': 1.5}"),
        ("institutional_volume_climax", "Detect accumulation during retail panic selling", "{'high_volume_threshold': 3.0, 'min_high_vol_days': 2, 'decline_threshold': 0.02}"),
        ("failed_breakdown_reversal", "Capitalize on bear traps and quick reversals", "{'breakdown_lookback': 5, 'reversal_days_max': 3, 'volume_confirmation': 1.2}"),
        ("earnings_expectation_reset", "Trade technical patterns after earnings uncertainty removed", "{'atr_spike_threshold': 1.5, 'normalization_threshold': 1.3, 'pattern_types': ['pullback', 'breakout']}"),
        ("elder_force_impulse", "Elder's Force Index + Impulse System combining price, volume, trend, and momentum", "{'force_index_period': 13, 'ema_period': 13, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9}")
    ]
    
    setup_count = 0
    for name, description, parameters in setups:
        cursor.execute("""
            INSERT OR REPLACE INTO setups (name, description, parameters)
            VALUES (?, ?, ?)
        """, (name, description, parameters))
        setup_count += 1
    
    print(f"✅ {setup_count} trade setups created")

def insert_risk_free_rate_metadata(cursor):
    """Insert risk-free rate source metadata."""
    
    rate_sources = [
        ("^IRX", "3-Month Treasury Constant Maturity Rate", "FRED/Yahoo Finance", "yield", 1),
        ("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", "Yahoo Finance", "etf_proxy", 2),
        ("^TNX", "10-Year Treasury Constant Maturity Rate", "FRED/Yahoo Finance", "yield", 3),
        ("^FVX", "5-Year Treasury Constant Maturity Rate", "FRED/Yahoo Finance", "yield", 4)
    ]
    
    for symbol, description, source, data_type, priority in rate_sources:
        cursor.execute("""
            INSERT OR REPLACE INTO rate_metadata (symbol, description, source, data_type, priority)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, description, source, data_type, priority))
    
    print("✅ Risk-free rate metadata created")

def bootstrap_market_data(cursor, bootstrap_level="core"):
    """Bootstrap market data for ETFs and stocks."""
    from data_cache import DataCache
    
    cache = DataCache()
    
    # Define ETF groups
    core_etfs = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLU", "XLP", "TLT"]  # Regime detection
    
    # Define stock groups for bootstrapping
    core_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "JNJ", "UNH"]  # Diversified core
    priority_stocks = ["AMD", "CRM", "NFLX", "SHOP", "ROKU", "ZM", "SNOW", "PLTR", "NET", "DDOG", 
                      "ZS", "MRNA", "GILD", "BAC", "GS", "XOM", "CVX", "FCX", "NEM", "UBER"]  # High-growth/momentum
    
    if bootstrap_level == "core":
        symbols_to_bootstrap = core_etfs
        print(f"📡 Bootstrapping core regime detection ETFs ({len(symbols_to_bootstrap)} symbols)...")
    elif bootstrap_level == "priority":
        # Add high-priority trading ETFs
        priority_etfs = ["ARKK", "EEM", "GLD", "IBB", "ICLN", "KWEB", "VTI", "EFA"]
        symbols_to_bootstrap = core_etfs + priority_etfs
        print(f"📡 Bootstrapping priority ETFs ({len(symbols_to_bootstrap)} symbols)...")
    elif bootstrap_level == "all":
        # Get all ETFs from database
        cursor.execute("SELECT symbol FROM instruments WHERE type IN ('ETF', 'ETN') ORDER BY symbol")
        symbols_to_bootstrap = [row[0] for row in cursor.fetchall()]
        print(f"📡 Bootstrapping all ETFs ({len(symbols_to_bootstrap)} symbols)...")
    elif bootstrap_level == "stocks_core":
        symbols_to_bootstrap = core_stocks
        print(f"📡 Bootstrapping core stocks ({len(symbols_to_bootstrap)} symbols)...")
    elif bootstrap_level == "stocks_priority":
        symbols_to_bootstrap = core_stocks + priority_stocks
        print(f"📡 Bootstrapping priority stocks ({len(symbols_to_bootstrap)} symbols)...")
    elif bootstrap_level == "stocks_all":
        # Get all stocks from database
        cursor.execute("SELECT symbol FROM instruments WHERE type = 'Stock' ORDER BY symbol")
        symbols_to_bootstrap = [row[0] for row in cursor.fetchall()]
        print(f"📡 Bootstrapping all stocks ({len(symbols_to_bootstrap)} symbols)...")
    else:
        print("❌ Invalid bootstrap level. Use: core, priority, all, stocks_core, stocks_priority, or stocks_all")
        return False
    
    success_count = 0
    
    for i, symbol in enumerate(symbols_to_bootstrap, 1):
        try:
            print(f"   [{i:2d}/{len(symbols_to_bootstrap)}] {symbol}...", end=" ")
            
            # Bootstrap with 2+ years of data
            data = cache._fetch_and_cache_data(symbol, "2y")
            
            if not data.empty:
                print(f"✅ {len(data)} days")
                success_count += 1
            else:
                print("❌ No data")
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}...")
    
    print(f"\n✅ Bootstrapped {success_count}/{len(symbols_to_bootstrap)} symbols successfully")
    return success_count > 0


def main():
    """Initialize the trading system database."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Initialize ETF Trading System Database")
    parser.add_argument("--bootstrap", 
                       choices=["core", "priority", "all", "stocks_core", "stocks_priority", "stocks_all"], 
                       help="Bootstrap market data (core=regime ETFs, priority=top ETFs, all=everything, stocks_core=core stocks, stocks_priority=priority stocks, stocks_all=all stocks)")
    parser.add_argument("--skip-data", action="store_true", 
                       help="Skip data bootstrapping (schema and symbols only)")
    parser.add_argument("--stocks-only", action="store_true",
                       help="Only load stocks data (skip ETFs)")
    
    args = parser.parse_args()
    
    db_path = "journal.db"
    etf_csv_path = "etf_list.csv"
    stock_csv_path = "stocks_list.csv"
    
    print("🚀 Initializing Trading System Database...")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create schema
        create_database_schema(cursor)
        
        # Load instrument data
        if not args.stocks_only:
            if load_etf_data(cursor, etf_csv_path):
                print(f"📊 ETF data loaded from {etf_csv_path}")
        
        if load_stock_data(cursor, stock_csv_path):
            print(f"📈 Stock data loaded from {stock_csv_path}")
        
        # Insert default setups
        insert_default_setups(cursor)
        
        # Insert risk-free rate metadata
        insert_risk_free_rate_metadata(cursor)
        
        # Commit changes
        conn.commit()
        
        # Bootstrap market data if requested
        if not args.skip_data:
            bootstrap_level = args.bootstrap or "core"  # Default to core
            print(f"\n🔄 Bootstrapping market data (level: {bootstrap_level})...")
            bootstrap_market_data(cursor, bootstrap_level)
        else:
            print("\n⏭️  Skipping data bootstrapping (use --bootstrap to include)")
        
        # Display summary
        cursor.execute("SELECT COUNT(*) FROM instruments WHERE type IN ('ETF', 'ETN')")
        etf_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM instruments WHERE type = 'Stock'")
        stock_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM setups")
        setup_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM price_data")
        price_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM indicators")
        indicator_records = cursor.fetchone()[0]
        
        print(f"\n📈 Database initialized successfully!")
        print(f"   • Database: {db_path}")
        print(f"   • ETFs loaded: {etf_count}")
        print(f"   • Stocks loaded: {stock_count}")
        print(f"   • Trade setups: {setup_count}")
        print(f"   • Price records: {price_records:,}")
        print(f"   • Indicator records: {indicator_records:,}")
        
        if price_records > 0:
            print(f"\n🎯 Ready to use! Try:")
            print(f"   python regime_detection.py    # Test regime detection")
            print(f"   python screener.py --help     # Explore screening options")
        else:
            print(f"\n💡 To add market data, run:")
            print(f"   python init_database.py --bootstrap core           # Essential ETFs")
            print(f"   python init_database.py --bootstrap priority       # Priority ETFs") 
            print(f"   python init_database.py --bootstrap all            # All ETFs")
            print(f"   python init_database.py --bootstrap stocks_core    # Core stocks")
            print(f"   python init_database.py --bootstrap stocks_priority # Priority stocks")
            print(f"   python init_database.py --bootstrap stocks_all     # All stocks")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()