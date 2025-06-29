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
            volatility_regime TEXT,
            trend_regime TEXT,
            sector_rotation TEXT,
            risk_on_off TEXT,
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
    
    print("✅ Database schema created successfully")

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

def insert_default_setups(cursor):
    """Insert default trade setups."""
    
    setups = [
        ("trend_pullback", "Pullback in trending market", "{'pullback_pct': 0.05, 'trend_sma': 200}"),
        ("breakout_continuation", "Breakout with volume confirmation", "{'volume_multiplier': 1.5, 'breakout_period': 20}"),
        ("oversold_mean_reversion", "Mean reversion from oversold levels", "{'rsi_threshold': 30, 'bb_position': 'lower'}"),
        ("regime_rotation", "Sector rotation based on regime change", "{'regime_change_threshold': 0.1}")
    ]
    
    for name, description, parameters in setups:
        cursor.execute("""
            INSERT OR REPLACE INTO setups (name, description, parameters)
            VALUES (?, ?, ?)
        """, (name, description, parameters))
    
    print("✅ Default trade setups created")

def main():
    """Initialize the trading system database."""
    
    db_path = "journal.db"
    csv_path = "etf_list.csv"
    
    print("🚀 Initializing ETF Trading System Database...")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create schema
        create_database_schema(cursor)
        
        # Load ETF data
        if load_etf_data(cursor, csv_path):
            print(f"📊 ETF data loaded from {csv_path}")
        
        # Insert default setups
        insert_default_setups(cursor)
        
        # Commit changes
        conn.commit()
        
        # Display summary
        cursor.execute("SELECT COUNT(*) FROM instruments WHERE type = 'ETF'")
        etf_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM setups")
        setup_count = cursor.fetchone()[0]
        
        print(f"\n📈 Database initialized successfully!")
        print(f"   • Database: {db_path}")
        print(f"   • ETFs loaded: {etf_count}")
        print(f"   • Trade setups: {setup_count}")
        print(f"   • Ready for individual stocks (future)")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()