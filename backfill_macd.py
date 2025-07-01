#!/usr/bin/env python3
"""
MACD backfill script for ETF trading system.

This script backfills MACD Line and MACD Histogram indicators for all ETFs
in the database for the last couple of years.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List
import logging
from data_cache import DataCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD Line and MACD Histogram."""
    if len(prices) < slow + signal:
        return pd.Series(index=prices.index, dtype=float), pd.Series(index=prices.index, dtype=float)
    
    # MACD Line = 12-period EMA - 26-period EMA
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    
    # Signal Line = 9-period EMA of MACD Line
    signal_line = macd_line.ewm(span=signal).mean()
    
    # MACD Histogram = MACD Line - Signal Line
    macd_histogram = macd_line - signal_line
    
    return macd_line, macd_histogram


def get_symbols_to_backfill(db_path: str = "journal.db") -> List[str]:
    """Get list of symbols that need MACD backfill."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT DISTINCT symbol FROM instruments WHERE type IN ('ETF', 'ETN')")
        return [row[0] for row in cursor.fetchall()]


def get_price_data(symbol: str, db_path: str = "journal.db") -> pd.DataFrame:
    """Get price data for a symbol from database."""
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT date, close 
            FROM price_data 
            WHERE symbol = ? 
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=[symbol])
        
        if df.empty:
            return df
            
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df


def backfill_macd_for_symbol(symbol: str, db_path: str = "journal.db") -> None:
    """Backfill MACD indicators for a single symbol."""
    logger.info(f"Backfilling MACD for {symbol}")
    
    # Get price data
    price_data = get_price_data(symbol, db_path)
    if price_data.empty:
        logger.warning(f"No price data found for {symbol}")
        return
    
    # Calculate MACD
    macd_line, macd_histogram = calculate_macd(price_data['close'])
    
    # Prepare data for insertion
    indicator_data = []
    for date_idx in price_data.index:
        if not pd.isna(macd_line.loc[date_idx]):
            indicator_data.append((
                symbol,
                date_idx.strftime('%Y-%m-%d'),
                'MACD_Line',
                float(macd_line.loc[date_idx])
            ))
        
        if not pd.isna(macd_histogram.loc[date_idx]):
            indicator_data.append((
                symbol,
                date_idx.strftime('%Y-%m-%d'),
                'MACD_Histogram',
                float(macd_histogram.loc[date_idx])
            ))
    
    # Insert into database
    if indicator_data:
        with sqlite3.connect(db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO indicators 
                (symbol, date, indicator_name, value)
                VALUES (?, ?, ?, ?)
            """, indicator_data)
            conn.commit()
            logger.info(f"Backfilled {len(indicator_data)} MACD values for {symbol}")
    else:
        logger.warning(f"No MACD values calculated for {symbol}")


def main():
    """Main backfill function."""
    logger.info("Starting MACD backfill process")
    
    # Get all symbols
    symbols = get_symbols_to_backfill()
    logger.info(f"Found {len(symbols)} symbols to process")
    
    # Process each symbol
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"Processing {symbol} ({i}/{len(symbols)})")
            backfill_macd_for_symbol(symbol)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    logger.info("MACD backfill completed")
    
    # Show final stats
    cache = DataCache()
    stats = cache.get_cache_stats()
    logger.info(f"Final cache stats: {stats}")


if __name__ == "__main__":
    main()