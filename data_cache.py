"""
Data caching engine for ETF trading system.

This module provides intelligent caching of market data and technical indicators
to minimize yfinance API calls while ensuring data freshness and healing.

Key features:
- Smart refresh strategy (always refresh last 5 trading days)
- Healing logic to ensure indicator dependencies are met
- Batch indicator calculation and caching
- Graceful fallback to yfinance when cache misses occur
"""

import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Set, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCache:
    """Intelligent data cache for market data and technical indicators."""
    
    def __init__(self, db_path: str = "journal.db"):
        self.db_path = db_path
        self._ensure_schema_updated()
    
    def _ensure_schema_updated(self) -> None:
        """Ensure caching tables exist in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if price_data table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'")
            if not cursor.fetchone():
                logger.warning("price_data table missing - run init_database.py to update schema")
    
    def get_cached_data(self, symbol: str, period: str = "6mo", 
                       force_refresh: bool = False) -> pd.DataFrame:
        """
        Get market data from cache, fetching from yfinance if needed.
        
        Args:
            symbol: Stock/ETF symbol
            period: Period for data (6mo, 1y, etc.)
            force_refresh: Force refresh from yfinance
            
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        if force_refresh:
            return self._fetch_and_cache_data(symbol, period)
        
        # Check if we need to refresh data
        if self._should_refresh_data(symbol, period):
            return self._fetch_and_cache_data(symbol, period)
        
        # Try to get from cache
        cached_data = self._get_cached_price_data(symbol, period)
        if cached_data is not None and len(cached_data) > 0:
            # Add technical indicators from cache
            cached_data = self._add_cached_indicators(cached_data, symbol)
            return cached_data
        
        # Fallback to yfinance
        logger.info(f"Cache miss for {symbol}, fetching from yfinance")
        return self._fetch_and_cache_data(symbol, period)
    
    def _should_refresh_data(self, symbol: str, period: str) -> bool:
        """Determine if data should be refreshed from yfinance."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if we have recent data (last 5 trading days)
            cursor.execute("""
                SELECT MAX(date) FROM price_data 
                WHERE symbol = ? AND date >= date('now', '-7 days')
            """, (symbol,))
            
            result = cursor.fetchone()
            if not result or not result[0]:
                return True  # No recent data
            
            last_date = datetime.fromisoformat(result[0]).date()
            today = datetime.now().date()
            
            # If last data is more than 2 days old (accounting for weekends)
            if (today - last_date).days > 2:
                return True
            
            return False
    
    def _get_cached_price_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Get price data from cache."""
        # Convert period to days for SQL query
        days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
        days = days_map.get(period, 180)
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT date, open, high, low, close, volume
                FROM price_data 
                WHERE symbol = ? AND date >= date('now', '-{} days')
                ORDER BY date
            """.format(days)
            
            df = pd.read_sql_query(query, conn, params=(symbol,), 
                                 parse_dates=['date'], index_col='date')
            
            if df.empty:
                return None
            
            # Rename columns to match yfinance format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return df
    
    def _add_cached_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add technical indicators from cache to price data."""
        if df.empty:
            return df
        
        with sqlite3.connect(self.db_path) as conn:
            # Get date range from the dataframe
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
            
            query = """
                SELECT date, indicator_name, value
                FROM indicators 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date, indicator_name
            """
            
            indicators_df = pd.read_sql_query(
                query, conn, params=(symbol, start_date, end_date),
                parse_dates=['date'], index_col='date'
            )
            
            if indicators_df.empty:
                return df
            
            # Pivot indicators to columns
            indicators_pivot = indicators_df.pivot_table(
                index='date', columns='indicator_name', values='value'
            )
            
            # Merge with price data
            df = df.join(indicators_pivot, how='left')
            
            return df
    
    def _fetch_and_cache_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from yfinance and cache it."""
        try:
            logger.info(f"Fetching {symbol} data from yfinance...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Cache price data
            self._cache_price_data(symbol, data)
            
            # Calculate and cache technical indicators
            data_with_indicators = self._calculate_and_cache_indicators(symbol, data)
            
            return data_with_indicators
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _cache_price_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Cache price data in database."""
        if data.empty:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Prepare data for insertion
            cache_data = []
            for date_idx, row in data.iterrows():
                cache_data.append((
                    symbol,
                    date_idx.strftime('%Y-%m-%d'),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume']) if not pd.isna(row['Volume']) else 0
                ))
            
            # Insert or replace price data
            conn.executemany("""
                INSERT OR REPLACE INTO price_data 
                (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, cache_data)
            
            conn.commit()
            logger.info(f"Cached {len(cache_data)} price records for {symbol}")
    
    def _calculate_and_cache_indicators(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and cache them."""
        if data.empty:
            return data
        
        # Calculate indicators
        data = self._add_technical_indicators(data)
        
        # Cache indicators
        indicator_data = []
        indicator_columns = ['SMA20', 'SMA50', 'SMA200', 'RSI', 'ATR', 'BB_Upper', 'BB_Lower', 'BB_Middle']
        
        for date_idx, row in data.iterrows():
            for indicator in indicator_columns:
                if indicator in row and not pd.isna(row[indicator]):
                    indicator_data.append((
                        symbol,
                        date_idx.strftime('%Y-%m-%d'),
                        indicator,
                        float(row[indicator])
                    ))
        
        if indicator_data:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO indicators 
                    (symbol, date, indicator_name, value)
                    VALUES (?, ?, ?, ?)
                """, indicator_data)
                
                conn.commit()
                logger.info(f"Cached {len(indicator_data)} indicator values for {symbol}")
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data."""
        if len(data) < 200:  # Need enough data for 200-day SMA
            logger.warning("Insufficient data for all indicators")
        
        try:
            # Moving averages
            data['SMA20'] = data['Close'].rolling(20).mean()
            data['SMA50'] = data['Close'].rolling(50).mean()
            data['SMA200'] = data['Close'].rolling(200).mean()
            
            # RSI
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            # ATR
            data['ATR'] = self._calculate_atr(data)
            
            # Bollinger Bands
            data['BB_Upper'], data['BB_Lower'], data['BB_Middle'] = self._calculate_bollinger_bands(data['Close'])
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, 
                                  num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return upper, lower, sma
    
    def update_market_data(self, symbols: Optional[List[str]] = None, 
                          force_full_refresh: bool = False) -> None:
        """
        Update market data with healing strategy.
        
        Args:
            symbols: List of symbols to update (None for all)
            force_full_refresh: Force full refresh of all data
        """
        if symbols is None:
            symbols = self._get_all_symbols()
        
        logger.info(f"Updating market data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            try:
                if force_full_refresh:
                    # Full refresh - get 2 years of data
                    self._fetch_and_cache_data(symbol, "2y")
                else:
                    # Smart refresh - check what we need
                    min_date = self._get_min_cached_date(symbol)
                    
                    if min_date is None:
                        # No cached data - full bootstrap
                        self._fetch_and_cache_data(symbol, "2y")
                    else:
                        # Healing strategy - refresh from safe point
                        heal_from = min_date - timedelta(days=200)  # Safety for SMA200
                        today = datetime.now().date()
                        
                        # Always refresh last 5 trading days
                        bleeding_edge = today - timedelta(days=7)
                        
                        refresh_from = max(heal_from, bleeding_edge)
                        
                        if refresh_from < today:
                            # Calculate period needed
                            days_needed = (today - refresh_from).days
                            period = "1y" if days_needed > 365 else "6mo" if days_needed > 180 else "3mo"
                            
                            self._fetch_and_cache_data(symbol, period)
                
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
        
        logger.info("Market data update completed")
    
    def _get_all_symbols(self) -> List[str]:
        """Get all symbols from instruments table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT symbol FROM instruments WHERE type IN ('ETF', 'ETN')")
            return [row[0] for row in cursor.fetchall()]
    
    def _get_min_cached_date(self, symbol: str) -> Optional[date]:
        """Get minimum cached date for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT MIN(date) FROM price_data WHERE symbol = ?
            """, (symbol,))
            
            result = cursor.fetchone()
            if result and result[0]:
                return datetime.fromisoformat(result[0]).date()
            return None
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count price records
            cursor.execute("SELECT COUNT(*) FROM price_data")
            price_count = cursor.fetchone()[0]
            
            # Count indicator records
            cursor.execute("SELECT COUNT(*) FROM indicators")
            indicator_count = cursor.fetchone()[0]
            
            # Count symbols with data
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM price_data")
            symbols_count = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute("SELECT MIN(date), MAX(date) FROM price_data")
            date_range = cursor.fetchone()
            
            return {
                "price_records": price_count,
                "indicator_records": indicator_count,
                "symbols_cached": symbols_count,
                "date_range": f"{date_range[0]} to {date_range[1]}" if date_range[0] else "No data"
            }


if __name__ == "__main__":
    # Test the data cache
    cache = DataCache()
    
    print("📊 Data Cache Test")
    print("=================")
    
    # Show current stats
    stats = cache.get_cache_stats()
    print(f"Current cache stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test with a few symbols
    test_symbols = ["SPY", "QQQ", "IWM"]
    
    print(f"\n🔄 Testing cache with symbols: {test_symbols}")
    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
        data = cache.get_cached_data(symbol, "3mo")
        print(f"  Retrieved {len(data)} days of data")
        if not data.empty:
            print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
            print(f"  Columns: {list(data.columns)}")
    
    # Show updated stats
    stats = cache.get_cache_stats()
    print(f"\nUpdated cache stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")