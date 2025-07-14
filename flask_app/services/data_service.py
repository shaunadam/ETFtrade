"""
Data Service Layer

Integrates data_cache.py and database operations with Flask application.
Provides web-friendly interface for data management operations.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Add parent directory to import CLI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_cache import DataCache
from flask_app.models import db, Instrument, PriceData, Indicator

class DataService:
    """Service for managing data operations in Flask app"""
    
    def __init__(self):
        # Use the same database path as Flask config
        from flask_app.config import Config
        # Extract the database path from the SQLAlchemy URI
        db_uri = Config.SQLALCHEMY_DATABASE_URI
        if db_uri.startswith('sqlite:///'):
            db_path = db_uri.replace('sqlite:///', '')
        else:
            # Fallback to root directory journal.db
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'journal.db')
        self.data_cache = DataCache(db_path)
    
    def get_cache_stats(self) -> Dict:
        """Get data cache statistics"""
        try:
            # Get cache stats from DataCache
            stats = self.data_cache.get_cache_stats()
            
            # Add database stats using raw SQLite queries
            import sqlite3
            with sqlite3.connect(self.data_cache.db_path) as conn:
                cursor = conn.cursor()
                
                # Count instruments
                cursor.execute("SELECT COUNT(*) FROM instruments")
                instruments_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM instruments WHERE type IN ('ETF', 'ETN')")
                etfs_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM instruments WHERE type = 'Stock'")
                stocks_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM price_data")
                price_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM indicators")
                indicator_records = cursor.fetchone()[0]
                
                # Get date ranges
                cursor.execute("SELECT MIN(date), MAX(date) FROM price_data")
                price_date_range = cursor.fetchone()
            
            web_stats = {
                'database': {
                    'total_instruments': instruments_count,
                    'etfs': etfs_count,
                    'stocks': stocks_count,
                    'price_records': price_records,
                    'indicator_records': indicator_records,
                    'price_date_range': {
                        'start': price_date_range[0] if price_date_range[0] else None,
                        'end': price_date_range[1] if price_date_range[1] else None
                    }
                },
                'cache': stats,
                'last_updated': datetime.now().isoformat()
            }
            
            return web_stats
            
        except Exception as e:
            return {
                'error': f"Failed to get cache stats: {str(e)}",
                'database': {
                    'total_instruments': 0,
                    'etfs': 0,
                    'stocks': 0,
                    'price_records': 0,
                    'indicator_records': 0,
                    'price_date_range': {
                        'start': None,
                        'end': None
                    }
                },
                'cache': {},
                'last_updated': datetime.now().isoformat()
            }
    
    def update_data(self, symbols: Optional[List[str]] = None, force_refresh: bool = False) -> Dict:
        """Update market data for symbols"""
        try:
            if symbols is None:
                # Get all instrument symbols from database
                import sqlite3
                with sqlite3.connect(self.data_cache.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT symbol FROM instruments ORDER BY symbol")
                    symbols = [row[0] for row in cursor.fetchall()]
            
            results = {
                'success': [],
                'failed': [],
                'total_processed': 0,
                'start_time': datetime.now().isoformat()
            }
            
            for symbol in symbols:
                try:
                    if force_refresh:
                        # Force full refresh
                        data = self.data_cache.get_cached_data(symbol, "2y", force_refresh=True)
                    else:
                        # Smart refresh (default behavior)
                        data = self.data_cache.get_cached_data(symbol)
                    
                    if not data.empty:
                        results['success'].append({
                            'symbol': symbol,
                            'records': len(data),
                            'date_range': {
                                'start': data.index.min().strftime('%Y-%m-%d'),
                                'end': data.index.max().strftime('%Y-%m-%d')
                            }
                        })
                    else:
                        results['failed'].append({
                            'symbol': symbol,
                            'error': 'No data returned'
                        })
                        
                except Exception as e:
                    results['failed'].append({
                        'symbol': symbol,
                        'error': str(e)
                    })
                
                results['total_processed'] += 1
            
            results['end_time'] = datetime.now().isoformat()
            results['success_rate'] = len(results['success']) / results['total_processed'] if results['total_processed'] > 0 else 0
            
            return results
            
        except Exception as e:
            return {
                'error': f"Data update failed: {str(e)}",
                'success': [],
                'failed': [],
                'total_processed': 0,
                'success_rate': 0
            }
    
    def get_symbol_data(self, symbol: str, period: str = "1y") -> Dict:
        """Get data for a specific symbol"""
        try:
            # Get price data from cache
            data = self.data_cache.get_cached_data(symbol, period)
            
            if data.empty:
                return {
                    'symbol': symbol,
                    'error': 'No data available',
                    'data': []
                }
            
            # Convert to web-friendly format
            price_data = []
            for date, row in data.iterrows():
                price_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(row['Open'], 2),
                    'high': round(row['High'], 2),
                    'low': round(row['Low'], 2),
                    'close': round(row['Close'], 2),
                    'volume': int(row['Volume'])
                })
            
            # Get technical indicators
            indicators = self.get_symbol_indicators(symbol, period)
            
            return {
                'symbol': symbol,
                'period': period,
                'data': price_data,
                'indicators': indicators,
                'total_records': len(price_data),
                'date_range': {
                    'start': data.index.min().strftime('%Y-%m-%d'),
                    'end': data.index.max().strftime('%Y-%m-%d')
                }
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f"Failed to get symbol data: {str(e)}",
                'data': []
            }
    
    def get_symbol_indicators(self, symbol: str, period: str = "1y") -> Dict:
        """Get technical indicators for a symbol"""
        try:
            # Calculate period start date
            end_date = datetime.now().date()
            if period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "6m":
                start_date = end_date - timedelta(days=180)
            elif period == "3m":
                start_date = end_date - timedelta(days=90)
            elif period == "1m":
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=365)
            
            # Get indicators from database
            indicators_query = Indicator.query.filter(
                Indicator.symbol == symbol,
                Indicator.date >= start_date,
                Indicator.date <= end_date
            ).order_by(Indicator.date).all()
            
            # Group by indicator name
            indicators_dict = {}
            for indicator in indicators_query:
                if indicator.indicator_name not in indicators_dict:
                    indicators_dict[indicator.indicator_name] = []
                
                indicators_dict[indicator.indicator_name].append({
                    'date': indicator.date.strftime('%Y-%m-%d'),
                    'value': round(indicator.value, 4)
                })
            
            return indicators_dict
            
        except Exception as e:
            return {
                'error': f"Failed to get indicators: {str(e)}"
            }
    
    def get_instruments(self, instrument_type: Optional[str] = None) -> List[Dict]:
        """Get instruments from database"""
        try:
            import sqlite3
            with sqlite3.connect(self.data_cache.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query based on instrument type
                if instrument_type:
                    if instrument_type.lower() == 'etf':
                        cursor.execute("""
                            SELECT id, symbol, name, type, sector, theme, geography, 
                                   leverage, volatility_profile, tags
                            FROM instruments 
                            WHERE type IN ('ETF', 'ETN') 
                            ORDER BY symbol
                        """)
                    else:
                        cursor.execute("""
                            SELECT id, symbol, name, type, sector, theme, geography, 
                                   leverage, volatility_profile, tags
                            FROM instruments 
                            WHERE type = ? 
                            ORDER BY symbol
                        """, (instrument_type,))
                else:
                    cursor.execute("""
                        SELECT id, symbol, name, type, sector, theme, geography, 
                               leverage, volatility_profile, tags
                        FROM instruments 
                        ORDER BY symbol
                    """)
                
                instruments = cursor.fetchall()
                
                return [
                    {
                        'id': inst[0],
                        'symbol': inst[1],
                        'name': inst[2],
                        'type': inst[3],
                        'sector': inst[4],
                        'theme': inst[5],
                        'geography': inst[6],
                        'leverage': inst[7],
                        'volatility_profile': inst[8],
                        'tags': inst[9].split(',') if inst[9] else []
                    }
                    for inst in instruments
                ]
            
        except Exception as e:
            return [{'error': f"Failed to get instruments: {str(e)}"}]