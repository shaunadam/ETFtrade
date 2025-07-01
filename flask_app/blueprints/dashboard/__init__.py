"""
Dashboard Blueprint

Main dashboard showing system status, market regime, recent trades,
and key performance metrics for the ETF/Stock trading system.
"""

import sys
import os
from flask import Blueprint, render_template, jsonify
from datetime import datetime, timedelta
import sqlite3

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import existing trading system modules
try:
    from regime_detection import detect_current_regime
    from data_cache import ETFDataCache
except ImportError as e:
    print(f"Warning: Could not import trading modules: {e}")
    detect_current_regime = None
    ETFDataCache = None

dashboard_bp = Blueprint('dashboard', __name__, template_folder='templates')

@dashboard_bp.route('/')
def index():
    """Main dashboard page"""
    try:
        # Get system status
        status = get_system_status()
        
        # Get current market regime
        regime = get_current_regime()
        
        # Get recent trades
        recent_trades = get_recent_trades()
        
        # Get cache statistics
        cache_stats = get_cache_statistics()
        
        return render_template('dashboard/index.html',
                             status=status,
                             regime=regime,
                             recent_trades=recent_trades,
                             cache_stats=cache_stats)
    
    except Exception as e:
        error_msg = f"Dashboard error: {str(e)}"
        print(error_msg)
        return render_template('dashboard/index.html',
                             error=error_msg,
                             status={},
                             regime={},
                             recent_trades=[],
                             cache_stats={})

@dashboard_bp.route('/api/status')
def api_status():
    """API endpoint for dashboard status updates"""
    try:
        return jsonify({
            'success': True,
            'status': get_system_status(),
            'regime': get_current_regime(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def get_system_status():
    """Get current system status information"""
    try:
        # Database connectivity
        db_path = get_database_path()
        database_connected = check_database_connection(db_path)
        
        # Count total symbols
        total_symbols = count_total_symbols(db_path) if database_connected else 0
        
        # Count open positions
        open_positions = count_open_positions(db_path) if database_connected else 0
        
        # Count total trades
        total_trades = count_total_trades(db_path) if database_connected else 0
        
        # Data freshness check
        data_fresh = check_data_freshness(db_path) if database_connected else False
        
        return {
            'database_connected': database_connected,
            'total_symbols': total_symbols,
            'open_positions': open_positions,
            'total_trades': total_trades,
            'data_fresh': data_fresh,
            'cache_healthy': check_cache_health()
        }
    except Exception as e:
        print(f"Error getting system status: {e}")
        return {
            'database_connected': False,
            'total_symbols': 0,
            'open_positions': 0,
            'total_trades': 0,
            'data_fresh': False,
            'cache_healthy': False
        }

def get_current_regime():
    """Get current market regime information"""
    try:
        if detect_current_regime is None:
            return {'error': 'Regime detection module not available'}
        
        regime_data = detect_current_regime()
        
        # Format regime data for display
        formatted_regime = {}
        regime_mapping = {
            'volatility_regime': {'low': 'Low Vol', 'medium': 'Med Vol', 'high': 'High Vol'},
            'trend_regime': {'bullish': 'Bullish', 'bearish': 'Bearish', 'sideways': 'Sideways'},
            'sector_rotation': {'growth': 'Growth', 'value': 'Value', 'balanced': 'Balanced'},
            'risk_on_off': {'risk_on': 'Risk On', 'risk_off': 'Risk Off', 'neutral': 'Neutral'}
        }
        
        for key, value in regime_data.items():
            if key in regime_mapping and value in regime_mapping[key]:
                formatted_regime[key] = {
                    'value': value,
                    'label': regime_mapping[key][value]
                }
        
        return formatted_regime
        
    except Exception as e:
        print(f"Error getting current regime: {e}")
        return {'error': f'Regime detection failed: {str(e)}'}

def get_recent_trades(limit=10):
    """Get recent trades from the database"""
    try:
        db_path = get_database_path()
        if not check_database_connection(db_path):
            return []
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get recent trades with instrument information
        query = """
        SELECT t.id, i.symbol, t.setup, t.entry_date, t.exit_date,
               t.entry_price, t.exit_price, t.r_actual,
               CASE WHEN t.exit_date IS NULL THEN 'Open' ELSE 'Closed' END as status
        FROM trades t
        JOIN instruments i ON t.instrument_id = i.id
        ORDER BY t.entry_date DESC
        LIMIT ?
        """
        
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        trades = []
        for row in rows:
            trades.append({
                'id': row['id'],
                'symbol': row['symbol'],
                'setup': row['setup'],
                'entry_date': row['entry_date'],
                'exit_date': row['exit_date'],
                'entry_price': row['entry_price'],
                'exit_price': row['exit_price'],
                'r_actual': row['r_actual'],
                'status': row['status']
            })
        
        conn.close()
        return trades
        
    except Exception as e:
        print(f"Error getting recent trades: {e}")
        return []

def get_cache_statistics():
    """Get data cache statistics"""
    try:
        if ETFDataCache is None:
            return {
                'total_symbols': 0,
                'total_records': 0,
                'total_indicators': 0,
                'cache_hit_rate': 'N/A'
            }
        
        cache = ETFDataCache()
        stats = cache.get_cache_statistics()
        
        return {
            'total_symbols': stats.get('total_symbols', 0),
            'total_records': stats.get('total_records', 0),
            'total_indicators': stats.get('total_indicators', 0),
            'cache_hit_rate': f"{stats.get('cache_hit_rate', 0):.1f}%"
        }
        
    except Exception as e:
        print(f"Error getting cache statistics: {e}")
        return {
            'total_symbols': 0,
            'total_records': 0,
            'total_indicators': 0,
            'cache_hit_rate': 'Error'
        }

def get_database_path():
    """Get the path to the journal database"""
    # Try to get from config first, fallback to parent directory
    try:
        from flask import current_app
        return current_app.config.get('DATABASE_PATH')
    except:
        # Fallback to parent directory
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        return os.path.join(current_dir, 'journal.db')

def check_database_connection(db_path):
    """Check if database connection is working"""
    try:
        if not os.path.exists(db_path):
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        cursor.fetchone()
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection check failed: {e}")
        return False

def count_total_symbols(db_path):
    """Count total symbols in the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM instruments")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0

def count_open_positions(db_path):
    """Count open trading positions"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_date IS NULL")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0

def count_total_trades(db_path):
    """Count total trades in the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0

def check_data_freshness(db_path):
    """Check if market data is fresh (updated within last 2 days)"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check for recent price data
        two_days_ago = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT COUNT(*) FROM price_data 
            WHERE updated_at >= ? 
        """, (two_days_ago,))
        
        recent_count = cursor.fetchone()[0]
        conn.close()
        
        return recent_count > 0
    except Exception:
        return False

def check_cache_health():
    """Check if data cache is healthy"""
    try:
        if ETFDataCache is None:
            return False
        
        cache = ETFDataCache()
        # Simple health check - try to get cache stats
        stats = cache.get_cache_statistics()
        return stats.get('total_records', 0) > 0
    except Exception:
        return False