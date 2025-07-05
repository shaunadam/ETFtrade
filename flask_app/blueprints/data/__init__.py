"""
Data Management Blueprint

Handles data updates, cache management, and ETF/Stock universe management.
"""

from flask import Blueprint, render_template, jsonify, request
from datetime import datetime
import sys
import os

# Add parent directory to import service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.data_service import DataService

data_bp = Blueprint('data', __name__, template_folder='templates')

@data_bp.route('/')
def index():
    """Data management main page"""
    try:
        data_service = DataService()
        
        # Get cache statistics
        cache_stats = data_service.get_cache_stats()
        
        # Get instruments for universe display
        instruments = data_service.get_instruments()
        
        return render_template('data/index.html', 
                             cache_stats=cache_stats,
                             instruments=instruments[:10],  # Show first 10 for preview
                             total_instruments=len(instruments))
    except Exception as e:
        return render_template('data/index.html', 
                             error=f"Failed to load data: {str(e)}")

@data_bp.route('/api/cache-stats')
def api_cache_stats():
    """API endpoint for cache statistics"""
    try:
        data_service = DataService()
        stats = data_service.get_cache_stats()
        
        return jsonify({
            'success': True,
            'data': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@data_bp.route('/api/update', methods=['POST'])
def api_update():
    """API endpoint for data updates"""
    try:
        data_service = DataService()
        
        # Get request parameters
        symbols = request.json.get('symbols', []) if request.json else []
        force_refresh = request.json.get('force_refresh', False) if request.json else False
        
        # If no symbols specified, use all instruments
        if not symbols:
            symbols = None
        
        result = data_service.update_data(symbols, force_refresh)
        
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@data_bp.route('/api/instruments')
def api_instruments():
    """API endpoint for instruments"""
    try:
        data_service = DataService()
        
        instrument_type = request.args.get('type')
        instruments = data_service.get_instruments(instrument_type)
        
        return jsonify({
            'success': True,
            'data': instruments,
            'total': len(instruments),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@data_bp.route('/api/symbol/<symbol>')
def api_symbol_data(symbol):
    """API endpoint for symbol data"""
    try:
        data_service = DataService()
        
        period = request.args.get('period', '1y')
        symbol_data = data_service.get_symbol_data(symbol, period)
        
        return jsonify({
            'success': True,
            'data': symbol_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500