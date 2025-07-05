"""
Backtest Blueprint

Backtesting engine with walk-forward analysis and performance metrics.
"""

from flask import Blueprint, render_template, jsonify, request
from datetime import datetime
import sys
import os

# Add parent directory to import service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.backtest_service import BacktestService

backtest_bp = Blueprint('backtest', __name__, template_folder='templates')

@backtest_bp.route('/')
def index():
    """Backtest main page"""
    try:
        backtest_service = BacktestService()
        
        # Get available setups
        setups = backtest_service.get_available_setups()
        
        # Get default parameters
        default_params = backtest_service.get_default_parameters()
        
        return render_template('backtest/index.html', 
                             setups=setups,
                             default_params=default_params,
                             page_title="Backtest Engine")
    except Exception as e:
        return render_template('backtest/index.html', 
                             error=f"Failed to load backtest data: {str(e)}")

@backtest_bp.route('/api/setups')
def api_setups():
    """API endpoint for available setups"""
    try:
        backtest_service = BacktestService()
        setups = backtest_service.get_available_setups()
        
        return jsonify({
            'success': True,
            'data': setups,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@backtest_bp.route('/api/default-params')
def api_default_params():
    """API endpoint for default parameters"""
    try:
        backtest_service = BacktestService()
        params = backtest_service.get_default_parameters()
        
        return jsonify({
            'success': True,
            'data': params,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@backtest_bp.route('/api/validate', methods=['POST'])
def api_validate():
    """API endpoint for parameter validation"""
    try:
        backtest_service = BacktestService()
        config = request.json if request.json else {}
        
        validation = backtest_service.validate_parameters(config)
        
        return jsonify({
            'success': True,
            'data': validation,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@backtest_bp.route('/api/run', methods=['POST'])
def api_run():
    """API endpoint for running backtests"""
    try:
        backtest_service = BacktestService()
        config = request.json if request.json else {}
        
        # Validate parameters first
        validation = backtest_service.validate_parameters(config)
        if not validation['valid']:
            return jsonify({
                'success': False,
                'error': 'Invalid parameters',
                'validation_errors': validation['errors']
            }), 400
        
        # Run backtest
        results = backtest_service.run_backtest(config)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500