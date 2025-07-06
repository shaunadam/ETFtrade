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

@backtest_bp.route('/api/config/presets')
def api_config_presets():
    """API endpoint for configuration presets"""
    try:
        backtest_service = BacktestService()
        presets = backtest_service.get_available_presets()
        
        return jsonify({
            'success': True,
            'data': presets,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@backtest_bp.route('/api/config/load', methods=['POST'])
def api_config_load():
    """API endpoint for loading configuration"""
    try:
        backtest_service = BacktestService()
        data = request.json if request.json else {}
        
        preset_name = data.get('preset_name')
        config_file = data.get('config_file')
        
        result = backtest_service.load_configuration(preset_name, config_file)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@backtest_bp.route('/api/config/save', methods=['POST'])
def api_config_save():
    """API endpoint for saving configuration"""
    try:
        backtest_service = BacktestService()
        data = request.json if request.json else {}
        
        config_dict = data.get('config', {})
        name = data.get('name', f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        result = backtest_service.save_configuration(config_dict, name)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@backtest_bp.route('/api/config/validate-full', methods=['POST'])
def api_config_validate_full():
    """API endpoint for full configuration validation"""
    try:
        backtest_service = BacktestService()
        config_dict = request.json if request.json else {}
        
        result = backtest_service.validate_configuration(config_dict)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@backtest_bp.route('/api/config/export', methods=['POST'])
def api_config_export():
    """API endpoint for exporting configuration"""
    try:
        backtest_service = BacktestService()
        config_dict = request.json if request.json else {}
        
        result = backtest_service.export_configuration(config_dict)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@backtest_bp.route('/api/config/saved')
def api_config_saved():
    """API endpoint for getting saved configurations"""
    try:
        backtest_service = BacktestService()
        configs = backtest_service.get_saved_configurations()
        
        return jsonify({
            'success': True,
            'data': configs,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@backtest_bp.route('/api/run-with-config', methods=['POST'])
def api_run_with_config():
    """API endpoint for running backtest with full configuration"""
    try:
        backtest_service = BacktestService()
        data = request.json if request.json else {}
        
        config_dict = data.get('config', {})
        ui_settings = data.get('ui_settings', {})
        
        # Validate configuration first
        validation = backtest_service.validate_configuration(config_dict)
        if not validation.get('success', False) or not validation.get('data', {}).get('valid', False):
            return jsonify({
                'success': False,
                'error': 'Configuration validation failed',
                'validation': validation
            }), 400
        
        # Run backtest with configuration
        results = backtest_service.run_backtest_with_config(config_dict, ui_settings)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500