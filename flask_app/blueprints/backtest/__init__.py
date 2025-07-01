"""
Backtest Blueprint

Backtesting engine with walk-forward analysis and performance metrics.
"""

from flask import Blueprint, render_template, jsonify

backtest_bp = Blueprint('backtest', __name__, template_folder='templates')

@backtest_bp.route('/')
def index():
    """Backtest main page"""
    return render_template('backtest/index.html')

@backtest_bp.route('/api/run', methods=['POST'])
def api_run():
    """API endpoint for running backtests"""
    return jsonify({
        'success': False,
        'message': 'Backtest functionality coming soon!'
    })