"""
Regime Blueprint

Market regime analysis and visualization functionality.
"""

from flask import Blueprint, render_template, jsonify

regime_bp = Blueprint('regime', __name__, template_folder='templates')

@regime_bp.route('/')
def index():
    """Regime analysis main page"""
    return render_template('regime/index.html')

@regime_bp.route('/api/current')
def api_current():
    """API endpoint for current regime data"""
    return jsonify({
        'success': False,
        'message': 'Regime analysis functionality coming soon!'
    })