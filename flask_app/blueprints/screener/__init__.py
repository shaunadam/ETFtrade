"""
Screener Blueprint

ETF and Stock screening functionality with regime-aware filtering.
"""

from flask import Blueprint, render_template, jsonify

screener_bp = Blueprint('screener', __name__, template_folder='templates')

@screener_bp.route('/')
def index():
    """Screener main page"""
    return render_template('screener/index.html')

@screener_bp.route('/api/scan', methods=['POST'])
def api_scan():
    """API endpoint for screening scans"""
    return jsonify({
        'success': False,
        'message': 'Screener functionality coming soon!'
    })