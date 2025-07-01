"""
Journal Blueprint

Trade journal functionality for tracking positions, performance, and analysis.
"""

from flask import Blueprint, render_template, jsonify

journal_bp = Blueprint('journal', __name__, template_folder='templates')

@journal_bp.route('/')
def index():
    """Journal main page"""
    return render_template('journal/index.html')

@journal_bp.route('/api/trades', methods=['GET', 'POST'])
def api_trades():
    """API endpoint for trade management"""
    return jsonify({
        'success': False,
        'message': 'Journal functionality coming soon!'
    })