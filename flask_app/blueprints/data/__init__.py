"""
Data Management Blueprint

Handles data updates, cache management, and ETF/Stock universe management.
"""

from flask import Blueprint, render_template, jsonify

data_bp = Blueprint('data', __name__, template_folder='templates')

@data_bp.route('/')
def index():
    """Data management main page"""
    return render_template('data/index.html')

@data_bp.route('/cache-stats')
def cache_stats():
    """Cache statistics page"""
    return render_template('data/cache_stats.html')

@data_bp.route('/update-data')
def update_data():
    """Data update page"""
    return render_template('data/update_data.html')

@data_bp.route('/api/update', methods=['POST'])
def api_update():
    """API endpoint for data updates"""
    return jsonify({
        'success': False,
        'message': 'Data update functionality coming soon!'
    })