"""
Regime Blueprint

Market regime analysis and visualization functionality.
"""

from flask import Blueprint, render_template, jsonify, request
from datetime import datetime
import sys
import os

# Add parent directory to import service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.regime_service import RegimeService

regime_bp = Blueprint('regime', __name__, template_folder='templates')

@regime_bp.route('/')
def index():
    """Regime analysis main page"""
    try:
        regime_service = RegimeService()
        
        # Get current regime data
        current_regime = regime_service.get_current_regime()
        
        # Get comprehensive analysis
        analysis = regime_service.get_regime_analysis()
        
        return render_template('regime/index.html', 
                             current_regime=current_regime,
                             analysis=analysis,
                             page_title="Market Regime Analysis")
    except Exception as e:
        return render_template('regime/index.html', 
                             error=f"Failed to load regime data: {str(e)}")

@regime_bp.route('/api/current')
def api_current():
    """API endpoint for current regime data"""
    try:
        regime_service = RegimeService()
        current_regime = regime_service.get_current_regime()
        
        return jsonify({
            'success': True,
            'data': current_regime,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@regime_bp.route('/api/history')
def api_history():
    """API endpoint for historical regime data"""
    try:
        regime_service = RegimeService()
        days = request.args.get('days', 90, type=int)
        
        history = regime_service.get_regime_history(days)
        
        return jsonify({
            'success': True,
            'data': history,
            'days': days,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@regime_bp.route('/api/analysis')
def api_analysis():
    """API endpoint for regime analysis"""
    try:
        regime_service = RegimeService()
        analysis = regime_service.get_regime_analysis()
        
        return jsonify({
            'success': True,
            'data': analysis,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@regime_bp.route('/api/update', methods=['POST'])
def api_update():
    """API endpoint to update regime data"""
    try:
        regime_service = RegimeService()
        result = regime_service.update_regime_data()
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500