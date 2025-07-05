"""
Screener Blueprint

ETF and Stock screening functionality with regime-aware filtering.
Integrates with screener.py and trade_setups.py through service layer.
"""

from flask import Blueprint, render_template, jsonify, request, flash, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import SelectField, FloatField, BooleanField, SubmitField
from wtforms.validators import NumberRange, Optional
import sys
import os

# Add parent directory to path for service imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services import ScreenerService, RegimeService, DataService

screener_bp = Blueprint('screener', __name__, template_folder='templates')

# Initialize services
screener_service = ScreenerService()
regime_service = RegimeService()
data_service = DataService()

class ScreenerForm(FlaskForm):
    """Form for screening parameters"""
    setup_name = SelectField('Trade Setup', choices=[], validators=[Optional()])
    instrument_type = SelectField('Instrument Type', choices=[
        ('', 'All Instruments'),
        ('etf', 'ETFs Only'),
        ('stock', 'Stocks Only')
    ], default='')
    min_confidence = FloatField('Minimum Confidence', 
                               validators=[NumberRange(min=0.0, max=1.0)], 
                               default=0.5)
    regime_filter = BooleanField('Apply Regime Filter', default=True)
    submit = SubmitField('Run Screening')

@screener_bp.route('/')
def index():
    """Screener main page"""
    # Initialize form with available setups
    form = ScreenerForm()
    
    # Get available setups for dropdown
    setups = screener_service.get_available_setups()
    form.setup_name.choices = [('', 'All Setups')] + [
        (setup['name'], setup['display_name']) 
        for setup in setups 
        if 'error' not in setup
    ]
    
    # Get current regime for display
    current_regime = regime_service.get_current_regime()
    
    # Get cache stats for data health check
    cache_stats = data_service.get_cache_stats()
    
    return render_template('screener/index.html', 
                         form=form,
                         current_regime=current_regime,
                         cache_stats=cache_stats)

@screener_bp.route('/scan', methods=['POST'])
def run_scan():
    """Run screening scan with form parameters"""
    form = ScreenerForm()
    
    # Populate setup choices
    setups = screener_service.get_available_setups()
    form.setup_name.choices = [('', 'All Setups')] + [
        (setup['name'], setup['display_name']) 
        for setup in setups 
        if 'error' not in setup
    ]
    
    if form.validate_on_submit():
        # Run screening with form parameters
        results = screener_service.run_screening(
            setup_name=form.setup_name.data if form.setup_name.data else None,
            instrument_type=form.instrument_type.data if form.instrument_type.data else None,
            regime_filter=form.regime_filter.data,
            min_confidence=form.min_confidence.data
        )
        
        if 'error' not in results:
            flash(f"Screening completed: {results['total_matches']} matches from {results['total_screened']} instruments", 'success')
        else:
            flash(f"Screening failed: {results['error']}", 'error')
        
        return render_template('screener/results.html', 
                             results=results, 
                             form=form)
    else:
        # Form validation failed
        flash('Invalid screening parameters', 'error')
        return redirect(url_for('screener.index'))

@screener_bp.route('/analyze/<symbol>')
def analyze_symbol(symbol):
    """Analyze a specific symbol"""
    setup_name = request.args.get('setup')
    
    analysis = screener_service.analyze_symbol(symbol, setup_name)
    
    return render_template('screener/analysis.html', 
                         analysis=analysis, 
                         symbol=symbol)

@screener_bp.route('/api/scan', methods=['POST'])
def api_scan():
    """API endpoint for screening scans"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Run screening with API parameters
        results = screener_service.run_screening(
            setup_name=data.get('setup_name'),
            instrument_type=data.get('instrument_type'),
            regime_filter=data.get('regime_filter', True),
            min_confidence=data.get('min_confidence', 0.5)
        )
        
        if 'error' not in results:
            return jsonify({
                'success': True,
                'data': results
            })
        else:
            return jsonify({
                'success': False,
                'error': results['error']
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'API scan failed: {str(e)}'
        }), 500

@screener_bp.route('/api/analyze/<symbol>')
def api_analyze(symbol):
    """API endpoint for symbol analysis"""
    try:
        setup_name = request.args.get('setup')
        
        analysis = screener_service.analyze_symbol(symbol, setup_name)
        
        if 'error' not in analysis:
            return jsonify({
                'success': True,
                'data': analysis
            })
        else:
            return jsonify({
                'success': False,
                'error': analysis['error']
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Symbol analysis failed: {str(e)}'
        }), 500

@screener_bp.route('/api/setups')
def api_setups():
    """API endpoint to get available setups"""
    try:
        setups = screener_service.get_available_setups()
        
        return jsonify({
            'success': True,
            'data': setups
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get setups: {str(e)}'
        }), 500

@screener_bp.route('/export/<format>')
def export_results(format):
    """Export screening results in different formats"""
    # This would implement CSV/JSON export functionality
    # For now, return a placeholder
    return jsonify({
        'success': False,
        'message': f'{format.upper()} export functionality coming soon!'
    })