"""
Screener Blueprint

ETF and Stock screening functionality with regime-aware filtering.
Integrates with screener.py and trade_setups.py through service layer.
"""

from flask import Blueprint, render_template, jsonify, request, flash, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import SelectField, FloatField, BooleanField, SubmitField
from wtforms.validators import NumberRange, Optional
import sys
import os

# Add parent directory to path for service imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services import ScreenerService, RegimeService, DataService, ChartService

screener_bp = Blueprint('screener', __name__, template_folder='templates')

# Initialize services
screener_service = ScreenerService()
regime_service = RegimeService()
data_service = DataService()
chart_service = ChartService()

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

@screener_bp.route('/results')
def results():
    """Display cached screening results"""
    # Get results from session
    results = session.get('last_screening_results')
    params = session.get('last_screening_params')
    
    if not results:
        flash('No screening results found. Please run a new screening.', 'warning')
        return redirect(url_for('screener.index'))
    
    # Create a minimal form object for template compatibility
    form = ScreenerForm()
    setups = screener_service.get_available_setups()
    form.setup_name.choices = [('', 'All Setups')] + [
        (setup['name'], setup['display_name']) 
        for setup in setups 
        if 'error' not in setup
    ]
    
    # Pre-populate form with parameters used for screening
    if params:
        form.setup_name.data = params.get('setup_name', '')
        form.instrument_type.data = params.get('instrument_type', '')
        form.regime_filter.data = params.get('regime_filter', True)
        form.min_confidence.data = params.get('min_confidence', 0.5)
    
    return render_template('screener/results.html', 
                         results=results, 
                         form=form)

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
            # Store results in session for results page
            session['last_screening_results'] = results
            session['last_screening_params'] = data
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
    try:
        # Get parameters from session or query parameters
        setup_name = request.args.get('setup')
        instrument_type = request.args.get('type')
        regime_filter = request.args.get('regime_filter', 'true').lower() == 'true'
        min_confidence = float(request.args.get('min_confidence', 0.5))
        
        # Run screening to get fresh results
        results = screener_service.run_screening(
            setup_name=setup_name,
            instrument_type=instrument_type,
            regime_filter=regime_filter,
            min_confidence=min_confidence
        )
        
        if 'error' in results:
            return jsonify({
                'success': False,
                'error': results['error']
            }), 500
        
        if format.lower() == 'csv':
            return export_csv(results)
        elif format.lower() == 'json':
            return export_json(results)
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported export format: {format}'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Export failed: {str(e)}'
        }), 500

def export_csv(results):
    """Export results as CSV"""
    import csv
    import io
    from flask import make_response
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    header = [
        'Symbol', 'Name', 'Type', 'Setup', 'Signal', 'Confidence', 
        'Entry Price', 'Target Price', 'Stop Loss', 'Risk/Reward Ratio',
        'Current Price', 'Sector'
    ]
    writer.writerow(header)
    
    # Write data rows
    for result in results.get('results', []):
        instrument = result.get('instrument', {})
        setup_name = result.get('setup_name') or (result.get('best_setup', {}).get('setup_name') if result.get('best_setup') else '')
        signal = result.get('signal') or (result.get('best_setup', {}).get('signal') if result.get('best_setup') else '')
        confidence = result.get('confidence', 0)
        entry_price = result.get('entry_price') or (result.get('best_setup', {}).get('entry_price') if result.get('best_setup') else '')
        target_price = result.get('target_price') or (result.get('best_setup', {}).get('target_price') if result.get('best_setup') else '')
        stop_loss = result.get('stop_loss') or (result.get('best_setup', {}).get('stop_loss') if result.get('best_setup') else '')
        
        # Calculate R/R ratio
        rr_ratio = ''
        if entry_price and target_price and stop_loss:
            try:
                reward = float(target_price) - float(entry_price)
                risk = float(entry_price) - float(stop_loss)
                if risk > 0:
                    rr_ratio = f"{reward/risk:.1f}:1"
            except (ValueError, ZeroDivisionError):
                pass
        
        row = [
            instrument.get('symbol', ''),
            instrument.get('name', ''),
            instrument.get('type', ''),
            setup_name.replace('_', ' ').title() if setup_name else '',
            signal,
            f"{confidence*100:.1f}%" if confidence else '',
            f"${entry_price:.2f}" if entry_price else '',
            f"${target_price:.2f}" if target_price else '',
            f"${stop_loss:.2f}" if stop_loss else '',
            rr_ratio,
            f"${result.get('current_price', ''):.2f}" if result.get('current_price') else '',
            instrument.get('sector', '')
        ]
        writer.writerow(row)
    
    output.seek(0)
    
    # Create response
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=screener_results.csv'
    
    return response

def export_json(results):
    """Export results as JSON"""
    from flask import make_response
    import json
    from datetime import datetime
    
    # Prepare export data
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'screening_parameters': results.get('screening_parameters', {}),
        'summary': {
            'total_matches': results.get('total_matches', 0),
            'total_screened': results.get('total_screened', 0)
        },
        'results': []
    }
    
    # Process results
    for result in results.get('results', []):
        export_result = {
            'instrument': result.get('instrument', {}),
            'setup_name': result.get('setup_name') or (result.get('best_setup', {}).get('setup_name') if result.get('best_setup') else ''),
            'signal': result.get('signal') or (result.get('best_setup', {}).get('signal') if result.get('best_setup') else ''),
            'confidence': result.get('confidence', 0),
            'entry_price': result.get('entry_price') or (result.get('best_setup', {}).get('entry_price') if result.get('best_setup') else None),
            'target_price': result.get('target_price') or (result.get('best_setup', {}).get('target_price') if result.get('best_setup') else None),
            'stop_loss': result.get('stop_loss') or (result.get('best_setup', {}).get('stop_loss') if result.get('best_setup') else None),
            'current_price': result.get('current_price'),
            'analysis_timestamp': result.get('analysis_timestamp')
        }
        export_data['results'].append(export_result)
    
    # Create response
    response = make_response(json.dumps(export_data, indent=2))
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Disposition'] = 'attachment; filename=screener_results.json'
    
    return response

@screener_bp.route('/api/chart/<symbol>')
def api_chart_data(symbol):
    """API endpoint for getting chart data"""
    try:
        period_days = request.args.get('period', 90, type=int)
        setup_name = request.args.get('setup')
        
        if setup_name:
            chart_data = chart_service.get_trade_setup_chart_data(symbol, setup_name, period_days)
        else:
            chart_data = chart_service.get_price_chart_data(symbol, period_days)
        
        if 'error' not in chart_data:
            return jsonify({
                'success': True,
                'data': chart_data
            })
        else:
            return jsonify({
                'success': False,
                'error': chart_data['error']
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get chart data: {str(e)}'
        }), 500

@screener_bp.route('/api/chart-config/<chart_type>')
def api_chart_config(chart_type):
    """API endpoint for getting chart configuration"""
    try:
        symbol = request.args.get('symbol', 'CHART')
        
        config = chart_service.get_chart_config(chart_type)
        layout = chart_service.get_chart_layout(symbol, chart_type)
        
        return jsonify({
            'success': True,
            'data': {
                'config': config,
                'layout': layout
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get chart config: {str(e)}'
        }), 500