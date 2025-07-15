"""
Journal Blueprint

Trade journal functionality for tracking positions, performance, and analysis.
"""

from flask import Blueprint, render_template, jsonify, request, flash, redirect, url_for
from services.journal_service import JournalService
from models import db, Instrument, Setup

journal_bp = Blueprint('journal', __name__, template_folder='templates')
journal_service = JournalService()

@journal_bp.route('/')
def index():
    """Journal main page with overview"""
    # Get summary data for dashboard
    open_positions = journal_service.get_open_positions()
    performance = journal_service.get_performance_metrics(days_back=90)
    recent_trades = journal_service.get_all_trades(limit=10)
    
    return render_template('journal/index.html', 
                         open_positions=open_positions,
                         performance=performance,
                         recent_trades=recent_trades)

@journal_bp.route('/positions')
def positions():
    """Open positions page"""
    positions_data = journal_service.get_open_positions()
    return render_template('journal/positions.html', positions=positions_data)

@journal_bp.route('/history')
def history():
    """Trade history page"""
    status_filter = request.args.get('status', 'all')
    limit = int(request.args.get('limit', 50))
    
    if status_filter == 'all':
        trades = journal_service.get_all_trades(limit=limit)
    else:
        trades = journal_service.get_all_trades(status=status_filter, limit=limit)
    
    return render_template('journal/history.html', trades=trades, status_filter=status_filter)

@journal_bp.route('/add')
def add_trade_form():
    """Add trade form"""
    instruments = Instrument.query.order_by(Instrument.symbol).all()
    setups = Setup.query.order_by(Setup.name).all()
    return render_template('journal/add_trade.html', instruments=instruments, setups=setups)

@journal_bp.route('/edit/<int:trade_id>')
def edit_trade_form(trade_id):
    """Edit trade form"""
    from models import Trade
    trade = Trade.query.get_or_404(trade_id)
    instruments = Instrument.query.order_by(Instrument.symbol).all()
    setups = Setup.query.order_by(Setup.name).all()
    return render_template('journal/edit_trade.html', trade=trade, instruments=instruments, setups=setups)

@journal_bp.route('/analytics')
def analytics():
    """Analytics and performance dashboard"""
    performance_30d = journal_service.get_performance_metrics(days_back=30)
    performance_90d = journal_service.get_performance_metrics(days_back=90)
    performance_365d = journal_service.get_performance_metrics(days_back=365)
    correlations = journal_service.get_trade_correlations(days_back=30)
    
    return render_template('journal/analytics.html',
                         performance_30d=performance_30d,
                         performance_90d=performance_90d,
                         performance_365d=performance_365d,
                         correlations=correlations)

# API Endpoints
@journal_bp.route('/api/trades', methods=['GET'])
def api_get_trades():
    """Get trades API"""
    status = request.args.get('status')
    limit = int(request.args.get('limit', 50))
    
    if status:
        trades = journal_service.get_all_trades(status=status, limit=limit)
    else:
        trades = journal_service.get_all_trades(limit=limit)
    
    return jsonify({'success': True, 'trades': trades})

@journal_bp.route('/api/trades', methods=['POST'])
def api_add_trade():
    """Add trade API"""
    trade_data = request.get_json()
    result = journal_service.add_trade(trade_data)
    return jsonify(result)

@journal_bp.route('/api/trades/<int:trade_id>', methods=['PUT'])
def api_update_trade(trade_id):
    """Update trade API"""
    trade_data = request.get_json()
    result = journal_service.update_trade(trade_id, trade_data)
    return jsonify(result)

@journal_bp.route('/api/trades/<int:trade_id>/close', methods=['POST'])
def api_close_trade(trade_id):
    """Close trade API"""
    data = request.get_json()
    exit_price = data.get('exit_price')
    exit_date = data.get('exit_date')
    exit_reason = data.get('exit_reason', '')
    
    if not exit_price:
        return jsonify({'success': False, 'error': 'Exit price is required'})
    
    result = journal_service.close_trade(trade_id, exit_price, exit_date, exit_reason)
    return jsonify(result)

@journal_bp.route('/api/positions', methods=['GET'])
def api_get_positions():
    """Get open positions API"""
    positions = journal_service.get_open_positions()
    return jsonify({'success': True, 'data': positions})

@journal_bp.route('/api/performance', methods=['GET'])
def api_get_performance():
    """Get performance metrics API"""
    days_back = int(request.args.get('days', 90))
    performance = journal_service.get_performance_metrics(days_back=days_back)
    return jsonify({'success': True, 'data': performance})

@journal_bp.route('/api/correlations', methods=['GET'])
def api_get_correlations():
    """Get trade correlations API"""
    days_back = int(request.args.get('days', 30))
    correlations = journal_service.get_trade_correlations(days_back=days_back)
    return jsonify({'success': True, 'data': correlations})

# Form submission endpoints
@journal_bp.route('/submit/add_trade', methods=['POST'])
def submit_add_trade():
    """Handle add trade form submission"""
    try:
        trade_data = {
            'symbol': request.form['symbol'],
            'entry_date': request.form['entry_date'],
            'entry_price': request.form['entry_price'],
            'size': request.form['size'],
            'stop_loss': request.form.get('stop_loss'),
            'target_price': request.form.get('target_price'),
            'setup_name': request.form.get('setup_name'),
            'r_planned': request.form.get('r_planned'),
            'commission': request.form.get('commission', 0),
            'notes': request.form.get('notes', ''),
            'entry_reason': request.form.get('entry_reason', ''),
            'regime_at_entry': request.form.get('regime_at_entry', '')
        }
        
        result = journal_service.add_trade(trade_data)
        
        if result['success']:
            flash(result['message'], 'success')
            return redirect(url_for('journal.positions'))
        else:
            flash(result['error'], 'error')
            return redirect(url_for('journal.add_trade_form'))
            
    except Exception as e:
        flash(f'Error adding trade: {str(e)}', 'error')
        return redirect(url_for('journal.add_trade_form'))

@journal_bp.route('/submit/edit_trade/<int:trade_id>', methods=['POST'])
def submit_edit_trade(trade_id):
    """Handle edit trade form submission"""
    try:
        trade_data = {
            'entry_price': request.form.get('entry_price'),
            'exit_price': request.form.get('exit_price'),
            'exit_date': request.form.get('exit_date'),
            'size': request.form.get('size'),
            'stop_loss': request.form.get('stop_loss'),
            'target_price': request.form.get('target_price'),
            'r_planned': request.form.get('r_planned'),
            'r_actual': request.form.get('r_actual'),
            'commission': request.form.get('commission'),
            'notes': request.form.get('notes', ''),
            'entry_reason': request.form.get('entry_reason', ''),
            'exit_reason': request.form.get('exit_reason', ''),
            'status': request.form.get('status')
        }
        
        # Remove empty values
        trade_data = {k: v for k, v in trade_data.items() if v != '' and v is not None}
        
        result = journal_service.update_trade(trade_id, trade_data)
        
        if result['success']:
            flash(result['message'], 'success')
            return redirect(url_for('journal.history'))
        else:
            flash(result['error'], 'error')
            return redirect(url_for('journal.edit_trade_form', trade_id=trade_id))
            
    except Exception as e:
        flash(f'Error updating trade: {str(e)}', 'error')
        return redirect(url_for('journal.edit_trade_form', trade_id=trade_id))