"""
Trade Service Layer

Integrates trade journal operations with Flask application.
Provides web-friendly interface for trade management.
"""

import sys
import os
from datetime import datetime, date
from typing import Dict, List, Optional

# Add parent directory to import CLI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask_app.models import db, Trade, Instrument, Setup, Snapshot

class TradeService:
    """Service for trade operations in Flask app"""
    
    def get_open_trades(self) -> List[Dict]:
        """Get all open trades with instrument details"""
        try:
            trades = Trade.query.filter(Trade.status == 'open').join(Instrument).all()
            
            return [
                {
                    'id': trade.id,
                    'symbol': trade.instrument.symbol,
                    'instrument_name': trade.instrument.name,
                    'instrument_type': trade.instrument.type,
                    'setup_name': trade.setup.name if trade.setup else None,
                    'entry_date': trade.entry_date.isoformat(),
                    'entry_price': trade.entry_price,
                    'size': trade.size,
                    'r_planned': trade.r_planned,
                    'regime_at_entry': trade.regime_at_entry,
                    'days_held': trade.days_held,
                    'notes': trade.notes
                }
                for trade in trades
            ]
            
        except Exception as e:
            return [{'error': f'Failed to get open trades: {str(e)}'}]
    
    def get_closed_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent closed trades"""
        try:
            trades = Trade.query.filter(
                Trade.status == 'closed'
            ).order_by(
                Trade.exit_date.desc()
            ).limit(limit).all()
            
            return [
                {
                    'id': trade.id,
                    'symbol': trade.instrument.symbol,
                    'instrument_name': trade.instrument.name,
                    'instrument_type': trade.instrument.type,
                    'setup_name': trade.setup.name if trade.setup else None,
                    'entry_date': trade.entry_date.isoformat(),
                    'exit_date': trade.exit_date.isoformat() if trade.exit_date else None,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'size': trade.size,
                    'r_planned': trade.r_planned,
                    'r_actual': trade.r_actual,
                    'pnl': self._calculate_pnl(trade),
                    'days_held': trade.days_held,
                    'regime_at_entry': trade.regime_at_entry,
                    'notes': trade.notes
                }
                for trade in trades
            ]
            
        except Exception as e:
            return [{'error': f'Failed to get closed trades: {str(e)}'}]
    
    def get_trade_by_id(self, trade_id: int) -> Optional[Dict]:
        """Get a specific trade by ID with all details"""
        try:
            trade = Trade.query.filter(Trade.id == trade_id).first()
            
            if not trade:
                return None
            
            # Get snapshots
            snapshots = Snapshot.query.filter(
                Snapshot.trade_id == trade_id
            ).order_by(Snapshot.date).all()
            
            snapshot_data = [
                {
                    'id': snap.id,
                    'date': snap.date.isoformat(),
                    'price': snap.price,
                    'notes': snap.notes,
                    'chart_path': snap.chart_path
                }
                for snap in snapshots
            ]
            
            return {
                'id': trade.id,
                'symbol': trade.instrument.symbol,
                'instrument_name': trade.instrument.name,
                'instrument_type': trade.instrument.type,
                'setup_name': trade.setup.name if trade.setup else None,
                'setup_description': trade.setup.description if trade.setup else None,
                'entry_date': trade.entry_date.isoformat(),
                'exit_date': trade.exit_date.isoformat() if trade.exit_date else None,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'r_planned': trade.r_planned,
                'r_actual': trade.r_actual,
                'pnl': self._calculate_pnl(trade),
                'days_held': trade.days_held,
                'regime_at_entry': trade.regime_at_entry,
                'status': trade.status,
                'notes': trade.notes,
                'snapshots': snapshot_data,
                'created_at': trade.created_at.isoformat(),
                'updated_at': trade.updated_at.isoformat()
            }
            
        except Exception as e:
            return {'error': f'Failed to get trade: {str(e)}'}
    
    def create_trade(self, trade_data: Dict) -> Dict:
        """Create a new trade"""
        try:
            # Validate required fields
            required_fields = ['symbol', 'entry_date', 'entry_price', 'size']
            for field in required_fields:
                if field not in trade_data:
                    return {'error': f'Missing required field: {field}'}
            
            # Get instrument
            instrument = Instrument.query.filter(
                Instrument.symbol == trade_data['symbol']
            ).first()
            
            if not instrument:
                return {'error': f'Instrument {trade_data["symbol"]} not found'}
            
            # Get setup if provided
            setup = None
            if trade_data.get('setup_name'):
                setup = Setup.query.filter(
                    Setup.name == trade_data['setup_name']
                ).first()
            
            # Parse entry date
            if isinstance(trade_data['entry_date'], str):
                entry_date = datetime.strptime(trade_data['entry_date'], '%Y-%m-%d').date()
            else:
                entry_date = trade_data['entry_date']
            
            # Create trade
            new_trade = Trade(
                instrument_id=instrument.id,
                setup_id=setup.id if setup else None,
                entry_date=entry_date,
                entry_price=float(trade_data['entry_price']),
                size=float(trade_data['size']),
                r_planned=trade_data.get('r_planned'),
                regime_at_entry=trade_data.get('regime_at_entry'),
                notes=trade_data.get('notes', '')
            )
            
            db.session.add(new_trade)
            db.session.commit()
            
            return {
                'success': True,
                'trade_id': new_trade.id,
                'message': f'Trade created for {trade_data["symbol"]}',
                'created_at': new_trade.created_at.isoformat()
            }
            
        except Exception as e:
            db.session.rollback()
            return {'error': f'Failed to create trade: {str(e)}'}
    
    def update_trade(self, trade_id: int, update_data: Dict) -> Dict:
        """Update an existing trade"""
        try:
            trade = Trade.query.filter(Trade.id == trade_id).first()
            
            if not trade:
                return {'error': 'Trade not found'}
            
            # Update allowed fields
            allowed_fields = [
                'exit_date', 'exit_price', 'r_actual', 'notes', 'status'
            ]
            
            for field in allowed_fields:
                if field in update_data:
                    if field == 'exit_date' and update_data[field]:
                        if isinstance(update_data[field], str):
                            setattr(trade, field, 
                                   datetime.strptime(update_data[field], '%Y-%m-%d').date())
                        else:
                            setattr(trade, field, update_data[field])
                    else:
                        setattr(trade, field, update_data[field])
            
            # If closing trade, calculate R actual if not provided
            if update_data.get('status') == 'closed' and trade.exit_price and not trade.r_actual:
                if trade.r_planned:
                    price_change = trade.exit_price - trade.entry_price
                    expected_change = trade.entry_price * (trade.r_planned / 100)
                    trade.r_actual = (price_change / expected_change) * trade.r_planned if expected_change != 0 else 0
            
            trade.updated_at = datetime.utcnow()
            db.session.commit()
            
            return {
                'success': True,
                'message': 'Trade updated successfully',
                'updated_at': trade.updated_at.isoformat()
            }
            
        except Exception as e:
            db.session.rollback()
            return {'error': f'Failed to update trade: {str(e)}'}
    
    def add_snapshot(self, trade_id: int, snapshot_data: Dict) -> Dict:
        """Add a snapshot to a trade"""
        try:
            trade = Trade.query.filter(Trade.id == trade_id).first()
            
            if not trade:
                return {'error': 'Trade not found'}
            
            # Parse snapshot date
            if isinstance(snapshot_data['date'], str):
                snapshot_date = datetime.strptime(snapshot_data['date'], '%Y-%m-%d').date()
            else:
                snapshot_date = snapshot_data['date']
            
            new_snapshot = Snapshot(
                trade_id=trade_id,
                date=snapshot_date,
                price=float(snapshot_data['price']),
                notes=snapshot_data.get('notes', ''),
                chart_path=snapshot_data.get('chart_path')
            )
            
            db.session.add(new_snapshot)
            db.session.commit()
            
            return {
                'success': True,
                'snapshot_id': new_snapshot.id,
                'message': 'Snapshot added successfully'
            }
            
        except Exception as e:
            db.session.rollback()
            return {'error': f'Failed to add snapshot: {str(e)}'}
    
    def get_trade_statistics(self) -> Dict:
        """Get trade statistics summary"""
        try:
            # Basic counts
            total_trades = Trade.query.count()
            open_trades = Trade.query.filter(Trade.status == 'open').count()
            closed_trades = Trade.query.filter(Trade.status == 'closed').count()
            
            # Closed trades analysis
            closed_trades_query = Trade.query.filter(Trade.status == 'closed')
            winning_trades = closed_trades_query.filter(Trade.r_actual > 0).count()
            losing_trades = closed_trades_query.filter(Trade.r_actual < 0).count()
            
            win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0
            
            # Average R values
            avg_r_planned = db.session.query(db.func.avg(Trade.r_planned)).filter(
                Trade.r_planned.isnot(None)).scalar() or 0
            avg_r_actual = db.session.query(db.func.avg(Trade.r_actual)).filter(
                Trade.r_actual.isnot(None)).scalar() or 0
            
            # Recent trades (last 30 days)
            thirty_days_ago = datetime.now().date() - timedelta(days=30)
            recent_trades = Trade.query.filter(
                Trade.entry_date >= thirty_days_ago
            ).count()
            
            return {
                'total_trades': total_trades,
                'open_trades': open_trades,
                'closed_trades': closed_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'avg_r_planned': round(avg_r_planned, 2),
                'avg_r_actual': round(avg_r_actual, 2),
                'recent_trades_30d': recent_trades,
                'calculated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Failed to get trade statistics: {str(e)}'}
    
    def _calculate_pnl(self, trade: Trade) -> Optional[float]:
        """Calculate P&L for a trade"""
        if trade.exit_price and trade.entry_price:
            return (trade.exit_price - trade.entry_price) * trade.size
        return None