"""
Journal Service Layer

Comprehensive trade journal operations with performance analysis and correlation tracking.
Provides web-friendly interface for trade management and portfolio overview.
"""

import sys
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy import and_, or_, desc, asc

# Add parent directory to import CLI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import db, Trade, Instrument, Setup, Snapshot, Correlation, PriceData

class JournalService:
    """Service for comprehensive trade journal operations"""
    
    def get_all_trades(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get trades with optional status filter"""
        try:
            query = Trade.query.join(Instrument).outerjoin(Setup)
            
            if status:
                query = query.filter(Trade.status == status)
            
            trades = query.order_by(desc(Trade.entry_date)).limit(limit).all()
            
            return [self._trade_to_dict(trade) for trade in trades]
            
        except Exception as e:
            return [{'error': f'Failed to get trades: {str(e)}'}]
    
    def get_open_positions(self) -> Dict:
        """Get summary of all open positions with portfolio metrics"""
        try:
            open_trades = Trade.query.filter(Trade.status == 'open').join(Instrument).all()
            
            positions = []
            total_value = 0
            total_pnl = 0
            sectors = {}
            
            for trade in open_trades:
                trade_dict = self._trade_to_dict(trade)
                positions.append(trade_dict)
                
                # Calculate position value
                position_value = trade.entry_price * trade.size
                total_value += position_value
                
                # Track P&L
                if trade.pnl_dollar:
                    total_pnl += trade.pnl_dollar
                
                # Track sector exposure
                sector = trade.instrument.sector or 'Unknown'
                if sector not in sectors:
                    sectors[sector] = {'count': 0, 'value': 0}
                sectors[sector]['count'] += 1
                sectors[sector]['value'] += position_value
            
            return {
                'positions': positions,
                'summary': {
                    'total_positions': len(positions),
                    'total_value': total_value,
                    'total_pnl': total_pnl,
                    'sector_breakdown': sectors
                }
            }
            
        except Exception as e:
            return {'error': f'Failed to get open positions: {str(e)}'}
    
    def add_trade(self, trade_data: Dict) -> Dict:
        """Add a new trade to the journal"""
        try:
            # Validate required fields
            required_fields = ['symbol', 'entry_date', 'entry_price', 'size']
            for field in required_fields:
                if field not in trade_data:
                    return {'success': False, 'error': f'Missing required field: {field}'}
            
            # Get or create instrument
            instrument = Instrument.query.filter_by(symbol=trade_data['symbol']).first()
            if not instrument:
                return {'success': False, 'error': f'Instrument {trade_data["symbol"]} not found'}
            
            # Get setup if provided
            setup = None
            if 'setup_name' in trade_data:
                setup = Setup.query.filter_by(name=trade_data['setup_name']).first()
            
            # Parse entry date
            if isinstance(trade_data['entry_date'], str):
                entry_date = datetime.strptime(trade_data['entry_date'], '%Y-%m-%d').date()
            else:
                entry_date = trade_data['entry_date']
            
            # Calculate r_planned if not provided but we have entry, stop, and target
            r_planned = float(trade_data.get('r_planned', 0)) or None
            entry_price = float(trade_data['entry_price'])
            stop_loss = float(trade_data.get('stop_loss', 0)) or None
            target_price = float(trade_data.get('target_price', 0)) or None
            
            if not r_planned and stop_loss and target_price and entry_price:
                r_planned = self._calculate_risk_reward_ratio(entry_price, stop_loss, target_price)
            
            # Create new trade
            trade = Trade(
                instrument_id=instrument.id,
                setup_id=setup.id if setup else None,
                entry_date=entry_date,
                entry_price=entry_price,
                size=float(trade_data['size']),
                stop_loss=stop_loss,
                target_price=target_price,
                r_planned=r_planned,
                commission=float(trade_data.get('commission', 0)),
                notes=trade_data.get('notes', ''),
                entry_reason=trade_data.get('entry_reason', ''),
                regime_at_entry=trade_data.get('regime_at_entry', ''),
                status='open'
            )
            
            db.session.add(trade)
            db.session.commit()
            
            return {
                'success': True, 
                'trade_id': trade.id,
                'message': f'Trade added successfully: {instrument.symbol}'
            }
            
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'error': f'Failed to add trade: {str(e)}'}
    
    def update_trade(self, trade_id: int, trade_data: Dict) -> Dict:
        """Update an existing trade"""
        try:
            trade = Trade.query.get(trade_id)
            if not trade:
                return {'success': False, 'error': 'Trade not found'}
            
            # Update fields if provided
            updatable_fields = [
                'entry_price', 'exit_price', 'size', 'stop_loss', 'target_price',
                'r_planned', 'r_actual', 'commission', 'notes', 'entry_reason', 
                'exit_reason', 'status'
            ]
            
            for field in updatable_fields:
                if field in trade_data:
                    value = trade_data[field]
                    if field in ['entry_price', 'exit_price', 'size', 'stop_loss', 'target_price', 'r_planned', 'r_actual', 'commission']:
                        value = float(value) if value else None
                    setattr(trade, field, value)
            
            # Handle entry date
            if 'entry_date' in trade_data:
                if isinstance(trade_data['entry_date'], str):
                    trade.entry_date = datetime.strptime(trade_data['entry_date'], '%Y-%m-%d').date()
                else:
                    trade.entry_date = trade_data['entry_date']
            
            # Handle exit date
            if 'exit_date' in trade_data:
                if isinstance(trade_data['exit_date'], str):
                    trade.exit_date = datetime.strptime(trade_data['exit_date'], '%Y-%m-%d').date()
                else:
                    trade.exit_date = trade_data['exit_date']
            
            # Auto-calculate r_planned if entry, stop, and target are all present but r_planned is not manually set
            if (not trade_data.get('r_planned') and 
                trade.entry_price and trade.stop_loss and trade.target_price):
                trade.r_planned = self._calculate_risk_reward_ratio(trade.entry_price, trade.stop_loss, trade.target_price)
            
            # Update P&L if trade is closed
            if trade.status == 'closed' and trade.exit_price:
                trade.update_pnl()
            
            trade.updated_at = datetime.utcnow()
            db.session.commit()
            
            return {
                'success': True,
                'message': f'Trade {trade_id} updated successfully'
            }
            
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'error': f'Failed to update trade: {str(e)}'}
    
    def close_trade(self, trade_id: int, exit_price: float, exit_date: Optional[date] = None, exit_reason: str = '') -> Dict:
        """Close a trade with exit details"""
        try:
            trade = Trade.query.get(trade_id)
            if not trade:
                return {'success': False, 'error': 'Trade not found'}
            
            if trade.status != 'open':
                return {'success': False, 'error': 'Trade is not open'}
            
            # Parse exit_date if it's a string
            if exit_date:
                if isinstance(exit_date, str):
                    trade.exit_date = datetime.strptime(exit_date, '%Y-%m-%d').date()
                else:
                    trade.exit_date = exit_date
            else:
                trade.exit_date = date.today()
            
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.status = 'closed'
            
            # Calculate final P&L
            trade.update_pnl()
            
            # Calculate R actual if planned R exists
            if trade.r_planned and trade.entry_price:
                if trade.stop_loss:
                    risk_per_share = abs(trade.entry_price - trade.stop_loss)
                    actual_per_share = exit_price - trade.entry_price
                    trade.r_actual = actual_per_share / risk_per_share if risk_per_share > 0 else 0
            
            trade.updated_at = datetime.utcnow()
            db.session.commit()
            
            return {
                'success': True,
                'pnl_dollar': trade.pnl_dollar,
                'pnl_percent': trade.pnl_percent,
                'r_actual': trade.r_actual,
                'message': f'Trade closed successfully'
            }
            
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'error': f'Failed to close trade: {str(e)}'}
    
    def get_trade_correlations(self, days_back: int = 30) -> List[Dict]:
        """Get correlation data for recent trades"""
        try:
            # Get recent trades
            cutoff_date = date.today() - timedelta(days=days_back)
            recent_trades = Trade.query.filter(Trade.entry_date >= cutoff_date).join(Instrument).all()
            
            if len(recent_trades) < 2:
                return []
            
            # Get unique symbols from recent trades
            symbols = list(set([trade.instrument.symbol for trade in recent_trades]))
            
            # Get correlation data
            correlations = []
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    # Get instruments
                    inst1 = Instrument.query.filter_by(symbol=symbol1).first()
                    inst2 = Instrument.query.filter_by(symbol=symbol2).first()
                    
                    # Get latest correlation
                    corr = Correlation.query.filter(
                        or_(
                            and_(Correlation.instrument1_id == inst1.id, Correlation.instrument2_id == inst2.id),
                            and_(Correlation.instrument1_id == inst2.id, Correlation.instrument2_id == inst1.id)
                        )
                    ).order_by(desc(Correlation.date_calculated)).first()
                    
                    if corr:
                        correlations.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation_30d': corr.correlation_30d,
                            'correlation_90d': corr.correlation_90d,
                            'date_calculated': corr.date_calculated.isoformat()
                        })
            
            return correlations
            
        except Exception as e:
            return [{'error': f'Failed to get correlations: {str(e)}'}]
    
    def get_performance_metrics(self, days_back: int = 365) -> Dict:
        """Calculate performance metrics for the portfolio"""
        try:
            # Get closed trades in the period
            cutoff_date = date.today() - timedelta(days=days_back)
            closed_trades = Trade.query.filter(
                and_(
                    Trade.status == 'closed',
                    Trade.exit_date >= cutoff_date
                )
            ).all()
            
            if not closed_trades:
                return {'total_trades': 0}
            
            # Calculate metrics
            total_trades = len(closed_trades)
            winning_trades = [t for t in closed_trades if (t.pnl_dollar or 0) > 0]
            losing_trades = [t for t in closed_trades if (t.pnl_dollar or 0) < 0]
            
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            
            total_pnl = sum([t.pnl_dollar or 0 for t in closed_trades])
            avg_win = sum([t.pnl_dollar for t in winning_trades]) / len(winning_trades) if winning_trades else 0
            avg_loss = sum([t.pnl_dollar for t in losing_trades]) / len(losing_trades) if losing_trades else 0
            
            profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades and avg_loss != 0 else float('inf')
            
            # R multiples
            r_actuals = [t.r_actual for t in closed_trades if t.r_actual is not None]
            avg_r = sum(r_actuals) / len(r_actuals) if r_actuals else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_r_multiple': avg_r,
                'period_days': days_back
            }
            
        except Exception as e:
            return {'error': f'Failed to calculate performance metrics: {str(e)}'}
    
    def _get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent price for a symbol from the data cache"""
        try:
            latest_price_data = PriceData.query.filter(
                PriceData.symbol == symbol
            ).order_by(PriceData.date.desc()).first()
            
            return latest_price_data.close if latest_price_data else None
        except Exception:
            return None
    
    def _calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, target_price: float) -> float:
        """Calculate risk/reward ratio from entry, stop loss, and target prices"""
        try:
            if not all([entry_price, stop_loss, target_price]):
                return None
            
            risk_per_share = abs(entry_price - stop_loss)
            reward_per_share = abs(target_price - entry_price)
            
            if risk_per_share == 0:
                return None
            
            return round(reward_per_share / risk_per_share, 2)
        except (ValueError, ZeroDivisionError):
            return None
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade model to dictionary for API responses"""
        # Get current price for open positions or use exit price for closed ones
        current_price = None
        if trade.status == 'open':
            current_price = self._get_latest_price(trade.instrument.symbol)
        else:
            current_price = trade.exit_price
        
        # Calculate current metrics
        original_position_value = trade.entry_price * trade.size if trade.entry_price and trade.size else 0
        current_position_value = current_price * trade.size if current_price and trade.size else 0
        current_pnl_dollar = (current_price - trade.entry_price) * trade.size if current_price and trade.entry_price and trade.size else 0
        current_pnl_percent = ((current_price - trade.entry_price) / trade.entry_price) * 100 if current_price and trade.entry_price else 0
        
        # Calculate current R multiple
        current_r_multiple = None
        if current_price and trade.entry_price and trade.stop_loss:
            risk_per_share = abs(trade.entry_price - trade.stop_loss)
            actual_move_per_share = current_price - trade.entry_price
            current_r_multiple = actual_move_per_share / risk_per_share if risk_per_share > 0 else 0
        
        return {
            'id': trade.id,
            'symbol': trade.instrument.symbol,
            'instrument_name': trade.instrument.name,
            'instrument_type': trade.instrument.type,
            'sector': trade.instrument.sector,
            'setup_name': trade.setup.name if trade.setup else None,
            'entry_date': trade.entry_date.isoformat(),
            'exit_date': trade.exit_date.isoformat() if trade.exit_date else None,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'current_price': current_price,
            'stop_loss': trade.stop_loss,
            'target_price': trade.target_price,
            'size': trade.size,
            'r_planned': trade.r_planned,
            'r_actual': trade.r_actual,
            'r_current': round(current_r_multiple, 2) if current_r_multiple is not None else None,
            'original_position_value': original_position_value,
            'current_position_value': current_position_value,
            'current_pnl_dollar': current_pnl_dollar,
            'current_pnl_percent': current_pnl_percent,
            'pnl_dollar': trade.pnl_dollar,
            'pnl_percent': trade.pnl_percent,
            'commission': trade.commission,
            'notes': trade.notes,
            'entry_reason': trade.entry_reason,
            'exit_reason': trade.exit_reason,
            'regime_at_entry': trade.regime_at_entry,
            'status': trade.status,
            'days_held': trade.days_held,
            'created_at': trade.created_at.isoformat() if trade.created_at else None,
            'updated_at': trade.updated_at.isoformat() if trade.updated_at else None
        }